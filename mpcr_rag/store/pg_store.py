"""pgvector store: filters + vectors in one SQL query, synced from SQLite.

SQLite stays the source of truth (ingest writes it; the handoff bundle ships it).
This store is a derived query index, the role Pinecone had: rebuild it any time with
``sync_from_sqlite()``, which only re-embeds text whose content changed.

Chunks per species (``mpcr.ficha_chunks.section``, ``ordinal``):
  distribution  the distribution paragraph, always ONE chunk — the text the BIP index
                embedded, so the backends stay comparable (p95 = 193 tokens)
  description   morphology + discussion, SPLIT into ~300-token windows. e5 truncates
                at 512 tokens and 25% of descriptions exceeded it, so the fruits and
                the diagnostic notes at the end of a long entry were not in the vector
                at all. 6,946 species -> 13,641 description chunks, 0% truncated.

A species is scored by its BEST matching chunk and returned once (the query fetches
top_k * OVERFETCH chunks, then groups).

Search is EXACT (sequential scan), deliberately: at ~20k vectors it takes
milliseconds (63 ms distribution, 131 ms description), and an approximate HNSW index
post-filters in pgvector 0.6, so a restrictive metadata filter can return fewer than
top_k hits. Add an index only if the catalog grows by an order of magnitude.

Run:  python -m mpcr_rag.store.pg_store [status|start|stop|sync|demo]
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time

import psycopg2
import psycopg2.extras

from .. import config
from ..schema import Ficha

SECTIONS = ("distribution", "description")

_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS mpcr;

CREATE TABLE IF NOT EXISTS mpcr.fichas (
    vector_id        text PRIMARY KEY,
    species          text NOT NULL,
    genus            text,
    family           text,
    volume           text,
    pages            text,
    elev_min         int,
    elev_max         int,
    elev_min_eff     int,          -- outlier-extended range, used for membership
    elev_max_eff     int,
    habits           text[] NOT NULL DEFAULT '{{}}',
    vertientes       text[] NOT NULL DEFAULT '{{}}',
    regions          text[] NOT NULL DEFAULT '{{}}',
    forest_types     text[] NOT NULL DEFAULT '{{}}',
    flowering_months int[]  NOT NULL DEFAULT '{{}}',
    endemic          boolean NOT NULL DEFAULT false,
    ficha            jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS fichas_family  ON mpcr.fichas (family);
CREATE INDEX IF NOT EXISTS fichas_habits  ON mpcr.fichas USING gin (habits);
CREATE INDEX IF NOT EXISTS fichas_regions ON mpcr.fichas USING gin (regions);

CREATE TABLE IF NOT EXISTS mpcr.ficha_chunks (
    vector_id    text NOT NULL REFERENCES mpcr.fichas ON DELETE CASCADE,
    section      text NOT NULL,
    ordinal      int  NOT NULL DEFAULT 0,   -- long sections are split; 0 = whole section
    text         text NOT NULL,
    content_hash text NOT NULL,    -- model + prefix + text; unchanged => not re-embedded
    embedding    vector({config.EMBED_DIM}) NOT NULL,
    PRIMARY KEY (vector_id, section, ordinal)
);

CREATE TABLE IF NOT EXISTS mpcr.meta (key text PRIMARY KEY, value text);
"""

_server = None   # keep the embedded server object alive for the process lifetime


def _running():
    from pgserver.utils import PostmasterInfo
    info = PostmasterInfo.read_from_pgdata(config.PGDATA)
    return info if info is not None and info.is_running() else None


def start() -> None:
    """Start the embedded server on the fixed port (no-op if already running).

    A brand-new data directory is initialized by pgserver first (which starts it on a
    random port); it is then restarted on ``config.PG_PORT``.
    """
    import pgserver
    from pgserver._commands import pg_ctl
    config.PGDATA.parent.mkdir(parents=True, exist_ok=True)
    if not (config.PGDATA / "PG_VERSION").exists():
        pgserver.get_server(config.PGDATA, cleanup_mode=None)
    info = _running()
    if info is not None and info.port == config.PG_PORT:
        return
    if info is not None:
        stop()
    pg_ctl(["-w", "-o", '-h "127.0.0.1"', "-o", f"-p {config.PG_PORT}",
            "-l", str(config.PGDATA / "log"), "start"], pgdata=config.PGDATA)


def stop() -> None:
    from pgserver._commands import pg_ctl
    if _running() is not None:
        pg_ctl(["-w", "-m", "fast", "stop"], pgdata=config.PGDATA)


def _url() -> str:
    global _server
    if config.PG_URL:
        return config.PG_URL
    if _server is None:
        import pgserver
        start()
        # Attaches to the server started above. cleanup_mode=None leaves it running
        # after this process exits, so the MCP server and scripts share one instance
        # instead of each paying a cold start.
        _server = pgserver.get_server(config.PGDATA, cleanup_mode=None)
    return _server.get_uri()


def connect():
    conn = psycopg2.connect(_url())
    with conn, conn.cursor() as cur:
        cur.execute(_SCHEMA)
        # Pre-ordinal stores hold one chunk per section, with long descriptions
        # truncated by the model. Drop them; the next sync re-embeds.
        cur.execute("""SELECT 1 FROM information_schema.columns
                       WHERE table_schema='mpcr' AND table_name='ficha_chunks'
                         AND column_name='ordinal'""")
        if cur.fetchone() is None:
            cur.execute("DROP TABLE mpcr.ficha_chunks")
            cur.execute(_SCHEMA)
    return conn


def _vec(v) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in v) + "]"


# e5 truncates at 512 tokens. Measured on this corpus, Spanish botanical prose runs
# about 3.07 characters per token, and 25% of description sections exceeded the limit:
# everything after the cut (often the fruits and the diagnostic notes) was invisible to
# search. Descriptions are therefore split on sentence boundaries into windows of
# ~300 tokens with one sentence of overlap, so a character group spanning a boundary is
# still embedded intact somewhere.
CHARS_PER_TOKEN = 3.07
CHUNK_TOKENS = 300
CHUNK_CHARS = int(CHUNK_TOKENS * CHARS_PER_TOKEN)      # ~920 characters
# chunks fetched per requested hit before grouping to one row per species:
# a species can occupy several of the nearest chunks.
OVERFETCH = 10

_SENTENCE = re.compile(r"(?<=[.;])\s+")


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Sentence-aware windows of at most ``max_chars``, overlapping by one sentence.
    A single sentence longer than the window is cut on whitespace rather than dropped."""
    if len(text) <= max_chars:
        return [text]
    pieces, out, cur = _SENTENCE.split(text), [], ""
    for s in pieces:
        while len(s) > max_chars:                       # pathological run-on sentence
            cut = s.rfind(" ", 0, max_chars) or max_chars
            out.append(s[:cut].strip())
            s = s[cut:].lstrip()
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = f"{cur} {s}"
        else:
            out.append(cur)
            cur = f"{out[-1].rsplit('. ', 1)[-1]} {s}".strip()[-max_chars:]   # 1-sentence overlap
    if cur:
        out.append(cur)
    return [c for c in (x.strip() for x in out) if c]


def chunk_texts(f: Ficha) -> list[tuple[str, int, str]]:
    """(section, ordinal, text) to embed. ``distribution`` stays ONE chunk — it mirrors
    the Pinecone record so the backends remain comparable, and it never exceeds the
    token limit (p95 = 193 tokens). ``description`` is split when long."""
    out = [("distribution", 0, f.distribution_paragraph or f.species)]
    desc = " ".join(s for s in (f.morphology, f.discussion) if s)
    if desc:
        out += [("description", i, part) for i, part in enumerate(split_text(desc))]
    return out


def _hash(text: str) -> str:
    return hashlib.sha256(f"{config.LOCAL_EMBED_MODEL}|passage|{text}".encode()).hexdigest()


def _eff(f: Ficha) -> tuple[int | None, int | None]:
    if f.elev_min is None:
        return None, None
    lo = f.elev_outlier_min if f.elev_outlier_min is not None else f.elev_min
    hi = f.elev_outlier_max if f.elev_outlier_max is not None else f.elev_max
    return lo, hi


def sync_from_sqlite(sqlite_path=None, *, batch_size: int = 16, verbose: bool = True) -> dict:
    """Mirror SQLite into Postgres and embed new or changed chunks only."""
    t0 = time.time()
    src = sqlite3.connect(sqlite_path or config.SQLITE_PATH)
    fichas = [Ficha.from_json(js) for (js,) in src.execute("SELECT ficha_json FROM fichas")]
    src.close()

    conn = connect()
    cur = conn.cursor()

    rows = []
    for f in fichas:
        lo, hi = _eff(f)
        rows.append((f.vector_id, f.species, f.genus, f.family, f.volume, f.pages,
                     f.elev_min, f.elev_max, lo, hi, f.habits, f.vertientes, f.regions,
                     f.forest_types, f.flowering_months, bool(f.endemic_cr), f.to_json()))
    psycopg2.extras.execute_values(cur, """
        INSERT INTO mpcr.fichas (vector_id, species, genus, family, volume, pages,
            elev_min, elev_max, elev_min_eff, elev_max_eff, habits, vertientes, regions,
            forest_types, flowering_months, endemic, ficha)
        VALUES %s
        ON CONFLICT (vector_id) DO UPDATE SET
            species=EXCLUDED.species, genus=EXCLUDED.genus, family=EXCLUDED.family,
            volume=EXCLUDED.volume, pages=EXCLUDED.pages, elev_min=EXCLUDED.elev_min,
            elev_max=EXCLUDED.elev_max, elev_min_eff=EXCLUDED.elev_min_eff,
            elev_max_eff=EXCLUDED.elev_max_eff, habits=EXCLUDED.habits,
            vertientes=EXCLUDED.vertientes, regions=EXCLUDED.regions,
            forest_types=EXCLUDED.forest_types, flowering_months=EXCLUDED.flowering_months,
            endemic=EXCLUDED.endemic, ficha=EXCLUDED.ficha
        """, rows, template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::text[],%s::text[],"
                            "%s::text[],%s::text[],%s::int[],%s,%s::jsonb)", page_size=500)

    ids = [f.vector_id for f in fichas]
    cur.execute("DELETE FROM mpcr.fichas WHERE NOT (vector_id = ANY(%s))", (ids,))
    removed = cur.rowcount
    conn.commit()

    cur.execute("SELECT vector_id, section, ordinal, content_hash FROM mpcr.ficha_chunks")
    have = {(v, s, o): h for v, s, o, h in cur.fetchall()}
    wanted = {(f.vector_id, s, o): t for f in fichas for s, o, t in chunk_texts(f)}

    stale = [k for k in have if k not in wanted]
    for vid, sec, ordinal in stale:
        cur.execute("DELETE FROM mpcr.ficha_chunks WHERE vector_id=%s AND section=%s "
                    "AND ordinal=%s", (vid, sec, ordinal))
    todo = [(k, t) for k, t in wanted.items() if have.get(k) != _hash(t)]
    conn.commit()

    if verbose:
        print(f"[pg] {len(fichas)} fichas ({removed} removed), {len(wanted)} chunks: "
              f"{len(todo)} to embed, {len(stale)} stale", flush=True)

    from . import embeddings
    step = 256
    for i in range(0, len(todo), step):
        part = todo[i:i + step]
        vecs = embeddings.embed_passages([t for _, t in part], batch_size=batch_size)
        psycopg2.extras.execute_values(cur, """
            INSERT INTO mpcr.ficha_chunks (vector_id, section, ordinal, text,
                                           content_hash, embedding)
            VALUES %s
            ON CONFLICT (vector_id, section, ordinal) DO UPDATE SET
                text=EXCLUDED.text, content_hash=EXCLUDED.content_hash,
                embedding=EXCLUDED.embedding
            """, [(vid, sec, o, t, _hash(t), _vec(v))
                  for ((vid, sec, o), t), v in zip(part, vecs)],
            template="(%s,%s,%s,%s,%s,%s::vector)")
        conn.commit()
        if verbose:
            done = min(i + step, len(todo))
            print(f"[pg] embedded {done}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)

    meta = {"embed_model": config.LOCAL_EMBED_MODEL, "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_fichas": str(len(fichas)), "n_chunks": str(len(wanted))}
    psycopg2.extras.execute_values(
        cur, "INSERT INTO mpcr.meta (key, value) VALUES %s "
             "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", list(meta.items()))
    conn.commit()
    conn.close()
    return {"fichas": len(fichas), "removed": removed, "chunks": len(wanted),
            "embedded": len(todo), "stale_deleted": len(stale), "seconds": round(time.time() - t0)}


def _where(constraints: dict) -> tuple[str, list]:
    """Same semantics as retriever.build_filter (Pinecone): list fields match when
    they contain the value; elevation is overlap on the outlier-extended range."""
    sql, args = [], []
    if constraints.get("habit"):
        sql.append("%s = ANY(f.habits)"); args.append(constraints["habit"])
    if constraints.get("elev_hi") is not None:
        sql.append("f.elev_min_eff IS NOT NULL AND f.elev_min_eff >= 0 AND f.elev_min_eff <= %s")
        args.append(constraints["elev_hi"])
    if constraints.get("elev_lo") is not None:
        sql.append("f.elev_max_eff >= %s"); args.append(constraints["elev_lo"])
    if constraints.get("vertiente"):
        sql.append("%s = ANY(f.vertientes)"); args.append(constraints["vertiente"])
    if constraints.get("region"):
        sql.append("%s = ANY(f.regions)"); args.append(constraints["region"])
    if constraints.get("forest_type"):
        sql.append("%s = ANY(f.forest_types)"); args.append(constraints["forest_type"])
    if constraints.get("family"):
        sql.append("f.family = %s"); args.append(constraints["family"])
    if constraints.get("flowering_month") is not None:
        sql.append("%s = ANY(f.flowering_months)"); args.append(int(constraints["flowering_month"]))
    if constraints.get("endemic") is not None:
        sql.append("f.endemic = %s"); args.append(bool(constraints["endemic"]))
    return (" AND ".join(sql) or "TRUE"), args


def search(query_text: str, *, top_k: int = 10, section: str = "distribution",
           conn=None, **constraints) -> list[dict]:
    """Semantic search with metadata filters. Hit dicts match pinecone_client.search:
    ``id``, ``score`` (cosine similarity) plus the filterable fields.

    A species can hold several chunks of one section (long descriptions are split), so
    it is scored by its BEST matching chunk and returned once."""
    from . import embeddings
    q = _vec(embeddings.embed_query(query_text))
    where, args = _where(constraints)
    own = conn is None
    conn = conn or connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                WITH hits AS (
                    SELECT c.vector_id, 1 - (c.embedding <=> %s::vector) AS score
                    FROM mpcr.ficha_chunks c JOIN mpcr.fichas f USING (vector_id)
                    WHERE c.section = %s AND {where}
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                )
                SELECT f.vector_id AS id, max(h.score) AS score,
                       f.species, f.genus, f.family, f.volume, f.pages,
                       f.elev_min, f.elev_max, f.habits, f.vertientes, f.regions,
                       f.forest_types, f.endemic, f.flowering_months
                FROM hits h JOIN mpcr.fichas f USING (vector_id)
                GROUP BY f.vector_id, f.species, f.genus, f.family, f.volume, f.pages,
                         f.elev_min, f.elev_max, f.habits, f.vertientes, f.regions,
                         f.forest_types, f.endemic, f.flowering_months
                ORDER BY score DESC
                LIMIT %s""", [q, section, *args, q, top_k * OVERFETCH, top_k])
            return [dict(r) for r in cur.fetchall()]
    finally:
        if own:
            conn.close()

def passages(species: str, query_text: str, *, top_k: int = 1,
             section: str | None = None, conn=None) -> list[dict]:
    """The chunks of ONE species ranked by similarity to ``query_text``.

    ``search`` ranks species; this ranks the text of a species already identified,
    to pick the passage that answers a question: the distribution paragraph, or a
    description window (morphology + discussion). ``section=None`` searches both.
    Hit dicts: ``species``, ``volume``, ``pages``, ``section``, ``ordinal``, ``text``,
    ``score`` (cosine similarity)."""
    from . import embeddings
    q = _vec(embeddings.embed_query(query_text))
    vector_id = species.strip().replace(" ", "_")
    sections = [section] if section else list(SECTIONS)
    own = conn is None
    conn = conn or connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT f.species, f.volume, f.pages, c.section, c.ordinal, c.text,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM mpcr.ficha_chunks c JOIN mpcr.fichas f USING (vector_id)
                WHERE c.vector_id = %s AND c.section = ANY(%s)
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s""", [q, vector_id, sections, q, top_k])
            return [dict(r) for r in cur.fetchall()]
    finally:
        if own:
            conn.close()

def status() -> dict:
    info = _running()
    out = {"running": info is not None, "pgdata": str(config.PGDATA)}
    if info is not None:
        out.update(host="127.0.0.1", port=info.port, user="postgres", password="(none)",
                   database="postgres", url=f"postgresql://postgres@127.0.0.1:{info.port}/postgres")
        conn = connect()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM mpcr.fichas")
            out["fichas"] = cur.fetchone()[0]
            cur.execute("SELECT section, count(*) FROM mpcr.ficha_chunks GROUP BY 1 ORDER BY 1")
            out["chunks"] = dict(cur.fetchall())
            cur.execute("SELECT key, value FROM mpcr.meta")
            out["meta"] = dict(cur.fetchall())
        conn.close()
    return out


def _demo() -> None:
    for sec, text in (("distribution", "bosque nuboso de altura en la Cordillera de Talamanca"),
                      ("description", "arbusto con flores moradas y hojas pubescentes")):
        print(f"\n--- {sec}: {text!r}")
        for h in search(text, top_k=5, section=sec):
            print(f"  {h['score']:.3f}  {h['species']:32} [{h['elev_min']}-{h['elev_max']} m]")
    print(json.dumps({"filtered": [h["species"] for h in search(
        "bosque de altura, robledales", top_k=5, elev_lo=2000, vertiente="Pacífico",
        endemic=True)]}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Local pgvector store")
    ap.add_argument("command", nargs="?", default="status",
                    choices=["status", "start", "stop", "sync", "demo"])
    cmd = ap.parse_args().command
    if cmd == "start":
        start()
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif cmd == "stop":
        stop()
        print("stopped")
    elif cmd == "sync":
        print(sync_from_sqlite())
    elif cmd == "demo":
        _demo()
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2))
