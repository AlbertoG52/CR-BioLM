"""Scientific synonyms -> species of the Manual catalog, from the GBIF backbone.

For each catalog species with a GBIF key (taxa_gbif.json), the synonyms of that key
are stored, plus the GBIF accepted name when it differs from the Manual's name.
Names that are already a catalog species are skipped.

The synonyms are GBIF's, not the Manual's: they may disagree with the Manual.

Run:  python -m mpcr_rag.evidence.synonyms fetch
      python -m mpcr_rag.evidence.synonyms resolve "Bombax pentandrum"
      python -m mpcr_rag.evidence.synonyms report
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pygbif import species as gbif_species

from ..store import pg_store
from . import taxa
from .vernacular import norm

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mpcr.synonym (
    name_norm     text NOT NULL,        -- lowercase, unaccented, for lookup
    synonym       text NOT NULL,        -- as published by GBIF
    species       text NOT NULL,        -- Manual name
    source        text NOT NULL,        -- GBIF_SYNONYM | GBIF_ACCEPTED
    source_detail text,                 -- GBIF taxon key
    fetched       date NOT NULL DEFAULT current_date,
    PRIMARY KEY (name_norm, species)
);
CREATE INDEX IF NOT EXISTS synonym_species ON mpcr.synonym (species);
"""


def _binomial(name: str | None) -> bool:
    return bool(name) and len(name.split()) == 2


def from_gbif(taxon_key: int, attempts: int = 4) -> list[str] | None:
    """Synonyms of a GBIF key. None means a network failure."""
    out, offset = [], 0
    while True:
        for i in range(attempts):
            try:
                page = gbif_species.name_usage(key=taxon_key, data="synonyms",
                                               limit=500, offset=offset)
                break
            except Exception:
                time.sleep(2 * (i + 1))
        else:
            return None
        out += [r.get("canonicalName") for r in page.get("results", [])]
        if page.get("endOfRecords", True):
            return out
        offset += 500


def fetch(workers: int = 4, verbose: bool = True) -> dict:
    tx = taxa.load()
    catalog = {norm(s) for s in tx}
    conn = pg_store.connect()
    cur = conn.cursor()
    cur.execute(_SCHEMA)
    conn.commit()
    todo = {sp: t for sp, t in tx.items() if t.get("taxon_key")}
    stats = {"species": 0, "rows": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(from_gbif, t["taxon_key"]): sp for sp, t in todo.items()}
        for fut in as_completed(futures):
            sp = futures[fut]
            names = fut.result()
            stats["species"] += 1
            if names is None:
                stats["failed"] += 1
                continue
            key = todo[sp]["taxon_key"]
            rows = [(n, "GBIF_SYNONYM") for n in names]
            rows.append((todo[sp].get("accepted_name"), "GBIF_ACCEPTED"))
            for name, source in rows:
                if not _binomial(name) or norm(name) in catalog:
                    continue
                cur.execute("""INSERT INTO mpcr.synonym
                    (name_norm, synonym, species, source, source_detail)
                    VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                            (norm(name), name, sp, source, f"GBIF taxon {key}"))
                stats["rows"] += cur.rowcount
            conn.commit()
            if verbose and stats["species"] % 250 == 0:
                print(f"  [{stats['species']}/{len(todo)}] rows={stats['rows']}", flush=True)
    conn.close()
    return stats


def resolve(name: str, *, conn=None) -> list[dict]:
    """Catalog species that carry this synonym."""
    own = conn is None
    conn = conn or pg_store.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT s.species, f.ficha->>'family', s.source
                           FROM mpcr.synonym s
                           JOIN mpcr.fichas f ON f.species = s.species
                           WHERE s.name_norm = %s ORDER BY s.species""", (norm(name),))
            rows = cur.fetchall()
    finally:
        if own:
            conn.close()
    return [{"species": s, "family": fam, "source": src} for s, fam, src in rows]


def all_synonyms(conn=None) -> list[dict]:
    """Every stored synonym with its catalog species."""
    own = conn is None
    conn = conn or pg_store.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT s.synonym, s.species, f.ficha->>'family', s.source
                           FROM mpcr.synonym s
                           JOIN mpcr.fichas f ON f.species = s.species
                           ORDER BY s.name_norm, s.species""")
            rows = cur.fetchall()
    finally:
        if own:
            conn.close()
    return [{"synonym": n, "species": s, "family": fam, "source": src}
            for n, s, fam, src in rows]


def report(conn=None) -> None:
    conn = conn or pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT name_norm), count(DISTINCT species) "
                    "FROM mpcr.synonym")
        rows, names, sp = cur.fetchone()
        print(f"synonym: {rows} rows, {names} distinct names, {sp} species")
        cur.execute("SELECT source, count(*) FROM mpcr.synonym GROUP BY 1 ORDER BY 2 DESC")
        print("  by source:", dict(cur.fetchall()))
        cur.execute("""SELECT count(*) FROM (SELECT name_norm FROM mpcr.synonym
                       GROUP BY 1 HAVING count(*) > 1) x""")
        print(f"  names shared by several species: {cur.fetchone()[0]}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("resolve")
    r.add_argument("name")
    sub.add_parser("report")
    a = ap.parse_args()

    if a.cmd == "fetch":
        print(json.dumps(fetch(workers=a.workers), ensure_ascii=False))
    elif a.cmd == "resolve":
        for c in resolve(a.name):
            print(f"  {c['species']:32} {(c['family'] or ''):18} via {c['source']}")
    else:
        report()
