"""
MPCR-RAG MCP server.

Exposes the Manual de Plantas de Costa Rica catalog as MCP tools callable by
Claude Code or any MCP client.  Tools are grouped by cost:

  L0 — free, instant, SQLite only, no API keys required
  L1 — seconds, I/O cached to disk, no LLM (description search: local embeddings)
  L2 — requires OPENROUTER_API_KEY, plus PINECONE_API_KEY when
       MPCR_VECTOR_BACKEND=pinecone (the pgvector backend needs no vector key)

Every tool returns a provenance envelope:
  {value, source, citation, confidence, caveat}
  source     : one of MPCR | GBIF | DEM | SINAC
  confidence : one of exact | estimated | insufficient
  caveat     : non-empty string when the caller should distrust the value

Run locally:
  python -m mpcr_rag.mcp.server          (stdio, for Claude Code)
  mcp dev mpcr_rag/mcp/server.py         (MCP Inspector)
"""
from __future__ import annotations

import dataclasses
import json
import os
import base64
import dataclasses
import threading
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer

from mpcr_rag import config
from mpcr_rag.query import intent as intent_mod
from mpcr_rag.query import gbif_map
from mpcr_rag.query.retriever import filter_all
from mpcr_rag.store import local_store

server = MCPServer("mpcr-rag", version="0.1.0")

# ---------------------------------------------------------------------------
# Shared lazy state — connections opened once per process lifetime
# ---------------------------------------------------------------------------

"""_conn = None
_index = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = local_store.connect(config.SQLITE_PATH)
    return _conn"""

_local = threading.local()   # one SQLite connection per worker thread
_index = None


def _get_conn():
    """SQLite connection for the current thread.

    The MCP SDK runs sync tools in a worker-thread pool, and a sqlite3 connection
    can only be used by the thread that opened it, so each thread opens its own.
    """
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _local.conn = local_store.connect(config.SQLITE_PATH)
    return conn

def _get_index():
    """Pinecone index handle, or None (pgvector connects on demand)."""
    global _index
    if _index is None:
        if config.VECTOR_BACKEND != "pinecone" or not os.environ.get("PINECONE_API_KEY"):
            return None
        from mpcr_rag.store import pinecone_client as pc
        _index = pc.ensure_index()
    return _index


# ---------------------------------------------------------------------------
# Provenance envelope
# ---------------------------------------------------------------------------

def _envelope(
    value,
    *,
    source: str,
    citation: str = "",
    confidence: str = "exact",
    caveat: str = "",
) -> dict:
    return {
        "value": value,
        "source": source,
        "citation": citation,
        "confidence": confidence,
        "caveat": caveat,
    }

def _freshness(info) -> tuple[dict, str]:
    """RR_56: when a species' GBIF points were fetched, for the envelope value, and
    the caveat to add when a due refresh failed and the previous points are served."""
    fields = {"updated_at": info.updated_at, "stale": info.stale}
    caveat = ""
    if info.stale:
        caveat = (f"Could not refresh the GBIF points; these are the ones fetched on "
                  f"{info.updated_at}.")
    return fields, caveat


def _join(*caveats: str) -> str:
    return " ".join(c for c in caveats if c)

def _ficha_summary(f) -> dict:
    return {
        "species": f.species,
        "family": f.family,
        "volume": f.volume,
        "pages": f.pages,
        "elev_min": f.elev_min,
        "elev_max": f.elev_max,
        "vertientes": f.vertientes,
        "regions": f.regions,
        "habits": f.habits,
        "endemic_cr": f.endemic_cr,
        "forest_types": f.forest_types,
        "flowering_months": f.flowering_months,
    }


# ---------------------------------------------------------------------------
# L0 — free, instant, SQLite only
# ---------------------------------------------------------------------------

@server.tool()
def get_vocabulary() -> dict:
    """Return the controlled vocabulary for all filter fields.

    Cost: L0 — free, instant, SQLite only.  Call this first to know which
    values are valid for the filter parameters of search_species and
    semantic_search.

    Returns the full set of distinct values present in the catalog for:
    habits, forest_types, regions, vertientes, families.
    """
    vocab = intent_mod.load_vocab(_get_conn())
    n = _get_conn().execute("SELECT COUNT(*) FROM fichas").fetchone()[0]
    return _envelope(
        vocab,
        source="MPCR",
        citation=f"Manual de Plantas de Costa Rica (Hammel et al.) — {n:,} species",
        confidence="exact",
    )


@server.tool()
def search_species(
    habit: Optional[str] = None,
    elev_lo: Optional[int] = None,
    elev_hi: Optional[int] = None,
    vertiente: Optional[str] = None,
    region: Optional[str] = None,
    forest_type: Optional[str] = None,
    family: Optional[str] = None,
    flowering_month: Optional[int] = None,
    endemic: Optional[bool] = None,
) -> dict:
    """Exhaustive structured search over the Manual catalog (no Pinecone, no top-k cap).

    Cost: L0 — free, instant, SQLite only.  All filters are AND-combined;
    omit a parameter to skip that filter.  Use get_vocabulary() to see valid
    values for habit, vertiente, region, forest_type, family.

    Args:
        habit: growth form e.g. "árbol", "arbusto", "epifita", "hierba"
        elev_lo: minimum elevation overlap (metres)
        elev_hi: maximum elevation overlap (metres)
        vertiente: "Caribe" or "Pacífico"
        region: canonical botanical region name e.g. "Cordillera de Talamanca"
        forest_type: e.g. "muy húmedo", "pluvial", "nuboso", "páramo"
        family: plant family e.g. "Lauraceae", "Orchidaceae"
        flowering_month: 1–12
        endemic: true = Costa Rica endemics only

    Returns a provenance envelope whose value is
      {count: int, species: [{species, family, elev_min, elev_max, vertientes, volume, pages}]}
    """
    candidates = filter_all(
        _get_conn(),
        habit=habit,
        elev_lo=elev_lo,
        elev_hi=elev_hi,
        vertiente=vertiente,
        region=region,
        forest_type=forest_type,
        family=family,
        flowering_month=flowering_month,
        endemic=endemic,
    )
    summaries = [_ficha_summary(f) for f in candidates]
    return _envelope(
        {"count": len(candidates), "species": summaries},
        source="MPCR",
        citation="Manual de Plantas de Costa Rica — structured fields only",
        confidence="exact",
        caveat="" if candidates else "No species matched the given filters.",
    )


@server.tool()
def get_species(name: str) -> dict:
    """Fetch the full ficha for a single species by scientific name.

    Cost: L0 — free, instant, SQLite only.

    Args:
        name: scientific name e.g. "Peltogyne purpurea" or "Talamancaster minusculus"

    Returns a provenance envelope whose value is the full Ficha JSON including
    distribution_paragraph, elev_min/max, vertientes, regions, parks, habits,
    Tomo and página.
    """
    vector_id = name.strip().replace(" ", "_")
    f = local_store.get(_get_conn(), vector_id)
    if f is None:
        return _envelope(
            None,
            source="MPCR",
            confidence="insufficient",
            caveat=f"Species '{name}' not found in catalog. Check spelling or use search_species().",
        )
    value = {
        **_ficha_summary(f),
        "distribution_paragraph": f.distribution_paragraph,
        "morphology": f.morphology,
        "discussion": f.discussion,
        "genus_description": f.genus_description,
        "common_names": f.common_names,
    }
    return _envelope(
        value,
        source="MPCR",
        citation=f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages}",
        confidence="exact",
        caveat=("genus_description is GENUS-level text: cite it as such, never as a "
                "statement about this species." if f.genus_description else ""),
    )


# ---------------------------------------------------------------------------
# L1 — seconds, I/O cached, no LLM
# ---------------------------------------------------------------------------

@server.tool()
def parse_distribution_text(
    habitat_raw: str,
    geographic_notes: str,
    species: str,
) -> dict:
    """Parse raw Manual distribution text into a structured DistributionFicha.

    Cost: L1 — pure computation, no network, no LLM.  The parser is heuristic
    (regex + gazetteer), not ML — confidence is "estimated".

    Args:
        habitat_raw: text before the elevation's semicolon boundary
          e.g. "Bosque muy húmedo y pluvial, 0–800(–1200) m"
        geographic_notes: text after the semicolon boundary
          e.g. "vert. Caribe, Cord. de Talamanca"
        species: scientific name (used for GBIF taxon resolution in the parser)

    Returns a provenance envelope whose value is the DistributionFicha as JSON.
    """
    from utils.distribution_map.parser import build_ficha
    df = build_ficha(
        habitat_raw=habitat_raw,
        geographic_notes=geographic_notes,
        species=species,
    )
    value = dataclasses.asdict(df)
    return _envelope(
        value,
        source="MPCR",
        citation="geo_parser + gazetteer (entities.csv, José Araya 2026)",
        confidence="estimated",
        caveat="Parser is rule-based. Check unresolved_tokens for place names it could not match.",
    )


@server.tool()
def get_occurrences(
    species: str,
    elev_lo: Optional[int] = None,
    elev_hi: Optional[int] = None,
    vertiente: Optional[str] = None,
    region: Optional[str] = None,
) -> dict:
    """Fetch and filter Costa Rica GBIF occurrences for a species.

    Cost: L1 — first call fetches from GBIF API and caches to disk; subsequent
    calls are instant.  No LLM, no Pinecone.

    Args:
        species: scientific name
        elev_lo: minimum elevation filter (metres)
        elev_hi: maximum elevation filter (metres)
        vertiente: "Caribe" or "Pacífico"
        region: botanical region name

    Returns a provenance envelope whose value is:
      {n, bbox, elev_mean, elev_std, elev_min_obs, elev_max_obs, region_breakdown}
    """
    pts = gbif_map.get_points(species)
    if pts.empty:
        return _envelope(
            {"n": 0},
            source="GBIF",
            citation="GBIF occurrence API — Costa Rica (hasCoordinate=true, hasGeospatialIssue=false)",
            confidence="insufficient",
            caveat=f"No GBIF occurrences found for '{species}' in Costa Rica.",
        )

    filtered = gbif_map.filter_points(pts, elev_lo=elev_lo, elev_hi=elev_hi,
                                      vertiente=vertiente, region=region)
    n = len(filtered)
    caveat = ""
    if n == 0:
        caveat = "Filters excluded all occurrences. Try relaxing elev_lo/elev_hi or vertiente."

    elev_col = filtered["Altitud"] if "Altitud" in filtered.columns else None
    value: dict = {"n": n}
    if elev_col is not None and not elev_col.dropna().empty:
        value["elev_mean"] = round(float(elev_col.mean()), 1)
        value["elev_std"] = round(float(elev_col.std()), 1)
        value["elev_min_obs"] = int(elev_col.min())
        value["elev_max_obs"] = int(elev_col.max())

    if "Nombre" in filtered.columns:
        value["region_breakdown"] = filtered["Nombre"].value_counts().to_dict()

    if "geometry" in filtered.columns and n > 0:
        b = filtered.total_bounds  # minx, miny, maxx, maxy
        value["bbox"] = {"lon_min": round(b[0], 4), "lat_min": round(b[1], 4),
                         "lon_max": round(b[2], 4), "lat_max": round(b[3], 4)}

    return _envelope(
        value,
        source="GBIF",
        citation="GBIF occurrence API — Costa Rica (hasCoordinate=true, hasGeospatialIssue=false)",
        confidence="estimated",
        caveat=caveat,
    )

@server.tool()
def get_occurrence_points(species: str, limit: int = 500) -> dict:
    """Cleaned Costa Rica GBIF presence points for a species, one entry per record.

    Cost: L1 — same disk cache as get_occurrences (the first call fetches from
    GBIF, later calls are instant until the cache expires). No LLM.

    The points are already cleaned by gbif_map: occurrenceStatus=PRESENT,
    hasCoordinate=true, hasGeospatialIssue=false, no (0,0), swapped lat/lon fixed,
    uncertainty <= ~10 km. elev_m comes from the DEM at the point, not from the GBIF record.

    Args:
        species: scientific name e.g. "Quercus costaricensis"
        limit: maximum number of points returned (the count n is always total)

    Returns a provenance envelope whose value is:
         {n, truncated, points: [{lat, lon, elev_m, region, vertiente, date}],
          updated_at, stale}
    date is the GBIF eventDate ('YYYY-MM-DD', 'YYYY-MM' or 'YYYY'), or None.
    updated_at: when the points were fetched from GBIF (UTC ISO-8601).
    stale: true when a refresh failed and the previous points are returned.
    """
    citation = "GBIF occurrence API — Costa Rica (occurrenceStatus=PRESENT, hasCoordinate=true, hasGeospatialIssue=false)"
    pts, info = gbif_map.get_points_info(species)
    fresh, fresh_caveat = _freshness(info)
    if pts.empty:
        return _envelope(
            {"n": 0, "truncated": False, "points": [], **fresh},
            source="GBIF",
            citation=citation,
            confidence="insufficient",
            caveat=_join(f"No GBIF occurrences found for '{species}' in Costa Rica.", fresh_caveat),
        )

    def _num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if v != v else v  # NaN -> None

    def _text(v):
        return v if isinstance(v, str) and v else None

    points = []
    for _, p in pts.head(limit).iterrows():
        elev = _num(p.get("Altitud"))
        points.append({
            "lat": round(p.geometry.y, 5),
            "lon": round(p.geometry.x, 5),
            "elev_m": None if elev is None else int(round(elev)),
            "region": _text(p.get("Nombre")),
            "vertiente": _text(p.get("Vertiente")),
            "date": _text(p.get("date")),
        })
    return _envelope(
        {"n": len(pts), "truncated": len(pts) > limit, "points": points, **fresh},
        source="GBIF",
        citation=citation,
        confidence="exact",
        caveat=fresh_caveat,
    )

@server.tool()
def render_species_map(species: str) -> dict:
    """Render the Manual-grounded distribution map for a single named species.

    Cost: L1 — re-derives DistributionFicha from the stored distribution paragraph
    (no LLM), then renders via the validated map pipeline with GBIF presence points.
    Map is cached by species vector_id.

    Args:
        species: scientific name e.g. "Peltogyne purpurea"

    Returns a provenance envelope whose value is:
      {map_path: str, n_pts: int, layer_summary: str}
    """
    f = local_store.get(_get_conn(), species.strip().replace(" ", "_"))
    if f is None:
        return _envelope(
            None,
            source="MPCR",
            confidence="insufficient",
            caveat=f"Species '{species}' not found in catalog.",
        )

    caveat = ""
    if f.elev_min is not None and f.elev_min == f.elev_max:
        caveat = (
            f"WARNING: Manual gives a single elevation point ({f.elev_min} m) — "
            "the DEM elevation mask will be empty (zero-width range). "
            "The map is rendered without a cyan elevation layer. "
            "Use get_occurrences() to cross-check with GBIF evidence."
        )

    map_path, n_pts = gbif_map.single_species_map(f)
    layers = [
        "CR base (provinces)",
        "Unmatched botanical regions (dark gray)",
        "Matched regions — Manual text (white outline)",
    ]
    if n_pts >= 5:
        layers.append(f"GBIF presence points (n={n_pts}, red dots)")
    layers += [
        "DEM elevation mask (cyan fill, within matched regions)",
        "Parks mentioned (orange outline)",
        "Protected areas filtered by elevation (amber outline)",
    ]
    return _envelope(
        {
            "map_path": str(map_path),
            "n_pts": n_pts,
            "layer_summary": " | ".join(layers),
        },
        source="MPCR+GBIF+DEM",
        citation=f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages}; "
                 "GBIF occurrences CR; DEM altitud_cr.tif; "
                 "Regiones Botánicas (José Araya, 2026)",
        confidence="estimated",
        caveat=caveat,
    )

@server.tool()
def get_species_map_image(species: str) -> dict:
    """The render_species_map PNG itself, base64-encoded, for clients that run on
    another machine or container and cannot open map_path.

    Cost: L1 — same map and cache as render_species_map (no LLM). The PNG travels
    inside the envelope, so expect a few hundred KB per call; MCP clients running
    on this machine should prefer render_species_map and read map_path.

    Args:
        species: scientific name e.g. "Peltogyne purpurea"

    Returns a provenance envelope whose value is:
      {png_base64: str, n_pts: int, layer_summary: str}
    or value None (confidence "insufficient") when no map can be rendered.
    """
    rendered = render_species_map(species)
    if rendered["value"] is None:
        return rendered
    png = Path(rendered["value"]["map_path"]).read_bytes()
    return _envelope(
        {
            "png_base64": base64.b64encode(png).decode("ascii"),
            "n_pts": rendered["value"]["n_pts"],
            "layer_summary": rendered["value"]["layer_summary"],
        },
        source=rendered["source"],
        citation=rendered["citation"],
        confidence=rendered["confidence"],
        caveat=rendered["caveat"],
    )


# ---------------------------------------------------------------------------
# L2 — requires PINECONE_API_KEY and OPENROUTER_API_KEY
# ---------------------------------------------------------------------------

@server.tool()
def search_by_description(query_text: str, family: Optional[str] = None,
                          habit: Optional[str] = None, top_k: int = 10) -> dict:
    """Find species whose Manual DESCRIPTION (morphology + discussion) matches a
    plant description, e.g. "arbusto con flores moradas y hojas pubescentes".

    Cost: L1 — local e5 embeddings over pgvector, no API key. Requires
    MPCR_VECTOR_BACKEND=pgvector and a synced store (build_catalog --pgvector).

    Returns candidates ranked by similarity. This is a shortlist for identification,
    not an identification: confirm against the morphology text of each candidate.
    """
    if config.VECTOR_BACKEND != "pgvector":
        return _envelope(None, source="MPCR", confidence="insufficient",
                         caveat="search_by_description requires MPCR_VECTOR_BACKEND=pgvector.")
    from mpcr_rag.store import pg_store
    constraints = {k: v for k, v in dict(family=family, habit=habit).items() if v is not None}
    hits = pg_store.search(query_text, top_k=top_k, section="description", **constraints)
    value = [{"score": round(float(h["score"]), 4), "species": h["species"],
              "family": h["family"], "habits": h["habits"],
              "volume": h["volume"], "pages": h["pages"]} for h in hits]
    return _envelope(
        value, source="MPCR", confidence="estimated",
        citation="Manual de Plantas de Costa Rica — morphology/discussion text, e5 embeddings",
        caveat="" if hits else "No species matched. Loosen family/habit filters.",
    )

@server.tool()
def get_relevant_passage(species: str, question: str, top_k: int = 1) -> dict:
    """The Manual passage of ONE species that best answers a question.

    Cost: L1 — local e5 embeddings over pgvector, no API key. Requires
    MPCR_VECTOR_BACKEND=pgvector and a synced store (build_catalog --pgvector).

    Unlike semantic_search and search_by_description, which rank SPECIES, this ranks
    the text of a species already identified: its distribution paragraph and its
    description windows (morphology + discussion). Use it to quote the passage that
    answers e.g. "¿de qué color son las flores?" instead of the distribution text.

    Args:
        species: scientific name e.g. "Quercus costaricensis"
        question: the user's question, in Spanish or English
        top_k: number of passages returned, best first (default 1)

    Returns a provenance envelope whose value is:
      {species, passages: [{section, ordinal, text, score}]}
    or value None (confidence "insufficient") when the species has no indexed text.
    """
    if config.VECTOR_BACKEND != "pgvector":
        return _envelope(None, source="MPCR", confidence="insufficient",
                         caveat="get_relevant_passage requires MPCR_VECTOR_BACKEND=pgvector.")
    from mpcr_rag.store import pg_store
    hits = pg_store.passages(species, question, top_k=top_k)
    if not hits:
        return _envelope(
            None, source="MPCR", confidence="insufficient",
            caveat=f"No indexed text for '{species}'. Check spelling or sync the store "
                   "(build_catalog --pgvector).",
        )
    best = hits[0]
    value = {
        "species": best["species"],
        "passages": [{"section": h["section"], "ordinal": h["ordinal"], "text": h["text"],
                      "score": round(float(h["score"]), 4)} for h in hits],
    }
    return _envelope(
        value, source="MPCR", confidence="estimated",
        citation=f"Manual de Plantas de Costa Rica, Tomo {best['volume']}, p. {best['pages']} "
                 "— e5 embeddings",
    )


@server.tool()
def where_to_see(species: str, limit: int = 8) -> dict:
    """Where a visitor could look for a species: protected areas holding occurrence
    records, the regions the Manual states, the regions the records add, and the
    flowering months.

    Cost: L0 - free, local tables (GBIF snapshot DOI 10.15468/dl.8yhee8 inside SINAC
    polygons, plus the Manual entry).

    Args:
        species: scientific name. For a common name, call resolve_common_name first:
            it may legitimately return several species ("Cortez amarillo" is three
            Handroanthus), and each has different places.

    The value keeps the two kinds of evidence apart:
      confirmed_places  parks with cleaned records of this species (observed)
      manual_regions    regions the Manual states (expert-stated range)
      record_regions    regions where records fall; those beyond manual_regions are
                        an extension the Manual does not mention
    A park with no records is NOT evidence of absence: collection effort is uneven,
    so never phrase an empty result as "it does not occur there".
    """
    from mpcr_rag.evidence.places import where_to_see as _wts
    f = local_store.get(_get_conn(), species.strip().replace(" ", "_"))
    if f is None:
        return _envelope(None, source="MPCR", confidence="insufficient",
                         caveat=f"Species '{species}' not found in the catalog.")
    w = _wts(f.species, limit=limit)
    value = {
        "species": f.species,
        "elev_min": f.elev_min, "elev_max": f.elev_max,
        "vertientes": f.vertientes,
        "flowering_months": f.flowering_months,
        "n_records": w["n_records"], "n_sites": w["n_cells"],
        "manual_regions": w["manual_regions"],
        "record_regions": w["record_regions"],
        "regions_beyond_manual": [r for r in w["record_regions"]
                                  if r not in set(w["manual_regions"])],
        "confirmed_places": w["confirmed_places"],
    }
    return _envelope(
        value, source="MPCR+GBIF+SINAC",
        citation=(f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages}; "
                  f"GBIF snapshot 10.15468/dl.8yhee8; SINAC protected areas"),
        confidence="exact" if w["confirmed_places"] else "insufficient",
        caveat=("GBIF records reflect collection effort, not abundance; absence of "
                "records in a park does not mean the species is absent there."),
    )


@server.tool()
def resolve_common_name(name: str, region: Optional[str] = None,
                        elev: Optional[int] = None) -> dict:
    """Map a common (vernacular) name to candidate species of the Manual catalog.

    Cost: L0 - free, local table lookup.

    Common names are ambiguous ("poro" is several Erythrina; "roble" is Quercus
    oleoides and also Tabebuia rosea), so this NEVER returns a single species
    silently. The envelope value is:
      {decision: "unique"|"preferred"|"ambiguous"|"none", species: str|None,
       candidates: [{species, family, elev_min, elev_max, n_records, sources}]}
    On "ambiguous", ask the user which species is meant, or answer for the group and
    say so; do not pick one. On "none", say the name is not in the catalog rather
    than guessing: global name indexes answer a different question (Tropicos returns
    the leek, Allium porrum, for "Poro").

    Matching is accent-sensitive first ("poro" the leek vs "poro" with an accent, the
    Erythrina), then accent-insensitive. Optional region and elevation from the
    question narrow the candidates.
    """
    from mpcr_rag.evidence import vernacular
    d = vernacular.resolve_decision(name, region=region, elev=elev)
    cands = [{"species": c["species"], "family": c["family"],
              "elev_min": c["elev_min"], "elev_max": c["elev_max"],
              "n_records": c["n_records"] or 0, "sources": sorted(c["sources"]),
              "cr_preferred": bool(c["preferred"]), "match": c["match"]}
             for c in d["candidates"]]
    value = {"decision": d["decision"], "species": d.get("species"), "candidates": cands}
    caveats = {
        "none": f"'{name}' is not in the vernacular table for this catalog "
                f"(Tomos II-VI). It may be a real name for a species outside the catalog.",
        "ambiguous": "Several species share this name. Ask which one, or answer for the "
                     "group; do not choose silently.",
        "preferred": "Chosen because it is the Costa Rica-preferred name AND the best "
                     "recorded candidate; other species share the name.",
        "unique": "",
    }
    return _envelope(
        value, source="MPCR+GBIF+iNaturalist",
        citation="mpcr.vernacular (Darwin Core VernacularName shape; source per row)",
        confidence="exact" if d["decision"] == "unique" else "estimated",
        caveat=caveats[d["decision"]],
    )


def _check_l2() -> str:
    missing = []
    if config.VECTOR_BACKEND == "pinecone" and not os.environ.get("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")
    if not os.environ.get("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    return ", ".join(missing)


@server.tool()
def semantic_search(
    query_text: str,
    habit: Optional[str] = None,
    elev_lo: Optional[int] = None,
    elev_hi: Optional[int] = None,
    vertiente: Optional[str] = None,
    region: Optional[str] = None,
    family: Optional[str] = None,
    endemic: Optional[bool] = None,
    top_k: int = 12,
) -> dict:
    """Semantic search over the Manual catalog (distribution-paragraph vectors).

    Cost: L2 — Pinecone backend needs PINECONE_API_KEY; pgvector backend runs
    locally.  Combines dense e5 retrieval with structured metadata filters.  Returns species
    ranked by relevance to the query text.

    Args:
        query_text: natural-language description e.g. "árbol de madera dura en bosque nuboso"
        habit, elev_lo, elev_hi, vertiente, region, family, endemic: same as search_species
        top_k: maximum results (default 12)

    Returns a provenance envelope whose value is a ranked list of species summaries.
    """
    missing = _check_l2()
    if "PINECONE_API_KEY" in missing:
        return _envelope(
            None,
            source="MPCR",
            confidence="insufficient",
            caveat=f"API key not configured: {missing}. semantic_search uses the Pinecone backend.",
        )

    from mpcr_rag.query.retriever import pattern_b
    constraints = {k: v for k, v in dict(
        habit=habit, elev_lo=elev_lo, elev_hi=elev_hi,
        vertiente=vertiente, region=region, family=family, endemic=endemic,
    ).items() if v is not None}

    results = pattern_b(
        query_text, top_k=top_k, conn=_get_conn(), index=_get_index(), **constraints
    )
    value = [{"score": round(score, 4), **_ficha_summary(f)} for f, score in results]
    return _envelope(
        value,
        source="MPCR",
        citation=f"Manual de Plantas de Costa Rica — e5 semantic index ({config.VECTOR_BACKEND})",
        confidence="estimated",
        caveat="" if results else "No results matched. Try broader filters or different query_text.",
    )


@server.tool()
def answer_question(question: str) -> dict:
    """Answer a natural-language question about Costa Rica flora.

    Cost: L2 — requires OPENROUTER_API_KEY (+ PINECONE_API_KEY on the Pinecone
    backend).  Routes to one
    of three fixed, citable recipes:
      A  (lookup)     — named species → Manual-grounded map + grounded text
      B  (list)       — geospatial filter → GBIF evidence map + grounded text
      B→A (superlative) — filter + deterministic extremal selection → single-species map

    Args:
        question: question in Spanish or English about CR flora
          e.g. "¿cuál arbusto endémico crece a mayor altitud en Talamanca?"

    Returns a provenance envelope whose value is:
      {text, mode, constraints, selector, map_path, n_pts, citations}
    where text is a grounded answer citing Manual (Tomo, página) for every claim.
    """
    missing = _check_l2()
    if missing:
        return _envelope(
            None,
            source="MPCR",
            confidence="insufficient",
            caveat=f"API keys not configured: {missing}.",
        )

    from mpcr_rag.query import answer as ans_mod
    a = ans_mod.answer(question, conn=_get_conn(), index=_get_index())

    citations = [
        f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages} ({f.species})"
        for f, _ in a["results"]
    ]
    value = {
        "text": a["text"],
        "mode": a["mode"],
        "constraints": a["constraints"],
        "selector": a.get("selector"),
        "map_path": str(a["map_path"]),
        "n_pts": a["n_pts"],
        "citations": citations,
    }
    return _envelope(
        value,
        source="MPCR+GBIF",
        citation="; ".join(citations) if citations else "Manual de Plantas de Costa Rica",
        confidence="estimated",
        caveat=(
            "Answer is grounded in Manual text only. GBIF occurrence data provides "
            "the map layer but does not alter the text. "
            "gbif_count selector is a collection-effort proxy, not real abundance."
            if a.get("selector") and "gbif_count" in (a.get("selector") or "") else ""
        ),
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@server.resource("mpcr://vocabulary")
def vocabulary_resource() -> str:
    """The controlled vocabulary — inject this into an agent's system prompt
    to guide filter value selection before calling search_species or semantic_search."""
    vocab = intent_mod.load_vocab(_get_conn())
    return json.dumps(vocab, ensure_ascii=False, indent=2)


@server.resource("mpcr://species/{vector_id}")
def species_resource(vector_id: str) -> str:
    """One ficha as addressable context.  vector_id is the species name with
    spaces replaced by underscores e.g. 'Peltogyne_purpurea'."""
    f = local_store.get(_get_conn(), vector_id)
    if f is None:
        return json.dumps({"error": f"Species '{vector_id}' not found."})
    return json.dumps({**_ficha_summary(f), "distribution_paragraph": f.distribution_paragraph},
                      ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server.run()
