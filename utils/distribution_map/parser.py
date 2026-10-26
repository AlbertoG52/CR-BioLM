"""
parser.py — Build a DistributionFicha from raw Manual de Plantas habitat text.

Primary path: gazetteer longest-match → rapidfuzz fuzzy fallback.
Optional LLM fallback for remaining unresolved tokens (pass enable_llm=True).

Public API:
    build_ficha(habitat_raw, geographic_notes, species, enable_llm) -> DistributionFicha
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

from .ficha import DistributionFicha, ElevationRange, EntityRef
from .gazetteer import (
    lookup, lookup_parks, fuzzy_lookup, normalize_text,
)
from .geo_parser import (
    parse_distribution_block,
    get_vertientes as _gp_vertientes,
    get_all_protected_areas as _gp_protected_areas,
    get_features_by_type as _gp_by_type,
)

# ---------------------------------------------------------------------------
# Elevation regex (ported from extract_habitat_from_pdf.py)
# ---------------------------------------------------------------------------
_RE_ELEV_RANGE = re.compile(
    r"""
    (?:\(([—–\-]?\s*\d{1,4})\s*[—–\-]\s*\))?   # leading outlier: (X–)
    (\d{1,4})\s*[—–\-]\s*(\d{1,4})\+?            # main range: X–Y  (trailing + ignored)
    -?-?                                           # optional trailing -- (e.g. 200--)
    (?:\s*\([—–\-]?\s*(\d{1,4})\+?\s*\))?        # trailing outlier: (–Z) or (-Z)
    \s*m\b
    """,
    re.VERBOSE | re.IGNORECASE,
)
_RE_ELEV_SINGLE = re.compile(r"\b(\d{2,4})\+?\s*m\b")
_RE_ELEV_APPROX = re.compile(r"ca\.?\s*(\d{1,4})\s*m\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Vertiente detection (simple, robust enough for normalized text)
# ---------------------------------------------------------------------------
_RE_CARIBE   = re.compile(r"vert(?:iente)?\.?\s*carib|caribena|caribeña", re.IGNORECASE)
_RE_PACIFICO = re.compile(r"vert(?:iente)?\.?\s*pac(?:if)?|pacifica", re.IGNORECASE)
_RE_AMBAS    = re.compile(r"ambas\s*vert", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Forest-type keywords in habitat_raw (before the semicolon separator)
# ---------------------------------------------------------------------------
# The Manual compresses forest types into a list sharing one "Bosque":
#   "Bosque húmedo, muy húmedo y pluvial, <elev>"
# so qualifiers are matched WITHOUT requiring a "bosque" prefix on each. Order
# matters: "muy húmedo" is matched (and its span reserved) before "húmedo" so the
# inner "húmedo" is not double-counted.
_FOREST_QUALIFIERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"muy\s+h[úu]medo",          re.I), "muy húmedo"),
    (re.compile(r"h[úu]medo",                re.I), "húmedo"),
    (re.compile(r"pluvial",                  re.I), "pluvial"),
    (re.compile(r"nuboso",                   re.I), "nuboso"),
    (re.compile(r"seco",                     re.I), "seco"),
    (re.compile(r"de\s+roble|robledal|roble", re.I), "roble"),
    (re.compile(r"p[áa]ramo",                re.I), "páramo"),
    (re.compile(r"manglar",                  re.I), "manglar"),
    (re.compile(r"matorral",                 re.I), "matorral"),
    (re.compile(r"premontano",               re.I), "premontano"),
]


def _parse_elevation(text: str) -> ElevationRange:
    """Extract elevation range from raw habitat text."""
    m = _RE_ELEV_RANGE.search(text)
    if m:
        lo_out, lo, hi, hi_out = m.groups()
        def _clean(v: Optional[str]) -> Optional[float]:
            if v is None:
                return None
            v = re.sub(r"[—–\-\s]", "", v)
            return float(v) if v.isdigit() or (v.lstrip("-").isdigit()) else None
        return ElevationRange(
            min_m        = _clean(lo),
            max_m        = _clean(hi),
            outlier_min_m= _clean(lo_out),
            outlier_max_m= _clean(hi_out),
        )

    # Approximate single value (ca. 1150 m) — use exact, let CSV supplement range
    m2 = _RE_ELEV_APPROX.search(text)
    if m2:
        v = float(m2.group(1))
        return ElevationRange(min_m=v, max_m=v)

    # Single value like "0–700 m" already handled above; bare "1300 m" means
    # we only know the ceiling, so leave it for the CSV supplement.
    # (Returning empty lets maps_only.py apply the catalog-extracted range.)
    return ElevationRange()

    return ElevationRange()


def _parse_forest_types(habitat_raw: str) -> list[str]:
    """Extract every forest-type descriptor from the compressed habitat clause.

    e.g. "Bosque húmedo, muy húmedo y pluvial, 600–1900 m" → [húmedo, muy húmedo, pluvial].
    """
    # Isolate the forest clause: from "Bosque" up to the elevation (first number).
    seg = habitat_raw
    mb = re.search(r"bosque\b", seg, re.I)
    if mb:
        rest = seg[mb.end():]
        me = re.search(r"\(?\s*\d", rest)      # elevation start, e.g. "(0–)600" or "600"
        seg = rest[:me.start()] if me else rest
    else:
        seg = seg.split(";")[0]

    found: list[str] = []
    used = [False] * len(seg)
    for pat, label in _FOREST_QUALIFIERS:
        for m in pat.finditer(seg):
            if any(used[m.start():m.end()]):    # already inside a longer match
                continue
            for i in range(m.start(), m.end()):
                used[i] = True
            if label not in found:
                found.append(label)
    return found


def _detect_vertientes(text: str) -> list[str]:
    """Return list of matched vertientes from geographic text."""
    raw = text
    ambas    = bool(_RE_AMBAS.search(raw))
    has_car  = bool(_RE_CARIBE.search(raw))
    has_pac  = bool(_RE_PACIFICO.search(raw))

    if ambas or (has_car and has_pac):
        return ["Caribe", "Pacífico"]
    if has_car:
        return ["Caribe"]
    if has_pac:
        return ["Pacífico"]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ficha(
    habitat_raw: str = "",
    geographic_notes: str = "",
    species: str = "",
    enable_llm: bool = False,
) -> DistributionFicha:
    """
    Parse habitat text into a structured DistributionFicha.

    Primary path: geo_parser (vertiente state machine) → entities.csv lookup
    → MOBOT gazetteer for localidad/region_informal features.
    Fuzzy fallback for anything not resolved by the above.

    Args:
        habitat_raw:       Raw habitat sentence from Manual de Plantas.
        geographic_notes:  Geographic context string (post-semicolon part).
        species:           Species name (for labelling only).
        enable_llm:        Call LLM for unresolved tokens (default off).
    """
    combined_text = f"{habitat_raw} {geographic_notes}".strip()

    # ── Elevation ────────────────────────────────────────────────────────────
    elevation = _parse_elevation(combined_text)

    # ── Forest types ─────────────────────────────────────────────────────────
    forest_types = _parse_forest_types(habitat_raw)

    # ── Structured geo_parser on geographic_notes ─────────────────────────────
    geo_text = geographic_notes.strip() if geographic_notes.strip() else combined_text
    structured_occs = parse_distribution_block(geo_text)

    # ── Vertientes from structured parser ─────────────────────────────────────
    vertientes = list(dict.fromkeys(_gp_vertientes(structured_occs)))
    if not vertientes:
        # Fallback to regex detection on full text
        vertientes = _detect_vertientes(combined_text)

    # ── Gazetteer lookup (entities.csv) on combined text ─────────────────────
    # This handles the canonical botanical regions, parks, etc.
    geo_matches = lookup(combined_text)

    def _to_entity_refs(hits: list[dict]) -> list[EntityRef]:
        seen: set[tuple] = set()
        refs = []
        for h in hits:
            key = (h["canonical_name"], h["hierarchy_level"])
            if key not in seen:
                seen.add(key)
                refs.append(EntityRef(
                    canonical_name  = h["canonical_name"],
                    hierarchy_level = h["hierarchy_level"],
                    source_span     = h.get("source_span", ""),
                ))
        return refs

    regions   = _to_entity_refs(
        geo_matches.get("cordillera", []) +
        geo_matches.get("llanura", []) +
        geo_matches.get("valle", []) +
        geo_matches.get("peninsula", []) +
        geo_matches.get("fila", []) +
        geo_matches.get("region_other", [])
    )
    parks     = _to_entity_refs(geo_matches.get("park", []))
    cantons   = _to_entity_refs(geo_matches.get("canton", []))
    districts = _to_entity_refs(geo_matches.get("district", []))

    # ── Supplement vertientes from entities.csv lookup ───────────────────────
    gazetteer_verts = [
        e["canonical_name"] for e in geo_matches.get("vertiente", [])
        if e["canonical_name"] not in vertientes
    ]
    vertientes = list(dict.fromkeys(vertientes + gazetteer_verts))

    # ── Per-occurrence lookup for geo_parser cordillera/region occurrences ───
    # When a plural "Cords. de X y Y" is expanded by geo_parser, the combined-text
    # scan may only consume "Cords. de X" and miss "Y" (orphaned without prefix).
    # Re-scan each individual occurrence name to catch these.
    _REGION_OCC_TYPES = frozenset({"cordillera", "llanura", "valle", "fila",
                                    "peninsula", "region_other", "region_informal"})
    existing_region_names = {r.canonical_name for r in regions}
    for occ in structured_occs:
        if occ["feature_type"] not in _REGION_OCC_TYPES:
            continue
        # Skip occurrences with a directional qualifier — the combined-text
        # scan already handled them (e.g. "S Fila Costeña" → Fila Costeña Sur).
        if occ.get("qualifier"):
            continue
        fname = occ["feature_name"].strip()
        occ_hits = lookup(fname)
        for level in ("cordillera", "llanura", "valle", "fila", "peninsula", "region_other"):
            for h in occ_hits.get(level, []):
                if h["canonical_name"] not in existing_region_names:
                    existing_region_names.add(h["canonical_name"])
                    regions.append(EntityRef(
                        canonical_name  = h["canonical_name"],
                        hierarchy_level = h["hierarchy_level"],
                        source_span     = fname,
                    ))

    # Pass occurrences through unchanged — rendering uses shapefile polygons only
    locality_occurrences: list[dict] = list(structured_occs)

    # ── Promote embedded protected areas to parks list ────────────────────────
    # e.g. "S Pen. de Nicoya (R.N.A. Cabo Blanco)" → add Cabo Blanco to parks
    for occ in locality_occurrences:
        for ep_name in occ.get("embedded_protected_areas", []):
            ep_hits = lookup(ep_name)
            ep_refs = _to_entity_refs(ep_hits.get("park", []))
            for ref in ep_refs:
                if ref not in parks:
                    parks.append(ref)

    # ── Unresolved tokens ─────────────────────────────────────────────────────
    unresolved: list[str] = []
    for occ in structured_occs:
        if occ["feature_type"] == "localidad":
            name = occ["feature_name"].strip()
            if len(name) >= 4:
                unresolved.append(name)

    # ── Fuzzy fallback for unresolved ─────────────────────────────────────────
    fuzzy_conf: dict[str, float] = {}
    if unresolved:
        fuzzy_hits = fuzzy_lookup(unresolved, cutoff=88)
        for level, hits in fuzzy_hits.items():
            for h in hits:
                ref = EntityRef(h["canonical_name"], h["hierarchy_level"],
                                h.get("source_span", ""))
                score = h.get("fuzzy_score", 0) / 100
                fuzzy_conf[h["canonical_name"]] = score
                if level in ("cordillera", "llanura", "valle", "peninsula",
                             "fila", "region_other"):
                    if ref not in regions:
                        regions.append(ref)
                        unresolved = [u for u in unresolved
                                      if u != h.get("source_span")]
                elif level == "park":
                    if ref not in parks:
                        parks.append(ref)
                        unresolved = [u for u in unresolved
                                      if u != h.get("source_span")]
                elif level == "canton":
                    if ref not in cantons:
                        cantons.append(ref)
                        unresolved = [u for u in unresolved
                                      if u != h.get("source_span")]

    # ── LLM fallback (off by default) ────────────────────────────────────────
    if enable_llm and unresolved:
        try:
            llm_result = _llm_resolve(unresolved, combined_text)
            for name in llm_result.get("regions", []):
                ref = EntityRef(name, "region_other", "llm")
                if ref not in regions:
                    regions.append(ref)
            unresolved = llm_result.get("still_unresolved", unresolved)
        except Exception as e:
            print(f"  [WARN] LLM fallback failed: {e}")

    # ── Directional ranges ("vert. Pac. desde P.N. Carara al S") ──────────────
    # The gazetteer finds the anchor and nothing else; see directional.py.
    from .directional import expand as _directional
    have = {r.canonical_name for r in regions}
    for name, _phrase in _directional(geographic_notes or combined_text):
        if name not in have:
            have.add(name)
            regions.append(EntityRef(name, "region_other", "directional"))

    # ── Confidence summary ────────────────────────────────────────────────────
    n_matched = len(regions) + len(parks) + len(cantons) + len(districts)
    confidence = {
        "overall": 1.0 if n_matched > 0 else 0.0,
        "geographic": min(1.0, n_matched / 3),
        "elevation": 1.0 if elevation.has_data() else 0.0,
        **fuzzy_conf,
    }

    return DistributionFicha(
        species               = species,
        habitat_raw           = habitat_raw,
        vertientes            = vertientes,
        regions               = regions,
        parks                 = parks,
        cantons               = cantons,
        districts             = districts,
        elevation             = elevation,
        forest_types          = forest_types,
        locality_occurrences  = locality_occurrences,
        confidence            = confidence,
        unresolved_tokens     = unresolved,
    )


def _llm_resolve(unresolved_tokens: list[str], context: str) -> dict:
    """
    LLM fallback for unresolved geographic tokens.
    Uses the project's existing OpenRouter client.
    Returns {"regions": [...], "still_unresolved": [...]}.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    try:
        import config as _cfg
        import requests, time, json

        prompt = (
            "You are a Costa Rica botanist. The following geographic tokens from a "
            "plant distribution description could not be identified automatically. "
            "For each token, identify which botanical region of Costa Rica it belongs to "
            "(e.g. Cordillera de Talamanca, Llanuras de Tortuguero, Península de Osa, etc.) "
            "or return null if unknown.\n\n"
            f"Context: {context[:400]}\n\n"
            f"Tokens: {unresolved_tokens}\n\n"
            'Respond ONLY with JSON: {"resolved": {"token": "region"}, "unresolved": [...]}'
        )

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {_cfg.OPENROUTER_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "meta-llama/llama-3.3-70b-instruct",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0},
            timeout=20,
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(text)
        regions = list(parsed.get("resolved", {}).values())
        still   = parsed.get("unresolved", [])
        return {"regions": [r for r in regions if r], "still_unresolved": still}
    except Exception as e:
        print(f"  [WARN] _llm_resolve error: {e}")
        return {"regions": [], "still_unresolved": unresolved_tokens}
