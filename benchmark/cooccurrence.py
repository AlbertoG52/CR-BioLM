"""The co-occurrence question: what else would I find growing with this plant?

This is the first question type whose answer is a SET the model generates, and the
first whose key is not a gold answer. Both departures are forced by the data.

A gold list does not work here. Species sharing a botanical region and an overlapping
elevation band with a given plant number 1,900 to 3,200 -- roughly a third of the flora
-- and occurrence data does not tighten it: co-recorded species in the same grid cells
run 1,200 to 2,500, because the cells are coarse and the well-collected ones hold
thousands of plants. Asked to name a plant that grows near *Clethra consimilis*, almost
any real Costa Rican species would be correct. The question would separate nothing.

So the key inverts. It does not ask whether a named plant is in a list of neighbours;
it asks whether that plant is POSSIBLE at all:

  exists        is the binomial in the catalogue? a fabricated name is the strongest
                signal available, and it costs nothing to detect
  region        does it share a botanical region with the subject?
  elevation     do their elevation bands overlap?

Co-occurrence is common, but INCOMPATIBILITY is rare and mechanical to detect. A model
that answers a Talamanca cloud-forest orchid with a Guanacaste dry-forest tree is
wrong in a way no gold list is needed to see, and that is exactly the error a system
without the evidence makes.

Two further properties matter for the thesis. The key needs only regions and elevation,
both of which the Manual states for 6,728 of 6,946 species, so unlike anything built on
occurrence counts this reaches the long tail. And it needs no expert.

Run:  python -m benchmark.cooccurrence --species "Clethra consimilis" --check "Quercus costaricensis, Bursera simaruba"
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from functools import lru_cache
from pathlib import Path

from mpcr_rag import config

N_REQUESTED = 3          # "menciona tres" -- enough to score, few enough to be natural

QUESTION = ("Si encuentro {sp} en el campo, ¿qué otras plantas podría ver creciendo en "
            "el mismo sitio? Mencione tres especies y diga por qué.")

# A binomial as it appears in prose: Genus species, optionally italicised or
# parenthesised. Deliberately strict about the capital, because lowercase Spanish words
# in sequence would otherwise match constantly.
_BINOMIAL = re.compile(r"\b([A-Z][a-zé]{3,})\s+([a-zé][a-zé\-]{2,})\b")

_STOP_GENUS = {"Manual", "Costa", "Parque", "Cordillera", "Reserva", "Nacional",
               "Según", "Vertiente", "Valle", "Isla", "Península", "Llanura", "Fila",
               "Puede", "Entre", "Estas", "Estos", "Otras", "Otros", "Algunas"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


@lru_cache(maxsize=1)
def _catalogue() -> tuple[dict, dict]:
    """(assessable, synonyms). Assessable species carry regions and an elevation band;
    synonyms map an older name onto the accepted one so a model using one is not
    treated as having made it up."""
    con = sqlite3.connect(config.SQLITE_PATH)
    out, syn = {}, {}
    for (fj,) in con.execute("SELECT ficha_json FROM fichas"):
        f = json.loads(fj)
        for s in (f.get("synonyms") or []):
            name = s if isinstance(s, str) else (s.get("name") if isinstance(s, dict) else None)
            if name:
                syn[_norm(name)] = f["species"]
        if f.get("elev_min") is None or not f.get("regions"):
            continue
        out[_norm(f["species"])] = (set(f["regions"]), f["elev_min"], f["elev_max"],
                                    f.get("family"), f["species"])
    con.close()
    return out, syn


@lru_cache(maxsize=1)
def _in_costa_rica() -> set:
    """Every species in the frozen GBIF snapshot: 13,504 of them.

    The Manual volumes we ingested are not the whole flora. Rhizophora mangle is a
    common Costa Rican mangrove and is absent from the catalogue; so is every fern,
    because no fern family appears in it at all. Scoring absence from the catalogue as
    a fabricated name would have punished answers for naming real plants that our
    ingest never covered -- 352 GBIF records for the mangrove, 192 for one fern.
    """
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT species FROM mpcr.occurrences WHERE species IS NOT NULL")
            return {_norm(r[0]) for r in cur.fetchall()}
    finally:
        conn.close()


def extract_species(text: str) -> list[str]:
    """Binomials named in an answer, in order, de-duplicated."""
    seen, out = set(), []
    for g, s in _BINOMIAL.findall(text or ""):
        if g in _STOP_GENUS:
            continue
        name = f"{g} {s}"
        if _norm(name) in seen:
            continue
        seen.add(_norm(name))
        out.append(name)
    return out


def check(subject: str, named: list[str], *, use_gbif: bool = True) -> list[dict]:
    """Per named species, one of three tiers.

    assessable     in the catalogue with regions and elevation -- compatibility decided
    outside_scope  real and recorded in Costa Rica, but not in the volumes we ingested;
                   not judged, and not held against the answer
    unrecognised   in neither reference. Still not proof of invention -- it may be a
                   synonym we do not carry, or a species never recorded here -- so it
                   is reported rather than scored as a fabrication.
    """
    cat, syn = _catalogue()
    cr = _in_costa_rica() if use_gbif else set()
    sub = cat.get(_norm(subject)) or cat.get(_norm(syn.get(_norm(subject), "")))
    results = []
    for name in named:
        key = _norm(name)
        row = cat.get(key) or cat.get(_norm(syn.get(key, "")))
        if row is None:
            tier = "outside_scope" if key in cr else "unrecognised"
            why = ("existe en Costa Rica pero fuera de los tomos procesados"
                   if tier == "outside_scope" else "no reconocida en ninguna referencia")
            results.append({"name": name, "tier": tier, "compatible": None, "why": why})
            continue
        regs, lo, hi, fam, canon = row
        if sub is None:
            results.append({"name": canon, "tier": "assessable", "compatible": None,
                            "why": "no se pudo evaluar: falta el sujeto"})
            continue
        s_regs, s_lo, s_hi, s_fam, _ = sub
        shared = sorted(regs & s_regs)
        elev_ok = lo <= s_hi and hi >= s_lo
        ok = bool(shared) and elev_ok
        why = ("compatible" if ok else
               "sin región botánica en común" if not shared else
               f"rango altitudinal disjunto ({lo}-{hi} m vs {s_lo}-{s_hi} m)")
        results.append({"name": canon, "tier": "assessable", "compatible": ok,
                        "shared_regions": shared, "elev_overlap": elev_ok,
                        "same_family": fam == s_fam, "why": why})
    return results


def verdict(results: list[dict]) -> str:
    """Scored on precision, not recall: everything named should be possible.

    Recall is meaningless when a third of the flora qualifies. What a system should be
    judged on is whether the plants it volunteers could actually be there.
    """
    if not results:
        return "abstained"
    judged = [r for r in results if r["tier"] == "assessable" and r["compatible"] is not None]
    if not judged:
        # Everything named sits outside what we can assess. That is a fact about our
        # ingest, not about the answer, so it is not scored either way.
        return "not_scored"
    ok = sum(1 for r in judged if r["compatible"])
    if ok == len(judged):
        return "correct" if len(judged) >= 2 else "partially_correct"
    return "partially_correct" if ok else "wrong"


def key_for(subject: str) -> dict | None:
    cat, syn = _catalogue()
    sub = cat.get(_norm(subject)) or cat.get(_norm(syn.get(_norm(subject), "")))
    if sub is None:
        return None
    regs, lo, hi, fam, canon = sub
    return {"criteria": "compatibility, not a gold list",
            "subject_regions": sorted(regs), "subject_elevation": [lo, hi],
            "n_requested": N_REQUESTED,
            "basis": ("each named species is scored on whether it exists in the Manual "
                      "and could occur with the subject: a shared botanical region and "
                      "an overlapping elevation band. Naming a plausible neighbour is "
                      "easy -- about a third of the flora qualifies -- so the measure "
                      "is precision over what the answer volunteers, never recall.")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Check co-occurrence compatibility")
    ap.add_argument("--species", required=True)
    ap.add_argument("--check", required=True, help="comma separated, or free answer text")
    args = ap.parse_args()
    # Always extract. The real pipeline only ever sees free prose, so the CLI must
    # exercise the same path -- splitting on commas tested nothing that runs.
    named = extract_species(args.check)
    k = key_for(args.species)
    print(f"SUBJECT {args.species}")
    print(f"  regions {k['subject_regions'] if k else '?'} | elevation "
          f"{k['subject_elevation'] if k else '?'}\n")
    res = check(args.species, named)
    mark = {True: "OK ", False: "NO ", None: "-- "}
    for r in res:
        print(f"  {mark[r['compatible']]} {r['tier']:14s} {r['name']:30s} {r['why']}")
    print(f"\nverdict: {verdict(res)}")


if __name__ == "__main__":
    main()


def neighbours(subject: str, limit: int = 12) -> list[dict]:
    """Species the Manual places in the same regions and elevation band as the subject.

    This is what "grounded" means for this question. A third of the flora qualifies, so
    the provider ranks rather than filters: species sharing more regions come first,
    then those whose elevation band sits closest to the subject's, then the same family.
    The point is a usable handful, not the whole set.
    """
    cat, syn = _catalogue()
    sub = cat.get(_norm(subject)) or cat.get(_norm(syn.get(_norm(subject), "")))
    if sub is None:
        return []
    s_regs, s_lo, s_hi, s_fam, s_canon = sub
    s_mid = (s_lo + s_hi) / 2
    scored = []
    for key, (regs, lo, hi, fam, canon) in cat.items():
        if canon == s_canon:
            continue
        shared = regs & s_regs
        if not shared or not (lo <= s_hi and hi >= s_lo):
            continue
        scored.append((len(shared), -abs((lo + hi) / 2 - s_mid), fam == s_fam,
                       {"species": canon, "family": fam, "elevation": [lo, hi],
                        "shared_regions": sorted(shared)}))
    scored.sort(key=lambda t: (-t[0], t[1], not t[2]))
    return [t[3] for t in scored[:limit]]
