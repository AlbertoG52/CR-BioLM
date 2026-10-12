"""What fraction of the flora the published record can settle, per question type.

Finding F6 in docs/EXPERIMENT_PLAN.md, and the only headline result that needs no model
run: a database query over all 6,946 species. It answers the question a benchmark of
elevation questions alone would hide — the Manual states where a plant grows for almost
every species and what it is good for for almost none.

The predicates here are IMPORTED from benchmark.questions, not reimplemented, so the
table describes the benchmark that actually exists. If a key rule changes there, this
table changes with it.

Two denominators, reported separately because they are not the same population:

  catalogue  all 6,946 species. Right for Manual-stated types.
  frame      species resolved to a GBIF accepted key, i.e. the ones occurrence evidence
             can be attributed to at all. Right for the park types.

Use mentions are counted at two scopes. The `uses` FIELD is empty for every species —
field_extractor leaves it to an LLM pass that has not been run — so a bare 0% would
conflate "the Manual is silent" with "we have not extracted it". Counting the Manual's
own use vocabulary resolves that, and the two scopes bracket the answer.

Run:  python -m benchmark.answerability [--csv]
Writes benchmark/data/answerability.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from mpcr_rag.ingest.field_extractor import _FAMILY_HABIT, _match_habits
from mpcr_rag.schema import Ficha
from mpcr_rag.store import pg_store

from .questions import _RANGE_BEYOND_CR, _USE_CUE, _eff

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "answerability.csv"

# Same park pool the keys are built from: terrestrial national parks and biological
# reserves with a DEM elevation profile.
_PARK_SQL = """SELECT code, elev_p05, elev_p95, regions
               FROM mpcr.places
               WHERE category IN ('PN','RB') AND elev_p05 IS NOT NULL
                 AND name !~* 'marin|isla'
               ORDER BY code"""


def _levels(f: Ficha, parks: list[dict], recorded: dict) -> dict[str, str]:
    """Per question type, the evidence level that settles it for this species, or
    absent from the dict when nothing in the record settles it.

    Mirrors benchmark.questions.for_species: a type appears here exactly when that
    function would emit a question of that type for this species.
    """
    out: dict[str, str] = {}

    # --- Manual-stated ------------------------------------------------------------
    if f.elev_min is not None:
        out["elevation"] = "stated"
    if f.vertientes:
        out["vertiente"] = "stated"
    if f.flowering_months:
        out["flowering"] = "stated"

    own = _match_habits((f.morphology or "")[:160])
    if _FAMILY_HABIT.get(f.family) and _FAMILY_HABIT[f.family] not in own:
        own = [_FAMILY_HABIT[f.family]] + own
    forms = [h for h in own if h != "terrestre"]
    if forms:
        out["growth_form"] = "stated"
    else:
        genus = [h for h in _match_habits((f.genus_description or "")[:160]) if h != "terrestre"]
        if len(genus) == 1:
            out["growth_form"] = "inferred_relatives"

    tail = f.distribution_paragraph.split(";", 1)[-1]
    if f.endemic_cr or _RANGE_BEYOND_CR.search(tail):
        out["endemism"] = "stated"

    eff = _eff(f)
    if eff is not None:
        out["cultivation"] = "inferred_data"

    # --- occurrence + polygon + DEM ----------------------------------------------
    regions = set(f.regions)
    has_confirmed = has_possible = has_unlikely = False
    for p in parks:
        n70, n_all = recorded.get(p["code"], (0, 0))
        region_overlap = bool(regions & set(p["regions"] or []))
        elev_overlap = eff is not None and p["lo"] <= eff[1] and p["hi"] >= eff[0]
        if n70 >= 1:
            has_confirmed = True
        elif n_all == 0 and region_overlap and elev_overlap:
            has_possible = True
        elif n_all == 0 and regions and eff is not None and not region_overlap and not elev_overlap:
            has_unlikely = True
    if has_confirmed:
        out["park_confirmed"] = "stated_by_records"
    if has_possible:
        out["park_possible"] = "inferred_data"
    if has_unlikely:
        out["park_unlikely"] = "inferred_data"

    # --- nothing in the record settles these -------------------------------------
    # `light` is always unanswerable from the record; `outside_record` is answerable
    # only in the sense that the Manual names a use, which is the measurement below.
    if _USE_CUE.search(f"{f.discussion} {f.distribution_paragraph}"):
        out["outside_record"] = "stated"

    return out


# Which denominator each type is a fraction of, and what settles it.
_DENOM = {
    "elevation":      ("catalogue", "Manual elevation range"),
    "vertiente":      ("catalogue", "Manual slope statement"),
    "flowering":      ("catalogue", "Manual phenology"),
    "growth_form":    ("catalogue", "species description, else an unambiguous genus"),
    "endemism":       ("catalogue", "ENDEMICA, or a range beyond Costa Rica"),
    "cultivation":    ("catalogue", "Manual elevation and slope"),
    "park_confirmed": ("frame",     "records inside the polygon since 1970"),
    "park_possible":  ("frame",     "region and elevation overlap, no records"),
    "park_unlikely":  ("frame",     "no region overlap, no elevation overlap, no records"),
    "light":          ("catalogue", "nothing in the record"),
    "outside_record": ("catalogue", "a use named in the Manual text"),
}

_ORDER = ["elevation", "vertiente", "flowering", "growth_form", "endemism", "cultivation",
          "park_confirmed", "park_possible", "park_unlikely", "outside_record", "light"]


def compute(verbose: bool = True) -> dict:
    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_PARK_SQL)
            parks = [dict(zip(["code", "lo", "hi", "regions"], r)) for r in cur.fetchall()]

            cur.execute("""SELECT f.species, f.ficha, t.taxon_key
                           FROM mpcr.fichas f
                           LEFT JOIN mpcr.taxa t ON t.species = f.species""")
            rows = cur.fetchall()

            cur.execute("SELECT species_key, place_code, n_since_1970, n_records "
                        "FROM mpcr.species_places")
            by_key: dict[int, dict] = {}
            for key, code, n70, n_all in cur.fetchall():
                by_key.setdefault(key, {})[code] = (n70, n_all)
    finally:
        conn.close()

    settled: Counter = Counter()
    by_level: dict[str, Counter] = {q: Counter() for q in _ORDER}
    n_catalogue = 0
    n_frame = 0
    use_narrow = use_broad = 0

    for species, ficha_json, taxon_key in rows:
        f = Ficha(**ficha_json)
        n_catalogue += 1
        if taxon_key:
            n_frame += 1
        lv = _levels(f, parks, by_key.get(taxon_key, {}))
        for qtype, level in lv.items():
            settled[qtype] += 1
            by_level[qtype][level] += 1
        # Use mentions at two scopes: the key's own scope, and the whole entry.
        if _USE_CUE.search(f"{f.discussion} {f.distribution_paragraph}"):
            use_narrow += 1
        if _USE_CUE.search(f.full_text or ""):
            use_broad += 1

    report = {"n_catalogue": n_catalogue, "n_frame": n_frame,
              "use_mentions_key_scope": use_narrow, "use_mentions_full_entry": use_broad,
              "types": []}
    for qtype in _ORDER:
        denom_name, basis = _DENOM[qtype]
        denom = n_frame if denom_name == "frame" else n_catalogue
        n = settled[qtype]
        report["types"].append({
            "qtype": qtype, "settled": n, "denominator": denom, "denominator_name": denom_name,
            "pct": round(100 * n / denom, 1),
            "levels": "; ".join(f"{k}={v}" for k, v in by_level[qtype].most_common()) or "-",
            "basis": basis,
        })

    if verbose:
        _print(report)
    return report


def _print(r: dict) -> None:
    print(f"catalogue {r['n_catalogue']} species; occurrence frame {r['n_frame']} "
          f"({100 * r['n_frame'] / r['n_catalogue']:.1f}% resolve to a GBIF accepted key)\n")
    print(f"{'question type':16s} {'settled':>8s} {'of':>7s} {'pct':>7s}  basis")
    for t in r["types"]:
        print(f"{t['qtype']:16s} {t['settled']:8d} {t['denominator']:7d} {t['pct']:6.1f}%  "
              f"{t['basis']}")
    print("\nlevels per type")
    for t in r["types"]:
        print(f"  {t['qtype']:16s} {t['levels']}")
    n = r["n_catalogue"]
    print("\nuse mentions (the `uses` FIELD is empty for all species; these count the "
          "Manual's own use vocabulary)")
    print(f"  key scope (discussion + distribution)  {r['use_mentions_key_scope']:5d}  "
          f"{100 * r['use_mentions_key_scope'] / n:5.1f}%")
    print(f"  whole entry (full_text)                {r['use_mentions_full_entry']:5d}  "
          f"{100 * r['use_mentions_full_entry'] / n:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-question-type answerability of the record")
    ap.add_argument("--csv", action="store_true", help=f"also write {OUT}")
    args = ap.parse_args()
    r = compute()
    if args.csv:
        DATA.mkdir(exist_ok=True)
        with open(OUT, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["qtype", "settled", "denominator",
                                               "denominator_name", "pct", "levels", "basis"])
            w.writeheader()
            w.writerows(r["types"])
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
