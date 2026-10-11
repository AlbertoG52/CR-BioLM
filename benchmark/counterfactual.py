"""The counterfactual condition: finding F10.

Premise traceability is not enough. A model can quote a true premise and then answer
from memory anyway -- "right for the wrong reasons", in McCoy et al.'s phrase. The only
cheap way to tell the difference is to change the fact and ask again.

So for each question we build a second pack with exactly ONE fact altered, and a second
key recomputed under that alteration. A system that reads the evidence should now be
correct against the COUNTERFACTUAL key and wrong against the real one. A system that
ignored the evidence answers identically either way, and scores high on the real key and
near zero on the counterfactual.

That makes evidence use measurable with no expert involved, and it applies to the
faithfulness benchmark as it stands -- it answers "does arm D actually use the pack, or
does it only look grounded", which nothing else in the design can tell us.

Closed-book arms have no pack to corrupt and are the built-in control: their
counterfactual score equals their real score by construction, and any difference is
noise in the measurement rather than signal.

Perturbations are large and implausible on purpose. A 30 m shift could be absorbed by
the checker's 200 m tolerance; moving a species 1,800 m up a mountain cannot be.

Not every question type can be perturbed. `light` and `outside_record` rest on the
record's silence, and there is no fact there to alter -- they are simply excluded, and
the excluded count is reported rather than hidden.

Run:  python -m benchmark.counterfactual --species "Clethra consimilis" --qtype elevation
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mpcr_rag.store import pg_store

from .evidence_pack import build as build_pack

DATA = Path(__file__).resolve().parent / "data"

ELEV_SHIFT_M = 1800          # far outside the checker's 200 m tolerance
MONTH_SHIFT = 6              # opposite half of the year
_OTHER_SLOPE = {"Caribe": "Pacífico", "Pacífico": "Caribe"}

# "1150-2200 m", "0-700 m", with the Manual's en dash and optional parenthetical outliers
_DASH = "[-‐‑‒–—―]"
_NUM = r"\d{1,2}[.,]?\d{0,3}"
_ELEV_TEXT = re.compile(
    rf"({_NUM})\s*{_DASH}\s*\(?\s*{_DASH}?\s*({_NUM})\s*\??\)?\s*m\b")


def _patch_elev_text(paragraph: str, shift_m: int) -> str:
    """Shift every elevation range stated in the Manual paragraph by the same amount, so
    the prose and the structured block agree about the counterfactual world."""
    def bump(tok: str) -> str:
        n = int(tok.replace(".", "").replace(",", ""))
        return str(n + shift_m)

    def repl(m: re.Match) -> str:
        return f"{bump(m.group(1))}–{bump(m.group(2))} m"

    return _ELEV_TEXT.sub(repl, paragraph)

# Perturbable AND internally consistent once perturbed. The pack carries both a
# structured block and the raw Manual paragraph, so altering only the structured value
# leaves the two contradicting each other -- and a careful model reading the paragraph
# would answer from the untouched text, which would show up as "ignores the evidence"
# when it is really "we corrupted the evidence badly". Elevation is patched in both
# places by _patch_elev_text; the park types are clean because the paragraph never
# mentions protected areas or record counts.
SUPPORTED = ("elevation", "cultivation", "park_confirmed", "park_possible", "park_unlikely")

# vertiente and flowering ARE perturbable, but the Manual paragraph states them in
# abbreviated prose ("vert. Carib.", "Fl. jun.-set.") that would need reliable text
# surgery to keep consistent. Excluded from F10 until that is written, rather than run
# with a self-contradicting pack.
EXCLUDED = {"light": "the record's silence is the point; no fact to alter",
            "outside_record": "the record's silence is the point; no fact to alter",
            "growth_form": "altering it would contradict the morphology block",
            "endemism": "altering it would contradict the distribution paragraph",
            "vertiente": "paragraph prose would contradict the structured block",
            "flowering": "paragraph prose would contradict the structured block"}


def plan(q: dict, ficha_facts: dict) -> dict | None:
    """What to alter for this question, and what the key becomes. None = not perturbable.

    Returns {overrides, place_overrides, key, altered} where `key` replaces q["key"] when
    scoring the counterfactual answer.
    """
    qtype, key = q["qtype"], q["key"]

    if qtype in ("elevation", "cultivation"):
        lo, hi = ficha_facts.get("elev_min"), ficha_facts.get("elev_max")
        if lo is None or hi is None:
            return None
        nlo, nhi = lo + ELEV_SHIFT_M, hi + ELEV_SHIFT_M
        ov = {"elev_min": nlo, "elev_max": nhi,
              "elev_outlier_min": None, "elev_outlier_max": None}
        para = ficha_facts.get("_distribution_paragraph")
        if para:
            patched = _patch_elev_text(para, ELEV_SHIFT_M)
            if patched == para:
                # The paragraph states its elevations in a form the patcher does not
                # recognise -- the Manual's parenthetical outliers, "0-100(-1000?) m",
                # are the usual case. Leaving the prose untouched would hand the model a
                # pack that contradicts itself, and a model that correctly followed the
                # prose would be recorded as ignoring the evidence. Refuse the question
                # instead; `report` counts refusals so the coverage is visible.
                return None
            ov["distribution_paragraph"] = patched
        if qtype == "elevation":
            new_key = {"min": nlo, "max": nhi, "outlier_min": None, "outlier_max": None}
        else:
            # Re-apply the keys' own rule under the shifted band. The setting elevation
            # is fixed by the question text, so moving the species out from under it
            # should turn "probable" into "improbable".
            elev = key.get("setting_elevation")
            slope = key.get("setting_slope")
            vert = ficha_facts.get("vertientes") or []
            in_core = elev is not None and nlo <= elev <= nhi
            slope_ok = (not vert) or (slope in vert)
            new_key = {**key,
                       "answer": ("probable" if in_core and slope_ok else
                                  "incierto" if (in_core and not slope_ok)
                                  or (elev is not None and nlo - 200 <= elev <= nhi + 200)
                                  else "improbable"),
                       "species_elevation": [nlo, nhi],
                       "species_elevation_effective": [nlo, nhi]}
        return {"overrides": ov, "place_overrides": None, "key": new_key,
                "altered": f"elevation {lo}-{hi} -> {nlo}-{nhi} m"}

    if qtype == "vertiente" and False:   # see EXCLUDED
        v = ficha_facts.get("vertientes") or []
        flipped = [_OTHER_SLOPE.get(x, x) for x in v]
        if not v or flipped == v:
            return None
        return {"overrides": {"vertientes": flipped}, "place_overrides": None,
                "key": {"vertientes": flipped},
                "altered": f"vertientes {v} -> {flipped}"}

    if qtype == "flowering" and False:   # see EXCLUDED
        mo = ficha_facts.get("flowering_months") or []
        if not mo:
            return None
        new = sorted({((m - 1 + MONTH_SHIFT) % 12) + 1 for m in mo})
        return {"overrides": {"flowering_months": new}, "place_overrides": None,
                "key": {"months": new, "months_es": None},
                "altered": f"flowering months {mo} -> {new}"}

    if qtype.startswith("park_"):
        # Flip the record count, which is what the keys' rule turns on: a park with
        # records is "sí"; the same park with none falls back to overlap.
        had = key.get("records_in_place") or 0
        if had >= 1:
            po = {"n_since_1970": 0, "n_records": 0}
            # with no records the verdict comes from overlap, which the pack states
            sp_r = set(ficha_facts.get("regions") or [])
            pk_r = set(key.get("place_regions") or [])
            eff = ficha_facts.get("elev_effective")
            p05, p95 = key.get("place_elevation_p05_p95") or [None, None]
            overlap_r = bool(sp_r & pk_r)
            overlap_e = bool(eff and p05 is not None and p05 <= eff[1] and p95 >= eff[0])
            ans = ("posible" if overlap_r and overlap_e else
                   "improbable" if not overlap_r and not overlap_e else "incierto")
            altered = f"records in place {had} -> 0"
        else:
            if key.get("answer") == "improbable":
                # Planting records in a park the pack still describes as overlapping
                # neither the species' regions nor its elevation makes the evidence
                # contradict itself. In the 20-species run both grounded models saw the
                # planted records, said so, and sided with the overlap facts -- which
                # was scored as ignoring the evidence, 0 of 11 and 1 of 10 followed. A
                # pack that disagrees with itself cannot test fidelity, so the question
                # is refused and counted with the other non-perturbable ones.
                return None
            po = {"n_since_1970": 7, "n_records": 7}
            ans = "sí"
            altered = "records in place 0 -> 7"
        return {"overrides": None, "place_overrides": po,
                "key": {**key, "answer": ans, "records_in_place": po["n_records"]},
                "altered": altered}

    return None


def build(cur, q: dict, level: str = "full") -> tuple | None:
    """(pack, counterfactual_key, altered) or None when the question is not perturbable."""
    real = build_pack(cur, q["species"], q["qtype"], level=level,
                      place_code=q["key"].get("place_code"))
    p = plan(q, real.facts)
    if not p:
        return None
    pack = build_pack(cur, q["species"], q["qtype"], level=level,
                      place_code=q["key"].get("place_code"),
                      overrides=p["overrides"], place_overrides=p["place_overrides"])
    return pack, p["key"], p["altered"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Show one counterfactual pack")
    ap.add_argument("--part", default="pilot")
    ap.add_argument("--species", required=True)
    ap.add_argument("--qtype", required=True)
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(DATA / f"questions_{args.part}.jsonl",
                                      encoding="utf-8") if l.strip()]
    q = next((x for x in qs if x["species"] == args.species and x["qtype"] == args.qtype), None)
    if not q:
        raise SystemExit("no such question in that part")

    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            got = build(cur, q)
            if not got:
                raise SystemExit(f"{args.qtype} is not perturbable")
            pack, key, altered = got
            print(f"QUESTION : {q['question']}")
            print(f"REAL KEY : {q['key']}")
            print(f"ALTERED  : {altered}")
            print(f"CF KEY   : {key}\n")
            print(pack.text())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
