"""What the expert review of tranche 1 found, counted from the returned workbook.

The workbook (expert_review_tramo1.xlsx, built by expert_sheet.py) showed a botanist 30
pairs of answers, each a free closed-book answer (A1F) and a grounded one (D) in random
order, with the record's key in plain words. Per pair the expert said which answer
deserved more trust, whether each had a botanical error, and whether the key was right.
Per species the expert also said whether the system's map was consistent with what is
known of the plant.

The checker's own verdicts are crossed with the expert's error flags here: an answer the
expert says is wrong should be caught by the direct verdict (anything but `correct`) or,
failing that, by a claim marked `contradicted`. Answers to the light question are left
out of that count because the type has no key and the checker does not score it.

Run:  python -m benchmark.expert_results
"""
from __future__ import annotations

import collections
import io
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SHEET = DATA / "expert_feedback" / "expert_review_tramo1_ARMANDO.xlsx"
GROUNDED, FREE = "D", "A1F"


def _yes(v) -> bool:
    return str(v or "").strip().lower().startswith("s")


def _verdicts() -> dict:
    out = {}
    for line in io.open(DATA / "verdicts_main.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if not r.get("error") and r.get("direct"):
            out[(r["qid"], r["arm"])] = r
    return out


def summarize() -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(SHEET, data_only=True)
    key = {k["n"]: k for k in (json.loads(l) for l in
                               io.open(DATA / "expert_review_key.jsonl", encoding="utf-8"))}
    ver = _verdicts()
    pref = collections.Counter()
    errors = collections.Counter()
    caught = collections.Counter()
    missed, key_wrong = [], []
    for row in wb["Revisión"].iter_rows(min_row=3, values_only=True):
        if row[0] is None:
            continue
        k = key[int(row[0])]
        choice = str(row[7] or "").strip()
        pref[{"A": k["A"], "B": k["B"]}.get(choice, "equal")] += 1
        for side, flag in (("A", row[8]), ("B", row[9])):
            if not _yes(flag):
                continue
            arm = k[side]
            errors[arm] += 1
            v = ver.get((k["qid"], arm), {})
            if v.get("direct") in ("not_scored", "beyond_record"):
                caught["unscoreable"] += 1
            elif v.get("direct") != "correct":
                caught["direct"] += 1
            elif any(c.get("verdict") == "contradicted" for c in v.get("claims") or []):
                caught["claim"] += 1
            else:
                missed.append({"species": k["species"], "qtype": k["qtype"], "arm": arm})
        if not _yes(row[10]):
            key_wrong.append({"species": k["species"], "qtype": k["qtype"],
                              "expert": row[11]})
    maps = collections.Counter()
    for row in wb["Fichas y mapas"].iter_rows(values_only=True):
        if str(row[0] or "").startswith("¿El mapa es consistente"):
            a = str(row[1] or "").strip().lower()
            maps["yes" if a == "si" else "no" if a == "no" else "partly"] += 1
    n = sum(pref.values())
    return {"pairs": n, "preferred": dict(pref), "errors": dict(errors),
            "scoreable_errors": sum(errors.values()) - caught["unscoreable"],
            "caught": dict(caught), "missed": missed, "keys_correct": n - len(key_wrong),
            "key_wrong": key_wrong, "maps": dict(maps)}


if __name__ == "__main__":
    print(json.dumps(summarize(), ensure_ascii=False, indent=1))
