"""Run the counterfactual condition (finding F10) and score it.

`counterfactual.py` builds a pack with one fact altered and the key recomputed under
that alteration. This module is the missing harness: it asks the grounded arms the same
question over the altered pack, and scores each answer twice, against the
COUNTERFACTUAL key and against the REAL one.

  followed   correct against the counterfactual key: the answer came from the evidence
  recalled   correct against the real key: the answer came from somewhere else, since
             the evidence in front of the model said otherwise

A system that reads its evidence should follow nearly always and recall nearly never.
This is the fidelity the thesis proposal defines -- an answer that is a faithful
rendering of the context it was given -- measured without anyone's judgement.

Generation and scoring live in one script because the set is small (one call per
perturbable question per arm) and neither stage is worth resuming separately; both
still skip rows already written.

Run:
  python -m benchmark.counterfactual_run --part main --species-file benchmark/data/species_20.json
  python -m benchmark.counterfactual_run --part main --report
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import io
import json
from pathlib import Path

from mpcr_rag.store import pg_store

from . import arms as A
from . import client, prompts
from . import verify as V
from .counterfactual import build as build_cf

DATA = Path(__file__).resolve().parent / "data"
ARMS = ("D", "F")           # the two grounded arms; closed-book arms have no pack to alter


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def generate(part: str, species: set[str] | None, max_cost: float, workers: int) -> None:
    out = DATA / f"answers_cf_{part}.jsonl"
    done = {(r["qid"], r["arm"]) for r in _jsonl(out) if not r.get("error")}
    qs = [q for q in _jsonl(DATA / f"questions_{part}.jsonl")
          if species is None or q["species"] in species]

    jobs, refused = [], collections.Counter()
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for q in qs:
            got = build_cf(cur, q)
            if not got:
                refused[q["qtype"]] += 1
                continue
            pack, cf_key, altered = got
            for arm in ARMS:
                if (q["qid"], arm) not in done:
                    jobs.append({"q": q, "arm": arm, "cf_key": cf_key, "altered": altered,
                                 "messages": prompts.build(q["question"], pack.text(),
                                                           family=A.ARMS[arm].prompt)})
    conn.close()
    print(f"perturbable questions: {len(jobs) // len(ARMS) if jobs else 0} to run; "
          f"not perturbable by type: {dict(refused)}")

    def execute(j: dict) -> dict:
        base = {"qid": j["q"]["qid"], "arm": j["arm"], "species": j["q"]["species"],
                "qtype": j["q"]["qtype"], "altered": j["altered"], "cf_key": j["cf_key"],
                "started_at": dt.datetime.now().isoformat(timespec="seconds")}
        try:
            res = client.call(A.ARMS[j["arm"]].model, j["messages"], pin=A.ARMS[j["arm"]].pin)
            return {**base, **res, "error": None}
        except Exception as e:
            return {**base, "answer_text": None, "error": f"{type(e).__name__}: {e}"}

    spent = 0.0
    with io.open(out, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(execute, jobs), 1):
            spent += float(row.get("cost_usd") or 0)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} spent=${spent:.4f}", flush=True)
            if spent > max_cost:
                raise SystemExit(f"cost ceiling ${max_cost:.2f} hit; re-run to resume")


def score(part: str, workers: int) -> None:
    out = DATA / f"verdicts_cf_{part}.jsonl"
    done = {(r["qid"], r["arm"]) for r in _jsonl(out) if not r.get("error")}
    qs = {q["qid"]: q for q in _jsonl(DATA / f"questions_{part}.jsonl")}
    todo = [a for a in _jsonl(DATA / f"answers_cf_{part}.jsonl")
            if not a.get("error") and a.get("answer_text") and (a["qid"], a["arm"]) not in done]

    def segment(a: dict):
        try:
            return a, V.extract(qs[a["qid"]]["question"], a["answer_text"]), None
        except Exception as e:
            return a, None, e

    with io.open(out, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(workers) as pool:
        for a, ext, err in pool.map(segment, todo):
            q = qs[a["qid"]]
            if err is not None:
                row = {"qid": a["qid"], "arm": a["arm"], "error": f"{type(err).__name__}: {err}"}
            else:
                # Only the direct verdict is meaningful here. The claim-level check
                # compares against the real record, which the altered pack contradicts
                # by construction.
                row = {"qid": a["qid"], "arm": a["arm"], "qtype": a["qtype"],
                       "species": a["species"], "altered": a["altered"],
                       "direct_cf": V.check_direct({**q, "key": a["cf_key"]},
                                                   a["answer_text"], ext),
                       "direct_real": V.check_direct(q, a["answer_text"], ext),
                       "stance": ext.get("stance"), "error": None}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"scored {len(todo)} -> {out}")


def report(part: str) -> None:
    rows = {(r["qid"], r["arm"]): r for r in _jsonl(DATA / f"verdicts_cf_{part}.jsonl")
            if not r.get("error")}
    real = {(r["qid"], r["arm"]): r["direct"] for r in _jsonl(DATA / f"verdicts_{part}.jsonl")
            if not r.get("error")}
    print(f"{'arm':4s} {'type':16s} {'n':>4s} {'followed':>9s} {'recalled':>9s} {'neither':>8s} "
          f"{'right on real pack':>19s}")
    for arm in ARMS:
        sub = [r for (q, a), r in rows.items() if a == arm]
        groups = {"ALL": sub}
        for r in sub:
            groups.setdefault(r["qtype"], []).append(r)
        for name in ["ALL"] + sorted(k for k in groups if k != "ALL"):
            g = groups[name]
            fol = sum(r["direct_cf"] == "correct" for r in g)
            rec = sum(r["direct_real"] == "correct" and r["direct_cf"] != "correct" for r in g)
            base = sum(real.get((r["qid"], arm)) == "correct" for r in g)
            print(f"{arm:4s} {name:16s} {len(g):4d} {fol:9d} {rec:9d} {len(g) - fol - rec:8d} "
                  f"{base:19d}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run and score the counterfactual condition")
    ap.add_argument("--part", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--species-file", default="")
    ap.add_argument("--max-cost", type=float, default=1.00)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", action="store_true", help="only print the table")
    args = ap.parse_args()
    if not args.report:
        species = (set(json.loads(Path(args.species_file).read_text(encoding="utf-8")))
                   if args.species_file else None)
        generate(args.part, species, args.max_cost, args.workers)
        score(args.part, 4)
    report(args.part)


if __name__ == "__main__":
    main()
