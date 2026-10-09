"""Scoring. One verdict row per answer row, appended to JSONL.

A separate stage from generation, and separately resumable, because the two have very
different costs and failure modes: generation spends money, scoring spends about five
seconds of local GPU per answer and is the longer wall-clock job on a full run. Nothing
here may ever trigger a re-generation.

What decides a verdict is `benchmark.verify`: the typed key plus deterministic parsers
for level 1, and claim segmentation followed by the same parsers for level 2. The only
model involved is the local segmenter, and it assigns no score (docs/EXPERIMENT_PLAN.md
sec. 2c).

Run:
  python -m benchmark.score --part pilot
  python -m benchmark.score --part pilot --arms G
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mpcr_rag.store import pg_store

from . import verify as V

DATA = Path(__file__).resolve().parent / "data"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score generated answers")
    ap.add_argument("--part", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--arms", default="", help="comma separated filter")
    ap.add_argument("--extractor", default=V.EXTRACTOR)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent segmenter calls; only helps if the Ollama server "
                         "allows it (OLLAMA_NUM_PARALLEL)")
    args = ap.parse_args()

    answers = _jsonl(DATA / f"answers_{args.part}.jsonl")
    questions = {q["qid"]: q for q in _jsonl(DATA / f"questions_{args.part}.jsonl")}
    out = DATA / f"verdicts_{args.part}.jsonl"
    # An errored verdict is not a scored one. Counting it as done would retire the
    # answer permanently: the 20 read-timeouts from the first pass would never be
    # retried, and the arms they fell in would quietly carry fewer observations than
    # the others. run.py already treats its own failures this way.
    done = set() if args.fresh else {
        (r["qid"], r["arm"], r.get("replicate", 0)) for r in _jsonl(out)
        if not r.get("error")}

    if args.arms:
        want = {a.strip() for a in args.arms.split(",")}
        answers = [a for a in answers if a["arm"] in want]

    todo = [a for a in answers
            if not a.get("error") and a.get("answer_text")
            and (a["qid"], a["arm"], a.get("replicate", 0)) not in done]
    if args.limit:
        todo = todo[:args.limit]

    skipped = sum(1 for a in answers if a.get("error") or not a.get("answer_text"))
    print(f"answers={len(answers)}  already scored={len(done)}  to score={len(todo)}  "
          f"unusable={skipped}")

    todo = [a for a in todo if a["qid"] in questions]

    def _segment(a: dict):
        # The only slow step, and the only one that runs off the main thread. It
        # returns the exception rather than raising so one timeout cannot take the
        # pool down with it.
        try:
            return a, V.extract(questions[a["qid"]]["question"], a["answer_text"],
                                args.extractor), None
        except Exception as e:
            return a, None, e

    conn = pg_store.connect()
    tally: collections.Counter = collections.Counter()
    fh = io.open(out, "a", encoding="utf-8")
    n = fail = 0
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        with conn.cursor() as cur:
            futures = [pool.submit(_segment, a) for a in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                a, ext, err = fut.result()
                q = questions[a["qid"]]
                try:
                    if err is not None:
                        raise err
                    v = V.verify_extracted(q, a["answer_text"], ext, cur)
                    row = {"qid": a["qid"], "arm": a["arm"],
                           "replicate": a.get("replicate", 0),
                           "species": a.get("species"), "qtype": a.get("qtype"),
                           "key_level": q.get("key_level"),
                           "key_review_priority": q.get("key_review_priority"),
                           "stratum": q.get("stratum"), "n_records": q.get("n_records"),
                           "n_global": q.get("n_global"),
                           "evidence_level": a.get("evidence_level"),
                           "direct": v["direct"], "stance": v["stance"],
                           "abstains": v["abstains"],
                           "claim_counts": v["claim_counts"], "claims": v["claims"],
                           "extractor": args.extractor, "error": None}
                    tally[(a["arm"], v["direct"])] += 1
                    n += 1
                except Exception as e:
                    row = {"qid": a["qid"], "arm": a["arm"],
                           "replicate": a.get("replicate", 0),
                           "error": f"{type(e).__name__}: {e}"}
                    fail += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  scored={n} fail={fail}  "
                          f"{(time.time() - t0) / i:.1f} s/answer", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        fh.close()
        conn.close()

    print(f"\nscored {n}, {fail} errors -> {out}")
    if tally:
        print("\nverdicts by arm")
        arms = sorted({a for a, _ in tally})
        verdicts = sorted({v for _, v in tally})
        print(f"  {'verdict':20s}" + "".join(f"{a:>8s}" for a in arms))
        for vd in verdicts:
            print(f"  {vd:20s}" + "".join(f"{tally[(a, vd)]:8d}" for a in arms))


if __name__ == "__main__":
    main()
