"""End-to-end retrieval evaluation: does the system find what the query asked for?

The grounding experiment (rag_vs_baseline) hands the model the correct Manual entry,
so it measures the value of the source passage, not the quality of retrieval. This
script closes that gap on the two stages the grounding experiment skips:

  1. Constraint parsing -- does the LLM intent parser recover the filter a human
     would have written for the question?
  2. Set retrieval     -- does the filter the parser produced select the same catalog
     entries as the gold filter?

Ground truth is the gold filter written by hand for each query. The gold ANSWER SET is
then computed deterministically by executing that gold filter over the catalog, so set
precision and recall isolate parser error from catalog error: a species missing from
the Manual cannot be penalised here, only a mis-parsed constraint can.

Filters are executed offline against SQLite rather than through Pinecone. Both apply
the same conjunctive constraints and return the same set; SQLite makes the run
reproducible without a vector database.

Run (needs OPENROUTER_API_KEY; ~30 calls to the intent parser):
    python -m mpcr_rag.eval.retrieval_eval \
        --db mpcr_rag/data/fichas.pre_trackB_20260912.sqlite
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

from ..query import intent as intent_mod
from ..query.retriever import filter_all
from ..store import local_store

FIELDS = ["habit", "forest_type", "vertiente", "region", "family",
          "elev_lo", "elev_hi", "flowering_month", "endemic"]


def _filter_of(d: dict) -> dict:
    """Keep only the structured filter fields, dropping unset ones."""
    return {k: d.get(k) for k in FIELDS if d.get(k) is not None}


def _species(conn, flt: dict) -> set[str]:
    return {f.species for f in filter_all(conn=conn, **flt)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--testset", default="mpcr_rag/eval/data/query_testset.json")
    ap.add_argument("--out", default="mpcr_rag/eval/results/retrieval_eval.csv")
    args = ap.parse_args()

    conn = local_store.connect(args.db)
    vocab = intent_mod.load_vocab(conn)
    cases = json.loads(Path(args.testset).read_text(encoding="utf-8"))["queries"]

    rows = []
    for c in cases:
        gold = _filter_of(c["gold"])
        parsed_full = intent_mod.parse_intent(c["q"], vocab=vocab)
        parsed = _filter_of(parsed_full)

        field_hits = {f: (gold.get(f) == parsed.get(f)) for f in FIELDS}
        gset, pset = _species(conn, gold), _species(conn, parsed)
        inter = gset & pset
        prec = len(inter) / len(pset) if pset else 0.0
        rec = len(inter) / len(gset) if gset else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        rows.append({
            "id": c["id"], "q": c["q"],
            "exact_filter": all(field_hits.values()),
            "fields_correct": sum(field_hits.values()),
            "n_gold": len(gset), "n_pred": len(pset),
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "gold_filter": json.dumps(gold, ensure_ascii=False),
            "parsed_filter": json.dumps(parsed, ensure_ascii=False),
            **{f"fld_{f}": field_hits[f] for f in FIELDS},
        })
        print(f"  [{c['id']:>2}] P={prec:.2f} R={rec:.2f} "
              f"fields={sum(field_hits.values())}/{len(FIELDS)}  {c['q'][:52]}",
              flush=True)

    n = len(rows)
    print(f"\nqueries                  : {n}")
    print(f"exact filter match       : {sum(r['exact_filter'] for r in rows)}/{n} "
          f"= {100 * sum(r['exact_filter'] for r in rows) / n:.0f}%")
    tot_fields = n * len(FIELDS)
    ok_fields = sum(r["fields_correct"] for r in rows)
    print(f"per-field accuracy       : {ok_fields}/{tot_fields} "
          f"= {100 * ok_fields / tot_fields:.1f}%")
    print(f"set precision (mean)     : {st.fmean(r['precision'] for r in rows):.3f}")
    print(f"set recall (mean)        : {st.fmean(r['recall'] for r in rows):.3f}")
    print(f"set F1 (mean)            : {st.fmean(r['f1'] for r in rows):.3f}")
    print(f"queries with recall 1.0  : {sum(1 for r in rows if r['recall'] == 1.0)}/{n}")

    print("\nper-field accuracy:")
    for f in FIELDS:
        k = sum(1 for r in rows if r[f"fld_{f}"])
        print(f"  {f:16s} {k:>2}/{n}  {100 * k / n:5.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-query results -> {out}")


if __name__ == "__main__":
    main()
