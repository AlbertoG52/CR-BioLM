"""Tables from verdict rows. Every number here is a count of rows, grouped.

The metrics are the ones section 5 of docs/EXPERIMENT_PLAN.md defines, and the reason
precision is never printed without coverage is that either one alone is gameable: a
system that abstains everywhere has perfect precision and no coverage, one that always
guesses has the reverse.

Scoreable excludes `not_scored` (no key yet, e.g. the light questions) and
`beyond_record`. Abstentions count against coverage but not against precision, except
on the questions where the record is silent -- there abstaining IS the correct answer,
and `silence` reports it separately.

Run:
  python -m benchmark.analyze --part pilot
  python -m benchmark.analyze --part pilot --by qtype
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

ANSWERED = ("correct", "partially_correct", "overstated", "understated", "wrong")
UNSCOREABLE = ("not_scored", "beyond_record")


def _dedupe(rows: list[dict]) -> list[dict]:
    """One row per (qid, arm, replicate), last write wins.

    Two scoring passes over the same answers append two verdict rows. They usually
    agree, but the local segmenter is not bit-reproducible under concurrency -- batch
    composition varies and float reduction order with it -- so 2 of 1,253 duplicate
    pairs in the 20-species run disagreed. Counting both would weight those answers
    twice and make a table depend on how many times scoring happened to run.
    """
    keep = {}
    for r in rows:
        if r.get("qid"):
            keep[(r["qid"], r.get("arm"), r.get("replicate", 0))] = r
    return list(keep.values()) if keep else rows


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


def _wilson(k: int, n: int) -> tuple[float, float]:
    """95% interval. Small runs produce small n, and a bare proportion over 12 items
    invites over-reading; printing the interval makes the uncertainty visible."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - hw) / d), min(1.0, (c + hw) / d))


def summarize(rows: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if not r.get("error") and r.get("direct"):
            by_arm[r["arm"]].append(r)

    out = {}
    for arm, rs in sorted(by_arm.items()):
        scoreable = [r for r in rs if r["direct"] not in UNSCOREABLE]
        answered = [r for r in scoreable if r["direct"] in ANSWERED]
        correct = [r for r in answered if r["direct"] == "correct"]
        # Finding F5. On a question the record cannot settle, `verify.check_direct`
        # scores a declined answer as `correct` and an asserted use as `beyond_record`
        # -- it does not use the `abstained` label there, because the checker cannot
        # tell an invention from true knowledge the Manual lacks, so that case is
        # routed to experts rather than marked wrong. The right pair of numbers is
        # therefore correct-vs-beyond_record over the silent questions, not the
        # abstention rate.
        silent_qs = [r for r in rs if r.get("key_level") == "none"]
        silent_ok = [r for r in silent_qs if r["direct"] == "correct"]
        silent_asserted = [r for r in silent_qs if r["direct"] == "beyond_record"]
        claims: collections.Counter = collections.Counter()
        for r in rs:
            claims.update(r.get("claim_counts") or {})
        total_claims = sum(claims.values())
        out[arm] = {
            "n": len(rs), "scoreable": len(scoreable), "answered": len(answered),
            "correct": len(correct),
            "precision": len(correct) / len(answered) if answered else None,
            "precision_ci": _wilson(len(correct), len(answered)),
            "coverage": len(answered) / len(scoreable) if scoreable else None,
            "silence_n": len(silent_qs),
            "silence_correct": len(silent_ok) / len(silent_qs) if silent_qs else None,
            "silence_asserted": len(silent_asserted) / len(silent_qs) if silent_qs else None,
            "claims": dict(claims),
            "contradiction_rate": claims["contradicted"] / total_claims if total_claims else None,
            "support_rate": claims["supported"] / total_claims if total_claims else None,
            "verdicts": collections.Counter(r["direct"] for r in rs),
        }
    return out


def _pct(v) -> str:
    return "  n/a" if v is None else f"{100 * v:5.1f}"


def print_main(summ: dict) -> None:
    print(f"{'arm':4s} {'n':>4s} {'scor':>5s} {'ans':>4s} {'prec%':>6s} {'95% CI':>14s} "
          f"{'cov%':>6s} {'silence_ok%':>12s} {'invented%':>10s} {'contra%':>8s}")
    for arm, s in summ.items():
        lo, hi = s["precision_ci"]
        ci = f"[{100*lo:4.0f},{100*hi:4.0f}]" if s["precision"] is not None else "       --     "
        print(f"{arm:4s} {s['n']:4d} {s['scoreable']:5d} {s['answered']:4d} "
              f"{_pct(s['precision']):>6s} {ci:>14s} {_pct(s['coverage']):>6s} "
              f"{_pct(s['silence_correct']):>12s} {_pct(s['silence_asserted']):>10s} "
              f"{_pct(s['contradiction_rate']):>8s}")


def print_verdicts(summ: dict) -> None:
    arms = list(summ)
    labels = sorted({v for s in summ.values() for v in s["verdicts"]})
    print(f"\n{'verdict':20s}" + "".join(f"{a:>6s}" for a in arms))
    for lab in labels:
        print(f"{lab:20s}" + "".join(f"{summ[a]['verdicts'][lab]:6d}" for a in arms))


def print_claims(summ: dict) -> None:
    arms = list(summ)
    cats = ("supported", "contradicted", "not_in_record", "unverifiable")
    print(f"\n{'claims':20s}" + "".join(f"{a:>6s}" for a in arms))
    for c in cats:
        print(f"{c:20s}" + "".join(f"{summ[a]['claims'].get(c, 0):6d}" for a in arms))


def print_by(rows: list[dict], field: str) -> None:
    arms = sorted({r["arm"] for r in rows if r.get("direct")})
    keys = sorted({str(r.get(field)) for r in rows if r.get("direct")})
    print(f"\nprecision by {field} (correct / answered)")
    print(f"{field:18s}" + "".join(f"{a:>10s}" for a in arms))
    for k in keys:
        cells = []
        for a in arms:
            sub = [r for r in rows if r["arm"] == a and str(r.get(field)) == k
                   and r.get("direct") in ANSWERED]
            ok = sum(1 for r in sub if r["direct"] == "correct")
            cells.append(f"{ok}/{len(sub)}" if sub else "-")
        print(f"{k:18s}" + "".join(f"{c:>10s}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize verdicts")
    ap.add_argument("--part", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--by", default="", help="also break down by this field, e.g. qtype")
    args = ap.parse_args()

    rows = _jsonl(DATA / f"verdicts_{args.part}.jsonl")
    if not rows:
        raise SystemExit(f"no verdicts for part={args.part}; run benchmark.score first")
    summ = summarize(rows)
    print(f"verdicts: {len(rows)} rows, {len(summ)} arms, "
          f"{len({r['species'] for r in rows if r.get('species')})} species\n")
    print_main(summ)
    print_verdicts(summ)
    print_claims(summ)
    if args.by:
        print_by(rows, args.by)
    print("\nprecision = correct / answered (abstentions excluded); coverage = answered / "
          "scoreable.\nNeither means anything alone -- see docs/EXPERIMENT_PLAN.md sec. 5.")


if __name__ == "__main__":
    main()
