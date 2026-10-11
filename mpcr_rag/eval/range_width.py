"""Does MPCR-GBIF agreement depend on how wide the Manual's elevation range is?

Reviewer 1 observed that a binary criterion ("is the GBIF median inside the Manual
range?") is easier to satisfy when the Manual range is wide. This script tests that
directly: it stratifies the validation set by Manual range width and reports the
agreement rate per stratum, alongside a continuous measure that does not saturate.

The continuous measure is the intersection-over-union of the GBIF core interval
[q05, q95] with the Manual range [l, u] -- the same IoU the paper already uses for
the grounding experiment, applied here to the external comparison.

Run:
    python -m mpcr_rag.eval.range_width
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

# Elevations are non-negative; a leading "-" here is the range separator, never a sign.
_NUM = re.compile(r"\d+")


def parse_range(s: str):
    """'100-300' / '0-500' / '800' -> (lo, hi); None if unparseable."""
    nums = [int(x) for x in _NUM.findall(s or "")]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums), max(nums)


def iou(a, b) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0:                      # both degenerate at the same point
        return 1.0 if inter >= 0 and a == b else 0.0
    return inter / union


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="mpcr_rag/eval/results/manual_vs_gbif.csv")
    ap.add_argument("--min-n", type=int, default=10)
    args = ap.parse_args()

    recs = []
    for r in csv.DictReader(open(args.csv, encoding="utf-8")):
        try:
            if int(float(r["n"])) < args.min_n:
                continue
            man = parse_range(r["manual"])
            if man is None:
                continue
            q05, q95 = float(r["q05"]), float(r["q95"])
            recs.append({
                "species": r["species"],
                "width": man[1] - man[0],
                "median_in": r["median_in"].strip().lower() == "true",
                "core_overlap": r["core_overlap"].strip().lower() == "true",
                "iou": iou((q05, q95), man),
            })
        except (ValueError, KeyError):
            continue

    n = len(recs)
    print(f"species with n >= {args.min_n}: {n}")
    print(f"median agreement overall : {sum(r['median_in'] for r in recs)}/{n} "
          f"= {100 * sum(r['median_in'] for r in recs) / n:.0f}%")
    print(f"core overlap overall     : {sum(r['core_overlap'] for r in recs)}/{n} "
          f"= {100 * sum(r['core_overlap'] for r in recs) / n:.0f}%")
    ious = [r["iou"] for r in recs]
    print(f"core IoU  median={st.median(ious):.2f}  mean={st.fmean(ious):.2f}  "
          f"min={min(ious):.2f}  max={max(ious):.2f}")
    print()

    # --- stratify by Manual range width (quartiles of the observed widths)
    recs.sort(key=lambda r: r["width"])
    q = [recs[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    print(f"{'stratum':22s} {'n':>4s} {'width (m)':>12s} {'median in':>10s} "
          f"{'core ovl':>9s} {'core IoU':>9s}")
    for i, grp in enumerate(q, 1):
        if not grp:
            continue
        w = [r["width"] for r in grp]
        mi = 100 * sum(r["median_in"] for r in grp) / len(grp)
        co = 100 * sum(r["core_overlap"] for r in grp) / len(grp)
        gi = st.median([r["iou"] for r in grp])
        print(f"Q{i} (narrowest+{i - 1:d})".ljust(22)
              + f" {len(grp):>4d} {min(w):>5.0f}-{max(w):<6.0f} {mi:>9.0f}% "
                f"{co:>8.0f}% {gi:>9.2f}")

    narrow = [r for r in recs if r["width"] <= 500]
    if narrow:
        mi = 100 * sum(r["median_in"] for r in narrow) / len(narrow)
        print(f"\nnarrow ranges (<= 500 m): {len(narrow)} species, "
              f"median agreement {mi:.0f}%, "
              f"core IoU median {st.median([r['iou'] for r in narrow]):.2f}")

    out = Path("mpcr_rag/eval/results/range_width.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\nper-species results -> {out}")


if __name__ == "__main__":
    main()
