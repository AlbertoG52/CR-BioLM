"""Every results table and figure in the paper, generated from the verdict rows.

Nothing in main.tex states a measured number about a model run by hand: the tables are
\\input from paper2/generated/, and re-running this script after a new run replaces
them. That is how the draft moves from the five-species shakedown to the 20-species
run to the main run without anyone retyping a percentage.

The counting is `benchmark.analyze.summarize`, imported rather than reimplemented, so
the paper and the console report cannot disagree.

Run from the repository root:
  python paper2/build_results.py --part pilot
  python paper2/build_results.py --part main --label "20-species preliminary run"
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.analyze import ANSWERED, UNSCOREABLE, summarize  # noqa: E402
from benchmark.arms import ARMS, MAP_QTYPES  # noqa: E402

DATA = ROOT / "benchmark" / "data"
OUT = Path(__file__).resolve().parent / "generated"
FIGS = Path(__file__).resolve().parent / "figs"

ORDER = ["G", "D", "F", "E", "A1", "C", "A1F", "CF", "DM", "FM"]
GENERATOR = {"openrouter:mistralai/mistral-large-4-0": "Mistral",
             "openrouter:google/gemma-4-31b-it": "Gemma", None: "template"}
EVIDENCE = {None: "none", "flora": "flora only", "full": "full pack", "map": "map only"}
PROMPT = {"shared": "careful", "minimal": "free", "map": "careful"}
PARK = ("park_confirmed", "park_possible", "park_unlikely")
QTYPES = ["elevation", "vertiente", "flowering", "growth_form", "endemism", "cultivation",
          "park_confirmed", "park_possible", "park_unlikely", "co_occurrence",
          "outside_record"]

# Edible-or-medicinal mentions, the narrow predicate the question is keyed on
# (questions._EDIBLE_MED_CUE). answerability.csv carries the broad "any use" count;
# this one was counted over all 6,946 entries on 2026-10-09 with the same regex.
EDIBLE_MEDICINAL = (133, 6946)


def _rows(part: str) -> list[dict]:
    """Verdict rows, one per (qid, arm, replicate): an errored row followed by its
    retry is one observation, not two."""
    keep: dict[tuple, dict] = {}
    for line in io.open(DATA / f"verdicts_{part}.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or not r.get("direct"):
            continue
        keep[(r["qid"], r["arm"], r.get("replicate", 0))] = r
    return list(keep.values())


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def _pct(v) -> str:
    return "--" if v is None else f"{100 * v:.1f}"


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: a binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def table_arms(summ: dict) -> str:
    lines = [r"\begin{tabular}{llllrrrrrr}", r"\toprule",
             r"Arm & Generator & Evidence & Prompt & $n$ & Answered & Precision & 95\% CI "
             r"& Coverage & Contrad. \\", r"\midrule"]
    for a in [x for x in ORDER if x in summ]:
        s, arm = summ[a], ARMS[a]
        lo, hi = s["precision_ci"]
        ci = f"[{100*lo:.0f}, {100*hi:.0f}]" if s["precision"] is not None else "--"
        lines.append(
            f"{a} & {GENERATOR[arm.model]} & {EVIDENCE[arm.evidence]} & "
            f"{PROMPT[arm.prompt] if arm.is_llm else '--'} & {s['scoreable']} & "
            f"{s['answered']} & {_pct(s['precision'])} & {ci} & {_pct(s['coverage'])} & "
            f"{_pct(s['contradiction_rate'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_qtype(rows: list[dict], arms: list[str]) -> str:
    lines = [r"\begin{tabular}{l" + "r" * len(arms) + "}", r"\toprule",
             "Question type & " + " & ".join(arms) + r" \\", r"\midrule"]
    for qt in QTYPES:
        cells = []
        for a in arms:
            sub = [r for r in rows if r["arm"] == a and r["qtype"] == qt
                   and r["direct"] in ANSWERED]
            ok = sum(r["direct"] == "correct" for r in sub)
            cells.append(f"{ok}/{len(sub)}" if sub else "--")
        if any(c != "--" for c in cells):
            lines.append(f"{_tex(qt)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_verdicts(summ: dict, arms: list[str]) -> str:
    labels = ["correct", "partially_correct", "overstated", "understated", "wrong",
              "abstained", "beyond_record", "not_scored"]
    cats = ["supported", "contradicted", "not_in_record", "unverifiable"]
    lines = [r"\begin{tabular}{l" + "r" * len(arms) + "}", r"\toprule",
             " & " + " & ".join(arms) + r" \\", r"\midrule",
             r"\multicolumn{" + str(len(arms) + 1) + r"}{l}{\emph{Direct verdicts (answers)}} \\"]
    for lab in labels:
        lines.append(f"{_tex(lab)} & " + " & ".join(str(summ[a]["verdicts"][lab]) for a in arms) + r" \\")
    lines += [r"\midrule",
              r"\multicolumn{" + str(len(arms) + 1) + r"}{l}{\emph{Claim verdicts (claims)}} \\"]
    for c in cats:
        lines.append(f"{_tex(c)} & " + " & ".join(str(summ[a]["claims"].get(c, 0)) for a in arms) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_silence(summ: dict, arms: list[str]) -> str:
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Arm & Silent questions & Declined or denied & Asserted a use \\", r"\midrule"]
    for a in arms:
        s = summ[a]
        n = s["silence_n"]
        if not n:
            continue
        ok = round((s["silence_correct"] or 0) * n)
        bad = round((s["silence_asserted"] or 0) * n)
        lines.append(f"{a} & {n} & {ok} & {bad} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


CONTRASTS = [  # (label, arm x, arm y, question types or None for all)
    ("Grounded vs.\\ closed book, free prompt", "D", "A1F", None),
    ("Grounded vs.\\ closed book, careful prompt", "D", "A1", None),
    ("Full pack vs.\\ flora only, park questions", "D", "E", PARK),
    ("Full pack vs.\\ flora only, all other types", "D", "E", "nonpark"),
    ("Text pack vs.\\ map, same model", "D", "DM", MAP_QTYPES),
    ("Text pack vs.\\ map, open weights", "F", "FM", MAP_QTYPES),
    ("Frontier vs.\\ open weights, both grounded", "D", "F", None),
    ("Model vs.\\ template, same evidence", "D", "G", None),
]


def table_contrasts(rows: list[dict]) -> str:
    """Paired on the question. An observation is 1 when the direct verdict is `correct`
    and 0 otherwise, abstentions included, over questions scoreable in both arms."""
    by = {(r["arm"], r["qid"]): r for r in rows if r["direct"] not in UNSCOREABLE}
    lines = [r"\begin{tabular}{p{6.4cm}rrrrrr}", r"\toprule",
             r"Contrast ($x$ vs.\ $y$) & Pairs & $x$ correct & $y$ correct & $x$ only & "
             r"$y$ only & $p$ \\", r"\midrule"]
    for label, x, y, qts in CONTRASTS:
        qids = sorted({q for (a, q) in by if a == x} & {q for (a, q) in by if a == y})
        if qts == "nonpark":
            qids = [q for q in qids if by[(x, q)]["qtype"] not in PARK]
        elif qts:
            qids = [q for q in qids if by[(x, q)]["qtype"] in qts]
        if not qids:
            continue
        cx = [by[(x, q)]["direct"] == "correct" for q in qids]
        cy = [by[(y, q)]["direct"] == "correct" for q in qids]
        b = sum(i and not j for i, j in zip(cx, cy))
        c = sum(j and not i for i, j in zip(cx, cy))
        p = _mcnemar_exact(b, c)
        ptxt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(f"{label} ({x}, {y}) & {len(qids)} & {sum(cx)} & {sum(cy)} & {b} & {c} "
                     f"& {ptxt} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_answerability() -> str:
    rows = list(csv.DictReader(io.open(DATA / "answerability.csv", encoding="utf-8")))
    rows.sort(key=lambda r: -float(r["pct"]))
    name = {"outside_record": "any use mentioned"}
    lines = [r"\begin{tabular}{lrrrl}", r"\toprule",
             r"Question type & Settled & Of & \% & Settled by \\", r"\midrule"]
    for r in rows:
        if r["qtype"] == "light":
            k, n = EDIBLE_MEDICINAL
            lines.append(f"edible or medicinal use & {k:,} & {n:,} & {100*k/n:.1f} & "
                         r"an edible or medicinal use named in the entry \\")
        lines.append(f"{_tex(name.get(r['qtype'], r['qtype']))} & {int(r['settled']):,} & "
                     f"{int(r['denominator']):,} & {float(r['pct']):.1f} & {_tex(r['basis'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines).replace(",", "{,}").replace("{,} ", ", ")


# ------------------------------------------------------------------------------ figures
# Categorical slots from the dataviz default palette, assigned by evidence condition in
# a fixed order. Generator is carried by marker shape and every point is labelled with
# its arm letter, so identity never rests on colour alone (the figure must survive a
# greyscale print).
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
EV_COLOR = {"full": "#2a78d6", None: "#eb6834", "flora": "#1baf7a", "map": "#4a3aa7"}
GEN_MARK = {"Mistral": "o", "Gemma": "s", "template": "D"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_answerability() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(io.open(DATA / "answerability.csv", encoding="utf-8")))
    items = [("any use mentioned" if r["qtype"] == "outside_record" else r["qtype"],
              float(r["pct"])) for r in rows]
    k, n = EDIBLE_MEDICINAL
    items.append(("edible or medicinal use", 100 * k / n))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh([i[0] for i in items], [i[1] for i in items], color="#2a78d6", height=0.62)
    for y, (_, v) in enumerate(items):
        ax.text(v + 1.2, y, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Species for which the record settles the question (%)", fontsize=9,
                  color=INK)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_answerability.pdf")
    plt.close(fig)


def fig_precision_coverage(summ: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    # Intervals are in the results table, not drawn: most arms answer everything, so
    # whiskers at coverage 100 would stack into one unreadable line.
    pts = []
    for a in [x for x in ORDER if x in summ]:
        s, arm = summ[a], ARMS[a]
        if s["precision"] is None:
            continue
        x, y = 100 * s["coverage"], 100 * s["precision"]
        ax.scatter([x], [y], s=62, marker=GEN_MARK[GENERATOR[arm.model]],
                   color=EV_COLOR[arm.evidence], edgecolor="white", linewidth=1.2, zorder=3)
        pts.append((a, x, y))
    # Arm letters sit in a column left of each cluster, spread so none overlap, with a
    # hairline back to the mark.
    for band in (lambda x: x >= 90, lambda x: x < 90):
        col = sorted([p for p in pts if band(p[1])], key=lambda p: p[2])
        last = -100.0
        for a, x, y in col:
            ly = max(y, last + 5.5)
            last = ly
            ax.annotate(a, (x, y), xytext=(x - 7, ly), textcoords="data", fontsize=8.5,
                        color=INK, ha="right", va="center",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5,
                                    "shrinkB": 5})
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Coverage: scoreable questions answered (%)", fontsize=9, color=INK)
    ax.set_ylabel("Precision when answering (%)", fontsize=9, color=INK)
    _style(ax)
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=7,
                      label=EVIDENCE[e]) for e, c in EV_COLOR.items()]
    handles += [Line2D([], [], marker=m, linestyle="", color=MUTED, markersize=7, label=g)
                for g, m in GEN_MARK.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, ncol=2,
              title="Evidence (colour) and generator (shape)", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_precision_coverage.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--label", default="", help="how the text names this run")
    args = ap.parse_args()

    rows = _rows(args.part)
    if not rows:
        raise SystemExit(f"no verdict rows for part={args.part}; run benchmark.score first")
    summ = summarize(rows)
    arms = [a for a in ORDER if a in summ]
    text_arms = [a for a in arms if ARMS[a].evidence != "map"]
    n_species = len({r["species"] for r in rows})
    n_questions = len({r["qid"] for r in rows})
    label = args.label or {"pilot": "five-species shakedown"}.get(args.part, args.part)

    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    files = {
        "tab_arms.tex": table_arms(summ),
        "tab_qtype.tex": table_qtype(rows, arms),
        "tab_verdicts.tex": table_verdicts(summ, arms),
        "tab_silence.tex": table_silence(summ, text_arms),
        "tab_contrasts.tex": table_contrasts(rows),
        "tab_answerability.tex": table_answerability(),
        "run_macros.tex": "\n".join([
            f"\\newcommand{{\\RunLabel}}{{{label}}}",
            f"\\newcommand{{\\RunSpecies}}{{{n_species}}}",
            f"\\newcommand{{\\RunQuestions}}{{{n_questions}}}",
            f"\\newcommand{{\\RunAnswers}}{{{len(rows):,}}}".replace(",", "{,}"),
        ]),
    }
    for name, body in files.items():
        (OUT / name).write_text(body + "\n", encoding="utf-8")
    fig_answerability()
    fig_precision_coverage(summ)
    print(f"{label}: {n_species} species, {n_questions} questions, {len(rows)} scored answers")
    print(f"wrote {len(files)} tables to {OUT} and 2 figures to {FIGS}")


if __name__ == "__main__":
    main()
