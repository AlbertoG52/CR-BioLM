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
  python paper2/build_results.py --part main --label "confirmatory run"       --species-file species_tranche1.json,species_tranche2.json

With --species-file the verdict rows are restricted to those species, and the tables of
the pre-registered analyses (benchmark/data/confirmatory/results.json), the map
comparison, the counterfactual condition and the expert review are written as well.
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

ORDER = ["G", "D", "F", "R", "E", "A1", "C", "A1F", "CF", "DM", "FM"]
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


def _rows(part: str, species: set[str] | None = None) -> list[dict]:
    """Verdict rows, one per (qid, arm, replicate): an errored row followed by its
    retry is one observation, not two."""
    keep: dict[tuple, dict] = {}
    for line in io.open(DATA / f"verdicts_{part}.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or not r.get("direct"):
            continue
        if species is not None and r["species"] not in species:
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


# ------------------------------------------------------------ the confirmatory tables

CONF = DATA / "confirmatory" / "results.json"
NAMES = {"D": "Mistral, full pack", "F": "Gemma, full pack", "A1F": "Mistral, free",
         "CF": "Gemma, free", "E": "Mistral, flora only", "DM": "Mistral, map only",
         "FM": "Gemma, map only", "R": "Gemma, orchestrator", "G": "template"}
CONTRAST_EN = {("D", "A1F"): r"Grounded vs.\ free closed book, Mistral",
               ("F", "CF"): r"Grounded vs.\ free closed book, Gemma",
               ("D", "E", 0): r"Full pack vs.\ flora only, park questions",
               ("D", "E", 1): r"Full pack vs.\ flora only, Manual-stated types",
               ("D", "DM"): r"Text pack vs.\ map, Mistral",
               ("F", "FM"): r"Text pack vs.\ map, Gemma",
               ("F", "D"): r"Gemma vs.\ Mistral, both grounded",
               ("F", "R"): r"Given route vs.\ orchestrator, Gemma"}
LONGTAIL_ARMS = ["A1F", "CF", "D", "F"]


def _num(v: int) -> str:
    return f"{v:,}".replace(",", "{,}")


def _signed(x: float, nd: int = 1) -> str:
    s = f"{x:+.{nd}f}"
    return ("$+$" if s[0] == "+" else "$-$") + s[1:]


def table_bootstrap(res: dict) -> str:
    """The pre-specified contrasts, with intervals from resampling species."""
    lines = [r"\begin{tabular}{p{6.0cm}rrrrl}", r"\toprule",
             r"Contrast ($x$ vs.\ $y$) & Species & Pairs & $x$ correct & $y$ correct & "
             r"Difference [95\% CI] \\", r"\midrule"]
    seen_de = 0
    for c in res["contrasts"]:
        k = (c["x"], c["y"])
        if k == ("D", "E"):
            k, seen_de = ("D", "E", seen_de), seen_de + 1
        lo, hi = c["ci95"]
        lines.append(f"{CONTRAST_EN[k]} ({c['x']}, {c['y']}) & {c['species']} & {c['pairs']} & "
                     f"{c['x_correct']} & {c['y_correct']} & {_signed(100 * c['difference'])} "
                     f"[{_signed(100 * lo)}, {_signed(100 * hi)}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _tier_share(rows: list[dict], arm: str, axis: str, tier: str) -> tuple[int, int]:
    pos = 1 if axis == "G" else 3          # stratum is written "G2L3"
    sub = [r for r in rows if r["arm"] == arm and r["direct"] not in UNSCOREABLE
           and r["stratum"][pos] == tier]
    return sum(r["direct"] == "correct" for r in sub), len(sub)


def table_longtail(rows: list[dict], res: dict) -> str:
    """Share correct (abstentions count as not correct) by tertile of local and of
    worldwide record count, and the fitted slope on log(1 + local records)."""
    m = res["model"]
    lines = [r"\begin{tabular}{lrrrrrrl}", r"\toprule",
             r" & \multicolumn{3}{c}{Local records ($L$)} & \multicolumn{3}{c}{Worldwide "
             r"records ($G$)} & Slope on \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Condition & low & mid & high & low & mid & high & $\log(1+L)$ [95\% CI] \\",
             r"\midrule"]
    for a in LONGTAIL_ARMS:
        cells = []
        for axis in ("L", "G"):
            for t in "123":
                k, n = _tier_share(rows, a, axis, t)
                cells.append(f"{100 * k / n:.0f}" if n else "--")
        est, se = m["slopes"][a]
        lines.append(f"{a} ({NAMES[a]}) & " + " & ".join(cells) +
                     f" & {_signed(est, 2)} [{_signed(est - 1.96 * se, 2)}, "
                     f"{_signed(est + 1.96 * se, 2)}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_flora_weighted(res: dict) -> str:
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Condition & Sample & Flora-weighted \\", r"\midrule"]
    for a, v in res["flora_weighted"].items():
        lines.append(f"{a} ({NAMES[a]}) & {100 * v['unweighted']:.1f} & "
                     f"{100 * v['flora_weighted']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_counterfactual(species: set[str]) -> str:
    """The altered-pack condition: the answer follows the altered fact (correct against
    the recomputed key), recalls the real one, or does neither."""
    keep = {}
    for line in io.open(DATA / "verdicts_cf_main.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if not r.get("error") and r["species"] in species:
            keep[(r["qid"], r["arm"])] = r
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Condition & Question type & $n$ & Followed the pack & Recalled the real fact "
             r"& Neither \\", r"\midrule"]
    for arm in ("D", "F"):
        sub = [r for (q, a), r in keep.items() if a == arm]
        groups = [("all", sub)] + [(qt, [r for r in sub if r["qtype"] == qt])
                                   for qt in ("elevation", "cultivation", "park_confirmed",
                                              "park_possible")]
        for name, g in groups:
            fol = sum(r["direct_cf"] == "correct" for r in g)
            rec = sum(r["direct_real"] == "correct" and r["direct_cf"] != "correct" for r in g)
            lines.append(f"{NAMES[arm] if name == 'all' else ''} & {_tex(name)} & {len(g)} & "
                         f"{fol} & {rec} & {len(g) - fol - rec} \\\\")
        if arm == "D":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_companions(species: set[str]) -> str:
    """Co-occurrence under two rules. The original: every species an answer names must
    share a botanical region and an elevation band with the subject. The stricter one an
    expert asked for: it must also share the subject's growth form. Every answer is
    re-checked under both, from its text; the H arms were run with the companion list in
    the pack filtered by growth form."""
    from benchmark.cooccurrence import check, extract_species, verdict
    answers = {}
    for line in io.open(DATA / "answers_main.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if (r.get("qtype") == "co_occurrence" and r["species"] in species
                and not r.get("error") and r.get("answer_text")):
            answers[(r["qid"], r["arm"])] = r

    def share(arm, strict):
        v = [verdict(check(r["species"], extract_species(r["answer_text"]),
                           same_habit=strict))
             for (_, a), r in answers.items() if a == arm]
        judged = [x for x in v if x not in ("abstained", "not_scored")]
        return f"{sum(x == 'correct' for x in judged)}/{len(judged)}"

    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Condition & Companion list in the pack & Region and elevation & "
             r"Also growth form \\", r"\midrule"]
    for arm, pack in (("A1F", "none"), ("CF", "none"), ("D", "original"), ("F", "original"),
                      ("R", "original"), ("DH", "filtered by growth form"),
                      ("FH", "filtered by growth form"), ("RH", "filtered by growth form")):
        name = NAMES.get(arm) or NAMES[arm[0] if arm != "RH" else "R"]
        lines.append(f"{arm} ({name}) & {pack} & {share(arm, False)} & {share(arm, True)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


MAP_SOURCES = [("full_heldout", "Ours, regions from half the records, scored on the other half"),
               ("full", "Ours, as drawn"), ("manual", "The Manual's stated range alone"),
               ("paid", "Mistral Large 4, from its own description"),
               ("open", "Gemma 4 31B, from its own description")]


def _mapcompare_rows() -> list[dict]:
    return [json.loads(l) for l in io.open(DATA / "mapcompare_scores_maps.jsonl",
                                             encoding="utf-8") if l.strip()]


def table_mapcompare() -> str:
    import statistics
    from scipy.stats import binomtest
    rows = _mapcompare_rows()
    lines = [r"\begin{tabular}{p{5.4cm}rrrrrr}", r"\toprule",
             r"Map & Maps & Records captured & Median capture & Median area & Lift & "
             r"Beats chance \\", r"\midrule"]
    for src, label in MAP_SOURCES:
        sub = [r for r in rows if r["source"] == src and r["has_map"] and r.get("area", 0) > 0]
        cap, tot = sum(r["captured"] for r in sub), sum(r["n_records"] for r in sub)
        rate = [r["captured"] / r["n_records"] for r in sub]
        lift = statistics.median(c / r["area"] for c, r in zip(rate, sub))
        beat = sum(binomtest(r["captured"], r["n_records"], min(r["area"], 1.0),
                             alternative="greater").pvalue < 0.05 for r in sub)
        lines.append(f"{label} & {len(sub)} & {100 * cap / tot:.0f} & "
                     f"{100 * statistics.median(rate):.0f} & "
                     f"{100 * statistics.median(r['area'] for r in sub):.1f} & "
                     f"{lift:.1f} & {beat} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def map_macros() -> list[str]:
    import statistics
    rows = _mapcompare_rows()
    out = [f"\\newcommand{{\\MapSpecies}}{{{len({r['species'] for r in rows})}}}",
           f"\\newcommand{{\\MapRecords}}{{{_num(sum(r['n_records'] for r in rows if r['source'] == 'manual'))}}}"]
    for src, tag in (("full_heldout", "Held"), ("paid", "Paid"), ("open", "Open"),
                     ("manual", "Manual")):
        sub = [r for r in rows if r["source"] == src and r["has_map"] and r.get("area", 0) > 0]
        out += [f"\\newcommand{{\\Map{tag}N}}{{{len(sub)}}}",
                f"\\newcommand{{\\Map{tag}Half}}"
                f"{{{sum(r['captured'] >= 0.5 * r['n_records'] for r in sub)}}}",
                f"\\newcommand{{\\Map{tag}Wide}}{{{sum(r['area'] >= 0.5 for r in sub)}}}"]
        for g, tier in zip("123", "ABC"):
            gs = [r["captured"] / r["n_records"] for r in sub if r["stratum"] == f"G{g}"]
            out.append(f"\\newcommand{{\\Map{tag}G{tier}}}{{{100 * statistics.median(gs):.0f}}}")
    return out


def expert_macros() -> list[str]:
    from benchmark.expert_results import summarize as expert
    e = expert()
    c, p, err, mp = e["caught"], e["preferred"], e["errors"], e["maps"]
    vals = {"Pairs": e["pairs"], "PrefGrounded": p.get("D", 0), "PrefFree": p.get("A1F", 0),
            "PrefEqual": p.get("equal", 0), "ErrGrounded": err.get("D", 0),
            "ErrFree": err.get("A1F", 0), "ErrScoreable": e["scoreable_errors"],
            "CaughtDirect": c.get("direct", 0), "CaughtClaim": c.get("claim", 0),
            "Caught": c.get("direct", 0) + c.get("claim", 0), "KeysRight": e["keys_correct"],
            "MapsYes": mp.get("yes", 0), "MapsPartly": mp.get("partly", 0),
            "MapsNo": mp.get("no", 0), "Maps": sum(mp.values())}
    return [f"\\newcommand{{\\Exp{k}}}{{{v}}}" for k, v in vals.items()]


def routing_macros(species: set[str]) -> list[str]:
    """What the orchestrator resolved, from the routing record kept on each answer of
    arm R, and its score on the hand-built set of hard questions (routing_set.py)."""
    last = {}
    for line in io.open(DATA / "answers_main.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["arm"] == "R" and r["species"] in species and not r.get("error"):
            last[r["qid"]] = r["routing"]
    parks = [v for q, v in last.items() if "__park_" in q]
    hard = [json.loads(l) for l in io.open(DATA / "routing_hard_result.jsonl", encoding="utf-8")]
    hard_ok = sum(r["route_ok"] and r["species_ok"] and r["place_ok"] for r in hard)
    return [f"\\newcommand{{\\RouteN}}{{{len(last)}}}",
            f"\\newcommand{{\\RouteOk}}{{{sum(v['route_ok'] for v in last.values())}}}",
            f"\\newcommand{{\\RouteSpeciesOk}}{{{sum(v['species_ok'] for v in last.values())}}}",
            f"\\newcommand{{\\RouteParks}}{{{len(parks)}}}",
            f"\\newcommand{{\\RouteParksOk}}{{{sum(bool(v['place_ok']) for v in parks)}}}",
            f"\\newcommand{{\\RouteHardN}}{{{len(hard)}}}",
            f"\\newcommand{{\\RouteHardOk}}{{{hard_ok}}}"]


def conf_macros(res: dict) -> list[str]:
    m = res["model"]
    g = m["global_slope"]
    return [f"\\newcommand{{\\LtN}}{{{_num(m['n'])}}}",
            f"\\newcommand{{\\LtChi}}{{{m['lrt_chisq']:.1f}}}",
            f"\\newcommand{{\\LtDf}}{{{m['lrt_df']}}}",
            f"\\newcommand{{\\LtP}}{{{m['lrt_p']:.3f}}}",
            f"\\newcommand{{\\LtGlobal}}{{{g[0]:.2f}}}",
            f"\\newcommand{{\\LtGlobalP}}{{{g[2]:.3f}}}"]


def fig_longtail(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    style = {"A1F": (EV_COLOR[None], "o", "-"), "CF": (EV_COLOR[None], "s", "--"),
             "D": (EV_COLOR["full"], "o", "-"), "F": (EV_COLOR["full"], "s", "--")}
    for ax, axis, title in ((axes[0], "L", "records in Costa Rica"),
                            (axes[1], "G", "records worldwide")):
        for a in LONGTAIL_ARMS:
            ys = []
            for t in "123":
                k, n = _tier_share(rows, a, axis, t)
                ys.append(100 * k / n if n else float("nan"))
            col, mk, ls = style[a]
            ax.plot([0, 1, 2], ys, color=col, marker=mk, linestyle=ls, linewidth=1.4,
                    markersize=6, label=f"{a} ({NAMES[a]})")
        ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
        ax.set_xlabel(f"Tertile of {title}", fontsize=9, color=INK)
        ax.set_ylim(0, 105)
        _style(ax)
    axes[0].set_ylabel("Questions answered correctly (%)", fontsize=9, color=INK)
    # The legend sits below both panels: inside either it covers the lines.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGS / "fig_longtail.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--label", default="", help="how the text names this run")
    ap.add_argument("--species-file", default="",
                    help="comma-separated species lists in benchmark/data to restrict to")
    args = ap.parse_args()

    species = None
    if args.species_file:
        species = set()
        for f in args.species_file.split(","):
            species |= set(json.loads((DATA / f.strip()).read_text(encoding="utf-8")))
    rows = _rows(args.part, species)
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
    macros = [f"\\newcommand{{\\RunLabel}}{{{label}}}",
              f"\\newcommand{{\\RunSpecies}}{{{n_species}}}",
              f"\\newcommand{{\\RunQuestions}}{{{_num(n_questions)}}}",
              f"\\newcommand{{\\RunAnswers}}{{{_num(len(rows))}}}"]
    files = {
        "tab_arms.tex": table_arms(summ),
        "tab_qtype.tex": table_qtype(rows, arms),
        "tab_verdicts.tex": table_verdicts(summ, arms),
        "tab_silence.tex": table_silence(summ, text_arms),
        "tab_contrasts.tex": table_contrasts(rows),
        "tab_answerability.tex": table_answerability(),
    }
    if species is not None:
        res = json.loads(CONF.read_text(encoding="utf-8"))
        files.update({
            "tab_bootstrap.tex": table_bootstrap(res),
            "tab_longtail.tex": table_longtail(rows, res),
            "tab_flora_weighted.tex": table_flora_weighted(res),
            "tab_counterfactual.tex": table_counterfactual(species),
            "tab_companions.tex": table_companions(species),
            "tab_mapcompare.tex": table_mapcompare(),
        })
        macros += conf_macros(res) + map_macros() + expert_macros() + routing_macros(species)
        fig_longtail(rows)
    files["run_macros.tex"] = "\n".join(macros)
    for name, body in files.items():
        (OUT / name).write_text(body + "\n", encoding="utf-8")
    fig_answerability()
    fig_precision_coverage(summ)
    print(f"{label}: {n_species} species, {n_questions} questions, {len(rows)} scored answers")
    print(f"wrote {len(files)} tables to {OUT} and figures to {FIGS}")


if __name__ == "__main__":
    main()
