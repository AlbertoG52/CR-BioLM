"""The pre-registered analyses, on the confirmatory set (tranches 1 and 2, 79 species).

Three analyses, in the order docs/EXPERIMENT_PLAN.md fixed them:

1. The long tail. A mixed-effects logistic regression of whether an answer is correct
   on the condition, the species' local and worldwide record counts, and the
   interaction of condition with local records, with random intercepts for species and
   for question type, since questions repeat within both:

       correct ~ arm * log1p(n_records) + log10(n_global) + (1 | species) + (1 | qtype)

   The claim is that the closed-book slope on local records is positive and the
   grounded slope is not: grounding makes accuracy stop depending on how well documented
   the plant is. Fitted with lme4::glmer, the reference implementation, from R. The
   interaction is tested by a likelihood-ratio test against the model without it.

2. Paired contrasts with intervals. For each pre-specified contrast, the difference in
   the share of questions answered correctly, with a 95% interval from a bootstrap that
   resamples SPECIES, not questions, because questions about the same species are not
   independent. 2,000 replicates, recorded seed.

3. The flora, not the sample. The sample was drawn with equal numbers per cell of the
   3x3 grid, so rare cells are over-represented. Each species is weighted by the size
   of its cell in the frame over the number sampled from it, which gives the precision
   one would expect over the whole flora.

"Correct" is the direct verdict `correct`; an abstention counts as not correct; the
`not_scored` and `beyond_record` verdicts are left out, as in every other table.

Run:  python -m benchmark.confirmatory       (writes benchmark/data/confirmatory/)
"""
from __future__ import annotations

import collections
import csv
import io
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper2"))
import build_results as B  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "confirmatory"
RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
SEED = 20261022
UNSCOREABLE = ("not_scored", "beyond_record")
PARK = ("park_confirmed", "park_possible", "park_unlikely")
MAP_TYPES = PARK + ("vertiente",)

CONTRASTS = [
    ("Con evidencia contra preguntado libremente (Mistral)", "D", "A1F", None),
    ("Con evidencia contra preguntado libremente (Gemma)", "F", "CF", None),
    ("Evidencia completa contra solo Manual, parques", "D", "E", lambda t: t in PARK),
    ("Evidencia completa contra solo Manual, lo que el Manual declara", "D", "E",
     lambda t: t not in PARK and t != "co_occurrence"),
    ("Texto contra mapa (Mistral)", "D", "DM", lambda t: t in MAP_TYPES),
    ("Texto contra mapa (Gemma)", "F", "FM", lambda t: t in MAP_TYPES),
    ("Modelo abierto contra frontera, con evidencia", "F", "D", None),
    ("Ruta dada contra orquestador (Gemma)", "F", "R", None),
]


def confirmatory_rows() -> list[dict]:
    species = set()
    for t in ("species_tranche1.json", "species_tranche2.json"):
        species |= set(json.loads((DATA / t).read_text(encoding="utf-8")))
    return [r for r in B._rows("main") if r["species"] in species]


def write_long_csv(rows: list[dict], path: Path) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "species", "qtype", "arm", "stratum", "n_records", "n_global",
                    "correct"])
        for r in rows:
            if r["direct"] in UNSCOREABLE:
                continue
            w.writerow([r["qid"], r["species"], r["qtype"], r["arm"], r["stratum"],
                        r.get("n_records") or 0, max(1, r.get("n_global") or 1),
                        int(r["direct"] == "correct")])


R_MODEL = r"""
lib <- Sys.getenv("R_LIBS_USER"); .libPaths(c(lib, .libPaths()))
suppressMessages(library(lme4))
d <- read.csv(commandArgs(TRUE)[1], encoding = "UTF-8")
d <- d[d$arm %in% c("D", "F", "A1F", "CF"), ]
d$arm <- factor(d$arm, levels = c("A1F", "CF", "D", "F"))
d$lrec <- log1p(d$n_records)
d$lglob <- log10(d$n_global)
ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
full <- glmer(correct ~ arm * lrec + lglob + (1 | species) + (1 | qtype),
              data = d, family = binomial, control = ctl)
null <- glmer(correct ~ arm + lrec + lglob + (1 | species) + (1 | qtype),
              data = d, family = binomial, control = ctl)
lrt <- anova(null, full)
co <- summary(full)$coefficients
v <- vcov(full)
slope <- function(arm) {
  b <- "lrec"; k <- paste0("arm", arm, ":lrec")
  if (arm == "A1F") return(c(co[b, 1], sqrt(v[b, b])))
  est <- co[b, 1] + co[k, 1]
  se <- sqrt(v[b, b] + v[k, k] + 2 * v[b, k])
  c(est, se)
}
out <- list(n = nrow(d), species = length(unique(d$species)),
            lrt_chisq = lrt$Chisq[2], lrt_df = lrt$Df[2], lrt_p = lrt$`Pr(>Chisq)`[2],
            global_slope = c(co["lglob", 1], co["lglob", 2], co["lglob", 4]),
            slopes = lapply(setNames(c("A1F", "CF", "D", "F"), c("A1F", "CF", "D", "F")), slope))
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 6))
"""


def fit_model(csv_path: Path) -> dict:
    script = OUT / "longtail_model.R"
    script.write_text(R_MODEL, encoding="utf-8")
    res = subprocess.run([RSCRIPT, str(script), str(csv_path)], capture_output=True,
                         text=True, encoding="utf-8")
    if res.returncode != 0:
        raise SystemExit(res.stderr[-2000:])
    return json.loads(res.stdout.strip().splitlines()[-1])


def bootstrap(rows: list[dict]) -> list[dict]:
    by = {(r["arm"], r["qid"]): r for r in rows if r["direct"] not in UNSCOREABLE}
    rng = random.Random(SEED)
    out = []
    for label, x, y, keep in CONTRASTS:
        qids = sorted({q for (a, q) in by if a == x} & {q for (a, q) in by if a == y})
        if keep:
            qids = [q for q in qids if keep(by[(x, q)]["qtype"])]
        per_sp: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for q in qids:
            per_sp[by[(x, q)]["species"]].append((int(by[(x, q)]["direct"] == "correct"),
                                                   int(by[(y, q)]["direct"] == "correct")))
        sp = sorted(per_sp)

        def diff(sample):
            pairs = [p for s in sample for p in per_sp[s]]
            return (sum(a for a, _ in pairs) - sum(b for _, b in pairs)) / len(pairs)

        point = diff(sp)
        reps = sorted(diff([rng.choice(sp) for _ in sp]) for _ in range(2000))
        out.append({"contrast": label, "x": x, "y": y, "pairs": len(qids),
                    "species": len(sp),
                    "x_correct": sum(a for s in sp for a, _ in per_sp[s]),
                    "y_correct": sum(b for s in sp for _, b in per_sp[s]),
                    "difference": point, "ci95": [reps[49], reps[1949]]})
    return out


def flora_weighted(rows: list[dict]) -> dict:
    sample = json.loads((DATA / "species_sample.json").read_text(encoding="utf-8"))
    frame = {c["id"]: c["frame_size"] for c in sample["cells"]}
    conf = {r["species"] for r in rows}
    n_cell = collections.Counter(i["stratum"] for i in sample["main"] if i["species"] in conf)
    cell_of = {i["species"]: i["stratum"] for i in sample["main"]}
    w_sp = {s: frame[cell_of[s]] / n_cell[cell_of[s]] for s in conf}
    out = {}
    for arm in ("G", "D", "F", "R", "E", "A1F", "CF", "DM", "FM"):
        ans = [r for r in rows if r["arm"] == arm and r["direct"] in B.ANSWERED]
        if not ans:
            continue
        w = [w_sp[r["species"]] for r in ans]
        c = [w_sp[r["species"]] * (r["direct"] == "correct") for r in ans]
        out[arm] = {"unweighted": sum(r["direct"] == "correct" for r in ans) / len(ans),
                    "flora_weighted": sum(c) / sum(w)}
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = confirmatory_rows()
    long_csv = OUT / "answers_long.csv"
    write_long_csv(rows, long_csv)
    model = fit_model(long_csv)
    contrasts = bootstrap(rows)
    weighted = flora_weighted(rows)
    (OUT / "results.json").write_text(json.dumps(
        {"model": model, "contrasts": contrasts, "flora_weighted": weighted},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"confirmatory set: {len({r['species'] for r in rows})} species, "
          f"{len({r['qid'] for r in rows})} questions\n")
    print(f"1. long tail: glmer on {model['n']} answers, {model['species']} species")
    print("   slope of log-odds of a correct answer per unit of log(1 + local records):")
    for arm, (est, se) in model["slopes"].items():
        lo, hi = est - 1.96 * se, est + 1.96 * se
        print(f"     {arm:4s} {est:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    g = model["global_slope"]
    print(f"   worldwide records, log10: {g[0]:+.3f} (se {g[1]:.3f}, p {g[2]:.3g})")
    print(f"   condition x local records, likelihood-ratio test: chi2 {model['lrt_chisq']:.1f}, "
          f"df {model['lrt_df']}, p {model['lrt_p']:.3g}\n")
    print("2. paired differences, share correct, 95% interval resampling species:")
    for c in contrasts:
        print(f"   {c['contrast']}: {c['x_correct']} v {c['y_correct']} of {c['pairs']} "
              f"({c['species']} species)  {100 * c['difference']:+.1f} points "
              f"[{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]")
    print("\n3. precision over the flora (weighted by cell size) against the sample:")
    for arm, v in weighted.items():
        print(f"   {arm:4s} sample {100 * v['unweighted']:5.1f}%   flora {100 * v['flora_weighted']:5.1f}%")


if __name__ == "__main__":
    main()
