"""Split the main sample into what has been seen and two confirmatory tranches.

The main sample is 99 species, 11 from each of the nine cells of global by local record
counts (sampler.py). Twenty of them were run as a shakedown and read closely, and the
checker and two key rules changed because of what was found. They are DEVELOPMENT data
from here on and are never pooled with what follows.

The other 79 have not been run under any arm. They are split, once and by a recorded
seed, into:

  tranche 1   3 species per cell, drawn at random within the cell  (27 species)
  tranche 2   the rest                                             (52 species)

The rule for pooling is fixed before tranche 1 is run, and written into the output file
so it travels with the data: if nothing in the checker, the keys, the prompts or the
evidence pack changes after tranche 1 has been read, the two tranches are pooled as one
confirmatory set of 79. If anything changes, tranche 1 joins the development data and
tranche 2 alone is confirmatory.

Run:  python -m benchmark.tranches      (writes species_tranche1.json, species_tranche2.json,
                                         tranches.json)
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SEED = 20261011
PER_CELL = 3


def build() -> dict:
    sample = json.loads((DATA / "species_sample.json").read_text(encoding="utf-8"))
    seen = set(json.loads((DATA / "species_20.json").read_text(encoding="utf-8")))
    by_cell: dict[str, list[str]] = collections.defaultdict(list)
    for item in sample["main"]:
        if item["species"] not in seen:
            by_cell[item["stratum"]].append(item["species"])

    rng = random.Random(SEED)
    t1, t2, cells = [], [], {}
    for cell in sorted(by_cell):
        pool = sorted(by_cell[cell])
        picked = sorted(rng.sample(pool, PER_CELL))
        t1 += picked
        t2 += [s for s in pool if s not in picked]
        cells[cell] = {"unseen": len(pool), "tranche1": picked}

    (DATA / "species_tranche1.json").write_text(
        json.dumps(sorted(t1), ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "species_tranche2.json").write_text(
        json.dumps(sorted(t2), ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "created": dt.date.today().isoformat(),
        "seed": SEED,
        "frame": "the 99-species main sample of species_sample.json (11 per cell of the "
                 "3x3 grid of global by local record counts)",
        "development": {"n": len(seen), "file": "species_20.json",
                        "why": "run and read during the shakedown; checker and key rules "
                               "were changed after reading them"},
        "tranche1": {"n": len(t1), "file": "species_tranche1.json",
                     "rule": f"{PER_CELL} species per cell, simple random sample without "
                             f"replacement within the cell, from the species not in the "
                             f"development set"},
        "tranche2": {"n": len(t2), "file": "species_tranche2.json",
                     "rule": "every remaining unseen species"},
        "pooling_rule": "If nothing in the checker, the keys, the prompts or the evidence "
                        "pack changes after tranche 1 has been read, tranches 1 and 2 are "
                        "pooled as the confirmatory set. If anything changes, tranche 1 "
                        "joins the development data and tranche 2 alone is confirmatory.",
        "arms": {"tranche1": "all, including A1", "tranche2": "all except A1 (cost)"},
        "cells": cells,
    }
    (DATA / "tranches.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return meta


if __name__ == "__main__":
    m = build()
    print(f"development {m['development']['n']}, tranche 1 {m['tranche1']['n']}, "
          f"tranche 2 {m['tranche2']['n']}")
    for cell, c in m["cells"].items():
        print(f"  {cell}  unseen={c['unseen']:2d}  tranche 1: {', '.join(c['tranche1'])}")
