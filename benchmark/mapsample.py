"""The sample for the map comparison: its own experiment, with its own rules.

The question-answering sample was drawn to include species with almost no records,
because that experiment is about what happens when documentation runs out. A map can
only be judged against records if there are records, so this sample is drawn
separately, and none of the 117 species of the other experiment is in it: their maps
were looked at while the comparison was being built, which makes them development data
here.

  frame     catalogue species with a resolved GBIF key, not an unnamed morphospecies,
            not sharing a key with another entry, not among the 117 already used, with
            at least 30 records in the frozen download, and with a Manual entry a map
            can be drawn from (a region or a slope stated)
  strata    tertiles of worldwide GBIF record count within the frame: how well known
            the species is, the axis the paper's question turns on
  size      100 per stratum, 300 in all
  draw      within a stratum the species are ordered by family and then by name, and
            every k-th is taken from a random start. Families are represented in
            proportion to their size and none is crowded out; the companion paper drew
            its samples the same way

Thirty records, not ten, because the system's own map is scored on records it did not
use: the regions are inferred from one half and the capture counted on the other, so
each half needs enough to count.

Run:  python -m benchmark.mapsample        (writes benchmark/data/maps_sample.json)
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np

from mpcr_rag import config
from mpcr_rag.store import pg_store

DATA = Path(__file__).resolve().parent / "data"
SEED = 20261012
MIN_RECORDS = 30
PER_STRATUM = 100


def build() -> dict:
    used = json.loads((DATA / "species_sample.json").read_text(encoding="utf-8"))
    used = {x["species"] for x in used["main"] + used["pilot"]}
    counts = json.loads((config.DATA_DIR / "global_counts_gbif.json")
                        .read_text(encoding="utf-8"))["counts"]
    conn = pg_store.connect()
    with conn.cursor() as cur:
        cur.execute(r"""
            WITH shared AS (SELECT taxon_key FROM mpcr.taxa WHERE taxon_key IS NOT NULL
                            GROUP BY taxon_key HAVING count(*) > 1)
            SELECT f.species, f.family, e.taxon_key, e.n_records
            FROM mpcr.fichas f JOIN mpcr.species_evidence e USING (species)
            WHERE e.taxon_key IS NOT NULL
              AND NOT (f.species ~ ' sp\. ?[0-9]')
              AND e.taxon_key NOT IN (SELECT taxon_key FROM shared)
              AND e.n_records >= %s
              AND (coalesce(jsonb_array_length(f.ficha->'regions'), 0) > 0
                   OR coalesce(jsonb_array_length(f.ficha->'vertientes'), 0) > 0)
            ORDER BY f.species""", (MIN_RECORDS,))
        frame = [{"species": s, "family": fam, "taxon_key": int(k), "n_records": n,
                  "n_global": int(counts[str(int(k))])}
                 for s, fam, k, n in cur.fetchall()
                 if s not in used and str(int(k)) in counts]
    conn.close()

    edges = [float(np.quantile([r["n_global"] for r in frame], q)) for q in (1 / 3, 2 / 3)]
    strata: dict[str, list[dict]] = collections.defaultdict(list)
    for r in frame:
        g = 1 if r["n_global"] <= edges[0] else (2 if r["n_global"] <= edges[1] else 3)
        r["stratum"] = f"G{g}"
        strata[r["stratum"]].append(r)

    rng = random.Random(SEED)
    sample = []
    for name in sorted(strata):
        pool = sorted(strata[name], key=lambda r: (r["family"], r["species"]))
        k = len(pool) / PER_STRATUM
        start = rng.random() * k
        sample += [pool[int(start + i * k)] for i in range(PER_STRATUM)]

    out = {"created": dt.date.today().isoformat(), "seed": SEED,
           "min_records": MIN_RECORDS, "per_stratum": PER_STRATUM,
           "frame_size": len(frame),
           "frame_families": len({r["family"] for r in frame}),
           "strata": {n: len(v) for n, v in sorted(strata.items())},
           "global_count_tertile_edges": edges,
           "excluded_already_used": len(used),
           "rule": "tertiles of worldwide GBIF records; within each, ordered by family "
                   "then name and every k-th taken from a random start",
           "sample": sample}
    (DATA / "maps_sample.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    return out


if __name__ == "__main__":
    o = build()
    s = o["sample"]
    fam = collections.Counter(r["family"] for r in s)
    print(f"frame {o['frame_size']} species in {o['frame_families']} families; "
          f"strata {o['strata']}; tertile edges {o['global_count_tertile_edges']}")
    print(f"sample {len(s)} species, {len(set(r['species'] for r in s))} distinct, "
          f"{len(fam)} families; largest: {fam.most_common(6)}")
    for g in ("G1", "G2", "G3"):
        sub = [r for r in s if r["stratum"] == g]
        print(f"  {g}: {len(sub)} species, {len({r['family'] for r in sub})} families, "
              f"records in CR median {int(np.median([r['n_records'] for r in sub]))}, "
              f"worldwide median {int(np.median([r['n_global'] for r in sub]))}")
