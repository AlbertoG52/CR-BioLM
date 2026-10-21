"""The map a person would get by asking a model where a plant grows.

Someone asks, in plain Spanish, where a species grows in Costa Rica: at what altitude,
on which slope, in which regions. A model answers in prose. That prose is, in effect, a
range map, and this module draws it and asks how good it is.

The comparison is built to differ in one thing only. The same rule turns a description
into a map for both sources:

    the botanical regions named, cut to the slopes named, cut to the elevation named

For the flora the three ingredients are the fields already extracted from the Manual.
For a model they are read from its answer by the same gazetteer and the same parsers
the checker uses on any answer. If a description names slopes and no region, the map is
every region on those slopes; if it names no elevation, the band is unrestricted. A
description that names neither a region nor a slope yields no map, and is counted.

Two numbers judge a map, and neither comes from the Manual:

    captured   the share of the species' records in the frozen download that fall
               inside the map (by the region, slope and terrain elevation stored with
               each record)
    area       the share of the country's land the map covers

They are read together. A map that captures everything by covering half the country
is not a good map; a small map that misses the records is a wrong one. Overlap with the
flora's map is reported too, as a secondary figure, since it scores a model against our
own source.

This is NOT the renderer of the companion paper, which also draws parks and marks
record-inferred regions. It is the range that renderer fills, computed on the terrain
grid so that areas can be counted.

Run:
  python -u -m benchmark.mapcompare --generate --models open
  python -u -m benchmark.mapcompare --report
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import io
import json
import random
import re
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import arms as A
from . import client, prompts
from . import verify as V

DATA = Path(__file__).resolve().parent / "data"
QUESTION = ("¿Dónde crece {sp} en Costa Rica? Indíqueme a qué altitud, en qué vertiente o "
            "vertientes y en qué regiones o cordilleras del país se encuentra.")
MODELS = {"open": A.OPEN, "paid": A.PAID}
SEED = 20261012
MIN_RECORDS = 10        # below this a capture rate is too coarse to mean anything


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


# ------------------------------------------------------------------------ the terrain

@lru_cache(maxsize=1)
def _grid():
    """Per land pixel of the terrain raster: region name, slope and elevation."""
    import rasterio
    from rasterio import features
    from utils.distribution_map.geodata import PATHS, load_regiones_botanicas

    regions = load_regiones_botanicas().to_crs(4326).reset_index(drop=True)
    with rasterio.open(PATHS["dem"]) as src:
        elev = src.read(1).astype("float32")
        if src.nodata is not None:
            elev[elev == src.nodata] = np.nan
        geoms = regions.to_crs(src.crs).geometry
        idx = features.rasterize(((g, i + 1) for i, g in enumerate(geoms)),
                                 out_shape=elev.shape, transform=src.transform, fill=0)
        _GEO["bounds"], _GEO["land"] = src.bounds, idx > 0
    land = idx > 0
    names = np.array([""] + list(regions["Nombre"]))[idx][land]
    slopes = np.array([""] + ["Caribe" if "arib" in v else "Pacífico"
                              for v in regions["Vertiente"]])[idx][land]
    return names, slopes, elev[land]


_GEO: dict = {}          # bounds and land mask of the terrain grid, filled by _grid()


def with_records(ing: dict, records: list[tuple]) -> dict:
    """The description widened the way the system widens the flora's: any region, and
    any slope, holding at least five of the given records is added."""
    reg = collections.Counter(r for r, _s, _e in records if r)
    slo = collections.Counter(s for _r, s, _e in records if s)
    return {"regions": sorted(set(ing["regions"]) | {r for r, n in reg.items() if n >= 5}),
            "slopes": (sorted(set(ing["slopes"]) | {s for s, n in slo.items() if n >= 5})
                       if ing["slopes"] else []),
            "elev": ing["elev"]}


def mask(ing: dict) -> np.ndarray | None:
    """The map of a description, as a boolean over the land pixels. None = no map."""
    if not ing["regions"] and not ing["slopes"]:
        return None
    names, slopes, elev = _grid()
    m = np.ones(len(names), dtype=bool)
    if ing["regions"]:
        m &= np.isin(names, list(ing["regions"]))
    if ing["slopes"]:
        m &= np.isin(slopes, list(ing["slopes"]))
    if ing["elev"]:
        m &= (elev >= ing["elev"][0]) & (elev <= ing["elev"][1])
    return m


# -------------------------------------------------------------------- the ingredients

# "It grows throughout the country" is a description of a range, and its map is the
# whole country (cut to whatever elevation is given). Read only from the regions a text
# happens to name, such an answer was being drawn narrower than what it said.
_EVERYWHERE = re.compile(
    r"tod[oa]s?\s+(?:el\s+pa[ií]s|el\s+territorio|las\s+regiones\s+del\s+pa[ií]s|"
    r"las\s+provincias|costa\s+rica)|a\s+lo\s+largo\s+y\s+ancho\s+del\s+pa[ií]s|"
    r"pr[áa]cticamente\s+todo\s+el\s+pa[ií]s", re.I)


_NUM = r"(\d{1,2}[.,]\d{3}|\d{1,4})"
_UP_TO = re.compile(
    r"(?:nivel del mar|\b0\s*m(?:etros)?)?\s*\bhasta\s+(?:los\s+|unos\s+|aproximadamente\s+|"
    r"cerca de\s+|alrededor de\s+)*" + _NUM + r"(?:\s*[–—-]\s*" + _NUM + r")?\s*"
    r"(?:m\b|metros|msnm|m\.s\.n\.m)", re.I)


def elevation_span(text: str) -> list[int] | None:
    """The full span of elevation a description gives, lowest to highest.

    The checker's parser reads "desde el nivel del mar hasta 1.500-2.000 m" as the range
    1500-2000: it takes the hedged upper bound for the whole range and never sees the
    sea level before it. For a single answer that is a detail; for a map it removes the
    lowlands of every species described that way, which one of the two models does
    often. Here an "up to" is an upper bound, hedged or not, and anything said to reach
    sea level starts at zero. Occasional extremes ("ocasionalmente hasta 2.500 m") are
    kept, as the Manual's outlier bounds are kept on our side.
    """
    spans = list(V.parse_elevations(text))
    for m in _UP_TO.finditer(text):
        top = max(V._num(g) for g in m.groups() if g)
        if top <= 4000:
            spans.append((0, top))
    if not spans:
        return None
    return [min(a for a, _ in spans), max(b for _, b in spans)]


def from_text(text: str) -> dict:
    spans = elevation_span(text)
    everywhere = bool(_EVERYWHERE.search(text))
    regions = (sorted(set(_grid()[0])) if everywhere else sorted(V._regions_named(text)))
    return {"regions": regions,
            "slopes": [] if everywhere else sorted(V.parse_slopes(text)),
            "everywhere": everywhere,
            "elev": spans}


def from_ficha(f) -> dict:
    lo = f.elev_outlier_min if f.elev_outlier_min is not None else f.elev_min
    hi = f.elev_outlier_max if f.elev_outlier_max is not None else f.elev_max
    if (lo is not None and lo == hi and f.elev_outlier_min is None
            and f.elev_outlier_max is None):
        # A single approximate value ("ca. 1400 m") is drawn as a band, as the renderer
        # now draws it (ElevationRange.band). Added after the comparison had been run;
        # it touches 2 of the 300 sampled species and both figures are reported.
        from utils.distribution_map.ficha import SINGLE_VALUE_MARGIN_M
        lo, hi = max(0, lo - SINGLE_VALUE_MARGIN_M), hi + SINGLE_VALUE_MARGIN_M
    return {"regions": sorted(f.regions or []), "slopes": sorted(f.vertientes or []),
            "elev": [lo, hi] if lo is not None and hi is not None else None}


def captured(ing: dict, records: list[tuple]) -> int:
    """How many records (region, slope, terrain elevation) fall inside the map."""
    n = 0
    for region, slope, elev in records:
        if ing["regions"] and region not in ing["regions"]:
            continue
        if ing["slopes"] and slope not in ing["slopes"]:
            continue
        if ing["elev"] and (elev is None or not ing["elev"][0] <= elev <= ing["elev"][1]):
            continue
        if region is None:
            continue
        n += 1
    return n


# ----------------------------------------------------------------------- generation

_ONLY: set | None = None      # set by --species-file to restrict a run to those species


def _species(part: str) -> list[dict]:
    if part == "maps":            # the map comparison's own sample (mapsample.py)
        items = json.loads((DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]
    else:
        items = json.loads((DATA / "species_sample.json").read_text(encoding="utf-8"))[part]
    return [i for i in items if _ONLY is None or i["species"] in _ONLY]


def generate(models: list[str], part: str, max_cost: float, workers: int) -> None:
    out = DATA / "answers_mapq.jsonl"
    done = {(r["species"], r["model"]) for r in _jsonl(out) if not r.get("error")}
    jobs = [(item, m) for item in _species(part) for m in models
            if (item["species"], m) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already on disk)", flush=True)

    def call(job):
        item, m = job
        spec = MODELS[m]
        base = {"species": item["species"], "model": m, "stratum": item["stratum"],
                "n_global": item.get("n_global"), "n_records": item.get("n_records")}
        try:
            res = client.call(spec, [{"role": "system", "content": prompts.MINIMAL},
                                     {"role": "user",
                                      "content": QUESTION.format(sp=item["species"])}],
                              pin=A.PIN.get(spec, ()))
            return {**base, "answer_text": res["answer_text"], "cost_usd": res.get("cost_usd"),
                    "served_by": res.get("served_by"), "error": None}
        except Exception as e:
            return {**base, "answer_text": None, "error": f"{type(e).__name__}: {e}"}

    spent = 0.0
    with io.open(out, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(call, jobs), 1):
            spent += float(row.get("cost_usd") or 0)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ${spent:.3f}", flush=True)
            if spent > max_cost:
                raise SystemExit(f"cost ceiling ${max_cost:.2f} hit; re-run to resume")


# --------------------------------------------------------------------------- scoring

def score(part: str) -> list[dict]:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store

    answers = collections.defaultdict(dict)
    for r in _jsonl(DATA / "answers_mapq.jsonl"):
        if not r.get("error") and r.get("answer_text"):
            answers[r["species"]][r["model"]] = r["answer_text"]

    rows = []
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for item in _species(part):
            sp = item["species"]
            cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (sp,))
            ficha = Ficha(**cur.fetchone()[0])
            cur.execute("""SELECT o.region, o.vertiente, o.elev_dem FROM mpcr.occurrences o
                           JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                           WHERE e.species = %s""", (sp,))
            records = [(reg, None if v is None else ("Caribe" if "arib" in v else "Pacífico"), e)
                       for reg, v, e in cur.fetchall()]
            sources = {"manual": from_ficha(ficha)}
            # The map the system actually draws adds the regions where records sit.
            # Scored on those same records it would be marking its own homework, so it
            # is scored twice: as drawn ("full"), and with the regions inferred from
            # one random half of the records and the capture counted on the other half
            # ("full_heldout", both halves in turn).
            sources["full"] = with_records(sources["manual"], records)
            for m, text in answers.get(sp, {}).items():
                sources[m] = from_text(text)
            base = mask(sources["manual"])
            rng = random.Random(f"{SEED}-{sp}")
            shuffled = records[:]
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            folds = [(shuffled[:half], shuffled[half:]), (shuffled[half:], shuffled[:half])]
            for name, ing in list(sources.items()) + [("full_heldout", None)]:
                if name == "full_heldout":
                    ings = [with_records(sources["manual"], train) for train, _ in folds]
                    mk = mask(ings[0])
                    row = {"species": sp, "stratum": item["stratum"], "source": name,
                           "ingredients": ings[0], "n_records": len(records),
                           "has_map": mk is not None}
                    if mk is not None:
                        row["area"] = float(np.mean([mask(i).mean() for i in ings]))
                        row["captured"] = sum(captured(i, test)
                                              for i, (_, test) in zip(ings, folds))
                    rows.append(row)
                    continue
                mk = mask(ing)
                row = {"species": sp, "stratum": item["stratum"], "source": name,
                       "ingredients": ing, "n_records": len(records),
                       "has_map": mk is not None}
                if mk is not None:
                    row["area"] = float(mk.mean())
                    row["captured"] = captured(ing, records)
                    if base is not None:
                        union = (mk | base).sum()
                        row["iou_manual"] = float((mk & base).sum() / union) if union else 0.0
                rows.append(row)
    conn.close()
    with io.open(DATA / f"mapcompare_scores_{part}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def report(part: str) -> None:
    rows = score(part)
    order = ["manual", "full", "full_heldout"]
    sources = order + sorted({r["source"] for r in rows} - set(order))
    print(f"{'source':8s} {'species':>8s} {'no map':>7s} {'with >=10 rec':>14s} "
          f"{'records captured':>17s} {'median capture':>15s} {'median area':>12s} "
          f"{'median IoU w/ manual':>21s}")
    for s in sources:
        sub = [r for r in rows if r["source"] == s]
        maps = [r for r in sub if r["has_map"]]
        rich = [r for r in maps if r["n_records"] >= MIN_RECORDS]
        cap = sum(r["captured"] for r in rich)
        tot = sum(r["n_records"] for r in rich)
        med_cap = statistics.median(r["captured"] / r["n_records"] for r in rich) if rich else 0
        med_area = statistics.median(r["area"] for r in maps) if maps else 0
        ious = [r["iou_manual"] for r in maps if "iou_manual" in r and s != "manual"]
        print(f"{s:13s}"[:13] + f" {len(sub):8d} {len(sub) - len(maps):7d} {len(rich):14d} "
              f"{cap:7d}/{tot:<7d}{100 * cap / tot if tot else 0:3.0f}% {100 * med_cap:14.0f}% "
              f"{100 * med_area:11.1f}% "
              f"{(f'{100 * statistics.median(ious):.0f}%' if ious else '-'):>21s}")
    from scipy.stats import binomtest
    print()
    print(f"{'source':13s} {'times better than a random map of its size (median)':>52s} "
          f"{'beats chance':>13s} {'at or below chance':>19s} {'captures >= half':>17s}")
    for s in sources:
        sub = [r for r in rows if r["source"] == s and r["has_map"]
               and r["n_records"] >= MIN_RECORDS and r.get("area", 0) > 0]
        if not sub:
            continue
        rate = [r["captured"] / r["n_records"] for r in sub]
        lift = statistics.median(c / r["area"] for c, r in zip(rate, sub))
        beat = sum(binomtest(r["captured"], r["n_records"], min(r["area"], 1.0),
                             alternative="greater").pvalue < 0.05 for r in sub)
        print(f"{s:13s} {lift:52.1f} {beat:9d}/{len(sub):<3d} "
              f"{sum(c <= r['area'] for c, r in zip(rate, sub)):15d}/{len(sub):<3d} "
              f"{sum(c >= 0.5 for c in rate):13d}/{len(sub):<3d}")

    # By how well known the species is worldwide: the paper's question, asked of maps.
    print("\nmedian capture by global-familiarity tier (species with >= 10 records)")
    print(f"{'source':8s}" + "".join(f"{'G' + g:>10s}" for g in "123"))
    for s in sources:
        cells = []
        for g in "123":
            sub = [r for r in rows if r["source"] == s and r["has_map"]
                   and r["n_records"] >= MIN_RECORDS and r["stratum"].startswith("G" + g)]
            cells.append(f"{100 * statistics.median(r['captured'] / r['n_records'] for r in sub):.0f}% "
                         f"(n={len(sub)})" if sub else "-")
        print(f"{s:13s}" + "".join(f"{c:>12s}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Maps from a model's description, scored")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--models", default="open", help="comma separated: open, paid")
    ap.add_argument("--part", default="main", choices=["pilot", "main", "maps"],
                    help="maps = the map comparison's own 300-species sample")
    ap.add_argument("--max-cost", type=float, default=0.60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--species-file", default="",
                    help="a JSON list of species; generate and report only for those")
    args = ap.parse_args()
    if args.species_file:
        global _ONLY
        _ONLY = set(json.loads(Path(args.species_file).read_text(encoding="utf-8")))
    if args.generate:
        generate([m.strip() for m in args.models.split(",")], args.part, args.max_cost,
                 args.workers)
    if args.report or not args.generate:
        report(args.part)


if __name__ == "__main__":
    main()
