"""Side-by-side pictures for the map comparison (benchmark/mapcompare.py).

Three panels per species, on the same terrain grid and with the same records drawn on
each: the map from the Manual's stated range, the map the system draws (the Manual plus
the regions where at least five records sit), and the map that follows from a model's
description. Each panel states how many records it captures and how much of the country
it covers, which are the two numbers the comparison reports.

Which species are shown is fixed by rule, not by eye: four drawn at random by a recorded
seed from the species with at least thirty records, the one where the model's map did
best against the Manual's, and any named with --also.

Run:  python -m benchmark.mapcompare_figs --model open --also "Calyptranthes pittieri"
Writes benchmark/data/mapcompare_examples/<species>.png
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import mapcompare as M

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "mapcompare_examples"
SEED = 20261012
LAND, FILL, DOT_IN, DOT_OUT, INK = "#e9ebe8", "#2a9d8f", "#1d3557", "#d1495b", "#1f2a28"
# Labels in Spanish for the experts, in English for the paper.
TEXT = {"es": {"cap": "capta {n} de {t} registros · cubre {a:.1f}% del país",
               "ours": "Nuestro mapa: Manual + regiones con registros",
               "manual": "Solo el rango que declara el Manual",
               "model": "{m}: su descripción", "noelev": "sin altitud",
               "all": "todo el país", "noreg": "sin regiones", "more": " y {k} más",
               "noslope": "sin vertiente"},
        "en": {"cap": "captures {n} of {t} records · covers {a:.1f}% of the country",
               "ours": "Ours: the Manual plus regions with records",
               "manual": "The Manual's stated range alone",
               "model": "{m}: from its own description", "noelev": "no elevation",
               "all": "whole country", "noreg": "no regions", "more": " and {k} more",
               "noslope": "no slope"}}
LANG = "es"


def _points(cur, species: str):
    cur.execute("""SELECT o.lon, o.lat, o.region, o.vertiente, o.elev_dem
                   FROM mpcr.occurrences o
                   JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                   WHERE e.species = %s""", (species,))
    return [(lon, lat, (reg, None if v is None else ("Caribe" if "arib" in v else "Pacífico"), e))
            for lon, lat, reg, v, e in cur.fetchall()]


def _panel(ax, title: str, ing: dict | None, pts: list) -> None:
    M._grid()
    land, b = M._GEO["land"], M._GEO["bounds"]
    extent = (b.left, b.right, b.bottom, b.top)
    ax.imshow(np.where(land, 1.0, np.nan), extent=extent, cmap=ListedColormap([LAND]),
              interpolation="nearest")
    n_in = 0
    area = 0.0
    if ing is not None and M.mask(ing) is not None:
        mk = M.mask(ing)
        full = np.zeros(land.shape, dtype=float)
        full[:] = np.nan
        full[land] = np.where(mk, 1.0, np.nan)
        ax.imshow(full, extent=extent, cmap=ListedColormap([FILL]), alpha=0.85,
                  interpolation="nearest")
        area = float(mk.mean())
    for lon, lat, rec in pts:
        inside = ing is not None and M.captured(ing, [rec]) == 1
        n_in += inside
        ax.scatter([lon], [lat], s=13, color=DOT_IN if inside else DOT_OUT,
                   edgecolor="white", linewidth=0.4, zorder=3)
    cap = TEXT[LANG]["cap"].format(n=n_in, t=len(pts), a=100 * area)
    ax.set_title(f"{title}\n{cap}", fontsize=9.5, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _describe(ing: dict) -> str:
    t = TEXT[LANG]
    elev = f"{ing['elev'][0]}–{ing['elev'][1]} m" if ing["elev"] else t["noelev"]
    r = ing["regions"]
    reg = (t["all"] if ing.get("everywhere") else t["noreg"] if not r
           else ", ".join(r) if len(r) <= 4
           else ", ".join(r[:3]) + t["more"].format(k=len(r) - 3))
    return f"{elev} · {', '.join(ing['slopes']) or t['noslope']} · {reg}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Example panels for the map comparison")
    ap.add_argument("--model", default="open")
    ap.add_argument("--also", default="", help="comma separated species to add")
    ap.add_argument("--models", default="",
                    help="comma separated models to show side by side (default: --model)")
    ap.add_argument("--species", default="",
                    help="comma separated species: draw exactly these, no random draw")
    ap.add_argument("--out", default="", help="subfolder for the images")
    ap.add_argument("--part", default="main", help="which scored sample to draw from")
    ap.add_argument("--lang", default="es", choices=sorted(TEXT))
    args = ap.parse_args()
    global LANG
    LANG = args.lang
    t = TEXT[LANG]
    models = [m.strip() for m in (args.models or args.model).split(",") if m.strip()]
    label = {"open": "Gemma 4 31B", "paid": "Mistral Large 4"}

    from mpcr_rag.store import pg_store
    rows = [json.loads(l) for l in io.open(DATA / f"mapcompare_scores_{args.part}.jsonl", encoding="utf-8")]
    by = {(r["species"], r["source"]): r for r in rows}
    rich = sorted(sp for (sp, src), r in by.items()
                  if src == args.model and r["n_records"] >= 30 and by[(sp, "manual")]["has_map"])
    if args.species:
        picked, best = [s.strip() for s in args.species.split(",") if s.strip()], None
    else:
        picked = random.Random(SEED).sample(rich, min(4, len(rich)))
        best = max(rich, key=lambda sp: by[(sp, args.model)].get("captured", 0)
                   - by[(sp, "manual")].get("captured", 0))
        for sp in [best] + [s.strip() for s in args.also.split(",") if s.strip()]:
            if sp not in picked:
                picked.append(sp)

    out_dir = OUT / args.out if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for sp in picked:
            pts = _points(cur, sp)
            recs = [p[2] for p in pts]
            manual = by[(sp, "manual")]["ingredients"]
            # Our map first: it is what the system delivers and the reference the others
            # are read against. The Manual's stated range alone is shown second, as the
            # diagnostic of what the record-based regions add.
            panels = [(t["ours"], M.with_records(manual, recs)), (t["manual"], manual)]
            panels += [(t["model"].format(m=label.get(m, m)), by[(sp, m)]["ingredients"])
                       for m in models if (sp, m) in by]
            fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 5.4), dpi=120)
            for ax, (title, ing) in zip(axes, panels):
                _panel(ax, title, ing, pts)
            fig.suptitle(sp, fontsize=13, style="italic", color=INK, y=0.99)
            lines = ["Manual: " + _describe(manual)] + [
                t.split(":")[0] + ": " + _describe(ing) for t, ing in panels[2:]]
            for k, line in enumerate(lines):
                fig.text(0.5, 0.012 + 0.032 * (len(lines) - 1 - k), line, ha="center",
                         fontsize=8, color=INK)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.86,
                                bottom=0.04 + 0.032 * len(lines), wspace=0.03)
            path = out_dir / (sp.replace(" ", "_") + ".png")
            fig.savefig(path, facecolor="white")
            plt.close(fig)
            why = "mejor caso del modelo" if sp == best else "al azar o solicitada"
            print(f"{sp}  ({why}) -> {path.name}")
    conn.close()


if __name__ == "__main__":
    main()
