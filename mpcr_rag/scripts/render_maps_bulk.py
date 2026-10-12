"""Bulk map renderer — generates PNG distribution maps for a random sample of species.

Usage:
    python -m mpcr_rag.scripts.render_maps_bulk            # 500 species (default)
    python -m mpcr_rag.scripts.render_maps_bulk 200        # custom sample size
    python -m mpcr_rag.scripts.render_maps_bulk 500 --seed 99

Maps are written to data/maps/ (same folder as the interactive renderer).
A CSV manifest (bulk_maps_manifest.csv) is written to that folder on completion.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import config
from ..query import gbif_map
from ..query.retriever import filter_all
from ..store import local_store


_MAPS_DIR = config.DATA_DIR / "maps"
_MAPS_DIR.mkdir(parents=True, exist_ok=True)

_MANIFEST = _MAPS_DIR / "bulk_maps_manifest.csv"
_WORKERS = 8


def _render_one(ficha) -> dict:
    t0 = time.time()
    try:
        map_path, n_pts = gbif_map.single_species_map(ficha)
        return {
            "species": ficha.species,
            "family": ficha.family,
            "volume": ficha.volume,
            "pages": ficha.pages,
            "elev_min": ficha.elev_min,
            "elev_max": ficha.elev_max,
            "endemic": ficha.endemic_cr,
            "n_gbif_pts": n_pts,
            "map_path": str(map_path),
            "elapsed_s": round(time.time() - t0, 1),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            "species": ficha.species,
            "family": ficha.family,
            "volume": ficha.volume,
            "pages": ficha.pages,
            "elev_min": ficha.elev_min,
            "elev_max": ficha.elev_max,
            "endemic": ficha.endemic_cr,
            "n_gbif_pts": 0,
            "map_path": "",
            "elapsed_s": round(time.time() - t0, 1),
            "status": "error",
            "error": str(exc),
        }


def main(n: int = 500, seed: int = 42) -> None:
    conn = local_store.connect(config.SQLITE_PATH)
    all_fichas = filter_all(conn)
    total = len(all_fichas)
    print(f"Catalog: {total:,} species total")

    if n > total:
        print(f"Requested {n} but catalog only has {total} — using all.")
        n = total

    rng = random.Random(seed)
    sample = rng.sample(all_fichas, n)
    print(f"Sample:  {n} species  (seed={seed})")
    print(f"Workers: {_WORKERS} threads")
    print(f"Output:  {_MAPS_DIR}\n")

    rows: list[dict] = []
    ok = err = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_render_one, f): f for f in sample}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if row["status"] == "ok":
                ok += 1
            else:
                err += 1
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (n - i) / rate if rate > 0 else 0
            status_char = "." if row["status"] == "ok" else "E"
            print(
                f"\r[{i:>4}/{n}] {status_char} ok={ok} err={err} "
                f"rate={rate:.1f}/s eta={eta:.0f}s   ",
                end="",
                flush=True,
            )

    print()  # newline after progress line

    # Write manifest sorted by species name
    rows.sort(key=lambda r: r["species"])
    fieldnames = ["species", "family", "volume", "pages", "elev_min", "elev_max",
                  "endemic", "n_gbif_pts", "map_path", "elapsed_s", "status", "error"]
    with _MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s")
    print(f"  ok:      {ok}")
    print(f"  errors:  {err}")
    print(f"  manifest: {_MANIFEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk species map renderer")
    parser.add_argument("n", nargs="?", type=int, default=500,
                        help="Number of species to render (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for species sample (default: 42)")
    args = parser.parse_args()
    main(n=args.n, seed=args.seed)
