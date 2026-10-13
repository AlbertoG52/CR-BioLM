"""Draw the question maps for a set of species ahead of a run, in parallel, and check them.

In tranche 1 the maps were drawn one at a time inside the generation run, which took
about three and a half of its five hours and produced nothing to look at until it was
over. This draws the same images beforehand with several processes, so that `run.py`
finds them on disk (`qmap.render(reuse=True)`), and so that a map with something
visibly wrong -- no elevation band, no records, a park that did not make it onto the
picture -- is found before any call is paid for.

The list of maps is built exactly as `run.py` builds it for the map arms: one per
spatial question, from the same pack facts. Nothing about the images changes; they are
only drawn earlier.

Run:  python -u -m benchmark.prerender --species-file benchmark/data/species_tranche2.json
Writes the images to benchmark/data/qmaps/ and a check sheet, qmaps_check_<name>.csv
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def _tasks(part: str, species: set[str]) -> list[dict]:
    from mpcr_rag.store import pg_store
    from . import arms as A
    from .evidence_pack import build as build_pack

    out, seen = [], set()
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for line in io.open(DATA / f"questions_{part}.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            q = json.loads(line)
            if q["species"] not in species or q["qtype"] not in A.MAP_QTYPES:
                continue
            pk = build_pack(cur, q["species"], q["qtype"], level="full",
                            place_code=q["key"].get("place_code"))
            regs = pk.facts.get("regions") if q["qtype"].startswith("park_") else None
            key = (q["species"], q["key"].get("place_code"), bool(regs))
            if key in seen:
                continue
            seen.add(key)
            out.append({"qid": q["qid"], "qtype": q["qtype"], "species": q["species"],
                        "place_code": q["key"].get("place_code"),
                        "elev_min": pk.facts.get("elev_min"),
                        "elev_max": pk.facts.get("elev_max"), "regions": regs,
                        "key_answer": q["key"].get("answer"),
                        "records_in_place": q["key"].get("records_in_place")})
    conn.close()
    return out


def _draw(t: dict) -> dict:
    # Imported inside the worker: each process needs its own matplotlib and its own
    # database connection.
    from .qmap import render
    t0 = time.time()
    try:
        path, shown = render(t["species"], t["place_code"], elev_min=t["elev_min"],
                             elev_max=t["elev_max"], highlight_regions=t["regions"],
                             reuse=not t.get("fresh"))
        return {**t, "path": str(path), "shown": shown, "seconds": round(time.time() - t0, 1),
                "error": None}
    except Exception as e:
        return {**t, "path": None, "shown": {}, "seconds": round(time.time() - t0, 1),
                "error": f"{type(e).__name__}: {e}"}


def _flags(r: dict) -> list[str]:
    """What a person should look at. None of these is necessarily an error -- a species
    with no parsed elevation has no band to draw -- but each is worth one glance."""
    s, out = r["shown"], []
    if r["error"]:
        return ["NO SE PUDO DIBUJAR: " + r["error"]]
    if not s.get("band_drawn"):
        out.append("sin franja altitudinal" + (" (la ficha no tiene altitud)"
                                               if r["elev_min"] is None else ""))
    if not s.get("n_points"):
        out.append("sin registros en el mapa")
    if r["place_code"] and not s.get("park_drawn"):
        out.append("el área consultada no quedó dibujada")
    if s.get("regions_missing"):
        out.append("regiones no encontradas: " + ", ".join(s["regions_missing"]))
    if r["qtype"] == "park_confirmed" and not s.get("n_points"):
        out.append("la clave dice que hay registros en el área y el mapa no muestra ninguno")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw and check question maps ahead of a run")
    ap.add_argument("--part", default="main", choices=["pilot", "main"])
    ap.add_argument("--species-file", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fresh", action="store_true",
                    help="redraw even where an image exists (after a change to the map)")
    args = ap.parse_args()

    name = Path(args.species_file).stem
    species = set(json.loads(Path(args.species_file).read_text(encoding="utf-8")))
    tasks = [{**t, "fresh": args.fresh} for t in _tasks(args.part, species)]
    # One process per species at a time: two maps of the same species drawn at once
    # collide on shared cache files.
    tasks.sort(key=lambda t: t["species"])
    print(f"{len(tasks)} maps for {len(species)} species, {args.workers} processes", flush=True)

    t0, rows = time.time(), []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(_draw, tasks), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f} s", flush=True)

    out = DATA / f"qmaps_check_{name}.csv"
    flagged = 0
    with io.open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["especie", "tipo", "area", "clave", "registros_en_el_mapa",
                    "franja_dibujada", "area_dibujada", "revisar", "archivo"])
        for r in sorted(rows, key=lambda r: (r["species"], r["qtype"])):
            fl = _flags(r)
            flagged += bool(fl)
            s = r["shown"]
            w.writerow([r["species"], r["qtype"], r["place_code"] or "", r["key_answer"] or "",
                        s.get("n_points", ""), "sí" if s.get("band_drawn") else "no",
                        "sí" if s.get("park_drawn") else ("" if not r["place_code"] else "no"),
                        "; ".join(fl), Path(r["path"]).name if r["path"] else ""])
    print(f"done in {time.time() - t0:.0f} s; {flagged} of {len(rows)} maps to look at -> {out}")


if __name__ == "__main__":
    main()
