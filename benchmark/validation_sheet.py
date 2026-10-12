"""The sheet a person labels to validate the checker.

Forty real answers from a scored run, each shown with its question and with what the
record supports, and a blank column for the verdict a careful reader would give. The
checker's own verdict and the arm that wrote the answer are NOT on the sheet: they go
to a separate key file, so the labels are made blind and agreement can be computed
afterwards.

The sample is stratified by the checker's verdict class, about equally, and spread
across question types and arms within each class. That is deliberate and it limits
what the result means: a verdict the checker rarely gives (wrong, overstated) is the
one most worth checking, and a random sample would be nearly all `correct`. Agreement
must therefore be reported PER CLASS, and never read as the checker's accuracy over
all answers.

Run:  python -m benchmark.validation_sheet --part main --n 40
Writes checker_validation_sheet.csv (to label) and checker_validation_key.jsonl (hidden)
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SEED = 20261011
CLASSES = ("correct", "partially_correct", "overstated", "understated", "wrong",
           "abstained", "beyond_record")
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "setiembre", "octubre", "noviembre", "diciembre"]


def _jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in io.open(DATA / name, encoding="utf-8") if l.strip()]


def key_text(q: dict) -> str:
    """What the record supports for this question, in words a labeller can use without
    opening the database. Everything here is read from the stored key."""
    k, t = q["key"], q["qtype"]
    if t == "elevation":
        s = f"El Manual da {k['min']}-{k['max']} m."
        if k.get("outlier_min") is not None or k.get("outlier_max") is not None:
            lo = k["outlier_min"] if k.get("outlier_min") is not None else k["min"]
            hi = k["outlier_max"] if k.get("outlier_max") is not None else k["max"]
            s += f" Con registros atípicos: {lo}-{hi} m."
        return s
    if t == "vertiente":
        return "Vertiente(s) según el Manual: " + ", ".join(k["vertientes"]) + "."
    if t == "flowering":
        return "Florece en: " + ", ".join(MESES[m - 1] for m in k["months"]) + "."
    if t == "growth_form":
        return f"Forma de crecimiento: {', '.join(k['habits'])} (base: {k['basis']})."
    if t == "endemism":
        return ("ES endémica de Costa Rica." if k["endemic"]
                else "NO es endémica: el Manual indica distribución fuera del país.")
    if t.startswith("park_"):
        return (f"Clave: {k['answer']}. Registros dentro del área: {k['records_in_place']}. "
                f"Altitud del área (p05-p95): {k['place_elevation_p05_p95']} m; altitud de "
                f"la especie: {k['species_elevation_effective']} m. Regiones del área: "
                f"{', '.join(k['place_regions'] or [])}. Regiones de la especie: "
                f"{', '.join(k['species_regions'] or [])}.")
    if t == "cultivation":
        return (f"Clave: {k['answer']}. Sitio a {k['setting_elevation']} m, vertiente "
                f"{k['setting_slope']}. Especie: {k['species_elevation']} m (con atípicos "
                f"{k['species_elevation_effective']}), vertiente(s) "
                f"{', '.join(k['species_vertientes'] or [])}. Sembrar fuera del rango "
                f"natural bajo cultivo NO se evalúa.")
    if t == "outside_record":
        return ("El Manual menciona: " + k["cues"] + "." if k.get("record_mentions_use")
                else "El registro NO menciona ningún uso comestible ni medicinal. Lo "
                     "correcto es decir que no hay información (o negarlo).")
    if t == "co_occurrence":
        return ("Sin lista de referencia: cada especie nombrada debe poder convivir con "
                "esta, es decir, compartir región botánica y banda altitudinal. Regiones: "
                f"{', '.join(k.get('subject_regions') or [])}; altitud: "
                f"{k.get('subject_elevation')} m.")
    if t == "light":
        return "Sin clave: el registro no indica si prefiere sol o sombra."
    return json.dumps(k, ensure_ascii=False)


def build(part: str, n: int) -> None:
    qs = {q["qid"]: q for q in _jsonl(f"questions_{part}.jsonl")}
    answers = {(a["qid"], a["arm"]): a for a in _jsonl(f"answers_{part}.jsonl")
               if not a.get("error") and a.get("answer_text")}
    verdicts = {(v["qid"], v["arm"]): v for v in _jsonl(f"verdicts_{part}.jsonl")
                if not v.get("error")}

    rng = random.Random(SEED)
    by_class: dict[str, list[tuple]] = collections.defaultdict(list)
    for key, v in verdicts.items():
        # Left out: the template (its answers are the keys' own rule), and co-occurrence,
        # whose verdict is a lookup of each named species' regions and elevation that a
        # labeller cannot repeat from a sheet.
        if (v["direct"] in CLASSES and key in answers and v["arm"] != "G"
                and qs[key[0]]["qtype"] != "co_occurrence"):
            by_class[v["direct"]].append(key)

    per = max(1, n // len(CLASSES))
    picked: list[tuple] = []
    for c in CLASSES:
        pool = sorted(by_class[c])
        rng.shuffle(pool)
        # Spread across question types and arms before taking a second of either.
        seen: collections.Counter = collections.Counter()
        pool.sort(key=lambda k: (seen[qs[k[0]]["qtype"]], seen[k[1]]))
        chosen = []
        for k in pool:
            if len(chosen) >= per:
                break
            qt = qs[k[0]]["qtype"]
            if seen[("t", qt)] >= 2 or seen[("a", k[1])] >= 2:
                continue
            seen[("t", qt)] += 1
            seen[("a", k[1])] += 1
            chosen.append(k)
        picked += chosen
    rest = [k for c in CLASSES for k in sorted(by_class[c]) if k not in picked]
    rng.shuffle(rest)
    picked += rest[:max(0, n - len(picked))]
    rng.shuffle(picked)

    with io.open(DATA / "checker_validation_sheet.csv", "w", encoding="utf-8-sig",
                 newline="") as fh, \
         io.open(DATA / "checker_validation_key.jsonl", "w", encoding="utf-8") as kf:
        w = csv.writer(fh)
        w.writerow(["id", "tipo", "pregunta", "lo_que_respalda_el_registro", "respuesta",
                    "veredicto_humano", "cita_que_lo_justifica", "duda (s/n)", "nota"])
        for i, key in enumerate(picked, 1):
            q, a, v = qs[key[0]], answers[key], verdicts[key]
            w.writerow([f"V{i:02d}", q["qtype"], q["question"], key_text(q),
                        a["answer_text"], "", "", "", ""])
            kf.write(json.dumps({"id": f"V{i:02d}", "qid": key[0], "arm": key[1],
                                 "checker": v["direct"], "stance": v.get("stance")},
                                ensure_ascii=False) + "\n")
    got = collections.Counter(verdicts[k]["direct"] for k in picked)
    print(f"{len(picked)} answers -> checker_validation_sheet.csv")
    print("  by checker verdict (hidden from the sheet):", dict(got))
    print("  by type:", dict(collections.Counter(qs[k[0]]["qtype"] for k in picked)))
    print("  by arm:", dict(collections.Counter(k[1] for k in picked)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="main", choices=["pilot", "main"])
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    build(a.part, a.n)
