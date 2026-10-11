"""The hard routing set: questions a router cannot solve by reading a template.

The benchmark questions are generated from one template per type, so their route is
nearly legible from the wording and the router scores 36 of 36 on them. That measures
nothing. This module builds the inputs a deployed system actually receives, in five
categories, each with a gold label that does not depend on any model's opinion:

  paraphrase        a benchmark question reworded the way a visitor or grower would ask
                    it; the gold route, species and place are the original's
  common_name       a question that uses a vernacular name instead of the binomial; gold
                    is every catalogue species that name denotes (one, or several)
  list              a question asking for a set of species; taken verbatim from the
                    30-query set of the companion paper; gold route catalog_search
  not_in_catalogue  a real plant the ingested volumes do not hold; the route is valid
                    and the species must come back UNRESOLVED, not guessed
  out_of_scope      not about Costa Rican plants at all

Only the paraphrases are written by a model, and not by the model that routes: the
frontier generator rewrites, the open-weights model is the router under test. Every row
is written with reviewed=false. The set is a DRAFT until a person has read it; the
review sheet is the CSV written beside it.

Run:  python -m benchmark.routing_set            (writes routing_hard.jsonl + .csv)
"""
from __future__ import annotations

import collections
import csv
import io
import json
import random
import re
from pathlib import Path

from . import client
from .arms import PAID, PIN
from .router import QTYPE_ROUTE, _norm

DATA = Path(__file__).resolve().parent / "data"
REPO = Path(__file__).resolve().parent.parent
SEED = 20261011
PER_TYPE = 5

_REWRITE = (
    "Reescribe la siguiente pregunta como la haría una persona común en Costa Rica (un "
    "turista, un agricultor, un estudiante): con otras palabras, otro orden y un tono "
    "coloquial. Conserva EXACTAMENTE el nombre científico, y el nombre del parque, el "
    "lugar y la altitud si aparecen. No agregues ni quites información. Devuelve solo la "
    "pregunta reescrita, en una línea.")

# Frames for the common-name questions. {n} is the vernacular name. They avoid an
# article before the name, because its gender is not something the table records.
_COMMON_FRAMES = [
    ("elevation", "¿A qué altura crece la planta que aquí llaman {n}?"),
    ("flowering", "¿En qué época del año florece lo que llaman {n}?"),
    ("vertiente", "Lo que conocemos como {n}, ¿se da del lado del Caribe o del Pacífico?"),
    ("uses", "¿Para qué sirve la planta conocida como {n}? ¿Es medicinal?"),
    ("growth_form", "La planta que llaman {n}, ¿es un árbol o un arbusto?"),
]

# The names are chosen by hand, not sampled, and the choice is the point: the first
# draft sampled the whole vernacular table and drew concatenated entries ("machetitos
# palo madre"), near-duplicates ("cocobola") and names from global indexes that nobody
# in Costa Rica uses that way. Only 183 of 6,946 catalogue species carry any common
# name, and 32 names come from a Costa Rica-specific source, so this block is small on
# purpose. Keys are the normalized form stored in the table; values are how the name is
# written in a question. Gold candidates are read from the table at build time.
_NAMES_ONE = {      # one catalogue species, name from the Costa Rica-specific source
    "bayo rojo": "bayo rojo", "caracoli": "caracolí", "caudillo": "caudillo",
    "cinco real": "cinco real", "corteza amarilla": "corteza amarilla",
    "encino amarillo": "encino amarillo", "hierba gallinera": "hierba gallinera",
    "madre de agua": "madre de agua", "poro extranjero": "poró extranjero",
    "roble sabana": "roble sabana",
}
_NAMES_MANY = {     # several catalogue species: the right answer is to return them all
    "pitahaya": "pitahaya", "espavel": "espavel", "cortez amarillo": "cortez amarillo",
    "poro": "poró", "nazareno": "nazareno", "cocobolo": "cocobolo", "ceiba": "ceiba",
    "encino": "encino",
}

# Real plants checked absent from the catalogue on 2026-10-11 (ferns and allies are not
# in the ingested volumes; the rest belong to volumes not yet ingested or are simply
# missing). `build` re-checks each and drops any that has since been added.
_ABSENT = [
    ("elevation", "¿A qué altura crece Cecropia obtusifolia en Costa Rica?", "Cecropia obtusifolia"),
    ("flowering", "¿En qué meses florece Cecropia peltata?", "Cecropia peltata"),
    ("vertiente", "¿Rhizophora mangle crece en el Caribe, en el Pacífico o en ambas costas?", "Rhizophora mangle"),
    ("growth_form", "¿Pteridium aquilinum es una hierba o un arbusto?", "Pteridium aquilinum"),
    ("elevation", "¿Hasta qué altitud llega Cyathea multiflora?", "Cyathea multiflora"),
    ("flowering", "¿Cuándo florece la Guarianthe skinneri?", "Guarianthe skinneri"),
    ("cultivation", "Tengo una finca en Turrialba, a unos 650 m. ¿Podría sembrar Coffea arabica ahí?", "Coffea arabica"),
    ("endemism", "¿Alsophila firma es endémica de Costa Rica?", "Alsophila firma"),
    ("light", "¿Selaginella pallescens prefiere sol o sombra?", "Selaginella pallescens"),
    ("uses", "¿Pinus caribaea tiene algún uso medicinal?", "Pinus caribaea"),
    ("endemism", "¿Zamia neurophyllidia es endémica del país?", "Zamia neurophyllidia"),
    ("uses", "¿La madera de Cedrela odorata sirve para algo más, es medicinal?", "Cedrela odorata"),
]

_OUT_OF_SCOPE = [
    "¿Dónde puedo ver quetzales en Costa Rica?",
    "¿A qué altura vive el jaguar en la Cordillera de Talamanca?",
    "¿Cuál es el hongo más venenoso de Costa Rica?",
    "¿En qué meses anidan las tortugas baulas en Guanacaste?",
    "¿Qué plantas crecen en el páramo de Colombia?",
    "¿Cuánto cuesta la entrada al Parque Nacional Manuel Antonio?",
    "¿Cuál es la capital de Costa Rica?",
    "¿Qué serpientes venenosas hay en el Parque Nacional Corcovado?",
    "¿Cómo se prepara un gallo pinto?",
    "¿Qué especies de ranas son endémicas de Costa Rica?",
    "¿A qué hora abre el Jardín Botánico Lankester?",
    "¿Qué árboles son típicos de los bosques de Alemania?",
    "¿Cuántos volcanes activos tiene Costa Rica?",
    "¿Qué mariposas se ven en Monteverde?",
    "¿Cómo está el clima hoy en Limón?",
]


def _pilot() -> list[dict]:
    return [json.loads(l) for l in io.open(DATA / "questions_pilot.jsonl", encoding="utf-8")
            if l.strip()]


def _paraphrases(rng: random.Random) -> tuple[list[dict], float]:
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for q in _pilot():
        by_type[q["qtype"]].append(q)
    out, cost = [], 0.0
    for qt in sorted(by_type):
        for q in rng.sample(by_type[qt], min(PER_TYPE, len(by_type[qt]))):
            res = client.call(PAID, [{"role": "system", "content": _REWRITE},
                                     {"role": "user", "content": q["question"]}],
                              pin=PIN[PAID])
            cost += res.get("cost_usd") or 0
            text = res["answer_text"].strip().splitlines()[0].strip().strip('"')
            # A rewrite that lost the binomial is not a paraphrase of this question;
            # it is dropped and counted rather than repaired by hand.
            if _norm(q["species"]) not in _norm(text):
                print(f"  dropped (binomial lost): {q['qid']}")
                continue
            out.append({"category": "paraphrase", "question": text, "source_qid": q["qid"],
                        "original": q["question"],
                        "gold": {"route": QTYPE_ROUTE.get(q["qtype"], q["qtype"]),
                                 "species": [q["species"]],
                                 "place_code": (q.get("key") or {}).get("place_code")}})
    return out, cost


def _common_names(cur, rng: random.Random) -> list[dict]:
    cur.execute("""SELECT v.name_norm, array_agg(DISTINCT v.species)
                   FROM mpcr.vernacular v JOIN mpcr.fichas f ON f.species = v.species
                   GROUP BY 1""")
    table = {n: sorted(sp) for n, sp in cur.fetchall()}
    out = []
    for i, (norm, shown) in enumerate([*_NAMES_ONE.items(), *_NAMES_MANY.items()]):
        species = table.get(norm, [])
        expect_many = norm in _NAMES_MANY
        if not species or (len(species) > 1) != expect_many:
            # The table moved under the hand-picked list; say so rather than key a
            # question against a mapping nobody chose.
            print(f"  dropped (table disagrees): {norm!r} -> {len(species)} species")
            continue
        route, frame = _COMMON_FRAMES[i % len(_COMMON_FRAMES)]
        out.append({"category": "common_name", "question": frame.format(n=shown),
                    "common_name": norm,
                    "gold": {"route": route, "species": species, "place_code": None}})
    return out


def _lists() -> list[dict]:
    t = json.loads((REPO / "mpcr_rag/eval/data/query_testset.json").read_text(encoding="utf-8"))
    out = []
    for q in t["queries"]:
        text = q.get("question") or q.get("query") or q.get("q")
        if text:
            out.append({"category": "list", "question": text,
                        "gold": {"route": "catalog_search", "species": [],
                                 "place_code": None}})
    return out


def _absent(cur) -> list[dict]:
    out = []
    for route, question, species in _ABSENT:
        cur.execute("SELECT 1 FROM mpcr.fichas WHERE lower(species) = lower(%s)", (species,))
        if cur.fetchone():
            print(f"  dropped (now in catalogue): {species}")
            continue
        out.append({"category": "not_in_catalogue", "question": question, "asked_about": species,
                    "gold": {"route": route, "species": [], "place_code": None}})
    return out


def _kept_paraphrases() -> list[dict]:
    """Paraphrases already written, if any. Rewording is the only step that costs
    money and the only one that is not reproducible call for call, so a rebuild of the
    other categories keeps them unless --fresh asks otherwise."""
    path = DATA / "routing_hard.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
    return [{k: v for k, v in r.items() if k not in ("id", "reviewed")}
            for r in rows if r["category"] == "paraphrase"]


def build(fresh: bool = False) -> Path:
    from mpcr_rag.store import pg_store
    rng = random.Random(SEED)
    conn = pg_store.connect()
    with conn.cursor() as cur:
        kept = [] if fresh else _kept_paraphrases()
        para, cost = (kept, 0.0) if kept else _paraphrases(rng)
        rows = (para + _common_names(cur, rng) + _lists() + _absent(cur)
                + [{"category": "out_of_scope", "question": q,
                    "gold": {"route": "out_of_scope", "species": [], "place_code": None}}
                   for q in _OUT_OF_SCOPE])
    conn.close()
    for i, r in enumerate(rows, 1):
        r["id"] = f"R{i:03d}"
        r["reviewed"] = False

    out = DATA / "routing_hard.jsonl"
    with io.open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with io.open(DATA / "routing_hard_review.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "category", "question", "original_question", "gold_route",
                    "gold_species", "gold_place_code", "ok (s/n)", "corrected_question",
                    "comment"])
        for r in rows:
            w.writerow([r["id"], r["category"], r["question"], r.get("original", ""),
                        r["gold"]["route"], "; ".join(r["gold"]["species"]),
                        r["gold"]["place_code"] or "", "", "", ""])
    print(f"{len(rows)} questions, paraphrase cost ${cost:.4f} -> {out}")
    print("  " + ", ".join(f"{k}={v}" for k, v in
                           collections.Counter(r["category"] for r in rows).items()))
    return out


if __name__ == "__main__":
    import sys
    build(fresh="--fresh" in sys.argv)
