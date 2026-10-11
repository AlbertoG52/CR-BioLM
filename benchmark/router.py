"""The router: from a bare question to a route, a species and a place.

Every grounded arm so far is ORACLE-ROUTED: the harness knows the question type and the
species because it generated the question, and hands `evidence_pack.build` both. A
deployed system knows neither. This module is the missing step, and it is objective 2
of the thesis proposal -- an orchestrator that reads the query and selects the tools.

The design is a hybrid, on purpose. The model does not plan free-form tool calls; it
chooses among a small catalogue of PREDEFINED ROUTES, each of which is an evidence
recipe that already exists and is already verified (one per question family in
`evidence_pack`). Two further routes are exits: `catalog_search` hands a multi-species
question to the structured search of the earlier system, and `out_of_scope` declines.
A fixed recipe is auditable in a way a free plan is not, and a wrong route is a
countable error rather than a transcript to read.

The model only proposes. What it extracts is resolved deterministically afterwards: the
species against the catalogue, the place against the protected-area table. A species
the catalogue does not hold is reported as unresolved, never guessed.

Gold labels cost nothing: every benchmark question was generated from a known type,
species and place, so the correct route is already recorded in the question files.

Run:
  python -m benchmark.router --ask "¿En qué meses florece Clethra consimilis?"
  python -m benchmark.router --eval --part pilot --per-type 3
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import unicodedata
from pathlib import Path

from . import client
from .arms import OPEN, PIN

DATA = Path(__file__).resolve().parent / "data"

# The open-weights generator routes. The proposal names an open model as the
# orchestrator, and a routing step that needed a frontier API would undo the
# deployability argument the grounded open-weights arm makes.
ROUTER_MODEL = OPEN

# route -> what it is for, in the words the model is shown.
ROUTES = {
    "elevation":      "a qué altitud crece una especie",
    "vertiente":      "en qué vertiente (Caribe, Pacífico o ambas) crece una especie",
    "flowering":      "en qué meses florece una especie",
    "growth_form":    "si una especie es árbol, arbusto, hierba, epífita o bejuco",
    "endemism":       "si una especie es endémica de Costa Rica",
    "park":           "si una especie se puede encontrar en un parque o área protegida "
                      "que la pregunta nombra",
    "cultivation":    "si una especie se podría sembrar en un sitio que la pregunta describe",
    "co_occurrence":  "qué otras plantas crecen junto a una especie",
    "uses":           "si una especie es comestible, medicinal o tiene algún uso",
    "light":          "si una especie prefiere sol o sombra",
    "catalog_search": "preguntas que piden una LISTA de especies según condiciones "
                      "(hábito, región, altitud, bosque), o la especie que más o menos "
                      "cumple algo",
    "out_of_scope":   "cualquier otra cosa: animales, hongos, otros países, o nada sobre "
                      "plantas de Costa Rica",
}

# Benchmark question type -> route. The three park types are ONE route: whether the
# answer is yes, possible or unlikely is decided by the evidence, not by the question.
QTYPE_ROUTE = {"park_confirmed": "park", "park_possible": "park", "park_unlikely": "park",
               "outside_record": "uses"}

_SYSTEM = (
    "Eres el enrutador de un sistema de consulta sobre la flora de Costa Rica. No "
    "respondes la pregunta: decides qué ruta la atiende y extraes los datos que esa ruta "
    "necesita. Devuelve SOLO un objeto JSON con estas claves:\n"
    '  "route": una de las rutas listadas\n'
    '  "species": el nombre científico tal como aparece en la pregunta, o null\n'
    '  "common_name": un nombre común citado en lugar del científico, o null\n'
    '  "place": el parque o área protegida que la pregunta nombra, o null\n'
    "Rutas:\n" + "\n".join(f"  {k}: {v}" for k, v in ROUTES.items())
)


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


def propose(question: str, spec: str = ROUTER_MODEL) -> dict:
    """The model's reading of the question. One call, temperature zero."""
    res = client.call(spec, [{"role": "system", "content": _SYSTEM},
                             {"role": "user", "content": f"PREGUNTA: {question}"}],
                      pin=PIN.get(spec, ()), max_tokens=400)
    text = re.sub(r"^```(?:json)?|```$", "", res["answer_text"].strip(), flags=re.M)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    if out.get("route") not in ROUTES:
        # An unparseable or invented route is a routing failure, recorded as such
        # rather than silently mapped to the nearest valid one.
        out["route"] = "invalid"
    out["_cost_usd"] = res.get("cost_usd")
    out["_latency_ms"] = res.get("latency_ms")
    return out


def resolve(cur, proposal: dict) -> dict:
    """Deterministic resolution of what the model extracted. Exact matches only."""
    species = place_code = None
    candidates: list[str] = []
    name = _norm(proposal.get("species") or "")
    if name:
        cur.execute("SELECT species FROM mpcr.fichas WHERE lower(species) = %s", (name,))
        row = cur.fetchone()
        species = row[0] if row else None
    common = _norm(proposal.get("common_name") or "")
    if species is None and common:
        # A vernacular name denotes one species or several ("cortez amarillo" is three).
        # One candidate resolves; several are returned as they are, because choosing
        # among them is the user's decision and not the router's.
        cur.execute("""SELECT DISTINCT v.species FROM mpcr.vernacular v
                       JOIN mpcr.fichas f ON f.species = v.species
                       WHERE v.name_norm = %s ORDER BY 1""", (common,))
        candidates = [r[0] for r in cur.fetchall()]
        if len(candidates) == 1:
            species = candidates[0]
    place = _norm(proposal.get("place") or "")
    if place:
        cur.execute("SELECT code, label FROM mpcr.places")
        hits = [c for c, label in cur.fetchall()
                if _norm(label) and (_norm(label) in place or place in _norm(label))]
        # Two polygons can share a name (the upstream layer labels two as Parque
        # Nacional Tortuguero); an ambiguous place is unresolved, not a coin toss.
        place_code = hits[0] if len(hits) == 1 else None
    return {"route": proposal.get("route"), "species": species, "place_code": place_code,
            "candidates": candidates or ([species] if species else []),
            "common_name": proposal.get("common_name")}


# Route -> the evidence recipe that serves it. The exits have none.
ROUTE_QTYPE = {"elevation": "elevation", "vertiente": "vertiente", "flowering": "flowering",
               "growth_form": "growth_form", "endemism": "endemism",
               "cultivation": "cultivation", "co_occurrence": "co_occurrence",
               "uses": "outside_record", "light": "light",
               # The three park types share one recipe; which verdict follows is for the
               # evidence to decide, so any of the three names the same blocks.
               "park": "park_possible"}


def recipe(r: dict) -> tuple[str, str, str | None] | None:
    """(species, pack question type, place code) when the route can be served, else
    None. A park route with no resolved place cannot be served: the recipe needs it."""
    qtype = ROUTE_QTYPE.get(r.get("route"))
    if not qtype or not r.get("species"):
        return None
    if r["route"] == "park" and not r.get("place_code"):
        return None
    return r["species"], qtype, r.get("place_code")


def route(cur, question: str, spec: str = ROUTER_MODEL) -> dict:
    p = propose(question, spec)
    return {**resolve(cur, p), "proposal": p}


# ------------------------------------------------------------------------- evaluation

def gold(part: str) -> list[dict]:
    out = []
    for line in io.open(DATA / f"questions_{part}.jsonl", encoding="utf-8"):
        if line.strip():
            q = json.loads(line)
            out.append({"qid": q["qid"], "question": q["question"], "qtype": q["qtype"],
                        "route": QTYPE_ROUTE.get(q["qtype"], q["qtype"]),
                        "species": q["species"],
                        "place_code": (q.get("key") or {}).get("place_code")})
    return out


def hard() -> list[dict]:
    """The hard set (benchmark/routing_set.py). Gold species is a LIST: empty when the
    species must come back unresolved, several when a common name is ambiguous."""
    out = []
    for line in io.open(DATA / "routing_hard.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out.append({"qid": r["id"], "question": r["question"], "qtype": r["category"],
                        "route": r["gold"]["route"], "species": r["gold"]["species"],
                        "place_code": r["gold"]["place_code"],
                        "reviewed": r.get("reviewed", False)})
    return out


def evaluate(name: str, per_type: int, spec: str) -> None:
    from mpcr_rag.store import pg_store
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for g in (hard() if name == "hard" else gold(name)):
        if isinstance(g["species"], str):
            g["species"] = [g["species"]]
        by_type[g["qtype"]].append(g)
    items = [g for qt in sorted(by_type) for g in by_type[qt][:per_type or None]]

    conn = pg_store.connect()
    rows, cost = [], 0.0
    with conn.cursor() as cur:
        for g in items:
            r = route(cur, g["question"], spec)
            cost += r["proposal"].get("_cost_usd") or 0
            # The exits carry no species, so there is nothing to resolve there and a
            # name the model happened to extract is not an error.
            exits = g["route"] in ("catalog_search", "out_of_scope")
            rows.append({**g, "got_route": r["route"], "got_species": r["candidates"],
                         "got_place": r["place_code"],
                         "route_ok": r["route"] == g["route"],
                         "species_ok": exits or sorted(r["candidates"]) == sorted(g["species"]),
                         "place_ok": (g["place_code"] is None
                                      or r["place_code"] == g["place_code"])})
    conn.close()

    out = DATA / f"routing_{name}_result.jsonl"
    with io.open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    unreviewed = sum(1 for r in rows if r.get("reviewed") is False)
    print(f"{n} questions, router={spec}, cost ${cost:.4f}"
          + (f"  [{unreviewed} NOT YET REVIEWED -- draft numbers]" if unreviewed else ""))
    print(f"  {'category':18s} {'n':>4s} {'route':>6s} {'species':>8s} {'place':>6s} {'all':>5s}")
    for qt in sorted(by_type):
        sub = [r for r in rows if r["qtype"] == qt]
        if not sub:
            continue
        print(f"  {qt:18s} {len(sub):4d} {sum(r['route_ok'] for r in sub):6d} "
              f"{sum(r['species_ok'] for r in sub):8d} {sum(r['place_ok'] for r in sub):6d} "
              f"{sum(r['route_ok'] and r['species_ok'] and r['place_ok'] for r in sub):5d}")
    full = sum(r["route_ok"] and r["species_ok"] and r["place_ok"] for r in rows)
    print(f"  {'TOTAL':18s} {n:4d} {sum(r['route_ok'] for r in rows):6d} "
          f"{sum(r['species_ok'] for r in rows):8d} {sum(r['place_ok'] for r in rows):6d} "
          f"{full:5d}")
    for r in rows:
        if not (r["route_ok"] and r["species_ok"] and r["place_ok"]):
            print(f"  MISS {r['qid']} [{r['qtype']}] {r['question'][:70]!r}: route "
                  f"{r['got_route']} (want {r['route']}), species {r['got_species']} "
                  f"(want {r['species']}), place {r['got_place']} (want {r['place_code']})")
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Route a question, or score the router")
    ap.add_argument("--ask", default="")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--part", default="pilot", choices=["pilot", "main", "hard"],
                    help="pilot/main: the templated benchmark questions; hard: the "
                         "paraphrase, common-name, list, absent and out-of-scope set")
    ap.add_argument("--per-type", type=int, default=3,
                    help="questions per benchmark type; 0 for all")
    ap.add_argument("--model", default=ROUTER_MODEL)
    args = ap.parse_args()
    if args.eval:
        evaluate(args.part, args.per_type, args.model)
    elif args.ask:
        from mpcr_rag.store import pg_store
        conn = pg_store.connect()
        with conn.cursor() as cur:
            print(json.dumps(route(cur, args.ask, args.model), ensure_ascii=False, indent=1))
        conn.close()


if __name__ == "__main__":
    main()
