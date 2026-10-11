"""The arms. A declarative registry plus two implementations.

The design (docs/EXPERIMENT_PLAN.md sec. 2 and 2c) is generator x evidence:

                  no evidence   live web   flora only   full pack
  Mistral Large 4      A1           B           E            D
  Gemma 4 31B          C            -           -            F
  template, no model   -            -           -            G

Seven arms, two generator models. An arm is a row of declarative settings, not a
module: the LLM arms differ only in model spec and evidence level, so expressing that
as data rather than as seven near-identical files removes the chance of one drifting
from another. The prompt itself is shared and its sameness is asserted in run.py.

Arm G is the floor: it answers from the same facts the pack carries, by implementing
each question type's own rule in Spanish. It is deliberately a FAIR floor -- it reads
`pack.facts`, which is built in the same pass as the prose blocks, so it can never be
accused of having been shown less than arms D and F. If G lands close to D, the honest
reading is that the task needs a database more than a language model, and finding F4
exists to say so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .evidence_pack import MONTHS_ES
from .questions import SETTINGS

PAID = "openrouter:mistralai/mistral-large-4-0"
OPEN = "openrouter:google/gemma-4-31b-it"

# Upstream pins. Mistral serves its own model and nothing else does, so the pin is
# trivial there. Gemma is served by 13 upstreams at fp4/fp8/bf16 -- Crusoe is bf16, so
# the open arm runs the published weights rather than a quantization of them.
PIN = {PAID: ("Mistral",), OPEN: ("Crusoe", "Novita")}


@dataclass(frozen=True)
class Arm:
    name: str
    model: str | None          # None = no model (template)
    evidence: str | None       # None | "flora" | "full"
    web: bool = False
    note: str = ""
    prompt: str = "shared"     # prompt family, see prompts.FAMILIES
    routed: bool = False       # True = the router picks species, recipe and place

    @property
    def pin(self) -> tuple[str, ...]:
        return PIN.get(self.model or "", ())

    @property
    def is_llm(self) -> bool:
        return self.model is not None


ARMS = {a.name: a for a in [
    Arm("A1", PAID, None, note="closed book, careful prompt"),
    Arm("A1F", PAID, None, prompt="minimal",
        note="closed book, FREE prompt: what a person actually gets when they just ask"),
    Arm("B",  PAID, None, web=True,
        note="web search; plumbing unverified, excluded from the first run"),
    Arm("C",  OPEN, None, note="closed book, open weights, careful prompt"),
    Arm("CF", OPEN, None, prompt="minimal",
        note="closed book, open weights, FREE prompt"),
    Arm("D",  PAID, "full", note="grounded, the system under test"),
    Arm("E",  PAID, "flora", note="ablation: flora text only"),
    Arm("F",  OPEN, "full", note="grounded, open weights"),
    Arm("DM", PAID, "map", prompt="map",
        note="grounded by MAP ONLY: same model as D, evidence as a picture"),
    Arm("FM", OPEN, "map", prompt="map",
        note="grounded by MAP ONLY, open weights: same model as F"),
    Arm("G",  None, "full", note="template, no model: the floor"),
    Arm("R",  OPEN, "full", routed=True,
        note="AGENT-ROUTED: same generator and evidence as F, but the router reads the "
             "bare question and chooses species, recipe and place. F is its oracle-"
             "routed twin, so F minus R is what routing errors cost"),
]}

# Arms runnable today. B waits on the search mechanism being confirmed.
READY = [n for n in ("G", "A1", "A1F", "C", "CF", "D", "E", "F", "DM", "FM", "R")]

# What the routed arm says when the router cannot serve the question: no species
# resolved, an exit route, or a park it could not identify. A deployed system that
# answered closed-book here would be passing an ungrounded answer off as grounded.
UNROUTED = ("No pude identificar con certeza la especie o el lugar de la pregunta en el "
            "registro consultado, así que no puedo responderla.")

# The map condition only applies where a map can carry the evidence: the asked-about
# area has to be drawable and the answer has to turn on geometry. Elsewhere a picture
# of a distribution says nothing a key tests, and running it would measure nothing.
MAP_QTYPES = ("park_confirmed", "park_possible", "park_unlikely", "vertiente")


# ---------------------------------------------------------------- arm G, the template

def _es_list(items: list[str]) -> str:
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


_ELEV_IN_Q = re.compile(r"unos\s+([\d.,]+)\s*m")


def _setting_from_question(question: str) -> tuple[int, str] | None:
    """The cultivation question states its setting in the prompt. Recovering it is a
    lookup against the same eleven settings the keys were built from, not inference."""
    m = _ELEV_IN_Q.search(question)
    if not m:
        return None
    elev = int(m.group(1).replace(".", "").replace(",", ""))
    for phrase, _e, slope in SETTINGS:
        if phrase.lower() in question.lower():
            return elev, slope
    return elev, ""


_ABSTAIN = ("El registro consultado no contiene información que permita responder "
            "esta pregunta.")


def template_answer(question: str, qtype: str, facts: dict) -> str:
    """Deterministic answer from the structured evidence. No model involved."""
    sp = facts.get("species", "la especie")
    cita = facts.get("cita", "")
    src = f" (fuente: {cita})" if cita else ""

    if qtype == "elevation":
        lo, hi = facts.get("elev_min"), facts.get("elev_max")
        if lo is None:
            return _ABSTAIN
        eff = facts.get("elev_effective")
        extra = ""
        if eff and (eff[0] != lo or eff[1] != hi):
            extra = f" Incluyendo registros atípicos, el rango va de {eff[0]} a {eff[1]} m."
        return f"{sp} crece entre {lo} y {hi} m de altitud{src}.{extra}"

    if qtype == "vertiente":
        v = facts.get("vertientes") or []
        if not v:
            return _ABSTAIN
        if len(v) > 1:
            return f"{sp} crece en ambas vertientes ({_es_list(v)}){src}."
        return f"{sp} crece en la vertiente {v[0]}{src}."

    if qtype == "flowering":
        mo = facts.get("flowering_months") or []
        if not mo:
            return _ABSTAIN
        return f"{sp} florece en {_es_list([MONTHS_ES[m - 1] for m in mo])}{src}."

    if qtype == "growth_form":
        gf = facts.get("growth_form")
        return f"{sp} es {gf}{src}." if gf else _ABSTAIN

    if qtype == "endemism":
        if facts.get("endemic_cr"):
            return f"Sí, {sp} es endémica de Costa Rica{src}."
        beyond = facts.get("range_beyond")
        if beyond:
            return (f"No, {sp} no es endémica de Costa Rica: el registro indica "
                    f"distribución más allá del país ({beyond}){src}.")
        return _ABSTAIN

    if qtype in ("park_confirmed", "park_possible", "park_unlikely"):
        p = facts.get("place")
        if not p:
            return _ABSTAIN
        where = p["label"]
        if p["n_since_1970"] >= 1:
            return (f"Sí. Hay {p['n_since_1970']} registro(s) de {sp} dentro del "
                    f"{where} desde 1970{src}.")
        if p["region_overlap"] and p["elev_overlap"]:
            return (f"Es posible. No hay registros de {sp} dentro del {where}, pero el "
                    f"área coincide con sus regiones botánicas ({_es_list(p['shared_regions'])}) "
                    f"y con su rango altitudinal. El esfuerzo de colecta es desigual, de modo "
                    f"que la ausencia de registros no es evidencia de ausencia{src}.")
        if not p["region_overlap"] and not p["elev_overlap"]:
            return (f"Es improbable. No hay registros de {sp} dentro del {where}, y el área "
                    f"no coincide ni con sus regiones botánicas ni con su rango "
                    f"altitudinal{src}.")
        return (f"Es incierto. No hay registros de {sp} dentro del {where}; el área coincide "
                f"parcialmente con su distribución conocida{src}.")

    if qtype == "cultivation":
        st = _setting_from_question(question)
        lo, hi = facts.get("elev_min"), facts.get("elev_max")
        if st is None or lo is None:
            return _ABSTAIN
        elev, slope = st
        eff = facts.get("elev_effective") or [lo, hi]
        vert = facts.get("vertientes") or []
        in_core = lo <= elev <= hi
        in_eff = eff[0] <= elev <= eff[1]
        slope_ok = (not vert) or (slope in vert) or not slope
        if in_core and slope_ok:
            v = "probable"
        elif in_eff or (in_core and not slope_ok):
            v = "incierto"
        else:
            v = "improbable"
        return (f"Es {v}. El registro ubica a {sp} entre {lo} y {hi} m"
                + (f", en la vertiente {_es_list(vert)}" if vert else "")
                + f", y el sitio indicado está a unos {elev} m. Sembrar fuera del rango "
                  f"natural bajo cultivo no está evaluado aquí{src}.")

    if qtype == "light":
        hab = facts.get("habitat_sentence")
        return (f"{_ABSTAIN} El registro describe su hábitat como: {hab}." if hab
                else _ABSTAIN)

    if qtype == "outside_record":
        cues = facts.get("use_cues")
        if cues:
            return (f"El registro menciona lo siguiente en relación con usos de {sp}: "
                    f"{cues}{src}. No contiene más detalle al respecto.")
        return _ABSTAIN

    return _ABSTAIN
