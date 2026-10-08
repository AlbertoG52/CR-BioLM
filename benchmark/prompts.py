"""One prompt, shared by every model arm. The evidence block is the only thing that
differs between them.

This is where the tier-ablation bug of the previous experiment would recur if we let
it. There, arms differed in more than the thing being ablated, so the ablation measured
the prompt. The defence here is structural rather than careful: there is a single
immutable system message, and an arm either appends an EVIDENCIA section to the user
message or it does not. Nothing else varies.

That makes parity *checkable*: every model arm must report the same
``prompt_template_hash``. If two arms disagree, the run is invalid, and `run.py`
refuses to start.

Note the abstention licence sits in the shared system message, so the closed-book arms
are given exactly the same permission to decline as the grounded ones. Licensing
abstention only where evidence is supplied would make finding F5 -- who invents an
answer when nothing supports one -- an artefact of the prompt.

The clause about evidence is present in both cases and simply has nothing to refer to
when no evidence is supplied. That is deliberate: identical text beats tailored text.
"""
from __future__ import annotations

import hashlib

SYSTEM = (
    "Eres un botánico especialista en la flora de Costa Rica. Responde la pregunta de "
    "la forma más veraz posible, en español, en un párrafo breve.\n"
    "\n"
    "Reglas:\n"
    "1. Si no tienes base suficiente para responder, dilo explícitamente en lugar de "
    "adivinar. Es preferible declarar que no se sabe antes que ofrecer un dato "
    "incierto como si fuera cierto.\n"
    "2. Cuando se te proporcione una sección EVIDENCIA, básate únicamente en ella y "
    "cita la fuente que allí aparece. No agregues datos que la evidencia no contenga.\n"
    "3. No inventes nombres de lugares, altitudes, meses ni usos.\n"
    "4. Responde a la pregunta concreta; no describas la planta en general."
)


def user_message(question: str, evidence: str | None = None) -> str:
    """The user turn. With evidence it carries an EVIDENCIA section; without it, the
    bare question. No other difference between arms."""
    if evidence:
        return f"EVIDENCIA:\n{evidence}\n\nPREGUNTA: {question}"
    return f"PREGUNTA: {question}"


def build(question: str, evidence: str | None = None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(question, evidence)}]


def template_hash() -> str:
    """Identifies the template, NOT the filled-in prompt. Every model arm in a run must
    report this same value; the evidence itself is tracked by the pack hash."""
    shape = SYSTEM + "||" + user_message("<Q>", "<E>") + "||" + user_message("<Q>")
    return hashlib.sha256(shape.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    print(f"template_hash = {template_hash()}\n")
    print("--- with evidence ---")
    for m in build("¿A qué altura crece X?", "altitud (Manual):\n950-1050 m"):
        print(f"[{m['role']}]\n{m['content']}\n")
    print("--- closed book ---")
    for m in build("¿A qué altura crece X?"):
        print(f"[{m['role']}]\n{m['content']}\n")
