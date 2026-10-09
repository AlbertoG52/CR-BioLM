"""Check an answer against the scientific record, at two levels.

1. DIRECT ANSWER vs the question's key (deterministic once the answer is parsed):
     correct · partially_correct · overstated (more certainty than the record
     supports) · understated · wrong · abstained · not_scored (no key yet, e.g. light)
     · beyond_record (asserts something the record is silent on; experts decide)

2. EVERY FACTUAL CLAIM in the answer vs the species' full record (Manual entry +
   snapshot occurrences), not only the question's key:
     supported      the record states it (or occurrence data shows it)
     contradicted   the record states otherwise (e.g. an elevation outside the range,
                    a park with no records whose elevation band cannot hold the species)
     not_in_record  checkable, but the record does not include it (a region the Manual
                    does not list and no record falls in). Absence in the record is NOT
                    proof of absence in the field, so this is kept apart from contradicted.
     unverifiable   no source in the record covers this kind of claim (uses, light,
                    fine morphology for now). True knowledge the Manual lacks lands here.

The answer text is split into claims by a language model (local by default, so no API
cost); every number, month, slope and growth form is then parsed from the claim TEXT
by the deterministic parsers below, not taken from the model's output.

Run:  python -m benchmark.verify --calibrate      (checks the checker, see CALIBRATION)
"""
from __future__ import annotations

import argparse
import io
import json
import re
import contextlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import requests

from llm import providers
from mpcr_rag.schema import Ficha
from mpcr_rag.store import pg_store

EXTRACTOR = "ollama:qwen3-vl:8b-instruct"
DATA = Path(__file__).resolve().parent / "data"
ELEV_TOL_M = 200

# ----------------------------------------------------------------------------- parsers

_MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
           "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11,
           "diciembre": 12}
_HABIT_WORDS = [(r"\b[áa]rbol|arbolito", "árbol"), (r"\barbust", "arbusto"),
                (r"\bhierba|herb[áa]ce", "hierba"), (r"hemiep[íi]fit|ep[íi]fit", "epífita"),
                (r"\bbejuco|liana|trepador|enredadera", "bejuco"), (r"\bpalma\b|palmera", "palma")]


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def _num(s: str) -> int:
    s = s.replace(" ", "")
    if re.fullmatch(r"\d{1,2}[.,]\d{3}", s):          # 1.500 / 1,500 -> 1500
        s = re.sub(r"[.,]", "", s)
    return int(float(s.replace(",", ".")))


_N = r"(\d{1,2}[.,]\d{3}|\d{1,4})"


def parse_elevations(text: str) -> list[tuple[int, int]]:
    """Elevation intervals in metres: "entre 900 y 1.500 m", "0–700 m", "hasta 2000 m",
    "por encima de 3000 m", "a unos 1200 m"."""
    t = text.replace("m.s.n.m.", "m").replace("msnm", "m")
    t = re.sub(r"\bnivel del mar\b", "0 m", t)
    t = re.sub(r"\b(?:los|unos|aproximadamente|aprox\.|cerca de|alrededor de)\s+", "", t)
    out = []
    for m in re.finditer(rf"(?:entre|de|desde)?\s*{_N}\s*(?:m\s*)?(?:–|-|a|y|hasta)\s*{_N}\s*(?:m\b|metros)", t):
        lo, hi = sorted((_num(m.group(1)), _num(m.group(2))))
        out.append((lo, hi))
    if out:
        return out
    for m in re.finditer(rf"(hasta|por debajo de|menos de|bajo)\s+{_N}\s*(?:m\b|metros)", t):
        out.append((0, _num(m.group(2))))
    for m in re.finditer(rf"(por encima de|sobre|más de|arriba de|mayores? a)\s+{_N}\s*(?:m\b|metros)", t):
        out.append((_num(m.group(2)), 4000))
    if out:
        return out
    for m in re.finditer(rf"{_N}\s*(?:m\b|metros)", t):
        v = _num(m.group(1))
        if 0 <= v <= 4000:
            out.append((v, v))
    return out


def parse_months(text: str) -> set[int]:
    t = _norm(text)
    if re.search(r"todo el ano|durante todo|a lo largo del ano|todos los meses", t):
        return set(range(1, 13))
    out: set[int] = set()
    names = "|".join(_norm(k) for k in _MONTHS)
    for a, b in re.findall(rf"({names})\s*(?:a|hasta|-|–)\s*({names})", t):
        i, j = _month(a), _month(b)
        out.update(range(i, j + 1) if i <= j else [*range(i, 13), *range(1, j + 1)])
    for n in re.findall(rf"\b({names})\b", t):
        out.add(_month(n))
    return out


def _month(name: str) -> int:
    return {_norm(k): v for k, v in _MONTHS.items()}[name]


def parse_slopes(text: str) -> set[str]:
    t = _norm(text)
    if re.search(r"ambas vertientes|ambas|las dos vertientes|ambos lados", t):
        return {"Caribe", "Pacífico"}
    out = set()
    if re.search(r"caribe|atlantic", t):
        out.add("Caribe")
    if re.search(r"pacific", t):
        out.add("Pacífico")
    return out


def parse_habits(text: str) -> set[str]:
    return {label for pat, label in _HABIT_WORDS if re.search(pat, text, re.I)}


# -------------------------------------------------------------------------- extraction

_PROMPT = """Analiza la RESPUESTA a una PREGUNTA sobre una planta. Devuelve SOLO JSON:
{{
  "abstains": true|false,
  "stance": "yes"|"possible"|"no"|"uncertain"|"none",
  "claims": [{{"text": "<una afirmación factual, copiada o parafraseada breve>",
               "type": "elevation"|"place"|"slope"|"phenology"|"growth_form"|"endemism"|"use"|"light"|"morphology"|"other"}}]
}}
Reglas:
- "abstains": true si la respuesta dice que no tiene información suficiente para responder la pregunta.
- "stance": la postura sobre la pregunta. "yes" = afirma que sí; "no" = afirma que no o que es improbable;
  "possible" = dice que es posible o probable pero no seguro; "uncertain" = no sabe o no puede confirmar;
  "none" = la pregunta no es de sí/no.
- "claims": cada afirmación factual SOBRE LA PLANTA por separado (altitud, lugares, vertiente, floración,
  forma de crecimiento, endemismo, usos, luz, morfología). No incluyas opiniones, consejos ni frases de cortesía.
  Para lugares, una afirmación por lugar.

PREGUNTA: {question}
RESPUESTA: {answer}
"""


def extract(question: str, answer: str, spec: str = EXTRACTOR) -> dict:
    prov = providers.resolve(spec)
    body = {"model": prov.model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": _PROMPT.format(question=question, answer=answer)}]}
    r = requests.post(prov.url, headers=prov.headers(), json=body, timeout=180 if prov.is_local else 60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M)
    parsed = json.loads(content)
    parsed.setdefault("claims", [])
    return parsed


# ------------------------------------------------------------------------------ record

@dataclass
class Record:
    ficha: Ficha
    gbif_regions: set[str] = field(default_factory=set)
    parks_with_records: dict[str, int] = field(default_factory=dict)   # normalized name -> n
    parks: dict[str, dict] = field(default_factory=dict)               # normalized name -> row

    @property
    def eff(self):
        f = self.ficha
        if f.elev_min is None:
            return None
        return (f.elev_outlier_min if f.elev_outlier_min is not None else f.elev_min,
                f.elev_outlier_max if f.elev_outlier_max is not None else f.elev_max)


def load_record(cur, species: str) -> Record:
    cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (species,))
    rec = Record(Ficha(**cur.fetchone()[0]))
    cur.execute("SELECT regions, taxon_key FROM mpcr.species_evidence WHERE species = %s", (species,))
    row = cur.fetchone()
    if row:
        rec.gbif_regions = set(row[0] or [])
        cur.execute("""SELECT p.name, s.n_records FROM mpcr.species_places s
                       JOIN mpcr.places p ON p.code = s.place_code WHERE s.species_key = %s""", (row[1],))
        rec.parks_with_records = {_norm(n): k for n, k in cur.fetchall()}
    cur.execute("SELECT name, label, elev_p05, elev_p95, regions FROM mpcr.places")
    rec.parks = {_norm(n): {"label": l, "lo": lo, "hi": hi, "regions": rg}
                 for n, l, lo, hi, rg in cur.fetchall()}
    return rec


def _regions_named(text: str) -> set[str]:
    """Canonical botanical regions named in free text, via the map pipeline's gazetteer."""
    from utils.distribution_map.parser import build_ficha
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_ficha(habitat_raw="", geographic_notes=text, species="")
    return {r.canonical_name for r in df.regions}


# ------------------------------------------------------------------------ claim checks

# A claim whose subject is the asked-about SITE, not the species. The cultivation
# questions state the farm's elevation in the prompt ("una finca ... unos 1400 m"), and
# a good grounded answer restates it. Checked naively, that elevation is compared with
# the species' range and scored "contradicted" -- penalising exactly the behaviour we
# want. Such a restatement is not a claim about the record, so it is not checkable:
# this is the "is this even a checkable statement" gate that AIS puts before support.
_SITE_SUBJECT = re.compile(
    r"\b(el|ese|este|dicho)\s+(sitio|lugar|punto|terreno)\b"
    r"|\bla\s+(finca|propiedad|zona|ubicaci[óo]n|localidad)\s+(indicada|se[ñn]alada|mencionada)?\b"
    r"|\bel\s+[áa]rea\s+indicada\b", re.I)


def check_claim(claim: dict, rec: Record) -> str:
    text, ctype = claim.get("text", ""), claim.get("type", "other")
    f = rec.ficha
    if ctype == "elevation" and _SITE_SUBJECT.search(text):
        return "unverifiable"
    # Route by content, not only by the model's label: "Crece en la vertiente Caribe"
    # typed as a place is a slope claim.
    if ctype in ("place", "other") and parse_slopes(text) and not _regions_named(text):
        ctype = "slope"
    if ctype == "elevation":
        ivs, eff = parse_elevations(text), rec.eff
        if not ivs or eff is None:
            return "unverifiable"
        lo, hi = min(a for a, _ in ivs), max(b for _, b in ivs)
        if hi < eff[0] - ELEV_TOL_M or lo > eff[1] + ELEV_TOL_M:
            return "contradicted"
        return "supported" if lo >= eff[0] - ELEV_TOL_M and hi <= eff[1] + ELEV_TOL_M else "contradicted"
    if ctype == "slope":
        s = parse_slopes(text)
        if not s or not f.vertientes:
            return "unverifiable"
        return "supported" if s <= set(f.vertientes) else "contradicted"
    if ctype == "phenology":
        m = parse_months(text)
        if not m or not f.flowering_months:
            return "not_in_record" if m else "unverifiable"
        overlap = len(m & set(f.flowering_months)) / len(m)
        return "supported" if overlap >= 0.75 else ("contradicted" if overlap < 0.25 else "not_in_record")
    if ctype == "growth_form":
        h = parse_habits(text)
        known = parse_habits(" ".join(f.habits))
        if not h or not known:
            return "unverifiable"
        return "supported" if h & known else "contradicted"
    if ctype == "endemism":
        t = _norm(text)
        says_endemic = "endemic" in t and not re.search(r"\bno\b[^.]{0,20}endemic", t)
        return "supported" if says_endemic == f.endemic_cr else "contradicted"
    if ctype == "place":
        t = _norm(text)
        for name, p in rec.parks.items():
            if name and len(name) > 4 and name in t:
                if rec.parks_with_records.get(name):
                    return "supported"
                eff = rec.eff
                if eff and p["lo"] is not None and (p["hi"] < eff[0] - ELEV_TOL_M or p["lo"] > eff[1] + ELEV_TOL_M):
                    return "contradicted"
                return "not_in_record"
        named = _regions_named(text)
        if not named:
            return "unverifiable"
        return "supported" if named & (set(f.regions) | rec.gbif_regions) else "not_in_record"
    return "unverifiable"          # use, light, morphology, other


# ------------------------------------------------------------------------ direct check

_STANCE_TABLE = {   # key answer -> {stance: verdict}
    "sí":        {"yes": "correct", "possible": "understated", "uncertain": "understated", "no": "wrong"},
    "posible":   {"possible": "correct", "uncertain": "correct", "yes": "overstated", "no": "understated"},
    "improbable": {"no": "correct", "uncertain": "partially_correct", "possible": "overstated", "yes": "wrong"},
    "probable":  {"yes": "correct", "possible": "correct", "uncertain": "understated", "no": "wrong"},
    "incierto":  {"possible": "correct", "uncertain": "correct", "yes": "overstated", "no": "overstated"},
}


def check_direct(q: dict, answer: str, ext: dict) -> str:
    key, qtype = q["key"], q["qtype"]
    if qtype == "light":
        return "not_scored"
    if qtype == "co_occurrence":
        # Scored on precision over what the answer volunteers, never recall: about a
        # third of the flora could co-occur with any given species, so naming a
        # neighbour is easy and only an impossible one is informative. Species outside
        # the ingested volumes are not held against the answer (cooccurrence.py).
        from .cooccurrence import check as co_check, extract_species, verdict as co_verdict
        return co_verdict(co_check(q["species"], extract_species(answer)))
    if qtype == "outside_record":
        claims_use = any(c.get("type") == "use" for c in ext["claims"])
        if not key["record_mentions_use"]:
            # The record is silent: an answer ASSERTING a use may be invented or may be
            # true knowledge the Manual lacks (napier grass as forage). The checker
            # cannot tell which, so it is routed to experts, not scored as wrong.
            #
            # But a DENIAL is not an assertion. "No se menciona ningún uso medicinal"
            # is the correct, grounded answer, and it produces a use-typed claim just
            # like an invention does. Counting those as assertions put every grounded
            # arm at an 80% invention rate in the pilot while they were in fact
            # reporting the record's silence exactly as intended -- a finding that was
            # entirely an artefact of this line. The stance distinguishes them.
            denies = ext.get("stance") == "no"
            return ("correct" if ext.get("abstains") or denies or not claims_use
                    else "beyond_record")
        return "understated" if ext.get("abstains") else "correct"
    if ext.get("abstains") and ext.get("stance") in ("uncertain", "none", None):
        return "abstained"
    if qtype.startswith("park_") or qtype == "cultivation":
        return _STANCE_TABLE[key["answer"]].get(ext.get("stance"), "abstained")
    if qtype == "elevation":
        ivs = parse_elevations(answer)
        if not ivs:
            return "abstained"
        lo, hi = min(a for a, _ in ivs), max(b for _, b in ivs)
        k_lo = key["outlier_min"] if key["outlier_min"] is not None else key["min"]
        k_hi = key["outlier_max"] if key["outlier_max"] is not None else key["max"]
        if hi < k_lo - ELEV_TOL_M or lo > k_hi + ELEV_TOL_M:
            return "wrong"
        within = lo >= k_lo - ELEV_TOL_M and hi <= k_hi + ELEV_TOL_M
        covers_core = lo <= key["min"] + ELEV_TOL_M and hi >= key["max"] - ELEV_TOL_M
        return "correct" if within and covers_core else "partially_correct"
    if qtype == "vertiente":
        s, k = parse_slopes(answer), set(key["vertientes"])
        return "abstained" if not s else ("correct" if s == k else ("partially_correct" if s & k else "wrong"))
    if qtype == "flowering":
        m, k = parse_months(answer), set(key["months"])
        if not m:
            return "abstained"
        precision, recall = len(m & k) / len(m), len(m & k) / len(k)
        return ("correct" if precision >= 0.75 and recall >= 0.5 else
                "wrong" if precision < 0.25 else "partially_correct")
    if qtype == "growth_form":
        h, k = parse_habits(answer), set(key["habits"])
        return "abstained" if not h else ("correct" if h & k and not (h - k) else
                                          "partially_correct" if h & k else "wrong")
    if qtype == "endemism":
        stance = ext.get("stance")
        if stance not in ("yes", "no"):
            return "abstained"
        return "correct" if (stance == "yes") == key["endemic"] else "wrong"
    return "not_scored"


def verify(q: dict, answer: str, cur, spec: str = EXTRACTOR) -> dict:
    ext = extract(q["question"], answer, spec)
    rec = load_record(cur, q["species"])
    claims = [{**c, "verdict": check_claim(c, rec)} for c in ext["claims"]]
    counts = {v: sum(1 for c in claims if c["verdict"] == v)
              for v in ("supported", "contradicted", "not_in_record", "unverifiable")}
    return {"qid": q["qid"], "direct": check_direct(q, answer, ext), "abstains": ext.get("abstains"),
            "stance": ext.get("stance"), "claims": claims, "claim_counts": counts}


# ------------------------------------------------------------------------- calibration

def calibrate(spec: str = EXTRACTOR) -> None:
    """Hand-written answers with the verdicts a careful human would assign. The checker
    must reproduce them before it is trusted on real answers."""
    cases = [json.loads(l) for l in (DATA / "verify_calibration.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = {json.loads(l)["qid"]: json.loads(l) for l in (DATA / "questions_pilot.jsonl").read_text(encoding="utf-8").splitlines()}
    conn = pg_store.connect()
    ok_direct = ok_claims = n_claim_checks = 0
    with conn.cursor() as cur:
        for c in cases:
            res = verify(qs[c["qid"]], c["answer"], cur, spec)
            hit = res["direct"] == c["expected_direct"]
            ok_direct += hit
            claim_note = ""
            if "expected_contradicted" in c:
                n_claim_checks += 1
                got = res["claim_counts"]["contradicted"] > 0
                ok_claims += got == c["expected_contradicted"]
                claim_note = f" | contradicted claim expected={c['expected_contradicted']} got={got}"
            print(f"[{'OK ' if hit else 'XX '}] {c['qid']:42} {c['label']:24} expected={c['expected_direct']:17} "
                  f"got={res['direct']:17}{claim_note}")
            if not hit or claim_note and (res["claim_counts"]["contradicted"] > 0) != c.get("expected_contradicted"):
                for cl in res["claims"]:
                    print(f"        {cl['verdict']:14} [{cl.get('type')}] {cl.get('text')}")
    conn.close()
    print(f"\ndirect verdicts: {ok_direct}/{len(cases)}   contradicted-claim detection: {ok_claims}/{n_claim_checks}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--extractor", default=EXTRACTOR)
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.extractor)
