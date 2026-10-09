"""Everyday questions with structured answer keys, generated per sampled species.

Each question carries the key an answer is checked against, where the key comes from,
and its evidence level (docs/EXTENSIONS.md): ``stated`` (the Manual says it),
``inferred_data`` (occurrences, elevation, regions), ``pending_expert`` (no record
source; experts label it), ``none`` (the record is silent — abstaining is correct).

Park questions come in three kinds so a yes/no key is never trivially "yes":
  confirmed  clean snapshot records inside the park (since 1970)
  possible   the park overlaps a Manual region AND the species' elevation band, no records
  unlikely   no Manual region overlap, no elevation overlap, no records
Parks matching none of these cleanly (e.g. region overlap without elevation overlap)
are not used: an ambiguous key would score noise.

Run:  python -m benchmark.questions [--part pilot|main]
Writes benchmark/data/questions_<part>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from mpcr_rag.ingest.field_extractor import _FAMILY_HABIT, _match_habits
from mpcr_rag.schema import Ficha
from mpcr_rag.store import pg_store

# Every key states what it is: the answer the scientific record supports. The Manual and
# the snapshot are incomplete for some species (a cosmopolitan weed collected once in
# Costa Rica), so a key can be right about the record and wrong about the field.
KEY_SCOPE = "record"   # Manual de Plantas de Costa Rica + GBIF snapshot, not field truth

DATA = Path(__file__).resolve().parent / "data"
SEED = 20260913

MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "setiembre", "octubre", "noviembre", "diciembre"]

# Cultivation scenarios: an everyday setting with its elevation. The key is inferred
# from the Manual elevation range and slope, so it is labelled inferred_data.
SETTINGS = [
    ("en Puntarenas, cerca del mar", 10, "Pacífico"),
    ("en Limón, en la costa Caribe", 20, "Caribe"),
    ("en Liberia, Guanacaste", 150, "Pacífico"),
    ("en San Carlos, en la zona norte", 300, "Caribe"),
    ("en Pérez Zeledón", 700, "Pacífico"),
    ("en Turrialba", 650, "Caribe"),
    ("en San José, en el Valle Central", 1150, "Pacífico"),
    ("en Monteverde", 1400, "Pacífico"),
    ("en Cartago", 1450, "Caribe"),
    ("en San Gerardo de Dota", 2200, "Pacífico"),
    ("en las faldas del Volcán Irazú", 2800, "Caribe"),
]

# Countries/regions in the Manual's range statement ("S Méx.–Pan.", "Pantrop.").
_RANGE_BEYOND_CR = re.compile(r"\b(Méx|Guat|Bel|Hond|Salv|Nic|Pan|Col|Ecua|Perú|Bol|Ven|Bras|Guayana|"
                              r"Antillas|EUA|Can\.|Amér|pantrop|Pantrop|cosmopolita|Viejo Mundo|"
                              r"Nativa|nativa|introd)", re.U)

# SINAC names are stored without accents; everyday phrasing should have them.
_ACCENTS = {"Volcan": "Volcán", "Poas": "Poás", "Irazu": "Irazú", "Chirripo": "Chirripó",
            "Rincon": "Rincón", "Barbara": "Bárbara", "Jose": "José", "Tapanti": "Tapantí",
            "Macizo de la Muerte": "Macizo de la Muerte", "Arenal Monteverde": "Arenal-Monteverde",
            "Juan Castro Blanco": "Juan Castro Blanco", "Carara": "Carara", "Cahuita": "Cahuita",
            "Piedras Blancas": "Piedras Blancas", "Guayabo": "Guayabo", "Diria": "Diriá",
            "Baulas": "Las Baulas", "Hitoy": "Hitoy", "Cerere": "Cerere", "Lomas de Barbudal": "Lomas de Barbudal"}


def _accented(label: str) -> str:
    return " ".join(_ACCENTS.get(w, w) for w in label.split(" "))


_USE_CUE = re.compile(r"\b(comestible|medicinal|madera|cultivad|cult\.|ornamental|venenos|t[óo]xic|"
                      r"frutos? comestibles|se usa|usad[ao]s?)\b", re.I)

# The question asks about EDIBILITY or MEDICINE, so its key must turn on those and not
# on any use whatever. Keyed off _USE_CUE, eight of the nine "the Manual states a use"
# cases were nothing of the kind: three ornamental, two madera, three a bare "se usa".
# Dalbergia retusa was keyed as stating a use because its entry mentions rosewood
# timber, so a system that correctly declined on edibility would score understated.
#
# _USE_CUE still measures "any use mentioned" for the catalogue-wide answerability
# figure, which is a different and also useful number: 249 species (3.6%) there against
# 133 (1.9%) here. The paper reports 1.9% for this question.
_EDIBLE_MED_CUE = re.compile(
    r"\b(comestibles?|medicinal(?:es)?|se come|comest\.|infusi[óo]n|remedio|"
    r"curativ[ao]s?|frutos? comestibles)\b", re.I)


def _ficha(cur, species: str) -> Ficha:
    cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (species,))
    return Ficha(**cur.fetchone()[0])


def _eff(f: Ficha):
    if f.elev_min is None or f.elev_max is None:
        return None
    lo = f.elev_outlier_min if f.elev_outlier_min is not None else f.elev_min
    hi = f.elev_outlier_max if f.elev_outlier_max is not None else f.elev_max
    return lo, hi


def _park_keys(cur, f: Ficha, taxon_key, rng) -> list[dict]:
    """Pick up to one confirmed, one possible and one unlikely park."""
    cur.execute("""SELECT code, label, category, elev_p05, elev_p95, regions
                   FROM mpcr.places WHERE category IN ('PN','RB') AND elev_p05 IS NOT NULL
                     AND name !~* 'marin|isla'      -- mostly-marine parks: odd plant questions
                   ORDER BY code""")
    parks = [dict(zip(["code", "label", "category", "lo", "hi", "regions"], r)) for r in cur.fetchall()]
    recorded = {}
    if taxon_key:
        cur.execute("SELECT place_code, n_since_1970, n_records FROM mpcr.species_places "
                    "WHERE species_key = %s", (taxon_key,))
        recorded = {c: (n70, n) for c, n70, n in cur.fetchall()}

    eff = _eff(f)
    regions = set(f.regions)
    confirmed, possible, unlikely = [], [], []
    for p in parks:
        n70, n_all = recorded.get(p["code"], (0, 0))
        region_overlap = bool(regions & set(p["regions"] or []))
        elev_overlap = eff is not None and p["lo"] <= eff[1] and p["hi"] >= eff[0]
        if n70 >= 1:
            confirmed.append((p, n70))
        elif n_all == 0 and region_overlap and elev_overlap:
            possible.append((p, 0))
        elif n_all == 0 and regions and eff is not None and not region_overlap and not elev_overlap:
            unlikely.append((p, 0))

    out = []
    for kind, pool, value in (("confirmed", confirmed, "sí"), ("possible", possible, "posible"),
                              ("unlikely", unlikely, "improbable")):
        if not pool:
            continue
        p, n = rng.choice(pool)
        out.append({
            "qtype": f"park_{kind}",
            "question": f"Si visito el {_accented(p['label'])}, ¿puedo encontrarme con {f.species}?",
            "key": {"answer": value, "place_code": p["code"], "records_in_place": n,
                    "place_elevation_p05_p95": [round(p["lo"]), round(p["hi"])],
                    "place_regions": p["regions"], "species_regions": f.regions,
                    "species_elevation_effective": list(eff) if eff else None},
            "key_level": "inferred_data" if kind != "confirmed" else "stated_by_records",
            "key_source": ["GBIF snapshot 10.15468/dl.8yhee8", "MPCR regions + elevation",
                           "SINAC protected areas", "DEM"],
        })
    return out


def for_species(cur, item: dict, rng) -> list[dict]:
    f = _ficha(cur, item["species"])
    sp = f.species
    cite = f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages}"
    qs = _park_keys(cur, f, item.get("taxon_key"), rng)

    if f.elev_min is not None:
        qs.append({"qtype": "elevation",
                   "question": f"¿A qué altura (en metros sobre el nivel del mar) crece {sp} en Costa Rica?",
                   "key": {"min": f.elev_min, "max": f.elev_max,
                           "outlier_min": f.elev_outlier_min, "outlier_max": f.elev_outlier_max},
                   "key_level": "stated", "key_source": [cite]})
    if f.vertientes:
        qs.append({"qtype": "vertiente",
                   "question": f"¿{sp} crece en la vertiente Caribe, en la Pacífica o en ambas?",
                   "key": {"vertientes": f.vertientes}, "key_level": "stated", "key_source": [cite]})
    if f.flowering_months:
        qs.append({"qtype": "flowering",
                   "question": f"¿En qué meses florece {sp} en Costa Rica?",
                   "key": {"months": f.flowering_months,
                           "months_es": [MONTHS[m - 1] for m in f.flowering_months]},
                   "key_level": "stated", "key_source": [cite]})
    # Growth form: the species' own description, else the genus description only when
    # it names a single form. Genus descriptions often list every form in the genus
    # ("a veces subescandentes o epífitas"), which made Stellaria media a vine/epiphyte.
    own = _match_habits((f.morphology or "")[:160])
    if _FAMILY_HABIT.get(f.family) and _FAMILY_HABIT[f.family] not in own:
        own = [_FAMILY_HABIT[f.family]] + own
    forms = [h for h in own if h != "terrestre"]
    level = "stated"
    if not forms:
        genus_forms = [h for h in _match_habits((f.genus_description or "")[:160]) if h != "terrestre"]
        forms, level = (genus_forms, "inferred_relatives") if len(genus_forms) == 1 else ([], None)
    if forms:
        qs.append({"qtype": "growth_form",
                   "question": f"¿{sp} es un árbol, un arbusto, una hierba, una epífita o un bejuco?",
                   "key": {"habits": forms,
                           "basis": "species description" if level == "stated"
                                    else "genus description (single growth form)"},
                   "key_level": level, "key_source": [cite]})
    tail = f.distribution_paragraph.split(";", 1)[-1]
    range_stated = bool(_RANGE_BEYOND_CR.search(tail))
    if f.endemic_cr or range_stated:
        qs.append({"qtype": "endemism",
                   "question": f"¿{sp} es endémica de Costa Rica?",
                   "key": {"endemic": f.endemic_cr,
                           "basis": "ENDÉMICA stated" if f.endemic_cr else "range beyond Costa Rica stated"},
                   "key_level": "stated", "key_source": [cite]})

    eff = _eff(f)
    if eff is not None:
        place, elev, slope = rng.choice(SETTINGS)
        in_core = f.elev_min <= elev <= f.elev_max
        in_eff = eff[0] <= elev <= eff[1]
        slope_ok = not f.vertientes or slope in f.vertientes
        verdict = ("probable" if in_core and slope_ok else
                   "incierto" if in_eff or (in_core and not slope_ok) else "improbable")
        qs.append({"qtype": "cultivation",
                   "question": f"Tengo una finca {place} (unos {elev} m de altitud). "
                               f"¿Podría sembrar {sp} ahí?",
                   "key": {"answer": verdict,
                           "basis": "natural elevation range and slope from the Manual; growing "
                                    "outside the natural range under cultivation is NOT tested — "
                                    "an answer saying so is not wrong",
                           "setting_elevation": elev, "setting_slope": slope,
                           "species_elevation": [f.elev_min, f.elev_max],
                           "species_elevation_effective": list(eff), "species_vertientes": f.vertientes},
                   "key_level": "inferred_data", "key_source": [cite, "elevation/slope rule"]})

    qs.append({"qtype": "light",
               "question": f"¿{sp} prefiere crecer a pleno sol o en sombra?",
               "key": {"answer": None,
                       "manual_habitat_text": f.distribution_paragraph.split(";")[0]},
               "key_level": "pending_expert", "key_source": ["expert labels (to collect)"]})

    use_text = " ".join(m.group(0) for m in
                        _EDIBLE_MED_CUE.finditer(f"{f.discussion} {f.distribution_paragraph}"))
    qs.append({"qtype": "outside_record",
               "question": f"¿{sp} es comestible o tiene algún uso medicinal?",
               "key": {"record_mentions_use": bool(use_text), "cues": use_text or None,
                       "expected": "answer only what the Manual states" if use_text
                                   else "abstain: the record is silent"},
               "key_level": "stated" if use_text else "none", "key_source": [cite]})

    for q in qs:
        q.update({"qid": f"{sp.replace(' ', '_')}__{q['qtype']}", "species": sp,
                  "stratum": item["stratum"], "n_records": item["n_records"],
                  "n_global": item.get("n_global"), "key_scope": KEY_SCOPE,
                  "key_review_priority": _review_priority(q, item)})
    return qs


def _review_priority(q: dict, item: dict) -> str:
    """Which keys experts should check first. Highest risk: absence-type keys
    ("improbable", out of range) for species well known worldwide but sparsely
    recorded here — the record is most likely incomplete exactly there."""
    absence_key = (q["qtype"] in ("park_unlikely", "cultivation")
                   or (q["qtype"] == "park_possible"))
    stratum = item.get("stratum", "")
    if absence_key and stratum.startswith("G3") and stratum.endswith("L1"):
        return "high"
    if q["key_level"] in ("inferred_data", "inferred_relatives"):
        return "normal"
    return "low"


def build(part: str) -> Path:
    sample = json.loads((DATA / "species_sample.json").read_text(encoding="utf-8"))
    rng = random.Random(f"{SEED}-{part}")
    conn = pg_store.connect()
    out = DATA / f"questions_{part}.jsonl"
    n = 0
    with conn.cursor() as cur, out.open("w", encoding="utf-8") as fh:
        for item in sample[part]:
            for q in for_species(cur, item, rng):
                fh.write(json.dumps(q, ensure_ascii=False) + "\n")
                n += 1
    conn.close()
    print(f"{n} questions for {len(sample[part])} species -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="pilot", choices=["pilot", "main"])
    build(ap.parse_args().part)
