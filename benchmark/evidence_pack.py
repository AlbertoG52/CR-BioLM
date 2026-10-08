"""The evidence pack: what the record holds about one species, bearing on one question,
serialized so a model can cite it and a reader can check it.

Two levels. The difference between them is exactly what arm E ablates:

  flora   the Manual entry alone -- what the published treatment says
  full    the Manual entry plus occurrence, protected-area, region and terrain evidence

Three rules this module follows, each for a reason:

1. **No model reasons spatially.** Region overlap, elevation-band overlap and
   point-in-polygon counts are computed here and serialized as facts (docs/
   EXPERIMENT_PLAN.md sec. 2b). For the park question types this means arm D is handed
   the inputs to the key's own rule, so its score there measures *faithful use of
   computed evidence*, not spatial inference. That is the claim the system makes, and
   the paper should say so rather than let it read as the model being clever.

2. **The pack never tells the model what to answer.** Where the record is silent --
   light preference, uses for most species -- the pack simply lacks the fact. It does
   not contain a sentence saying "the record is silent", because the shared prompt
   already licenses abstention for every arm. Putting it in the pack would hand the
   grounded arm the abstention that finding F5 is trying to measure.

3. **The genus description is included only when it names a single growth form**, the
   same rule the keys use. Genus descriptions list every form in the genus, which is
   what once made *Stellaria media* a vine. It is also the largest block by far, so
   dropping it in the usual case keeps the prompt well inside a local model's context.

Run:  python -m benchmark.evidence_pack --species "Heliconia rodriguezii" --qtype park_possible
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from mpcr_rag.ingest.field_extractor import _match_habits
from mpcr_rag.schema import Ficha

from .questions import _RANGE_BEYOND_CR, _USE_CUE
from mpcr_rag.store import pg_store

MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "setiembre", "octubre", "noviembre", "diciembre"]

MORPHOLOGY_CHARS = 900          # enough to characterise the plant, not the whole plate
EFFORT_CAVEAT = ("El esfuerzo de colecta es desigual: la ausencia de registros en un "
                 "área protegida no es evidencia de ausencia de la especie.")


@dataclass
class Pack:
    """Evidence for one (species, question type) at one level.

    ``blocks`` is the prose a model reads. ``facts`` is the same evidence as structured
    values, for the template arm G. They are populated from one pass over one ficha, so
    arm G is provably reading the same evidence as arms D and F rather than a
    convenient subset -- which is what makes G a fair floor instead of a straw man.
    """
    species: str
    qtype: str
    level: str
    blocks: list[tuple[str, str]] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def add(self, heading: str, body: str | None) -> None:
        if body:
            self.blocks.append((heading, str(body).strip()))

    def text(self) -> str:
        return "\n\n".join(f"{h}:\n{b}" for h, b in self.blocks)

    @property
    def hash(self) -> str:
        """Stable over the content, so a re-run is provably the same evidence."""
        payload = json.dumps({"species": self.species, "qtype": self.qtype,
                              "level": self.level, "blocks": self.blocks,
                              "providers": sorted(self.providers)},
                             ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_row(self) -> dict:
        return {"species": self.species, "qtype": self.qtype, "level": self.level,
                "evidence_pack_hash": self.hash, "evidence_providers": self.providers,
                "n_blocks": len(self.blocks), "n_chars": len(self.text())}


# --------------------------------------------------------------------------------------
# loaders


def _ficha(cur, species: str) -> Ficha:
    cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (species,))
    row = cur.fetchone()
    if not row:
        raise KeyError(f"no ficha for {species!r}")
    return Ficha(**row[0])


def _taxon_key(cur, species: str) -> int | None:
    cur.execute("SELECT taxon_key FROM mpcr.taxa WHERE species = %s", (species,))
    row = cur.fetchone()
    return row[0] if row else None


def _occurrence(cur, species: str) -> dict | None:
    cur.execute("""SELECT n_records, n_cells, n_specimens, n_observations, first_year,
                          last_year, elev_dem_p05, elev_dem_p50, elev_dem_p95,
                          regions, vertientes, record_months
                   FROM mpcr.species_evidence WHERE species = %s""", (species,))
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    cols = ["n_records", "n_cells", "n_specimens", "n_observations", "first_year",
            "last_year", "elev_p05", "elev_p50", "elev_p95", "regions", "vertientes",
            "months"]
    return dict(zip(cols, row))


def _place(cur, code: str) -> dict | None:
    cur.execute("""SELECT code, label, category, elev_p05, elev_p95, regions, vertientes
                   FROM mpcr.places WHERE code = %s""", (code,))
    row = cur.fetchone()
    if not row:
        return None
    return dict(zip(["code", "label", "category", "lo", "hi", "regions", "vertientes"], row))


def _records_in_place(cur, taxon_key: int | None, code: str) -> tuple[int, int]:
    if not taxon_key:
        return 0, 0
    cur.execute("""SELECT n_since_1970, n_records FROM mpcr.species_places
                   WHERE species_key = %s AND place_code = %s""", (taxon_key, code))
    row = cur.fetchone()
    return (row[0] or 0, row[1] or 0) if row else (0, 0)


# --------------------------------------------------------------------------------------
# blocks


def _eff(f: Ficha) -> tuple[int, int] | None:
    if f.elev_min is None or f.elev_max is None:
        return None
    lo = f.elev_outlier_min if f.elev_outlier_min is not None else f.elev_min
    hi = f.elev_outlier_max if f.elev_outlier_max is not None else f.elev_max
    return lo, hi



def _range_beyond(f: Ficha) -> str | None:
    """Only the part of the distribution that names territory OUTSIDE Costa Rica.

    The whole tail after the habitat clause still lists Costa Rican regions, so quoting
    it to justify "not endemic" drags in local place names that then read as unsupported
    distribution claims. Return the narrowest segment that actually carries the
    beyond-CR evidence.
    """
    tail = f.distribution_paragraph.split(";", 1)[-1]
    segs = [s.strip() for s in re.split(r"(?<=\.)\s+", tail) if s.strip()]
    hits = [s for s in segs if _RANGE_BEYOND_CR.search(s)]
    return min(hits, key=len) if hits else None


def _cite(f: Ficha) -> str:
    return f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages}"


def _elevation_text(f: Ficha) -> str | None:
    if f.elev_min is None:
        return None
    out = f"{f.elev_min}-{f.elev_max} m"
    eff = _eff(f)
    if eff and (eff[0] != f.elev_min or eff[1] != f.elev_max):
        out += f" (incluyendo registros atípicos: {eff[0]}-{eff[1]} m)"
    return out


def _growth_form_text(f: Ficha) -> str | None:
    """Species description first. The genus only when it names a single form -- the rule
    that keeps a genus listing every habit from mislabelling the species."""
    own = [h for h in _match_habits((f.morphology or "")[:160]) if h != "terrestre"]
    if own:
        return f"{', '.join(own)} (descripción de la especie)"
    genus = [h for h in _match_habits((f.genus_description or "")[:160]) if h != "terrestre"]
    if len(genus) == 1:
        return f"{genus[0]} (descripción del género; nombra una sola forma de crecimiento)"
    return None


def _park_block(f: Ficha, place: dict, n70: int, n_all: int) -> str:
    """The crux. A pack that said only "registros en el parque: 0" would push every arm
    to answer no, when the key says "posible" on region and elevation overlap. So the
    computed overlaps go in, with the collection-effort caveat."""
    eff = _eff(f)
    sp_regions = set(f.regions or [])
    pk_regions = set(place["regions"] or [])
    region_overlap = bool(sp_regions & pk_regions)
    elev_overlap = eff is not None and place["lo"] <= eff[1] and place["hi"] >= eff[0]
    shared = sorted(sp_regions & pk_regions)

    lines = [
        f"área protegida: {place['label']}",
        f"altitud del área (DEM, p05-p95): {round(place['lo'])}-{round(place['hi'])} m",
        f"regiones botánicas del área: {', '.join(place['regions'] or []) or 'sin datos'}",
        f"registros de la especie dentro del área: {n_all} en total, {n70} desde 1970",
        f"altitud de la especie (Manual): {_elevation_text(f) or 'sin datos'}",
        f"regiones botánicas de la especie (Manual): {', '.join(f.regions or []) or 'sin datos'}",
        f"traslape de regiones: {'sí' if region_overlap else 'no'}"
        + (f" ({', '.join(shared)})" if shared else ""),
        f"traslape altitudinal: {'sí' if elev_overlap else 'no'}",
        EFFORT_CAVEAT,
    ]
    return "\n".join(lines)


def _occurrence_block(occ: dict) -> str:
    months = ", ".join(MONTHS_ES[m - 1] for m in sorted(occ["months"] or [])) or "sin datos"
    lines = [
        f"registros limpios en Costa Rica: {occ['n_records']} "
        f"({occ['n_specimens'] or 0} especímenes, {occ['n_observations'] or 0} observaciones)",
        f"celdas ocupadas: {occ['n_cells']}",
        f"años: {occ['first_year']}-{occ['last_year']}",
        f"altitud de los registros (DEM, p05/p50/p95): "
        f"{round(occ['elev_p05'] or 0)}/{round(occ['elev_p50'] or 0)}/{round(occ['elev_p95'] or 0)} m",
        f"regiones con registros: {', '.join(occ['regions'] or []) or 'sin datos'}",
        f"vertientes con registros: {', '.join(occ['vertientes'] or []) or 'sin datos'}",
        f"meses con registros: {months}",
    ]
    return "\n".join(lines)


# Which blocks each question type gets. Keeping this explicit beats a catch-all pack:
# an elevation question does not need a parks table, and a shorter pack is a cheaper and
# more faithful one.
_FLORA_BLOCKS = {
    "elevation":      ("cita", "distribución y hábitat", "altitud"),
    "cultivation":    ("cita", "distribución y hábitat", "altitud", "vertientes"),
    "vertiente":      ("cita", "distribución y hábitat", "vertientes", "regiones"),
    "flowering":      ("cita", "distribución y hábitat", "floración"),
    "growth_form":    ("cita", "forma de crecimiento", "morfología"),
    "endemism":       ("cita", "distribución y hábitat", "endemismo"),
    "light":          ("cita", "distribución y hábitat", "morfología"),
    "outside_record": ("cita", "distribución y hábitat", "notas"),
    "park_confirmed": ("cita", "distribución y hábitat", "altitud", "regiones"),
    "park_possible":  ("cita", "distribución y hábitat", "altitud", "regiones"),
    "park_unlikely":  ("cita", "distribución y hábitat", "altitud", "regiones"),
}

# Occurrence-derived blocks, added only at level "full".
_FULL_EXTRA = {
    "elevation":      ("registros de ocurrencia",),
    "cultivation":    ("registros de ocurrencia",),
    "vertiente":      ("registros de ocurrencia",),
    "flowering":      ("registros de ocurrencia",),
    "growth_form":    (),
    "endemism":       (),
    "light":          (),
    "outside_record": (),
    "park_confirmed": ("área protegida", "registros de ocurrencia"),
    "park_possible":  ("área protegida", "registros de ocurrencia"),
    "park_unlikely":  ("área protegida", "registros de ocurrencia"),
}

# Occurrence evidence FILLS GAPS; it does not compete with a stated fact.
#
# Without this gate the grounded arm is penalised for using evidence we handed it.
# Cryptocentrum flavum: the elevation key is the Manual's 950-1050 m, while the records
# put the species at 423-1248 m, in Valle del Coto Brus rather than the Manual's
# Cordillera de Talamanca, flowering in June rather than December. All three are real
# facts about the snapshot and all three contradict a key the Manual defines. Handing
# them over would measure which source a model privileges, not whether it answers
# truthfully.
#
# So for a question whose key is a stated Manual fact, occurrence evidence enters only
# when the Manual is silent -- which is also a real advantage for the grounded arm, and
# a measurable one. For the park types the occurrence data IS the key's basis, so it is
# always included.
#
# Consequence, and it matches the pre-registered design (plan sec. 4b): arms D and E
# receive identical evidence wherever the Manual states the answer, so the ablation
# bites on the park types and on the gaps. That is the claim the evidence layer makes --
# it answers what the book cannot -- rather than a convenience.
_OCC_GATE = {
    "elevation":      lambda f: f.elev_min is None,
    "cultivation":    lambda f: f.elev_min is None,
    "vertiente":      lambda f: not f.vertientes,
    "flowering":      lambda f: not f.flowering_months,
    "park_confirmed": lambda f: True,
    "park_possible":  lambda f: True,
    "park_unlikely":  lambda f: True,
}


def build(cur, species: str, qtype: str, *, level: str = "full",
          place_code: str | None = None) -> Pack:
    """Assemble the pack for one question. ``place_code`` is required for park types."""
    if level not in ("flora", "full"):
        raise ValueError(f"level must be 'flora' or 'full', got {level!r}")
    if qtype not in _FLORA_BLOCKS:
        raise ValueError(f"unknown qtype {qtype!r}")

    f = _ficha(cur, species)
    pack = Pack(species=species, qtype=qtype, level=level)
    pack.providers.append("MPCR catalogue")

    eff_rng = _eff(f)
    pack.facts = {
        "species": f.species, "cita": _cite(f), "family": f.family,
        "elev_min": f.elev_min, "elev_max": f.elev_max,
        "elev_effective": list(eff_rng) if eff_rng else None,
        "vertientes": list(f.vertientes or []), "regions": list(f.regions or []),
        "forest_types": list(f.forest_types or []),
        "flowering_months": list(f.flowering_months or []),
        "growth_form": _growth_form_text(f), "endemic_cr": bool(f.endemic_cr),
        "range_tail": f.distribution_paragraph.split(";", 1)[-1].strip(),
        "range_beyond": _range_beyond(f),
        "habitat_sentence": f.distribution_paragraph.split(";")[0].strip(),
        "use_cues": " ".join(m.group(0) for m in
                             _USE_CUE.finditer(f"{f.discussion} {f.distribution_paragraph}")) or None,
    }

    wanted = list(_FLORA_BLOCKS[qtype])
    if level == "full":
        wanted += list(_FULL_EXTRA[qtype])

    for name in wanted:
        if name == "cita":
            pack.add("cita", f"{f.species} {f.authority or ''}".strip()
                     + f"\nfamilia: {f.family}\nfuente: {_cite(f)}")
        elif name == "distribución y hábitat":
            pack.add("distribución y hábitat (texto del Manual)", f.distribution_paragraph)
        elif name == "altitud":
            pack.add("altitud (Manual)", _elevation_text(f))
        elif name == "vertientes":
            pack.add("vertientes (Manual)", ", ".join(f.vertientes or []) or None)
        elif name == "regiones":
            pack.add("regiones botánicas (Manual)", ", ".join(f.regions or []) or None)
        elif name == "floración":
            pack.add("meses de floración (Manual)",
                     ", ".join(MONTHS_ES[m - 1] for m in f.flowering_months) or None
                     if f.flowering_months else None)
        elif name == "forma de crecimiento":
            pack.add("forma de crecimiento (Manual)", _growth_form_text(f))
        elif name == "morfología":
            pack.add("morfología (Manual)", (f.morphology or "")[:MORPHOLOGY_CHARS] or None)
        elif name == "endemismo":
            tail = f.distribution_paragraph.split(";", 1)[-1].strip()
            pack.add("endemismo (Manual)",
                     ("ENDÉMICA de Costa Rica" if f.endemic_cr else None)
                     or (f"distribución más allá de Costa Rica: {tail}" if tail else None))
        elif name == "notas":
            pack.add("notas y discusión (Manual)", (f.discussion or "")[:MORPHOLOGY_CHARS] or None)
        elif name == "registros de ocurrencia":
            if not _OCC_GATE.get(qtype, lambda _f: False)(f):
                continue
            occ = _occurrence(cur, species)
            if occ:
                pack.add("registros de ocurrencia (GBIF, instantánea 10.15468/dl.8yhee8)",
                         _occurrence_block(occ))
                pack.providers += ["GBIF snapshot 10.15468/dl.8yhee8", "DEM"]
        elif name == "área protegida":
            if not place_code:
                raise ValueError(f"{qtype} needs place_code")
            place = _place(cur, place_code)
            if place:
                n70, n_all = _records_in_place(cur, _taxon_key(cur, species), place_code)
                pack.add("área protegida consultada", _park_block(f, place, n70, n_all))
                sp_r, pk_r = set(f.regions or []), set(place["regions"] or [])
                pack.facts["place"] = {
                    "label": place["label"], "code": place["code"],
                    "elev_p05": round(place["lo"]), "elev_p95": round(place["hi"]),
                    "regions": list(place["regions"] or []),
                    "n_records": n_all, "n_since_1970": n70,
                    "region_overlap": bool(sp_r & pk_r),
                    "shared_regions": sorted(sp_r & pk_r),
                    "elev_overlap": bool(eff_rng is not None
                                         and place["lo"] <= eff_rng[1] and place["hi"] >= eff_rng[0]),
                }
                pack.providers += ["SINAC protected areas", "MPCR regions + elevation",
                                   "GBIF snapshot 10.15468/dl.8yhee8", "DEM"]

    # de-duplicate while keeping order, so the hash is stable
    seen: set[str] = set()
    pack.providers = [p for p in pack.providers if not (p in seen or seen.add(p))]
    return pack


def main() -> None:
    ap = argparse.ArgumentParser(description="Print one evidence pack")
    ap.add_argument("--species", required=True)
    ap.add_argument("--qtype", required=True, choices=sorted(_FLORA_BLOCKS))
    ap.add_argument("--level", default="full", choices=["flora", "full"])
    ap.add_argument("--place-code", default=None)
    args = ap.parse_args()

    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            p = build(cur, args.species, args.qtype, level=args.level,
                      place_code=args.place_code)
    finally:
        conn.close()
    print(f"--- {p.species} / {p.qtype} / level={p.level} / hash={p.hash} ---\n")
    print(p.text())
    print(f"\n--- providers: {', '.join(p.providers)}")
    print(f"--- {len(p.text())} chars, ~{len(p.text()) / 3.07:.0f} tokens")


if __name__ == "__main__":
    main()
