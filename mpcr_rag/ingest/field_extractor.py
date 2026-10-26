"""RawFicha → Ficha: structured field extraction.

The geospatial fields (elevation, vertientes, regions, forest_types) are produced
by CR-BioLM's existing map pipeline: the distribution paragraph splits on its
``m;`` boundary into the ``(habitat_raw, geographic_notes)`` pair that
``parser.build_ficha`` consumes — so RAG fields are identical to map fields.

On top of that we add the *deterministic* enrichment fields (endemism, phenology),
which are clean regex on the same paragraph. The soft fields that regex can't get
reliably (habit, common_names, uses, global_range) are left for the LLM pass
(``llm_enrich.py``) — every field is independently nullable.

Run as a module for a regression-style readout over the first corpus PDF:
    python -m mpcr_rag.ingest.field_extractor
"""
from __future__ import annotations

import json
import re

from utils.distribution_map.parser import build_ficha

from .. import config
from ..schema import Ficha, RawFicha
from .ficha_segmenter import segment

# --- family (genus-resolved, NOT file-derived) ----------------------------- #
# Several source PDFs (esp. Vol VI, split by fixed page ranges rather than by
# family, and the Vol II/III "FamilyA_FamilyB" monocot bundles) span MULTIPLE
# families per file; the filename only names the first (or first+last) one, so
# every species after a family transition inherited the wrong label. genus is
# extracted from the actual species binomial (reliable), so resolving family
# from genus via the GBIF backbone is the authoritative fix — confirmed the
# underlying text extraction (distribution paragraph, hence habit/elevation/
# region) is unaffected; only this per-file family tag was wrong.
_GENUS_FAMILY_CACHE_PATH = config.DATA_DIR / "genus_family_gbif.json"
_GENUS_FAMILY: dict[str, str | None] | None = None


def _load_genus_family_cache() -> dict[str, str | None]:
    global _GENUS_FAMILY
    if _GENUS_FAMILY is None:
        if _GENUS_FAMILY_CACHE_PATH.exists():
            _GENUS_FAMILY = json.loads(_GENUS_FAMILY_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _GENUS_FAMILY = {}
    return _GENUS_FAMILY


def _resolve_family(genus: str, fallback: str) -> str:
    """GBIF-backbone family for `genus`, cached; falls back to the file-derived
    label only when GBIF has no exact genus match (rare: ~3% of genera)."""
    if not genus:
        return fallback
    cache = _load_genus_family_cache()
    if genus in cache:
        return cache[genus] or fallback
    from pygbif import species as gbif_species
    try:
        r = gbif_species.name_backbone(scientificName=genus, taxonRank="genus", kingdom="Plantae")
        mt = r.get("matchType") or r.get("diagnostics", {}).get("matchType")
        fam = None
        if mt == "EXACT":
            fam = {c["rank"]: c["name"] for c in r.get("classification", [])}.get("FAMILY")
    except Exception:
        fam = None
    cache[genus] = fam
    _GENUS_FAMILY_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return fam or fallback


# --- phenology ------------------------------------------------------------- #
_MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}
_ENDEMIC = re.compile(r"end[ée]mic", re.IGNORECASE)

# --- growth form (habit) --------------------------------------------------- #
# Each (pattern, normalized label); a ficha can carry several ("Arbusto o árbol").
# Broadened to cover monocot/herb vocabulary (grasses, orchids, aroids, palms…).
_HABIT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[áa]rbol|arbolito", re.I), "árbol"),
    (re.compile(r"\barbust|arbustiv", re.I), "arbusto"),
    (re.compile(r"\bhierba|herb[áa]ce|c[eé]spit|gramin", re.I), "hierba"),
    (re.compile(r"\bbejuco|liana|trepador|enredadera|escandent|voluble", re.I), "bejuco"),
    (re.compile(r"hemiep[íi]fit", re.I), "hemiepífita"),
    (re.compile(r"\bep[íi]fit", re.I), "epífita"),
    (re.compile(r"\bterrestre", re.I), "terrestre"),
    (re.compile(r"\bacu[áa]tic|palustre|flotante|sumergid", re.I), "acuática"),
    (re.compile(r"sufr[úu]tic|subarbust", re.I), "sufrútice"),
    (re.compile(r"\bpalmas?\b|palmera|arecoid", re.I), "palma"),
    (re.compile(r"par[áa]sit", re.I), "parásita"),
]

# Block openers that signal a morphological description (not distribution/key).
_DESC_OPEN = re.compile(
    r"^(Arbusto|Árbol|Arbol|Hierba|Bejuco|Liana|Epíf|Hemiep|Sufr|Planta|Hojas|"
    r"Pseudobulbo|Rizoma|Tallos?|Culmos?|Cañas?|Palma|Tr[eé]pa|C[eé]spit|"
    r"Roseta|Bulbo|Cormo|Estolon)", re.I)


def _description_text(raw: RawFicha) -> str:
    """The morphological description block (where the growth form is stated).

    Prefers the segmented morphology; otherwise the first description-like block
    before the distribution paragraph (monocot descriptions often open with words
    like 'Pseudobulbos'/'Rizoma'/'Culmos' that aren't classic habit keywords)."""
    if raw.morphology:
        return raw.morphology
    for b in raw.blocks:
        if b is raw.distribution_paragraph:
            break
        if len(b) > 30 and _DESC_OPEN.match(b):
            return b
    # last resort: first sizable non-distribution block
    for b in raw.blocks:
        if b is not raw.distribution_paragraph and len(b) > 30:
            return b
    return raw.distribution_paragraph or ""


def _match_habits(text: str) -> list[str]:
    out: list[str] = []
    for pat, label in _HABIT_PATTERNS:
        if pat.search(text) and label not in out:
            out.append(label)
    return out


# Families whose defining growth form is never stated as a keyword in the text
# (palms describe themselves as "Planta con tallos…", never "palma").
_FAMILY_HABIT = {"Arecaceae": "palma"}


def _extract_habits(raw: RawFicha) -> list[str]:
    """Growth forms from the species description; if none are stated at species
    level (common for monocots: orchids, grasses), inherit from the genus. A
    family-defining growth form (Arecaceae → palma) is added unconditionally."""
    h = _match_habits(_description_text(raw)[:160])
    if not h and raw.genus_description:
        h = _match_habits(raw.genus_description[:160])
    fam_h = _FAMILY_HABIT.get(raw.family)
    if fam_h and fam_h not in h:
        h = [fam_h] + h
    return h


def split_distribution(paragraph: str) -> tuple[str, str]:
    """Split on the elevation's ``m;`` boundary into (habitat_raw, geographic_notes).

    ``geographic_notes`` is trimmed before the phenology/range/voucher tail so the
    geo parser only sees place names.
    """
    m = re.search(r"m\s*;", paragraph)
    if m:
        habitat_raw = paragraph[: m.start() + 1].strip()   # keep the 'm', drop ';'
        rest = paragraph[m.end():].strip()
    else:
        habitat_raw, rest = "", paragraph
    geo = re.split(r"\bFls?\.\s|\bFr\.\s", rest)[0].strip()
    return habitat_raw, geo


def _months(marker: str, text: str) -> list[int]:
    """Parse the month list following 'Fl.' / 'Fr.' (e.g. 'ene., abr.–ago., oct.')."""
    m = re.search(marker + r"\.?\s*([a-z][a-z.,–\-\s]*)", text)
    if not m:
        return []
    seg = m.group(1)
    out: set[int] = set()
    for a, b in re.findall(r"([a-z]{3})\.?\s*[–\-]\s*([a-z]{3})", seg):
        if a in _MONTHS and b in _MONTHS:
            i, j = _MONTHS[a], _MONTHS[b]
            out.update(range(i, j + 1) if i <= j
                       else list(range(i, 13)) + list(range(1, j + 1)))
    for t in re.findall(r"\b([a-z]{3})\b", seg):
        if t in _MONTHS:
            out.add(_MONTHS[t])
    return sorted(out)


def extract(raw: RawFicha) -> Ficha:
    """Build a fully-extracted Ficha from a RawFicha (deterministic fields only)."""
    par = raw.distribution_paragraph or ""
    habitat_raw, geo_notes = split_distribution(par)

    df = build_ficha(habitat_raw=habitat_raw, geographic_notes=geo_notes,
                     species=raw.species)

    elev = df.elevation
    return Ficha(
        species=raw.species,
        authority=raw.authority,
        genus=raw.genus,
        family=_resolve_family(raw.genus, raw.family),
        volume=raw.volume,
        pages=str(raw.page),
        elev_min=int(elev.min_m) if elev.min_m is not None else None,
        elev_max=int(elev.max_m) if elev.max_m is not None else None,
        elev_outlier_min=int(elev.outlier_min_m) if elev.outlier_min_m is not None else None,
        elev_outlier_max=int(elev.outlier_max_m) if elev.outlier_max_m is not None else None,
        vertientes=list(df.vertientes),
        regions=[r.canonical_name for r in df.regions],
        forest_types=list(df.forest_types),
        distribution_paragraph=par,
        habits=_extract_habits(raw),
        endemic_cr=bool(_ENDEMIC.search(par)),
        flowering_months=_months("Fl", par),
        fruiting_months=_months("Fr", par),
        morphology=raw.morphology or "",
        discussion=raw.discussion or "",
        genus_description=raw.genus_description or "",
        full_text="\n".join(s for s in (raw.header_block, raw.morphology, par,
                                        raw.discussion) if s),
    )


_RANGE_IN_NAME = re.compile(r"\.(\d{1,4})[_-](\d{1,4})\.pdf$", re.I)


def page_offset(pdf_name: str) -> int:
    """The book page of the first page of a corpus PDF.

    The segmenter numbers pages from 0 within each file, and that index was being
    stored as the page. It is not the book page: the files are cut from the volumes,
    and their names carry the book pages they hold ("Lemnaceae_Musaceae
    MPCRv2.618_674.pdf"). Calathea donnell-smithii was cited as Tomo II, p. 21; it is on
    p. 639 (618 + 21). Found by an expert reviewing citations, and checked against the
    page numbers printed on the PDFs: the offset is the first page in the file name, and
    1 for the whole-volume files, whose printed numbering starts at 1.
    """
    m = _RANGE_IN_NAME.search(pdf_name)
    return int(m.group(1)) if m else 1


def volume_label(entry: dict) -> str:
    """Volume IV ships as two files with separate pagination; the second is IV(2), and
    a page number cited without saying which part could not be found."""
    return "IV(2)" if "IV(2)" in entry["path"].name else entry["volume"]


def extract_corpus_pdf(entry: dict) -> list[Ficha]:
    raws = segment(str(entry["path"]), volume=entry["volume"], family=entry["family"])
    offset, volume = page_offset(entry["path"].name), volume_label(entry)
    out = []
    for r in raws:
        if r.distribution_paragraph:
            f = extract(r)
            f.pages = str(offset + int(r.page))
            f.volume = volume
            out.append(f)
    return out


if __name__ == "__main__":
    from .. import config
    fichas = extract_corpus_pdf(config.CORPUS[0])

    n = len(fichas)
    n_elev = sum(1 for f in fichas if f.elev_min is not None)
    n_vert = sum(1 for f in fichas if f.vertientes)
    n_reg = sum(1 for f in fichas if f.regions)
    n_for = sum(1 for f in fichas if f.forest_types)
    n_end = sum(1 for f in fichas if f.endemic_cr)
    print(f"{n} fichas extracted")
    print(f"  elevation : {n_elev}/{n} ({100*n_elev//n}%)")
    print(f"  vertientes: {n_vert}/{n} ({100*n_vert//n}%)")
    print(f"  regions   : {n_reg}/{n} ({100*n_reg//n}%)")
    print(f"  forest    : {n_for}/{n} ({100*n_for//n}%)")
    print(f"  endemic   : {n_end}/{n}")
    print("\n--- sample ---")
    for f in fichas[:12]:
        print(f"\n{f.species}  [{f.elev_min}–{f.elev_max} m]  endemic={f.endemic_cr}")
        print(f"  vert:   {f.vertientes}")
        print(f"  forest: {f.forest_types}")
        print(f"  flower: {f.flowering_months}")
        print(f"  regions: {[r for r in f.regions]}")
