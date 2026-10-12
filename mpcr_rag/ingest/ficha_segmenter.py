"""Segment a Manual de Plantas de Costa Rica volume PDF into per-species fichas.

The Manual is a flora: each species entry ("ficha") has a near-regular anatomy:

    1. Header:        ``Aiouea costaricensis (Mez) Kosterm., <protologue>. 1938.``
    2. Synonyms+names ``Bellota costaricensis Mez, ... 1889. Ira, Mangle, Quizarra.``
    3. Morphology:    ``Arbusto o arbol, 3-30 m, las ramitas ...``
    4. Distribution:  ``Bosque muy humedo, ... 600-2500 m; ambas verts. ... Fl. ene. CR.``
    5. Discussion:    ``Aiouea costaricensis se reconoce por ...``

Only (1) the header and (4) the distribution paragraph are reliable anchors; ANY
other section may be absent for a given species. So segmentation keys solely on
those two, and every other field is independently optional.

The distribution paragraph later splits on its ``m;`` boundary into exactly the
``(habitat_raw, geographic_notes)`` pair that ``parser.build_ficha`` consumes, so
downstream extraction reuses the existing map pipeline verbatim.

Run as a module to segment the first PDF in ``config.CORPUS``:
    python -m mpcr_rag.ingest.ficha_segmenter
"""
from __future__ import annotations

import re

import fitz  # PyMuPDF

from ..schema import RawFicha


# --------------------------------------------------------------------------- #
# Block classification patterns
# --------------------------------------------------------------------------- #

# A species header opens a block as "<Genus> <epithet>" whose protologue carries a
# `volume: page` citation + a 4-digit year near the start — a discriminator no
# distribution paragraph or discussion shares. Unnamed entries are "<Genus> sp. N.".
_YEAR = r"(?:1[6789]\d\d|20\d\d)"
_HEADER_NAMED = re.compile(r"^([A-Z][a-záéíóúñ-]+)\s+([a-záéíóúñ-]{3,})\b")
_HEADER_SP = re.compile(r"^([A-Z][a-záéíóúñ]+)\s+sp\.\s*\d+")
_CITATION = re.compile(r"\d{1,3}(?:\(\d+[a-z]?\))?:\s*\d{1,4}")  # "46: 73" / "10(5): 49"
_YEAR_RE = re.compile(_YEAR)

# Works cited without a volume ("L., Sp. pl. 342. 1753.", "Kunth, Nov. gen. sp. t. 467.
# 1822.") have no "vol: page", so the colon rule above missed ~1,200 species headers:
# their description and distribution were appended to the previous species and the
# species never reached the catalog. Accepted only with an authority right after the
# binomial — discussion prose continues in lowercase ("... difiere de") and can cite
# years, an authority opens with a capital or a parenthesis.
_PAGE_YEAR = re.compile(r"\b\d{1,4}[a-z]?[.,]\s*" + _YEAR + r"\b")


def _authority_ok(block: str, binomial_end: int) -> bool:
    """The text between the binomial and the first comma must read as an authority:
    it opens with a capital or "(" ("L.,", "(Benth.) Kuntze,", "Mez 2 ,") and carries
    no citation numbers. This separates real protologues from look-alikes that also
    have a binomial, a citation and a year:
      figure credits   "Alternanthera paronychioides Flora brasiliensis 5(1): 185, …"
      prose            "Es probable que el género Conyza sea polifilético, …"
      bracketed notes  "Rhynchospora pedersenii Guagl. [Darwiniana 39: 321. 2001; …]"
    """
    rest = block[binomial_end:binomial_end + 90]
    comma = rest.find(",")
    if comma < 0 or "[" in rest[:comma]:
        return False
    authority = rest[:comma].strip()
    # lowercase name particles open real authorities: "van der Werff", "de Nevers",
    # "hort. ex Jacobi"
    return (bool(re.match(r"^(?:\(|[A-Z]|(?:van|von|de|der|den|du|la|le|d’|d'|hort\.|ex)\s)",
                          authority))
            and ":" not in authority and not re.search(r"\d{2,}", authority))

# Figure credits under illustrations carry a binomial + citation + year and parsed as
# species headers, creating duplicate fichas that swallowed the next genus's text.
_FIG_CREDIT = re.compile(
    r"^[A-Z][a-záéíóúñ-]+\s+[a-záéíóúñ-]+\s+"
    r"(?:Cortes[íi]a|Modificad|Tomad|Reproducid|Adaptad|Redibujad|Ilustr|Dibuj|Fotograf)"
)

# "Salacia sp. 1 es parecida a ..." — discussion prose, not an unnamed-species header.
_SP_PROSE = re.compile(r"^\S+\s+sp\.\s*\d+\s*,?\s+[a-záéíóúñ]")

# Capitalized Spanish words that open non-header blocks (distribution/morph/discussion).
_GENUS_STOP = {
    "Bosque", "Se", "Las", "Los", "El", "La", "En", "Por", "Como", "Esta", "Este",
    "Arbusto", "Árbol", "Arbol", "Hierba", "Bejuco", "Liana", "Planta", "Plantas",
    "Hojas", "Fls", "Frs", "Infl", "Infls", "Lámina", "Láminas", "Yemas", "Ramitas",
    "Vegetativamente", "Entre", "Sin", "Con", "Burger", "Werff",
    # sentence openers in genus notes ("Algunos autores …", "Es probable que …")
    "Algunos", "Algunas", "Es", "Existen", "Hay", "Aunque", "Según", "Otros", "Otras",
    "Varios", "Varias", "Muchos", "Muchas", "Todas", "Todos",
}

# Genus header block: the genus name plus its species count ("Allium Ca. 700 spp.,",
# "Guzmania 126 spp.,", "Apteria 1 sp.,", OCR-mangled "Ca 25 spp.,") or the treatment
# author ("Lepanthes Por C. A. Luer"). Blocks reach this point with linebreaks already
# flattened by _clean, so the previous "Genus\nPor|Ca." pattern never matched: genus
# boundaries went undetected and every species in a PDF inherited the FIRST genus's
# description (and, through it, a wrong inherited habit).
_GENUS_HEADER = re.compile(
    r"^(?:[A-Z][a-záéíóúñ]+\s+)?(?:Ca\.?\s*|Quizás\s+|Aprox\.?\s*|Más de\s+)?"
    r"\d+(?:[–-]\d+)?\s+spp?\.[\s,;]"
    r"|^[A-Z][a-záéíóúñ]+\s+Por\s+[A-Z]"
)
# Where the genus description starts when it shares a block with the species-count
# sentence ("Ca. 10–15 spp., pantrop.; 3 spp. en CR. Hierbas decumbentes. …").
_GENUS_DESC_START = re.compile(
    r"\.\s+(?=(?:Hierbas?|Arbust|Árbol|Arbol|Plantas?|Bejucos?|Lianas?|Epíf|Terrestres?|"
    r"Tallos?|Hojas|Sufr|Subarbust|Palmas?|Rizomas?|Culmos?)\b)")

# Herbarium voucher that closes a distribution paragraph: "( Hammel 19343 ; CR, INB, MO)".
_VOUCHER = re.compile(r"\([^()]*\d[^()]*\b[A-Z]{2,5}\s*\)")
_DISCUSSION_CUE = re.compile(r"\bse (?:reconoce|distingue|caracteriza|diferencia)|\bdifiere\b")
# The genus name often stands alone in its own block, followed by the treatment's
# bibliography and only then the species-count block.
_GENUS_NAME = re.compile(r"^[A-Z][a-záéíóúñ]+$")

# Bibliography lines in a genus preamble: "Davis, J. I. 1978. Systematics of ...".
_BIBLIO = re.compile(r"^[A-Z][A-Za-záéíóúñ'’-]+,\s+(?:[A-Z][a-z]?\.\s*-?\s*)+.{0,120}?\b"
                     + _YEAR + r"[a-z]?\.")

# Figure labels that sit between text blocks: "Fig. 12", or a bare binomial.
_CAPTION = re.compile(r"^Fig(?:ura)?\.?\s*\d|^[A-Z][a-záéíóúñ]+\s+[a-záéíóúñ-]+\.?$")

# Word hyphenation left by line joins ("pu- bescentes", "Veg- etativamente").
# Only lowercase-to-lowercase, so ranges ("4–9"), en dashes and "Méx.–Hond." survive.
_HYPHEN_BREAK = re.compile(r"([a-záéíóúñü])[-­]\s+([a-záéíóúñü])")

# Dichotomous-key block: carries leader dots.
_KEY_BLOCK = re.compile(r"\.\s\.\s\.|\.{4,}")

# Distribution paragraph detection. The elevation detector is deliberately loose
# (open/atypical ranges exist: "(100–)600–2500", "0–200+", "3200–"); precise min/max
# parsing happens later via parser._parse_elevation, not here.
_ELEV_M = re.compile(r"\d[\d\s().+?–—-]*\s*m\b")   # '?' = uncertain-elevation marker
# A real distribution block opens with a habitat/elevation term (NEVER a habit word)
# and carries a strong slope/endemism cue. NB: bare "ambas"/"región" are rejected —
# they appear in MORPHOLOGY ("ambas caras", "región basal"); require "ambas vert".
_FOREST_START = re.compile(r"^(Bosque|Sabana|Manglar|Páramo|Charral|Vegetaci|Matorral|Pastizal"
                           r"|Arrecifes|Playas|Aguas|Lagunas|Pantanos|Charcas|Humedales)")
_STRONG_DIST = re.compile(r"vert\.|verts\.|ambas\s+vert|END[ÉE]MIC|ENDEMIC")

# Morphology block opener (habit / description start).
_MORPH = re.compile(
    r"^(Arbusto|Árbol|Arbol|Hierba|Bejuco|Liana|Epífit|Hemiep|Sufrút|Planta|Hojas)"
)

# A leading clave number ("1 Ramitas ...", "3 Yemas ...") marks a key fragment.
_NUM_KEY = re.compile(r"^\d{1,2}\s+\S")

# Page furniture to drop.
_FURNITURE = re.compile(r"^\d+\s+Manual de Plantas|^MPCRv|Manual de Plantas de Costa Rica\s*$")


# Stray C0/C1 control characters injected by the OCR (e.g. "2500\x04 m") break
# regexes and would pollute embedded text + geo_parser input — strip them first.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _clean(text: str) -> str:
    """Strip OCR control chars, then flatten linebreaks + hyphenation into prose."""
    text = _CTRL.sub("", text)
    text = text.replace("-\n", "").replace("­\n", "")
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def dehyphenate(text: str) -> str:
    """Rejoin words split by line-end hyphenation ("pu- bescentes" -> "pubescentes").

    Also applied to the distribution paragraph, which feeds geo_parser. Measured
    before adoption (2026-09-12): 1,806 of 6,946 paragraphs carry a break; parsing them
    de-hyphenated GAINS 79 region matches over 64 species ("Cord. de Tala- manca" was
    invisible to the gazetteer) and loses none.
    """
    if not text:
        return text
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return re.sub("­(?=\\S)", "", text)          # stray soft hyphen: "­Caraigres"


_BARE_UPPER = re.compile(r"^\s*\d[\d\s.,]*\s*m\s*;")          # "2800 m; ambas verts. ..."
_DANGLING = re.compile(r"\d\s*[–—-]\s*$")                     # "... montanos, 600–"
_HABITAT_OPEN = re.compile(r"(?:^|(?<=[.;]\s))(Bosque|Sabana|Manglar|Páramo|Charral|Vegetaci"
                           r"|Matorral|Pastizal|Arrecifes|Playas|Aguas|Lagunas|Pantanos"
                           r"|Charcas|Humedales)")


def _rejoin_split_range(morphology: str | None, dist: str | None):
    """Put back together a distribution paragraph cut in the middle of its elevation.

    A block break can fall between the two bounds of the range: the description block
    then ends "... bosques montanos, 600–" and the distribution block opens "2800 m;
    ambas verts. ...". The habitat sentence and the lower bound were filed as
    description, the paragraph kept a bare upper bound that parses as no elevation at
    all, and the map lost its elevation band (Calyptranthes pittieri, found when an
    expert review sheet showed the map without one). Both ends must match, so an intact
    entry is never touched.
    """
    if not morphology or not dist or not _BARE_UPPER.match(dist) \
            or not _DANGLING.search(morphology):
        return morphology, dist
    opens = [m.start(1) for m in _HABITAT_OPEN.finditer(morphology)]
    if not opens:
        return morphology, dist
    cut = opens[-1]
    return morphology[:cut].rstrip(), morphology[cut:].rstrip() + dist.lstrip()


def _is_caption(block: str) -> bool:
    """Figure label, not prose. A short description ("Hierba anual.") has the same
    two-word shape as a bare binomial, so habit/description openers are excluded."""
    if len(block) >= 80 or not _CAPTION.match(block):
        return False
    first = block.split()[0].rstrip(".")
    return first not in _GENUS_STOP and not _MORPH.match(block)


def _append(section: str | None, block: str) -> str:
    """Append a block to a multi-block section, rejoining a word split at the page
    break ("hori-" + "zontales")."""
    if not section:
        return block
    if section.endswith(("-", "­")) and block[:1].islower():
        return section[:-1] + block
    return f"{section} {block}"


def extract_blocks(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_index, clean_block_text)] in reading order, furniture removed.

    Uses "dict" mode (keeps morphology and distribution as distinct blocks) and a
    plain reading-order sort by (y, x). The Manual is single-column with figures, so
    a y-first order is correct — an earlier (left/right-half, y) sort wrongly assumed
    two columns and scrambled pages where an image sits beside the text.
    """
    doc = fitz.open(pdf_path)
    out: list[tuple[int, str]] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        ph = page.rect.height
        blocks = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:                    # skip image blocks
                continue
            txt = " ".join(s["text"] for line in b["lines"] for s in line["spans"])
            blocks.append((b["bbox"][0], b["bbox"][1], txt))
        blocks.sort(key=lambda t: (round(t[1]), t[0]))   # reading order: y then x
        for x0, y0, txt in blocks:
            t = txt.strip()
            if not t or _FURNITURE.search(t):
                continue
            # Running-name footers / PDF stamps: short text hugging top/bottom edge.
            if len(t) < 40 and (y0 < 100 or y0 > ph - 80):
                continue
            out.append((pno, _clean(t)))
    doc.close()
    return out


def _is_species_header(block: str) -> tuple[str, str, str] | None:
    """Return (genus, binomial, authority) if the block opens a species ficha."""
    if _FIG_CREDIT.match(block):
        return None
    m = _HEADER_SP.match(block)
    if m and not _SP_PROSE.match(block):
        genus = m.group(1)
        sp = re.match(r"^\S+\s+(sp\.\s*\d+)", block).group(1)
        return genus, f"{genus} {sp}", ""
    m = _HEADER_NAMED.match(block)
    if m and m.group(1) not in _GENUS_STOP:
        head = block[:140]                   # protologue sits at the very start
        colon_cited = bool(_CITATION.search(head) and _YEAR_RE.search(head))
        page_cited = bool(_PAGE_YEAR.search(block[:160]))
        if (colon_cited or page_cited) and _authority_ok(block, m.end(2)):
            genus, epithet = m.group(1), m.group(2)
            authority = block[m.end(2):].split(".")[0].strip(" ,")
            return genus, f"{genus} {epithet}", authority
    return None


def segment(pdf_path: str, *, volume: str = "", family: str = "") -> list[RawFicha]:
    """Split a volume PDF into RawFicha records (one per species entry)."""
    blocks = extract_blocks(pdf_path)
    fichas: list[RawFicha] = []
    current: RawFicha | None = None
    genus_desc: str | None = None   # genus preamble text, inherited by its species

    for pno, block in blocks:
        is_genus_name = bool(_GENUS_NAME.match(block)) and block not in _GENUS_STOP
        if is_genus_name or (_GENUS_HEADER.match(block) and not _is_species_header(block)):
            # The genus preamble (description, key) no longer belongs to the
            # previous species; without this it was appended as its "discussion".
            current = None
            genus_desc = None
            desc = _GENUS_DESC_START.search(block) if not is_genus_name else None
            if desc:
                genus_desc = block[desc.end():]
            continue

        hdr = _is_species_header(block)
        if hdr and current is not None and not current.blocks:
            # Two headers with nothing between them. Every real entry has a
            # description or distribution before the next, so one of the two is not
            # an entry: after a full protologue the second is a synonym line in its
            # own block ("Cynosurus aegyptius L., Sp. pl. 72. 1753."); after a bare
            # label ("Licaria sp. 2", a figure tag) the label is the noise.
            if _CITATION.search(current.header_block[:160]) or _PAGE_YEAR.search(current.header_block[:160]):
                current.header_block = f"{current.header_block} {block}"
                continue
            fichas.pop()
        if hdr:
            # Every header is accepted, as before: the genus-match guard never
            # fired (genus boundaries were undetected), and making it effective now
            # would silently drop species whenever a genus header is missed.
            genus, binomial, authority = hdr
            current = RawFicha(
                species=binomial, authority=authority, genus=genus,
                header_block=block, volume=volume, family=family, page=pno,
                genus_description=dehyphenate(genus_desc or ""),
            )
            fichas.append(current)
            continue

        is_key = bool(_KEY_BLOCK.search(block) or _NUM_KEY.match(block))
        is_caption = _is_caption(block) or bool(_FIG_CREDIT.match(block))

        if current is None:
            # family/genus preamble: keep its prose, skip keys, figure labels and
            # bibliography lines
            if not is_key and not is_caption and not _BIBLIO.match(block):
                genus_desc = _append(genus_desc, block)
            continue

        # Distribution = has elevation, does NOT open with a habit word (that's
        # morphology), and carries a strong slope/forest/endemism cue.
        is_dist = (bool(_ELEV_M.search(block)) and not _MORPH.match(block)
                   and bool(_STRONG_DIST.search(block) or _FOREST_START.match(block)))
        if not is_dist and is_key:
            continue  # leftover dichotomous-key fragment

        current.blocks.append(block)
        if current.distribution_paragraph is None and is_dist:
            current.distribution_paragraph = block
        elif is_caption:
            continue
        elif current.distribution_paragraph is None:
            # Everything between header and distribution is the description; it
            # often spans blocks (a page break mid-description).
            current.morphology = _append(current.morphology, block)
        else:
            if current.discussion is None and not _VOUCHER.search(current.distribution_paragraph):
                # The distribution paragraph was cut at a block break: its tail
                # (phenology, range, voucher) opens this block, followed by the
                # discussion ("Austral., N. Z. ( Rodríguez 2149 ; CR, MO) Crepis … se
                # caracteriza"). Without this the flowering months and range were
                # filed as discussion.
                v = _VOUCHER.search(block, 0, 700)
                if v and not _DISCUSSION_CUE.search(block[:v.start()]):
                    current.distribution_paragraph = _append(
                        current.distribution_paragraph, block[:v.end()].strip())
                    block = block[v.end():].strip()
                    if not block:
                        continue
            current.discussion = _append(current.discussion, block)

    for f in fichas:
        f.morphology, f.distribution_paragraph = _rejoin_split_range(
            f.morphology, f.distribution_paragraph)
        # A figure label (the species' own name) glued inside a block: "Cords. de
        # Tilarán, Central y Sanicula liberta de Talamanca" hid Cordillera Central from
        # the parser. The species never names itself inside these two sections.
        own_label = re.compile(r"\s*" + re.escape(f.species) + r"\s*")
        if f.morphology:
            f.morphology = dehyphenate(own_label.sub(" ", f.morphology).strip())
        if f.distribution_paragraph:
            f.distribution_paragraph = dehyphenate(
                own_label.sub(" ", f.distribution_paragraph).strip())
        if f.discussion:
            # a figure credit glued to the start of the discussion block:
            # "Scleranthus annuus Flora von Deutschland t. 201. 1886 Esta sp. …"
            credit = re.match(re.escape(f.species) + r"\s+(?:[A-Z(])[^()]{0,160}?\b" + _YEAR
                              + r"(?:[–-]\d{2,4})?(?:\[[^\]]*\])?\.?\s+(?=[A-ZÁÉÍÓÚ])",
                              f.discussion)
            if credit and not re.match(re.escape(f.species) + r"\s+(?:se|es)\b", f.discussion):
                f.discussion = f.discussion[credit.end():]
        f.discussion = dehyphenate(f.discussion) if f.discussion else None
    return _dedupe(fichas)


def _dedupe(fichas: list[RawFicha]) -> list[RawFicha]:
    """Collapse duplicate headers (running-footer repeats at genus transitions).

    Keep, per species, the entry with a distribution paragraph, then with a
    description; block count only breaks the remaining ties. Ranking on block count
    first favoured a duplicate that had swallowed a neighbour's text over the real
    entry. Drop empties (a footer that matched a header but captured nothing).
    """
    best: dict[str, RawFicha] = {}

    def rank(f: RawFicha):
        return bool(f.distribution_paragraph), bool(f.morphology), len(f.blocks)

    for f in fichas:
        if not f.blocks and not f.distribution_paragraph:
            continue
        cur = best.get(f.species)
        if cur is None or rank(f) > rank(cur):
            best[f.species] = f
    return list(best.values())


def _report(fichas: list[RawFicha]) -> None:
    n_dist = sum(1 for f in fichas if f.distribution_paragraph)
    print(f"{len(fichas)} fichas | {n_dist} with distribution paragraph "
          f"| {len(fichas) - n_dist} WITHOUT (missing-field cases)")
    for f in fichas:
        flag = "" if f.distribution_paragraph else "   <-- NO DIST PARAGRAPH"
        print(f"\n### {f.species}  (p.{f.page}){flag}")
        if f.distribution_paragraph:
            print(f"    {f.distribution_paragraph[:200]}")


if __name__ == "__main__":
    from .. import config
    entry = config.CORPUS[0]
    print(f"Segmenting {entry['family']} (Vol {entry['volume']})\n{entry['path']}\n")
    _report(segment(str(entry["path"]), volume=entry["volume"], family=entry["family"]))
