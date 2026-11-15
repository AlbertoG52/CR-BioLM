"""Common names -> species, federated from existing providers.

Nothing here invents a lexicon. Names are collected from providers and stored in the
shape of the Darwin Core VernacularName extension (vernacularName, language,
countryCode, isPreferredName, source), so every candidate an answer rests on keeps its
provenance and can be exported as a checklist later.

Sources, in order of trust (country-specific first):
  MPCR       names printed in the Manual entry              -> cite Tomo and page
  INAT_CR    iNaturalist name preferred for Costa Rica      -> what people say today
  INAT_ES    other Spanish iNaturalist names                -> candidates
  GBIF       GBIF vernacular names (Spanish)                -> recall, no locality
  TROPICOS   Missouri Botanical Garden, REVERSE index only: Tropicos has no
             species -> names method, but /Name/Search accepts an undocumented
             ``commonname`` parameter. Global and with no country field, so it buys
             recall at the cost of locality ("nazareno" also returns Tibouchina
             urvilleana, which in Costa Rica is Peltogyne purpurea). Needs
             TROPICOS_API_KEY; a website account is not enough.

A common name is ambiguous by nature ("poro" is several Erythrina; "roble" is Quercus
and also Tabebuia rosea), so ``resolve`` returns RANKED CANDIDATES with provenance and
never silently picks one.

Run:  python -m mpcr_rag.evidence.vernacular fetch --pilot --expert
      python -m mpcr_rag.evidence.vernacular resolve "poro" [--region ...] [--elev 200]
      python -m mpcr_rag.evidence.vernacular report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import requests

from ..store import pg_store
from . import taxa

INAT_CR_PLACE = 6924                    # iNaturalist place id for Costa Rica
UA = {"User-Agent": "CR-BioLM thesis research"}
TROPICOS_KEY = os.environ.get("TROPICOS_API_KEY", "")

# Country-specific sources first. Tropicos sits with GBIF, not above: its common-name
# index is global and carries no country field, so it lists names used anywhere in the
# Americas ("nazareno" -> Tibouchina urvilleana, which in Costa Rica is Peltogyne
# purpurea). It buys recall, not locality.
TRUST = {"MPCR": 0, "INAT_CR": 1, "INAT_ES": 2, "GBIF": 3, "TROPICOS": 3}
CR_SPECIFIC = TRUST["INAT_CR"]        # trust <= this means "a Costa Rican source said so"
DOMINANCE_RATIO = 0.5                 # see resolve_decision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mpcr.vernacular (
    name_norm     text NOT NULL,        -- lowercase, unaccented, for lookup
    vernacular    text NOT NULL,        -- as published
    species       text NOT NULL,        -- Manual name
    language      text,                 -- ISO 639 where known
    country_code  text,                 -- 'CR' when the source says so
    is_preferred  boolean NOT NULL DEFAULT false,
    source        text NOT NULL,        -- MPCR | TROPICOS | INAT_CR | INAT_ES | GBIF
    source_detail text,                 -- dataset, taxon id, Tomo/page
    fetched       date NOT NULL DEFAULT current_date,
    PRIMARY KEY (name_norm, species, source)
);
CREATE INDEX IF NOT EXISTS vernacular_name ON mpcr.vernacular (name_norm);
CREATE INDEX IF NOT EXISTS vernacular_species ON mpcr.vernacular (species);
"""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------- providers

def from_inat(species: str, accepted: str | None = None) -> list[dict]:
    """iNaturalist names; the Costa Rica-preferred one is marked INAT_CR."""
    queries = [species] + ([accepted] if accepted and accepted != species else [])
    for query in queries:
        try:
            r = requests.get("https://api.inaturalist.org/v1/taxa",
                             params={"q": query, "rank": "species", "locale": "es",
                                     "preferred_place_id": INAT_CR_PLACE, "all_names": "true"},
                             headers=UA, timeout=30)
        except Exception:
            return []
        if not r.ok:
            continue
        hit = next((x for x in r.json().get("results", []) if x.get("name") == query), None)
        if not hit:
            continue
        out, preferred = [], hit.get("preferred_common_name")
        detail = f"iNat taxon {hit['id']}"
        if preferred:
            out.append({"vernacular": preferred, "language": "spa", "country_code": "CR",
                        "is_preferred": True, "source": "INAT_CR", "source_detail": detail})
        for n in hit.get("names") or []:
            if (n.get("lexicon") or "").lower() not in ("spanish", "espanol", "español", "es", "spa"):
                continue
            if preferred and norm(n.get("name")) == norm(preferred):
                continue
            out.append({"vernacular": n["name"], "language": "spa", "country_code": None,
                        "is_preferred": False, "source": "INAT_ES", "source_detail": detail})
        return out
    return []


_YEAR_CITE = re.compile(r"(?:1[6789]\d\d|20\d\d)(?:\[[^\]]*\])?\s*\.\s*")
_SYN_CUE = re.compile(r"[A-Z]\.\s|\bsensu\b|\bnon\b|\bvar\.|\bsubsp\.|\bnom\.|\bex\b|\d|&")
_AUTHOR_TAIL = re.compile(
    r"\b(Kunth|Lindl|Schltr|Pittier|Standl|Vahl|Jacq|Rudd|Urb|Mez|Donn|Sm|Griseb|DC|Benth|"
    r"Hook|Willd|Poepp|Endl|Miq|Sw|Poir|Rchb|Cogn|Klotzsch|Planch|Baker|Presl|Steud|Nees|"
    r"Aubl|Gaertn|Blume|Wedd|Seem|Oerst|Hemsl|Rose|Britton|Nash|Trel|Woodson|Croat|Grayum|"
    r"Hammel|Burger|Werff|G[oó]mez|Grimes|Pollard|Fl)\b\.?$")
_HYPHEN_SPLIT = re.compile(r"([a-záéíóúñ])-\s+([a-záéíóúñ])")
_SMALL_CAPS = re.compile(r"\b([A-ZÁÉÍÓÚÑ])\s+([A-ZÁÉÍÓÚÑ]{2,})")


def from_mpcr(ficha) -> list[dict]:
    """PROTOTYPE extractor for the names printed in the Manual header.

    The names sit after the last citation year, comma separated:
    "... Standl. Cortez , Cortez amarillo , Guayacan ."

    Measured on the full catalog: ~36% of species yield at least one name, and
    author surnames, flora abbreviations and synonyms still leak through. This is a
    placeholder so the MPCR layer exists end to end; the bachelor team's extractor,
    validated on a gold set (target precision >= 0.95), replaces this function
    without changing anything downstream.
    """
    head = (ficha.full_text or "").splitlines()[0] if ficha.full_text else ""
    head = _HYPHEN_SPLIT.sub(lambda m: m.group(1) + m.group(2), head)
    parts = _YEAR_CITE.split(head)
    if len(parts) < 2:
        return []
    # "A NTORCHA DE B RASIL" -> "Antorcha de Brasil" (small caps lost by the OCR)
    tail = _SMALL_CAPS.sub(lambda m: m.group(1) + m.group(2).lower(), parts[-1]).strip(" .")
    out = []
    for piece in re.split(r"\s*,\s*", tail):
        p = piece.strip(" .")
        if not p:
            continue
        if _SYN_CUE.search(p) or _AUTHOR_TAIL.search(p):
            # a real name can trail a synonym: "... non Kunth. Poro"
            p = re.split(r"(?<=[a-zA-Z])\.\s+", p)[-1].strip(" .")
            if not p or _SYN_CUE.search(p) or _AUTHOR_TAIL.search(p):
                continue
        p = re.sub(r"\s*[\[\]]\s*", " ", p).strip()
        if len(p) < 3 or len(p.split()) > 4 or not re.match(r"^[A-ZÁÉÍÓÚÑ]", p):
            continue
        out.append({"vernacular": p, "language": "spa", "country_code": "CR",
                    "is_preferred": False, "source": "MPCR",
                    "source_detail": f"Manual de Plantas de Costa Rica, "
                                     f"Tomo {ficha.volume}, p. {ficha.pages}"})
    return out


def from_gbif(taxon_key: int | None) -> list[dict]:
    if not taxon_key:
        return []
    from pygbif import species as gbif_species
    try:
        rows = gbif_species.name_usage(key=taxon_key, data="vernacularNames").get("results", [])
    except Exception:
        return []
    out, seen = [], set()
    for v in rows:
        if (v.get("language") or "").lower() not in ("spa", "es"):
            continue
        n = v.get("vernacularName")
        if not n or norm(n) in seen:
            continue
        seen.add(norm(n))
        out.append({"vernacular": n, "language": "spa", "country_code": v.get("country"),
                    "is_preferred": False, "source": "GBIF",
                    "source_detail": (v.get("source") or "")[:200]})
    return out


def tropicos_by_common_name(name: str) -> list[str]:
    """Scientific names Tropicos lists under a common name (reverse lookup).

    Tropicos has no species -> common names method: its documented API covers
    Search, Summary, Synonyms, AcceptedNames, Distributions, References, Images,
    ChromosomeCounts, HigherTaxa and Specimens. The web UI's common-name search is
    reachable through an UNDOCUMENTED ``commonname`` parameter on /Name/Search, and
    that is what this uses.

    Results are GLOBAL and include genus-rank rows, so the caller must intersect them
    with the catalog: "Poro" returns Allium porrum (the leek) among others, while
    "Cortez amarillo" returns exactly the three Handroanthus the Manual names.
    Needs TROPICOS_API_KEY (a website account is not enough).
    """
    if not TROPICOS_KEY or not name.strip():
        return []
    try:
        r = requests.get("https://services.tropicos.org/Name/Search",
                         params={"commonname": name.strip(), "apikey": TROPICOS_KEY,
                                 "format": "json"}, headers=UA, timeout=40)
        rows = r.json() if r.ok and r.text.strip().startswith("[") else []
    except Exception:
        return []
    out = []
    for x in rows:
        sci = (x.get("ScientificName") or "").strip()
        # species rank only: genus-level rows ("Tabebuia", "Cecropia") are not answers
        if sci and len(sci.split()) >= 2 and x.get("NameId"):
            out.append(sci)
    return out


def _catalog_index(conn) -> dict[str, str]:
    """Scientific name -> catalog species, including the accepted names of synonyms,
    so a Tropicos result under a current name still matches the Manual's name."""
    with conn.cursor() as cur:
        cur.execute("SELECT species FROM mpcr.fichas")
        idx = {s: s for (s,) in cur.fetchall()}
        cur.execute("SELECT species, accepted_name FROM mpcr.taxa WHERE accepted_name IS NOT NULL")
        for species, accepted in cur.fetchall():
            idx.setdefault(accepted, species)
    return idx


def expand_from_tropicos(names: list[str] | None = None, *, verbose: bool = True,
                         pause: float = 0.5) -> dict:
    """For every common name already in the store, ask Tropicos which other catalog
    species carry it, and record those as source=TROPICOS.

    This is how a name-keyed index becomes useful here: the Manual and iNaturalist
    give names per species; Tropicos gives species per name, which finds catalog
    species the other sources never associated with that name.
    """
    conn = pg_store.connect()
    cur = conn.cursor()
    cur.execute(_SCHEMA)
    conn.commit()
    if names is None:
        cur.execute("SELECT DISTINCT vernacular FROM mpcr.vernacular ORDER BY 1")
        names = [r[0] for r in cur.fetchall()]
    idx = _catalog_index(conn)
    stats = {"names": 0, "hits": 0, "new_rows": 0}
    for i, name in enumerate(names, 1):
        found = tropicos_by_common_name(name)
        matched = {idx[s] for s in found if s in idx}
        stats["names"] += 1
        stats["hits"] += bool(matched)
        for species in sorted(matched):
            cur.execute("""INSERT INTO mpcr.vernacular
                (name_norm, vernacular, species, language, country_code, is_preferred,
                 source, source_detail)
                VALUES (%s,%s,%s,'spa',NULL,false,'TROPICOS',%s)
                ON CONFLICT DO NOTHING""",
                        (norm(name), name, species,
                         "Tropicos /Name/Search?commonname (global index)"))
            stats["new_rows"] += cur.rowcount
        conn.commit()
        if verbose and (i % 25 == 0 or matched):
            print(f"  [{i}/{len(names)}] {name:28} Tropicos={len(found):3} "
                  f"in catalog={len(matched)}", flush=True)
        time.sleep(pause)
    conn.close()
    return stats


# ----------------------------------------------------------------------- store

def fetch(species_list: list[str], verbose: bool = True, mpcr: bool = True) -> dict:
    tx = taxa.load()
    conn = pg_store.connect()
    cur = conn.cursor()
    cur.execute(_SCHEMA)
    conn.commit()
    stats = {"species": 0, "rows": 0, "with_cr_preferred": 0, "no_names": 0}
    for i, sp in enumerate(species_list, 1):
        t = tx.get(sp) or {}
        rows = from_inat(sp, t.get("accepted_name")) + from_gbif(t.get("taxon_key"))
        if mpcr:
            from ..store import local_store
            from .. import config as _cfg
            f = local_store.get(local_store.connect(_cfg.SQLITE_PATH), sp.replace(" ", "_"))
            if f:
                rows = from_mpcr(f) + rows
        for r in rows:
            cur.execute("""INSERT INTO mpcr.vernacular
                (name_norm, vernacular, species, language, country_code, is_preferred,
                 source, source_detail)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (norm(r["vernacular"]), r["vernacular"], sp, r["language"],
                         r["country_code"], r["is_preferred"], r["source"], r["source_detail"]))
        conn.commit()
        stats["species"] += 1
        stats["rows"] += len(rows)
        stats["no_names"] += not rows
        pref = next((r["vernacular"] for r in rows if r["source"] == "INAT_CR"), None)
        stats["with_cr_preferred"] += bool(pref)
        if verbose:
            print(f"  [{i:3}/{len(species_list)}] {sp:34} {len(rows):3} names | "
                  f"CR-preferred: {pref or '-'}", flush=True)
        time.sleep(1.0)            # iNaturalist asks for about one request per second
    conn.close()
    return stats


# -------------------------------------------------------------------- resolver

def resolve(name: str, *, conn=None, region: str | None = None,
            elev: int | None = None, limit: int = 8) -> list[dict]:
    """Ranked candidate species for a common name, with provenance.

    Ranking: source trust first (the Manual and the Costa Rica-preferred iNaturalist
    name beat a generic Spanish name), then how often the species is actually recorded
    in Costa Rica. Constraints taken from the question (region, elevation) filter the
    candidates when that leaves any. Never returns a single species silently.
    """
    own = conn is None
    conn = conn or pg_store.connect()
    sql = """
        SELECT v.species,
               bool_or(v.is_preferred)            AS preferred,
               array_agg(DISTINCT v.source)       AS sources,
               max(v.vernacular)                  AS as_published,
               f.ficha->>'family'                 AS family,
               (f.ficha->>'elev_min')::int        AS elev_min,
               (f.ficha->>'elev_max')::int        AS elev_max,
               f.regions                          AS regions,
               e.n_records                        AS n_records
        FROM mpcr.vernacular v
        JOIN mpcr.fichas f ON f.species = v.species
        LEFT JOIN mpcr.species_evidence e ON e.species = v.species
        WHERE {clause}
        GROUP BY v.species, f.ficha, f.regions, e.n_records"""
    try:
        with conn.cursor() as cur:
            # Always unaccented (RR_35): "jicaro" and "jícaro" must give the same
            # candidates. Matching the accented spelling first let a single unaccented
            # row ("Jicaro", one species) hide the other four "Jícaro". The JOIN with
            # mpcr.fichas keeps only catalog species, so the leek that "poro" names in
            # Mexico (the reason for the old accent-sensitive pass) never shows up.
            cur.execute(sql.format(clause="v.name_norm = %s"), (norm(name),))
            cols = [c.name for c in cur.description]
            typed = name.strip().lower()
            cands = []
            for r in cur.fetchall():
                c = dict(zip(cols, r))
                c["match"] = "exact" if (c["as_published"] or "").lower() == typed else "unaccented"
                cands.append(c)
    finally:
        if own:
            conn.close()

    for c in cands:
        c["trust"] = min(TRUST.get(s, 9) for s in c["sources"])
    if region:
        kept = [c for c in cands if region in (c["regions"] or [])]
        cands = kept or cands
    if elev is not None:
        kept = [c for c in cands if c["elev_min"] is not None
                and c["elev_min"] - 200 <= elev <= (c["elev_max"] or 0) + 200]
        cands = kept or cands
    cands.sort(key=lambda c: (c["trust"], -(c["n_records"] or 0)))
    return cands[:limit]


def resolve_decision(name: str, **kw) -> dict:
    """resolve() plus an explicit decision the caller must act on.

      none        the name is not in the store -> say so, do not guess
      unique      one candidate -> proceed, but name the species in the answer
      preferred   several candidates, but exactly one comes from a Costa Rica-specific
                  authority (Manual, Tropicos CR, iNaturalist CR-preferred) AND is at
                  least as well recorded here as the others -> proceed with it, saying
                  which and why. Both conditions are needed: iNaturalist marks "Ceiba"
                  as the Costa Rica name of Spirotheca rosea (76 records) because
                  Ceiba pentandra (200 records) is filed under the indigenous "Shkuli",
                  so trust alone would pick the less likely plant.
      ambiguous   several equally plausible -> ask back or answer for the group; never
                  pick silently

    Candidates are always species of THIS catalog: a global name index answers a
    different question (Tropicos returns Allium porrum, the leek, for "Poro").
    """
    cands = resolve(name, **kw)
    if not cands:
        return {"decision": "none", "name": name, "candidates": []}
    if len(cands) == 1:
        return {"decision": "unique", "name": name, "species": cands[0]["species"],
                "candidates": cands}
    best, rest = cands[0], cands[1:]
    best_records = best["n_records"] or 0
    # A Costa Rica-specific name beats candidates that only a global index lists,
    # unless such a rival is far better recorded here (twice as many records), which
    # is the Ceiba case: iNaturalist calls Spirotheca rosea (76 records) "Ceiba"
    # because Ceiba pentandra (200) sits under the indigenous "Shkuli".
    if (best["trust"] <= CR_SPECIFIC
            and all(c["trust"] > best["trust"] for c in rest)
            and best_records >= DOMINANCE_RATIO * max((c["n_records"] or 0) for c in rest)):
        return {"decision": "preferred", "name": name, "species": best["species"],
                "candidates": cands}
    return {"decision": "ambiguous", "name": name, "candidates": cands}


def names_for(species: str, conn=None) -> list[str]:
    """Common names stored for one catalog species, best first.

    Order: names more sources agree on; then names that point to fewer species
    (an ambiguous "Jícaro" goes after "Olla de mono"); then names printed in the
    Manual; then the one iNaturalist prefers for Costa Rica, which is often an
    indigenous name ("Kós", "Shkulí"). One spelling per name, the Manual's when it
    has it. Empty when the table is missing or the database is down: callers
    treat common names as optional.
    """
    own = conn is None
    try:
        conn = conn or pg_store.connect()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT (array_agg(v.vernacular ORDER BY v.source = 'MPCR' DESC,
                                                        v.vernacular))[1]
                FROM mpcr.vernacular v
                WHERE v.species = %s
                GROUP BY v.name_norm
                ORDER BY count(DISTINCT v.source) DESC,
                         (SELECT count(DISTINCT w.species) FROM mpcr.vernacular w
                          WHERE w.name_norm = v.name_norm) ASC,
                         bool_or(v.source = 'MPCR') DESC,
                         bool_or(v.is_preferred AND v.country_code = 'CR') DESC,
                         min(v.vernacular)""", (species,))
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        if own and conn is not None:
            conn.close()


def report(conn=None) -> None:
    conn = conn or pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT name_norm), count(DISTINCT species) "
                    "FROM mpcr.vernacular")
        rows, names, sp = cur.fetchone()
        print(f"vernacular: {rows} rows, {names} distinct names, {sp} species")
        cur.execute("SELECT source, count(*) FROM mpcr.vernacular GROUP BY 1 ORDER BY 2 DESC")
        print("  by source:", dict(cur.fetchall()))
        cur.execute("""SELECT name_norm, count(DISTINCT species) n, array_agg(DISTINCT species) sp
                       FROM mpcr.vernacular GROUP BY 1 HAVING count(DISTINCT species) > 1
                       ORDER BY n DESC, 1 LIMIT 12""")
        amb = cur.fetchall()
        print(f"  ambiguous names (within the fetched species): {len(amb)}")
        for n, k, sps in amb:
            print(f"    {n:24} -> {k}: {', '.join(sorted(sps)[:4])}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--pilot", action="store_true", help="the 18 pilot species")
    f.add_argument("--expert", action="store_true", help="the 21 expert-validated species")
    f.add_argument("--species-file")
    r = sub.add_parser("resolve")
    r.add_argument("name")
    r.add_argument("--region")
    r.add_argument("--elev", type=int)
    sub.add_parser("report")
    a = ap.parse_args()

    if a.cmd == "fetch":
        names: list[str] = []
        if a.pilot:
            s = json.loads((Path(__file__).resolve().parents[2] / "benchmark" / "data"
                            / "species_sample.json").read_text(encoding="utf-8"))
            names += [x["species"] for x in s["pilot"]]
        if a.expert:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from utils.compare_expert_maps import SPECIES
            names += list(SPECIES)
        if a.species_file:
            names += [l.strip() for l in Path(a.species_file).read_text(encoding="utf-8").splitlines()
                      if l.strip()]
        seen: set[str] = set()
        names = [n for n in names if not (n in seen or seen.add(n))]
        print(json.dumps(fetch(names), ensure_ascii=False))
    elif a.cmd == "resolve":
        for c in resolve(a.name, region=a.region, elev=a.elev):
            print(f"  {c['species']:32} {(c['family'] or ''):18} "
                  f"[{c['elev_min']}-{c['elev_max']} m] CR={c['n_records'] or 0:5} "
                  f"via {','.join(sorted(c['sources']))}{'  (CR-preferred)' if c['preferred'] else ''}")
    else:
        report()
