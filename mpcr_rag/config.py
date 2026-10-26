"""Configuration for MPCR-RAG: corpus manifest, models, paths.

Expand the corpus by appending entries to ``CORPUS`` — nothing else changes.
Each volume should be validated (Milestone 1-style spike) before it is added.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- repo / data roots ------------------------------------------------------
RAG_ROOT = Path(__file__).resolve().parent
DATA_DIR = RAG_ROOT / "data"
SQLITE_PATH = DATA_DIR / "fichas.sqlite"
FICHAS_JSON_DIR = DATA_DIR / "fichas_json"

MANUAL_ROOT = Path(
    r"C:\Users\Jose\Documents\Tesis\raw_data\Manual de Especies\Nelson"
    r"\wetransfer_manual-plantas-cr_2026-05-29_1701\Manual Plantas de Costa Rica"
)

# --- corpus manifest (auto-discovered) --------------------------------------
# Scan the validated-format volumes, skipping non-content PDFs (front matter,
# indices, identification keys). VII/VIII are excluded for now — different OCR/
# layout era (2014 MBOT/InDesign) that needs separate tuning.
_VOL_DIRS = {
    "MPCRv2 PDFs": "II", "MPCRv3 PDFs": "III",
    "Vol IV": "IV", "Vol V": "V", "Vol VI": "VI",
}
_SKIP = re.compile(
    r"\bFM\b|ix_xviii|vii_xvi|index|indices|clave|gimno|\.ps1|z\dmpcr", re.I
)


# Vol VI continuation files use abbreviated prefixes → map back to the real family.
_FAMILY_FIX = {
    "LauraPersea": "Lauraceae", "MalpStigmaphyllon": "Malpighiaceae",
    "MalvaModiola": "Malvaceae", "MelaTococa": "Melastomataceae",
    "MelastoMiconia": "Melastomataceae", "MelastomHenriettella": "Melastomataceae",
}


def _family_from_name(name: str) -> str:
    stem = re.sub(r"^\d+\s*", "", name.rsplit(".pdf", 1)[0])   # strip leading "01 "
    m = re.match(r"([A-Za-zé_]+)", stem)
    fam = m.group(1) if m else ""
    if fam.upper().startswith("MPCR"):
        return ""
    return _FAMILY_FIX.get(fam, fam)


def _discover_corpus() -> list[dict]:
    out: list[dict] = []
    for sub, vol in _VOL_DIRS.items():
        folder = MANUAL_ROOT / sub
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.pdf")):
            if _SKIP.search(p.name):
                continue
            out.append({"path": p, "volume": vol,
                        "family": _family_from_name(p.name) or f"Vol {vol}"})
    return out


CORPUS: list[dict] = _discover_corpus()

# --- embeddings / vector store ---------------------------------------------
# "pgvector" = local Postgres synced from SQLite; same e5-large model, run locally.
#              Default since 2026-09-13: passed eval/results/vector_parity.md
#              (self-retrieval 100% on both; filtered-query overlap@10 0.92).
# "pinecone" = hosted index the BIP paper was evaluated on (tag bip-2026-submission).
#              Stale after the 6,946-species rebuild (holds 5,791, old habits), so it
#              is only for reproducing the BIP numbers.
VECTOR_BACKEND = os.environ.get("MPCR_VECTOR_BACKEND", "pgvector")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX = "mpcr-fichas"
EMBED_MODEL = "multilingual-e5-large"   # Pinecone hosted inference
EMBED_DIM = 1024                        # e5-large; pin in methods section

# Same weights as Pinecone's hosted multilingual-e5-large, so the two backends are
# comparable (vector_parity.py measures how closely).
LOCAL_EMBED_MODEL = os.environ.get("MPCR_LOCAL_EMBED_MODEL", "intfloat/multilingual-e5-large")

# pgvector connection. With MPCR_PG_URL unset, an embedded Postgres (pgserver) is
# started on MPCR_PGDATA. The data dir must NOT live under the Drive-synced repo:
# sync + a live database = lock conflicts and corruption.
PG_URL = os.environ.get("MPCR_PG_URL", "")
PGDATA = Path(os.environ.get(
    "MPCR_PGDATA", Path.home() / "Documents" / "Tesis" / "pgdata" / "mpcr"))
# Fixed so GUI clients (VS Code, DBeaver, pgAdmin) keep a working saved connection;
# pgserver alone picks a random port on every start on Windows. 5432 is left free for
# a system-wide Postgres.
PG_PORT = int(os.environ.get("MPCR_PG_PORT", "5433"))

# --- GBIF points cache (query/gbif_cache.py, RR_56) --------------------------
# Days before a species' cached GBIF points are fetched again. 0 = never: the cache
# stays frozen, as the paper's experiments assume. The web app sets it (e.g. 30).
GBIF_CACHE_MAX_DAYS = float(os.environ.get("MPCR_GBIF_CACHE_MAX_DAYS") or 0)
# After a failed refresh, hours before GBIF is asked again for that species.
GBIF_RETRY_FAILED_HOURS = float(os.environ.get("MPCR_GBIF_RETRY_FAILED_HOURS") or 1)

# --- LLM enrichment (reuses CR-BioLM OpenRouter setup) ----------------------
ENRICH_MODEL = os.environ.get("MPCR_ENRICH_MODEL", "openai/gpt-4o-mini")

DATA_DIR.mkdir(exist_ok=True)
FICHAS_JSON_DIR.mkdir(exist_ok=True)
