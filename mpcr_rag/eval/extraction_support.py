"""Textual-support audit of the structured extraction.

Coverage says whether a field was populated; it says nothing about whether the value
is right. This audit asks a narrower, checkable question for every indexed entry:
is each extracted value literally supported by the distribution paragraph it came
from?

It is an automated consistency check, not expert annotation. It catches values the
parser invented or mis-assigned; it cannot catch a value that is textually supported
but botanically wrong.

Run (use the build the paper reports, NOT the current one):
    python -m mpcr_rag.eval.extraction_support \
        --db mpcr_rag/data/fichas.pre_trackB_20260912.sqlite
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import unicodedata
from pathlib import Path


def _norm(s: str) -> str:
    """Lowercase, strip accents, collapse dashes and whitespace."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[‐-―−]", "-", s)
    return re.sub(r"\s+", " ", s).lower()


# Slope cues. "ambas verts." licenses either slope (see geo_parser).
_BOTH = re.compile(r"ambas\s+verts?\.?", re.I)
_CARIB = re.compile(r"vert\w*\.?\s*carib|carib(?:e|ena|eno)\b", re.I)
_PACIF = re.compile(r"vert\w*\.?\s*pac|pacif(?:ico|ica)\b", re.I)
_ENDEM = re.compile(r"endemic", re.I)          # matches ENDEMICA / ENDÉMICA once normed
_INT = re.compile(r"\d+")

# NOTE: botanical regions are deliberately NOT audited here. They are canonicalised
# against a gazetteer, so the stored value routinely differs from the source wording
# ("region de Golfo Dulce" -> "Peninsula de Osa - Golfito", "Cords. de Guanacaste y
# Central" -> two separate canonical names). Literal string support is therefore the
# wrong test: it reports synonym mappings as failures. Auditing region assignment
# needs the gazetteer's synonym table, which is out of scope for this check.


def audit_one(f: dict) -> dict:
    text = f.get("distribution_paragraph") or ""
    t = _norm(text)
    nums = set(_INT.findall(t))

    # --- elevation: every parsed bound must appear as an integer token in the text
    elev_vals = [f.get("elev_min"), f.get("elev_max"),
                 f.get("elev_outlier_min"), f.get("elev_outlier_max")]
    elev_vals = [v for v in elev_vals if v is not None]
    elev_ok = all(str(int(v)) in nums for v in elev_vals) if elev_vals else None

    # --- slope: each recorded slope needs its own cue, or an "ambas verts." licence
    verts = f.get("vertientes") or []
    if verts:
        both = bool(_BOTH.search(t))
        slope_ok = all(
            both or (_CARIB.search(t) if v.startswith("Carib") else _PACIF.search(t))
            for v in verts
        )
        slope_ok = bool(slope_ok)
    else:
        slope_ok = None

    # --- endemism: the flag must agree with the ENDEMICA marker
    endem_flag = bool(f.get("endemic_cr"))
    endem_ok = (endem_flag == bool(_ENDEM.search(t)))

    return {"species": f.get("species"), "family": f.get("family"),
            "elev_ok": elev_ok, "slope_ok": slope_ok,
            "endem_ok": endem_ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="mpcr_rag/eval/results/extraction_support.csv")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    fichas = [json.loads(r[0]) for r in con.execute("SELECT ficha_json FROM fichas")]

    # This audit compares stored values against the source paragraph. The published
    # catalog has that paragraph stripped for copyright reasons, so refuse rather than
    # report a meaningless 0%: every value would look unsupported because there is no
    # text to support it against.
    if not any(f.get("distribution_paragraph") for f in fichas):
        raise SystemExit(
            "This catalog carries no distribution paragraphs, so textual support "
            "cannot be audited. "
            "Run it against a catalog built from the Manual PDFs "
            "(see README, 'Data not included')."
        )

    rows = [audit_one(f) for f in fichas]
    n = len(rows)

    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return "n/a"
        ok = sum(1 for v in vals if v)
        return f"{ok}/{len(vals)} = {100 * ok / len(vals):.1f}%"

    print(f"entries audited      : {n}")
    print(f"elevation supported  : {rate('elev_ok')}")
    print(f"slope supported      : {rate('slope_ok')}")
    print(f"endemism agrees      : {rate('endem_ok')}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"per-entry results -> {out}")


if __name__ == "__main__":
    main()
