"""GBIF 'most likely' map for Pattern B.

For a geospatial question, render the real GBIF occurrence points of the matching
species, filtered through the SAME constraints the map pipeline applies — altitude
(DEM), botanical region, and vertiente. The result answers "where would I actually
find these?" with occurrence evidence, not keyword guesses.

Reuses CR-BioLM machinery: GBIFExtractor (fetch), extraer_altitud (DEM elevation),
and the regiones-botánicas shapefile (which carries Nombre + Vertiente) for the
region/vertiente spatial join. GBIF points are cached per species on disk
(live API → reproducible + fast re-runs).
"""
from __future__ import annotations

import json
import unicodedata
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pygbif import occurrences, species as gbif_species

import config as cr_config                       # repo-root config (DEM, shapefiles)
from utils.distribution_map.parser import build_ficha
from utils.distribution_map.renderer import generate_distribution_map
from utils.geoprocesamiento import extraer_altitud

# Costa Rica bounding box (padded) — for lat/lon-swap and gross-error detection.
_CR_BOX = (-86.0, 7.9, -82.5, 11.3)   # (lon_min, lat_min, lon_max, lat_max)
_MAX_UNCERTAINTY_M = 10_000           # drop records less precise than ~10 km

from .. import config
from ..schema import Ficha

_CACHE = config.DATA_DIR / "gbif_cache"
_CACHE.mkdir(exist_ok=True)
_MAPS = config.DATA_DIR / "maps"
_MAPS.mkdir(exist_ok=True)

# Frozen, citable GBIF download snapshot (CR Tracheophyta, hasCoordinate=true,
# hasGeospatialIssue=false) — DOI 10.15468/dl.8yhee8, see mpcr_rag/paper/README.md.
# Used as the source for occurrence COUNTS (ranking many candidates), instead of
# hitting the live occurrence-search API once per candidate.
_SNAPSHOT_DIR = config.DATA_DIR / "gbif_snapshot"
_SNAPSHOT_DIR.mkdir(exist_ok=True)
_SNAPSHOT_KEY = "0048944-260519110011954"
_SNAPSHOT_COUNTS_CACHE = _SNAPSHOT_DIR / "species_counts.json"

_REGIONS_4326: gpd.GeoDataFrame | None = None
_SPECIES_COUNTS: dict[str, int] | None = None


def _regions() -> gpd.GeoDataFrame:
    global _REGIONS_4326
    if _REGIONS_4326 is None:
        g = gpd.read_file(cr_config.REGIONES_BOTANICAS_SHP).to_crs(4326)
        _REGIONS_4326 = g[["Nombre", "Vertiente", "geometry"]]
    return _REGIONS_4326


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).lower()


def resolve_taxon(species: str) -> tuple[int | None, str | None, bool]:
    """Resolve a Manual name to its GBIF ACCEPTED taxonKey + canonical name.

    Returns (taxon_key, accepted_name, is_synonym). Free-text scientificName search
    is unreliable (fuzzy, lumps synonyms) — we key on the accepted backbone taxon.
    """
    try:
        nb = gbif_species.name_backbone(scientificName=species)
    except Exception:
        return None, None, False
    if "usage" not in nb:
        return None, None, False
    # Require an EXACT backbone match — rejects fuzzy/higher-rank hits and unnamed
    # morphospecies ("Genus sp. 4"), which otherwise resolve to bogus taxa.
    if nb.get("diagnostics", {}).get("matchType") != "EXACT":
        return None, None, False
    if nb.get("synonym") and "acceptedUsage" in nb:
        acc = nb["acceptedUsage"]
        return int(acc["key"]), acc.get("canonicalName"), True
    return int(nb["usage"]["key"]), nb["usage"].get("canonicalName"), False


def _fetch_clean(species: str) -> gpd.GeoDataFrame:
    """CR occurrences for a species via accepted taxonKey, coordinate-cleaned.

    Filters: GBIF hasGeospatialIssue=False; drops zero/near-zero coords, lat/lon
    swaps (point outside CR but swap inside), and records coarser than ~10 km.
    """
    key, _accepted, _syn = resolve_taxon(species)
    cols = ["lon", "lat", "geometry"]
    if key is None:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs="EPSG:4326")

    recs: list[dict] = []
    for offset in range(0, 900, 300):                       # cap ~900 records
        r = occurrences.search(taxonKey=key, country="CR", hasCoordinate=True,
                               hasGeospatialIssue=False, limit=300, offset=offset)
        recs.extend(r.get("results", []))
        if offset + 300 >= r.get("count", 0):
            break

    lo_x, lo_y, hi_x, hi_y = _CR_BOX
    rows = []
    for rec in recs:
        lon, lat = rec.get("decimalLongitude"), rec.get("decimalLatitude")
        if lon is None or lat is None:
            continue
        if abs(lon) < 0.01 and abs(lat) < 0.01:             # (0,0)
            continue
        unc = rec.get("coordinateUncertaintyInMeters")
        if unc is not None and unc > _MAX_UNCERTAINTY_M:
            continue
        in_box = lo_x <= lon <= hi_x and lo_y <= lat <= hi_y
        if not in_box:
            if lo_x <= lat <= hi_x and lo_y <= lon <= hi_y:  # lat/lon swapped → fix
                lon, lat = lat, lon
            else:
                continue                                     # gross out-of-CR error
        rows.append({"lon": lon, "lat": lat})

    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                            crs="EPSG:4326")


def _load_snapshot_counts() -> dict[str, int]:
    """Lazily fetch (once) and cache the frozen download's per-species occurrence
    counts. The zip (~155 MB, 1.01M CR Tracheophyta records) is downloaded once and
    kept under data/gbif_snapshot/ (gitignored); only the small aggregated JSON
    (species -> count) is reloaded on subsequent runs."""
    global _SPECIES_COUNTS
    if _SPECIES_COUNTS is not None:
        return _SPECIES_COUNTS
    if _SNAPSHOT_COUNTS_CACHE.exists():
        _SPECIES_COUNTS = json.loads(_SNAPSHOT_COUNTS_CACHE.read_text(encoding="utf-8"))
        return _SPECIES_COUNTS

    zip_path = _SNAPSHOT_DIR / f"{_SNAPSHOT_KEY}.zip"
    if not zip_path.exists():
        occurrences.download_get(_SNAPSHOT_KEY, path=str(_SNAPSHOT_DIR))

    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh, sep="\t", usecols=["species"], dtype=str, on_bad_lines="skip")
    counts = df["species"].value_counts().to_dict()
    _SNAPSHOT_COUNTS_CACHE.write_text(json.dumps(counts, ensure_ascii=False), encoding="utf-8")
    _SPECIES_COUNTS = counts
    return counts


def count_only(species: str) -> int:
    """CR occurrence count from the frozen, citable GBIF download snapshot (DOI
    10.15468/dl.8yhee8) — for ranking many candidates cheaply (the superlative
    'gbif_count' selector). Same cleaning level as the live API would give
    (hasCoordinate/hasGeospatialIssue flags only, no extra local cleaning) — just a
    local lookup instead of one occurrence-search call per candidate, and pinned to
    a reproducible snapshot instead of a live, growing database.
    """
    _key, accepted, _syn = resolve_taxon(species)
    name = accepted or species
    return int(_load_snapshot_counts().get(name, 0))


def get_points(species: str) -> gpd.GeoDataFrame:
    """Cached CR GBIF points for a species, tagged with Altitud + Nombre + Vertiente."""
    cache_file = _CACHE / f"{species.replace(' ', '_')}.geojson"
    if cache_file.exists():
        return gpd.read_file(cache_file)

    pts = _fetch_clean(species)
    if pts.empty:
        pts = gpd.GeoDataFrame({"lon": [], "lat": [], "Altitud": [], "Nombre": [],
                                "Vertiente": [], "species": []},
                               geometry=[], crs="EPSG:4326")
    else:
        pts = extraer_altitud(pts, cr_config.DEM_PATH)        # adds 'Altitud'
        pts = gpd.sjoin(pts, _regions(), how="left", predicate="within")
        pts = pts.drop(columns=[c for c in ("index_right",) if c in pts.columns])
        pts["species"] = species
    pts.to_file(cache_file, driver="GeoJSON")
    return pts


def filter_points(pts: gpd.GeoDataFrame, *, elev_lo=None, elev_hi=None,
                  vertiente=None, region=None) -> gpd.GeoDataFrame:
    """Apply the map-pipeline constraints to occurrence points."""
    if pts.empty:
        return pts
    m = pd.Series(True, index=pts.index)
    if elev_lo is not None:
        m &= pts["Altitud"].notna() & (pts["Altitud"] >= elev_lo)
    if elev_hi is not None:
        m &= pts["Altitud"].notna() & (pts["Altitud"] <= elev_hi)
    if vertiente:
        vkey = _norm(vertiente)[:4]                            # 'cari' / 'paci'
        m &= pts["Vertiente"].apply(lambda v: vkey in _norm(v))
    if region:
        rkey = _norm(region)
        m &= pts["Nombre"].apply(lambda n: rkey in _norm(n) or _norm(n) in rkey)
    return pts[m]


def most_likely_map(species_scores: list[tuple[Ficha, float]], *, query_text: str,
                    out_path: Path | None = None, **constraints) -> tuple[Path, int]:
    """Render the filtered GBIF occurrences of the matching species onto CR."""
    frames = []
    for f, _score in species_scores:
        pts = filter_points(
            get_points(f.species),
            elev_lo=constraints.get("elev_lo"), elev_hi=constraints.get("elev_hi"),
            vertiente=constraints.get("vertiente"), region=constraints.get("region"),
        )
        if not pts.empty:
            frames.append(pts.assign(species=f.species))

    regions = _regions()
    fig, ax = plt.subplots(figsize=(8, 9))
    regions.plot(ax=ax, color="#eeeeee", edgecolor="#cccccc", linewidth=0.4)

    n_pts = 0
    if frames:
        allpts = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
        n_pts = len(allpts)
        species_list = sorted(allpts["species"].unique())
        cmap = plt.get_cmap("tab20", max(1, len(species_list)))
        for i, sp in enumerate(species_list):
            sub = allpts[allpts["species"] == sp]
            sub.plot(ax=ax, color=cmap(i), markersize=18, alpha=0.75,
                     edgecolor="black", linewidth=0.2, label=f"{sp} (n={len(sub)})")
        ax.legend(fontsize=6, loc="lower left", framealpha=0.9, title="Especies")

    band = []
    lo, hi = constraints.get("elev_lo"), constraints.get("elev_hi")
    if lo is not None and hi is not None:
        band.append(f"{lo}–{hi} m")
    elif lo is not None:
        band.append(f"≥{lo} m")
    elif hi is not None:
        band.append(f"≤{hi} m")
    if constraints.get("vertiente"):
        band.append(f"vert. {constraints['vertiente']}")
    if constraints.get("region"):
        band.append(constraints["region"])
    # Source credit: the botanical-region base map is a digitization of the
    # phytogeographic map drawn by M. V. Castro for the Manual de Plantas de CR.
    fig.text(
        0.5, 0.01,
        "Fuentes: Regiones botánicas digitalizadas de M. V. Castro "
        "(Manual de Plantas de Costa Rica) · Presencias GBIF",
        ha="center", fontsize=5.5, color="#787878", transform=fig.transFigure,
    )
    ax.set_title(f"GBIF — presencias más probables\n{query_text}\n"
                 f"[{'  ·  '.join(band)}]  n={n_pts} puntos", fontsize=9)
    ax.set_axis_off()

    out_path = out_path or (_MAPS / f"mostlikely_{abs(hash(query_text)) % 10**8}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path, n_pts


def single_species_map(f: Ficha, *, out_path: Path | None = None) -> tuple[Path, int]:
    """Render the validated Manual-grounded distribution map for ONE named species.

    Re-derives the rich DistributionFicha (regions w/ hierarchy, parks, elevation
    range) from the same distribution paragraph stored at ingestion — no LLM, no
    re-ingest — and renders it with renderer.generate_distribution_map (the
    pipeline validated against Armando's expert maps), with GBIF points overlaid
    as the confirmatory layer. This is the Pattern-A map; Pattern B's multi-species
    "most likely" evidence map (above) is a deliberately different, GBIF-only view.
    """
    from ..ingest.field_extractor import split_distribution

    habitat_raw, geo_notes = split_distribution(f.distribution_paragraph)
    df = build_ficha(habitat_raw=habitat_raw, geographic_notes=geo_notes, species=f.species)
    pts = get_points(f.species)

    out_path = out_path or (_MAPS / f"manual_{f.vector_id}.png")
    generate_distribution_map(df, out_path, presencias_gdf=pts if not pts.empty else None)
    return out_path, len(pts)


if __name__ == "__main__":
    from .retriever import pattern_b, vector_index
    from ..store import local_store

    idx = vector_index()
    conn = local_store.connect(config.SQLITE_PATH)

    q = "especies endémicas sobre 2000 m en la vertiente Pacífico"
    res = pattern_b("bosque de altura, robledales", elev_lo=2000, vertiente="Pacífico",
                    endemic=True, conn=conn, index=idx)
    print(f"{len(res)} species matched:", [f.species for f, _ in res])
    path, n = most_likely_map(res, query_text=q, elev_lo=2000, vertiente="Pacífico")
    print(f"map → {path}  ({n} filtered GBIF points)")
