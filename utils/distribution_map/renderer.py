"""
renderer.py — Distribution map renderer (Phase 3).

Takes a DistributionFicha and produces a PNG using a narrowing strategy:
  country → vertiente → region → park → canton → district

The narrowest non-empty scope drives:
  - the clip geometry used for elevation masking
  - the map bounding box (zoom level)

Layers (bottom-to-top):
  1. Unmatched botanical regions          — dark gray
  2. Matched botanical regions            — muted per-region color
  3. Named parks from ficha              — orange fill
  4. Canton / district highlight         — indigo fill (if applicable)
  5. Elevation mask within narrowest scope — cyan
  6. Protected areas clipped to regions  — amber outline
  7. GBIF presence points                — red dots
  + Dual legend boxes + Fuentes footer
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from shapely.ops import unary_union
from pathlib import Path

from shapely.geometry import box
from .ficha import DistributionFicha, EntityRef
from .geodata import (
    load_regiones_botanicas, load_protected_areas,
    load_cantones, load_distritos, load_provincias, filter_pa_to_regions, PATHS,
)
from .gazetteer import _load_entities, normalize_text, lookup


# ---------------------------------------------------------------------------
# Entity → GDF resolution
# ---------------------------------------------------------------------------

def _entity_target(canonical_name: str) -> tuple[str, str, str] | None:
    """Return (target_shapefile, target_attribute, target_value) for an entity."""
    df = _load_entities()
    rows = df[df["canonical_name"] == canonical_name]
    if rows.empty:
        return None
    r = rows.iloc[0]
    ts = r["target_shapefile"].strip()
    ta = r["target_attribute"].strip()
    tv = r["target_value"].strip()
    return (ts, ta, tv) if (ts and ta and tv) else None


_LOADERS = {
    "regiones_botanicas": load_regiones_botanicas,
    "protected_areas":    load_protected_areas,
    "cantones":           load_cantones,
    "distritos":          load_distritos,
}



def _resolve_to_gdf(entities: list[EntityRef]) -> gpd.GeoDataFrame | None:
    """
    Resolve a list of EntityRef objects to their combined GeoDataFrame.
    Uses entities.csv to find (shapefile, attribute, value) for each entity.
    """
    gdfs = []
    for entity in entities:
        target = _entity_target(entity.canonical_name)
        if target is None:
            continue
        ts, ta, tv = target
        loader = _LOADERS.get(ts)
        if loader is None:
            continue
        source = loader()
        if ta not in source.columns:
            continue
        norm_tv = normalize_text(tv)
        mask = source[ta].apply(normalize_text) == norm_tv
        hit = source[mask].copy()
        if not hit.empty:
            gdfs.append(hit)
    if not gdfs:
        return None
    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return combined


# Qualifier map: single-letter → fraction of bounding box to keep
# N → top half (ymin..ymax), S → bottom half (ymin..ymid), etc.
_QUALIFIER_HALF: dict[str, str] = {
    "N": "north",
    "S": "south",
    "E": "east",
    "O": "west",
    "W": "west",
}

_REGION_SHAPEFILE_TYPES = frozenset({
    "cordillera", "llanura", "valle", "fila", "peninsula",
    "region_other", "region_informal",
})


def _clip_geom_to_half(geom, qualifier: str):
    """Clip a shapely geometry to the N/S/E/W half of its own bounding box."""
    side = _QUALIFIER_HALF.get(qualifier.upper())
    if side is None:
        return geom
    xmin, ymin, xmax, ymax = geom.bounds
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    if side == "north":
        clip = box(xmin, ymid, xmax, ymax)
    elif side == "south":
        clip = box(xmin, ymin, xmax, ymid)
    elif side == "east":
        clip = box(xmid, ymin, xmax, ymax)
    else:  # west
        clip = box(xmin, ymin, xmid, ymax)
    result = geom.intersection(clip)
    return result if not result.is_empty else geom


def _resolve_regions_with_qualifiers(
    ficha: DistributionFicha,
) -> gpd.GeoDataFrame | None:
    """
    Resolve matched botanical regions, applying N/S/E/O half-clipping when
    a geo_parser occurrence carries a direction qualifier (e.g. 'S Pen. de Nicoya').

    Two special rules:
    - When a qualified occurrence also has embedded_protected_areas, the region
      polygon is suppressed entirely — the park polygon already shows the exact
      location, so drawing the half-peninsula too is misleading.
    - The half-clip is further intersected with the Costa Rica land boundary
      (union of all regiones_botanicas) to avoid spilling into the ocean.

    Returns a GeoDataFrame in the same CRS as regiones_botanicas.
    If no locality_occurrences carry qualifiers, falls back to _resolve_to_gdf().
    """
    # Build a mapping: canonical_name → (qualifier, has_embedded_pa)
    qualifier_by_canonical: dict[str, str] = {}
    suppress_canonical: set[str] = set()

    # Only suppress when the region is a "pinpoint-able" feature where the park
    # polygon provides more precision than showing the half-region.
    # Broad features (llanura, cordillera) should still show — the park is an
    # additional highlight inside them, not a replacement.
    _SUPPRESSABLE_TYPES = frozenset({"peninsula", "fila", "valle", "region_other"})

    for occ in ficha.locality_occurrences:
        q = occ.get("qualifier")
        if not q:
            continue
        ftype = occ.get("feature_type", "")
        if ftype not in _REGION_SHAPEFILE_TYPES:
            continue
        fname = occ.get("feature_name", "")
        if not fname:
            continue
        has_ep = bool(occ.get("embedded_protected_areas"))
        hits = lookup(fname)
        for level_hits in hits.values():
            for h in level_hits:
                cn = h["canonical_name"]
                if cn not in qualifier_by_canonical:
                    qualifier_by_canonical[cn] = q
                if has_ep and ftype in _SUPPRESSABLE_TYPES:
                    suppress_canonical.add(cn)

    # No qualifiers found → plain resolution
    if not qualifier_by_canonical:
        return _resolve_to_gdf(ficha.regions)

    # Land boundary mask (CR-only, metric) to prevent ocean spill
    regiones = load_regiones_botanicas()
    land_metric = unary_union(regiones.to_crs("EPSG:5367").geometry)

    # Resolve each entity, applying clip or suppression
    gdfs = []
    for entity in ficha.regions:
        # Skip regions where the park polygon already gives precise location
        if entity.canonical_name in suppress_canonical:
            continue

        target = _entity_target(entity.canonical_name)
        if target is None:
            continue
        ts, ta, tv = target
        loader = _LOADERS.get(ts)
        if loader is None:
            continue
        source = loader()
        if ta not in source.columns:
            continue
        norm_tv = normalize_text(tv)
        mask = source[ta].apply(normalize_text) == norm_tv
        hit = source[mask].copy()
        if hit.empty:
            continue

        qualifier = qualifier_by_canonical.get(entity.canonical_name)
        if qualifier:
            hit_metric = hit.to_crs("EPSG:5367")
            clipped_geoms = []
            for g in hit_metric.geometry:
                # Ensure validity before clipping (avoids TopologyException)
                if not g.is_valid:
                    g = g.buffer(0)
                half = _clip_geom_to_half(g, qualifier)
                try:
                    land_fixed = land_metric if land_metric.is_valid else land_metric.buffer(0)
                    land_clipped = half.intersection(land_fixed)
                    clipped_geoms.append(land_clipped if not land_clipped.is_empty else half)
                except Exception:
                    clipped_geoms.append(half)
            hit_metric = hit_metric.copy()
            hit_metric["geometry"] = clipped_geoms
            hit = hit_metric.to_crs(source.crs)

        gdfs.append(hit)

    if not gdfs:
        return None
    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return combined


# ---------------------------------------------------------------------------
# Map bounds
# ---------------------------------------------------------------------------

_CR_BOUNDS = (-86.1, 7.9, -82.4, 11.3)   # (xmin, ymin, xmax, ymax) in EPSG:4326

# Minimum map window: always show at least this fraction of CR's lon/lat span.
# Prevents maps from zooming into a tiny corner when only one small region matches.
_CR_LON_SPAN = _CR_BOUNDS[2] - _CR_BOUNDS[0]   # ~3.7°
_CR_LAT_SPAN = _CR_BOUNDS[3] - _CR_BOUNDS[1]   # ~3.4°
_MIN_LON_SPAN = _CR_LON_SPAN * 0.55            # always show ≥55% of CR width
_MIN_LAT_SPAN = _CR_LAT_SPAN * 0.55            # always show ≥55% of CR height

_SCOPE_MARGIN = {
    "country":   0.03,
    "vertiente": 0.04,
    "region":    0.05,
    "park":      0.08,
    "canton":    0.12,
    "district":  0.15,
}


def _compute_bounds(clip_geom, scope: str) -> tuple[float, float, float, float]:
    return _CR_BOUNDS


# ---------------------------------------------------------------------------
# Elevation helpers
# ---------------------------------------------------------------------------

def _park_overlaps_elevation_range(
    park_gdf: gpd.GeoDataFrame,
    dem_path: Path,
    elev_min: float,
    elev_max: float,
) -> bool:
    """Return True if any DEM pixel within the park polygon falls in [elev_min, elev_max]."""
    try:
        with rasterio.open(dem_path) as src:
            dem_nodata = src.nodata if src.nodata is not None else -9999
            proj = park_gdf.to_crs(src.crs) if park_gdf.crs != src.crs else park_gdf
            geoms = [mapping(g) for g in proj.geometry if g is not None and g.is_valid]
            if not geoms:
                return False
            out_arr, _ = rio_mask(src, geoms, crop=True, nodata=dem_nodata)
            elev = out_arr[0].astype(float)
            elev[elev == dem_nodata] = np.nan
            return bool((~np.isnan(elev) & (elev >= elev_min) & (elev <= elev_max)).any())
    except Exception:
        return True   # on error, render rather than silently drop


# ---------------------------------------------------------------------------
# Elevation overlay
# ---------------------------------------------------------------------------

def _overlay_elevation(
    ax, clip_gdf: gpd.GeoDataFrame, dem_path: Path,
    elev_min: float, elev_max: float, alpha: float = 0.82,
) -> None:
    """Mask DEM to clip_gdf polygons; paint in-range pixels cyan."""
    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_nodata = src.nodata if src.nodata is not None else -9999

        proj = clip_gdf.to_crs(dem_crs) if clip_gdf.crs != dem_crs else clip_gdf
        geoms = [mapping(g) for g in proj.geometry if g is not None and g.is_valid]
        if not geoms:
            return
        try:
            out_arr, out_transform = rio_mask(src, geoms, crop=True, nodata=dem_nodata)
        except Exception as e:
            print(f"  [WARN] Elevation mask failed: {e}")
            return

        elev = out_arr[0].astype(float)
        elev[elev == dem_nodata] = np.nan

        h, w   = elev.shape
        rgba   = np.zeros((h, w, 4), dtype=np.float32)
        in_rng = (~np.isnan(elev)) & (elev >= elev_min) & (elev <= elev_max)
        if not in_rng.any():
            print(f"  [WARN] No DEM pixels in elevation range [{elev_min}–{elev_max}]")
            return
        rgba[in_rng] = [0.13, 0.85, 0.93, alpha]

        left   = out_transform.c
        top    = out_transform.f
        right  = left + out_transform.a * w
        bottom = top  + out_transform.e * h
        ax.imshow(rgba, extent=[left, right, bottom, top],
                  origin="upper", aspect="auto", zorder=5, interpolation="nearest")


def _overlay_elevation_outlier(
    ax, clip_gdf: gpd.GeoDataFrame, dem_path: Path,
    elev_min: float, elev_max: float,
    main_min: float, main_max: float,
) -> None:
    """Semi-transparent hatched overlay for outlier elevation pixels."""
    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_nodata = src.nodata if src.nodata is not None else -9999

        proj  = clip_gdf.to_crs(dem_crs) if clip_gdf.crs != dem_crs else clip_gdf
        geoms = [mapping(g) for g in proj.geometry if g is not None and g.is_valid]
        if not geoms:
            return
        try:
            out_arr, out_transform = rio_mask(src, geoms, crop=True, nodata=dem_nodata)
        except Exception as e:
            print(f"  [WARN] Outlier elevation mask failed: {e}")
            return

        elev = out_arr[0].astype(float)
        elev[elev == dem_nodata] = np.nan

        h, w = elev.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        in_outlier = (
            (~np.isnan(elev)) &
            (elev >= elev_min) & (elev <= elev_max) &
            ~((elev >= main_min) & (elev <= main_max))
        )
        if not in_outlier.any():
            return
        rgba[in_outlier] = [0.13, 0.85, 0.93, 0.38]

        left   = out_transform.c
        top    = out_transform.f
        right  = left + out_transform.a * w
        bottom = top  + out_transform.e * h
        ax.imshow(rgba, extent=[left, right, bottom, top],
                  origin="upper", aspect="auto", zorder=4, interpolation="nearest")

        try:
            plot_gdf = clip_gdf.to_crs("EPSG:4326") if clip_gdf.crs.to_epsg() != 4326 else clip_gdf
            union_geom = unary_union(plot_gdf.geometry)
            gpd.GeoDataFrame(geometry=[union_geom], crs="EPSG:4326").plot(
                ax=ax, facecolor="none", edgecolor="#22d3ee",
                linewidth=0.0, hatch="////", alpha=0.30, zorder=6,
            )
        except Exception as e:
            print(f"  [WARN] Outlier hatch failed: {e}")


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------

def _build_right_legend(
    ficha: DistributionFicha,
    matched_parks_gdf: gpd.GeoDataFrame | None,
    pa_filtered: gpd.GeoDataFrame | None,
    presencias_gdf,
    highlight_gdf: gpd.GeoDataFrame | None,
    scope: str,
    gbif_inferred_gdf=None,
) -> list:
    patches = []

    if scope not in ("country", "vertiente"):
        patches.append(mpatches.Patch(color="#2e3440", label="Fuera del rango geográfico"))
        patches.append(mpatches.Patch(
            facecolor="none", edgecolor="#e2e8f0", linewidth=1.0,
            label="Zona geográfica (rango altitudinal en cyan)",
        ))

    if ficha.elevation.has_data():
        elev_min = ficha.elevation.min_m
        elev_max = ficha.elevation.max_m
        patches.append(mpatches.Patch(
            color="#22d3ee", label=f"Hábitat óptimo ({int(elev_min)}–{int(elev_max)} m)",
        ))
        if ficha.elevation.outlier_min_m is not None:
            out_lo = ficha.elevation.outlier_min_m
            patches.append(mpatches.Patch(
                facecolor="#22d3ee", alpha=0.35, edgecolor="#22d3ee", linewidth=0.8,
                label=f"Reg. atípicos inf. ({int(out_lo)}–{int(elev_min)} m)", hatch="////",
            ))
        if ficha.elevation.outlier_max_m is not None:
            out_hi = ficha.elevation.outlier_max_m
            patches.append(mpatches.Patch(
                facecolor="#22d3ee", alpha=0.35, edgecolor="#22d3ee", linewidth=0.8,
                label=f"Reg. atípicos sup. ({int(elev_max)}–{int(out_hi)} m)", hatch="////",
            ))

    if presencias_gdf is not None and not presencias_gdf.empty:
        patches.append(mpatches.Patch(
            color="#ff4444", label=f"Presencias GBIF (n={len(presencias_gdf)})",
        ))

    if gbif_inferred_gdf is not None and not gbif_inferred_gdf.empty:
        n_inferred = len(gbif_inferred_gdf)
        patches.append(mpatches.Patch(
            facecolor="none", edgecolor="#fb923c", linewidth=1.0,
            linestyle="--",
            label=f"Región inferida por GBIF (n={n_inferred})",
        ))

    if matched_parks_gdf is not None and not matched_parks_gdf.empty:
        n_parks = len(ficha.parks)
        patches.append(mpatches.Patch(
            facecolor="none", edgecolor="#f97316", linewidth=1.0,
            label=f"Parques mencionados (n={n_parks})",
        ))

    if highlight_gdf is not None and not highlight_gdf.empty:
        if scope == "canton":
            label = f"Cantón(es) mencionado(s) (n={len(ficha.cantons)})"
        else:
            label = f"Distrito(s) mencionado(s) (n={len(ficha.districts)})"
        patches.append(mpatches.Patch(
            facecolor="none", edgecolor="#818cf8", linewidth=1.0,
            label=label,
        ))

    if pa_filtered is not None and not pa_filtered.empty:
        patches.append(mpatches.Patch(
            facecolor="none", edgecolor="#fbbf24", linewidth=0.8,
            label=f"Áreas Protegidas (n={len(pa_filtered)})",
        ))
    return patches


def _build_left_legend(
    ficha: DistributionFicha, matched_regions_gdf: gpd.GeoDataFrame | None,
) -> list:
    patches = []

    vert_colors = {"Caribe": "#29b6f6", "Pacífico": "#4caf50"}
    if ficha.vertientes:
        for v in ficha.vertientes:
            patches.append(mpatches.Patch(
                color=vert_colors.get(v, "#78909c"),
                label=f"▶ Vertiente del {v}",
            ))
    else:
        patches.append(mpatches.Patch(color="#4b5563", label="▶ Sin vertiente especificada"))

    if ficha.regions:
        shown = sorted({r.canonical_name for r in ficha.regions})[:8]
        for name in shown:
            patches.append(mpatches.Patch(
                facecolor="none", edgecolor="#e2e8f0", linewidth=1.0,
                label=f"  {name}",
            ))
        extra = len(set(r.canonical_name for r in ficha.regions)) - 8
        if extra > 0:
            patches.append(mpatches.Patch(
                facecolor="none", edgecolor="#6b7280", linewidth=0.8,
                label=f"  ... (+{extra} más)",
            ))
    return patches


# ---------------------------------------------------------------------------
# GBIF cluster flip
# ---------------------------------------------------------------------------

_GBIF_CLUSTER_MIN = 5   # minimum GBIF points in a region to flip it


def _gbif_inferred_regions(
    presencias_gdf,
    matched_regions_gdf: gpd.GeoDataFrame | None,
    regiones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    """
    Find botanical regions that contain ≥ _GBIF_CLUSTER_MIN GBIF occurrence
    points but were NOT already matched by the Manual text.

    Returns a GeoDataFrame of those extra regions (same CRS as regiones),
    or None if none qualify.
    """
    if presencias_gdf is None or presencias_gdf.empty:
        return None

    # Reproject GBIF points to match regiones CRS
    pts = presencias_gdf.to_crs(regiones.crs) if presencias_gdf.crs != regiones.crs else presencias_gdf

    # Spatial join: which botanical region does each point fall in?
    joined = gpd.sjoin(pts[["geometry"]], regiones[["geometry", "Nombre"]], how="left", predicate="within")

    # Count points per region name
    counts = joined.groupby("Nombre").size()
    qualifying = counts[counts >= _GBIF_CLUSTER_MIN].index.tolist()

    if not qualifying:
        return None

    # Remove any already matched by the Manual
    already_matched: set[str] = set()
    if matched_regions_gdf is not None and not matched_regions_gdf.empty:
        already_matched = set(matched_regions_gdf["Nombre"].dropna().tolist())

    new_names = [n for n in qualifying if n not in already_matched]
    if not new_names:
        return None

    result = regiones[regiones["Nombre"].isin(new_names)].copy()
    return result if not result.empty else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_distribution_map(
    ficha: DistributionFicha,
    output_path: Path | str,
    presencias_gdf=None,
) -> Path:
    """
    Render a distribution map for the given Ficha and save as PNG.

    Returns the output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load base shapefiles ──────────────────────────────────────────────
    regiones = load_regiones_botanicas()
    pa_all   = load_protected_areas()

    # ── Resolve entity geometries ─────────────────────────────────────────
    matched_regions_gdf   = _resolve_regions_with_qualifiers(ficha) if ficha.regions else None
    matched_parks_gdf     = _resolve_to_gdf(ficha.parks)      if ficha.parks     else None
    matched_cantons_gdf   = _resolve_to_gdf(ficha.cantons)    if ficha.cantons   else None
    matched_districts_gdf = _resolve_to_gdf(ficha.districts)  if ficha.districts else None

    # GBIF cluster flip: regions not in manual text but with ≥5 occurrence points
    gbif_inferred_gdf = _gbif_inferred_regions(presencias_gdf, matched_regions_gdf, regiones)

    # Clip park polygons to CR land boundary — removes marine-area extensions
    # (e.g. Cabo Blanco RNA has a 824 km² ocean polygon in the shapefile)
    if matched_parks_gdf is not None and not matched_parks_gdf.empty:
        land_union = unary_union(
            [g if g.is_valid else g.buffer(0) for g in regiones.geometry]
        )
        parks_metric = matched_parks_gdf.to_crs("EPSG:5367")
        land_metric  = unary_union(
            [g if g.is_valid else g.buffer(0)
             for g in regiones.to_crs("EPSG:5367").geometry]
        )
        clipped = parks_metric.geometry.apply(
            lambda g: (g if g.is_valid else g.buffer(0)).intersection(land_metric)
        )
        parks_metric = parks_metric.copy()
        parks_metric["geometry"] = clipped
        matched_parks_gdf = parks_metric.to_crs(matched_parks_gdf.crs)
        matched_parks_gdf = matched_parks_gdf[~matched_parks_gdf.geometry.is_empty]

    # Apply vertiente filter to botanical regions
    if matched_regions_gdf is not None and "vert_norm" in matched_regions_gdf.columns:
        if len(ficha.vertientes) == 1:
            vert = "carib" if ficha.vertientes[0] == "Caribe" else "pacifico"
            vf = matched_regions_gdf[matched_regions_gdf["vert_norm"] == vert]
            if not vf.empty:
                matched_regions_gdf = vf

    # ── Determine scope and clip geometry ─────────────────────────────────
    scope = ficha.effective_scope()

    if scope == "district" and matched_districts_gdf is not None:
        clip_gdf = matched_districts_gdf
    elif scope == "canton" and matched_cantons_gdf is not None:
        clip_gdf = matched_cantons_gdf
    elif scope == "park":
        # Clip to union of: matched regions + parks
        gdfs_for_clip = []
        if matched_regions_gdf is not None and not matched_regions_gdf.empty:
            gdfs_for_clip.append(matched_regions_gdf[["geometry"]])
        if matched_parks_gdf is not None and not matched_parks_gdf.empty:
            pk = matched_parks_gdf.to_crs(regiones.crs)[["geometry"]]
            gdfs_for_clip.append(pk)
        if gdfs_for_clip:
            clip_gdf = gpd.GeoDataFrame(
                pd.concat(gdfs_for_clip, ignore_index=True), crs=regiones.crs,
            )
        elif matched_parks_gdf is not None:
            clip_gdf = matched_parks_gdf
        else:
            clip_gdf = regiones
    elif scope == "region" and matched_regions_gdf is not None:
        clip_gdf = matched_regions_gdf
    elif scope == "vertiente" and ficha.vertientes:
        vert_gdfs = []
        for v in ficha.vertientes:
            vn = "carib" if v == "Caribe" else "pacifico"
            vert_gdfs.append(regiones[regiones["vert_norm"] == vn])
        clip_gdf = gpd.GeoDataFrame(pd.concat(vert_gdfs), crs=regiones.crs) if vert_gdfs else regiones
    else:
        clip_gdf = regiones

    # Ensure clip_gdf is in the same CRS as regiones for extent calculation
    if clip_gdf.crs != regiones.crs:
        clip_gdf = clip_gdf.to_crs(regiones.crs)
    # Make geometries valid before union (avoids TopologyException on bad shapefiles)
    valid_geoms = [g if g.is_valid else g.buffer(0) for g in clip_gdf.geometry]
    clip_geom = unary_union(valid_geoms)

    # ── Figure setup ──────────────────────────────────────────────────────
    xmin, ymin, xmax, ymax = _compute_bounds(clip_geom, scope)
    lon_span = xmax - xmin
    lat_span = ymax - ymin
    fig_w = 9
    fig_h = fig_w * (lat_span / lon_span) * 1.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.subplots_adjust(bottom=0.09)

    # ── Layer 0: CR country base fill ─────────────────────────────────────
    # The botanical-regions shapefile covers only ~93% of CR, leaving gaps that
    # appear as pure black background. Drawing the province polygons first
    # fills those gaps with the same dark-gray tone as unmatched regions.
    provincias_gdf = load_provincias()
    if provincias_gdf.crs != regiones.crs:
        provincias_gdf = provincias_gdf.to_crs(regiones.crs)
    provincias_gdf.plot(
        ax=ax, color="#2e3440", edgecolor="none", alpha=0.7, zorder=0,
    )

    # ── Layer 1: Unmatched regions (context) ──────────────────────────────
    if matched_regions_gdf is not None and not matched_regions_gdf.empty:
        matched_idx = set(matched_regions_gdf.index.tolist())
        unmatched_gdf = regiones[~regiones.index.isin(matched_idx)].copy()
    else:
        unmatched_gdf = regiones.copy()

    if not unmatched_gdf.empty:
        unmatched_gdf.plot(
            ax=ax, color="#2e3440", edgecolor="#4c566a",
            linewidth=0.4, alpha=0.7, zorder=1,
        )

    # ── Layer 2: Matched botanical regions — outline only ────────────────
    # The DEM elevation mask (Layer 4) provides the colored fill;
    # a bright outline here marks the boundary without adding visual noise.
    if matched_regions_gdf is not None and not matched_regions_gdf.empty:
        matched_regions_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#e2e8f0",
            linewidth=0.9, alpha=0.85, zorder=2,
        )

    # ── Layer 2.5: GBIF-inferred regions — dashed amber outline ─────────
    # Regions not mentioned in the Manual but supported by ≥5 GBIF points.
    # DEM mask is applied to these too (Layer 4); outline distinguishes them
    # from Manual-matched regions.
    if gbif_inferred_gdf is not None and not gbif_inferred_gdf.empty:
        gbif_plot = gbif_inferred_gdf.to_crs(regiones.crs) if gbif_inferred_gdf.crs != regiones.crs else gbif_inferred_gdf
        gbif_plot.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#fb923c",
            linewidth=1.1, alpha=0.80, zorder=3,
            linestyle="--",
        )

    # ── Layer 3: Named parks — only if they overlap the elevation sweet spot ──
    # Filter out parks where no DEM pixel falls in the species elevation range;
    # those add visual noise without informational value.
    if matched_parks_gdf is not None and not matched_parks_gdf.empty:
        dem_path_check = PATHS["dem"]
        if ficha.elevation.has_data() and dem_path_check.exists():
            keep_rows = []
            for i, row in matched_parks_gdf.iterrows():
                single = gpd.GeoDataFrame([row], crs=matched_parks_gdf.crs)
                if _park_overlaps_elevation_range(
                    single, dem_path_check,
                    ficha.elevation.min_m, ficha.elevation.max_m,
                ):
                    keep_rows.append(i)
            matched_parks_gdf = matched_parks_gdf.loc[keep_rows]

        if not matched_parks_gdf.empty:
            parks_plot = (
                matched_parks_gdf.to_crs(regiones.crs)
                if matched_parks_gdf.crs != regiones.crs else matched_parks_gdf
            )
            parks_plot.plot(
                ax=ax, facecolor="none", edgecolor="#f97316",
                linewidth=1.2, alpha=0.85, zorder=6,
            )

    # ── Layer 3.5: Canton / district highlight (indigo) ───────────────────
    highlight_gdf = None
    if scope == "canton" and matched_cantons_gdf is not None:
        highlight_gdf = (
            matched_cantons_gdf.to_crs(regiones.crs)
            if matched_cantons_gdf.crs != regiones.crs else matched_cantons_gdf
        )
        highlight_gdf.plot(
            ax=ax, facecolor="none", edgecolor="#818cf8",
            linewidth=1.2, alpha=0.85, zorder=6,
        )
    elif scope == "district" and matched_districts_gdf is not None:
        highlight_gdf = (
            matched_districts_gdf.to_crs(regiones.crs)
            if matched_districts_gdf.crs != regiones.crs else matched_districts_gdf
        )
        highlight_gdf.plot(
            ax=ax, facecolor="none", edgecolor="#818cf8",
            linewidth=1.2, alpha=0.85, zorder=6,
        )

    # ── Layer 4: Elevation mask within narrowest scope ────────────────────
    elev_mask_gdf = (
        highlight_gdf if highlight_gdf is not None
        else matched_parks_gdf if (matched_parks_gdf is not None and not matched_parks_gdf.empty)
        else matched_regions_gdf
    )
    if elev_mask_gdf is None:
        elev_mask_gdf = regiones

    # Merge parks + locality buffers into regions for elevation mask
    if (scope == "park" and matched_parks_gdf is not None
            and matched_regions_gdf is not None and not matched_regions_gdf.empty):
        parks_reproj = matched_parks_gdf.to_crs(matched_regions_gdf.crs)
        stub = parks_reproj[["geometry"]].assign(
            Nombre="", Vertiente="", vert_norm="pacifico"
        )
        elev_mask_gdf = gpd.GeoDataFrame(
            pd.concat([matched_regions_gdf, stub], ignore_index=True),
            crs=matched_regions_gdf.crs,
        )

    # Merge GBIF-inferred regions into elevation mask
    if gbif_inferred_gdf is not None and not gbif_inferred_gdf.empty:
        gbif_reproj = gbif_inferred_gdf.to_crs(elev_mask_gdf.crs)
        gbif_stub = gbif_reproj[["geometry"]].assign(
            Nombre="", Vertiente="", vert_norm=""
        )
        elev_mask_gdf = gpd.GeoDataFrame(
            pd.concat([elev_mask_gdf, gbif_stub], ignore_index=True),
            crs=elev_mask_gdf.crs,
        )

    dem_path = PATHS["dem"]
    if ficha.elevation.has_data() and dem_path.exists() and not elev_mask_gdf.empty:
        elev_min = ficha.elevation.min_m
        elev_max = ficha.elevation.max_m
        _overlay_elevation(ax=ax, clip_gdf=elev_mask_gdf, dem_path=dem_path,
                           elev_min=elev_min, elev_max=elev_max)

        if ficha.elevation.outlier_min_m is not None or ficha.elevation.outlier_max_m is not None:
            out_lo = ficha.elevation.outlier_min_m if ficha.elevation.outlier_min_m is not None else elev_min
            out_hi = ficha.elevation.outlier_max_m if ficha.elevation.outlier_max_m is not None else elev_max
            _overlay_elevation_outlier(
                ax=ax, clip_gdf=elev_mask_gdf, dem_path=dem_path,
                elev_min=out_lo, elev_max=out_hi,
                main_min=elev_min, main_max=elev_max,
            )

    # ── Layer 5: Protected areas clipped to matched regions, filtered by elevation ──
    pa_filtered = None
    if matched_regions_gdf is not None and not matched_regions_gdf.empty:
        pa_filtered = filter_pa_to_regions(pa_all, matched_regions_gdf)
        if pa_filtered is not None and not pa_filtered.empty:
            # Drop PAs with no pixels in the species elevation range
            if ficha.elevation.has_data() and dem_path.exists():
                keep = [
                    i for i, row in pa_filtered.iterrows()
                    if _park_overlaps_elevation_range(
                        gpd.GeoDataFrame([row], crs=pa_filtered.crs),
                        dem_path, ficha.elevation.min_m, ficha.elevation.max_m,
                    )
                ]
                pa_filtered = pa_filtered.loc[keep]
        if pa_filtered is not None and not pa_filtered.empty:
            pa_filtered.plot(
                ax=ax, facecolor="none", edgecolor="#fbbf24",
                linewidth=0.7, alpha=0.60, zorder=8,
            )

    # ── Layer 6: GBIF presence points ────────────────────────────────────
    if presencias_gdf is not None and not presencias_gdf.empty:
        ax.scatter(
            presencias_gdf.geometry.x,
            presencias_gdf.geometry.y,
            s=12, c="#ff4444", edgecolors="#ffaaaa",
            linewidths=0.4, alpha=0.85, zorder=10,
        )

    # ── Title + x-label ───────────────────────────────────────────────────
    sp_display = ficha.species.replace("_", " ")
    ax.set_title(sp_display, color="white", fontsize=13, fontweight="bold", pad=4)
    ax.set_xlabel(
        f"Hábitat potencial — Manual de Plantas CR  |  Elevación: {ficha.elevation.label()}",
        color="#9ca3af", fontsize=8,
    )

    # ── Legends ───────────────────────────────────────────────────────────
    right_patches = _build_right_legend(
        ficha, matched_parks_gdf, pa_filtered, presencias_gdf, highlight_gdf, scope,
        gbif_inferred_gdf=gbif_inferred_gdf,
    )
    left_patches = _build_left_legend(ficha, matched_regions_gdf)

    if right_patches:
        leg_right = ax.legend(
            handles=right_patches, loc="lower right", fontsize=7.5,
            facecolor="#1f2937", labelcolor="white",
            framealpha=0.9, edgecolor="#374151",
        )
        ax.add_artist(leg_right)

    if left_patches:
        ax.legend(
            handles=left_patches, loc="lower left", fontsize=7.0,
            facecolor="#1f2937", labelcolor="white",
            framealpha=0.9, edgecolor="#374151",
        )

    # ── Fuentes footer ────────────────────────────────────────────────────
    fig.text(
        0.5, 0.015,
        "Fuentes: Regiones botánicas digitalizadas de M. V. Castro (Manual de Plantas de Costa Rica) · "
        "Áreas Silvestres Protegidas (SINAC) · "
        "Altitud (SRTM, INEC) · Presencias GBIF",
        ha="center", fontsize=5.8, color="#6b7280",
        transform=fig.transFigure,
    )

    # ── Axis chrome ───────────────────────────────────────────────────────
    ax.tick_params(colors="#6b7280", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    fig.savefig(output_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Distribution map saved → {output_path}")
    return output_path
