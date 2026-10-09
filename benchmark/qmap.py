"""Question-specific maps for the multimodal evidence condition.

Separate from `utils/distribution_map/renderer.py` on purpose. That renderer produces
the expert-facing map validated against Armando's 21 maps, and it stays exactly as it
is: it is a deliverable, it is human-readable, and it is a published artefact. This
module is a stimulus for an experiment, and the two have opposite requirements.

Looking at the expert map showed why a separate one is needed. Asked whether a species
occurs in Parque Nacional La Amistad, a vision model given that map could not answer it:

  * no protected area is drawn or labelled, so the park in the question is simply not on
    the image, and the model would have to recall where it is -- turning a map-reading
    test into a closed-book geography test;
  * the legend reads "Vertiente del Pacifico" and "Cordillera de Talamanca", which for a
    slope or region question prints the answer on the picture;
  * the optimal-habitat entry is listed even when the DEM yields no pixels in range.

So this renderer inverts those three. The asked-about area is drawn and named; nothing
in the legend states a region, a slope or anything else a key turns on; and the
elevation band appears only when it actually covers something.

What the image is allowed to carry is exactly the evidence the text pack carries, in
visual form: where the records are, where the asked-about area is, and which ground lies
in the species' elevation band. A model should be able to answer from it, and the
comparison with the text pack is then about the interface, not about who was told more.

Run:  python -m benchmark.qmap --species "Cryptocentrum flavum" --place-code P18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.mask import mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mpcr_rag.evidence.places import _places                      # noqa: E402
from mpcr_rag.query.gbif_map import get_points                    # noqa: E402
from utils.distribution_map.geodata import (PATHS,                # noqa: E402
                                            load_regiones_botanicas)

MAPS = Path(__file__).resolve().parent / "data" / "qmaps"
MAPS.mkdir(parents=True, exist_ok=True)

LAND = "#3a3f4b"
LAND_EDGE = "#5a606e"
BG = "#161a22"
PARK = "#ffc14d"
POINTS = "#ff4d4d"
BAND = "#38e1d6"


def _elevation_band(regions: gpd.GeoDataFrame, lo: float, hi: float):
    """Ground inside the species' elevation band, as a boolean raster + extent.

    Returns None when the band covers nothing, so the legend never advertises a layer
    that was not drawn -- the failure the expert map showed on a 950-1050 m species.
    """
    dem_path = PATHS.get("dem")
    if not dem_path or not Path(dem_path).exists() or lo is None or hi is None:
        return None
    with rasterio.open(dem_path) as src:
        geom = [regions.to_crs(src.crs).union_all().__geo_interface__]
        # The DEM is int16. Asking mask() to fill with NaN silently produces garbage,
        # because NaN has no int16 representation -- which is why this layer came back
        # empty for a 950-1050 m species while 2.4% of the raster sits in that band.
        # Mask with the raster's own nodata, then move to float.
        arr, transform = mask(src, geom, crop=True, filled=True)
        raw = arr[0]
        band = raw.astype("float32")
        if src.nodata is not None:
            band[raw == src.nodata] = np.nan
        inside = np.isfinite(band) & (band >= lo) & (band <= hi)
        if not inside.any():
            return None
        h, w = band.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (w, h)
        return inside, (left, right, bottom, top)


def render(species: str, place_code: str | None = None, *,
           elev_min: float | None = None, elev_max: float | None = None,
           out_path: Path | None = None) -> tuple[Path, dict]:
    """Render the stimulus. Returns the path and what the image actually shows, so a
    run can record whether the park and the band made it onto the picture."""
    regions = load_regiones_botanicas().to_crs(4326)
    pts = get_points(species)
    shown = {"species": species, "place_code": place_code,
             "n_points": 0 if pts is None or pts.empty else len(pts),
             "park_drawn": False, "park_label": None, "band_drawn": False}

    fig, ax = plt.subplots(figsize=(9.5, 10), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Context only: no region is named, coloured or distinguished from any other.
    regions.plot(ax=ax, facecolor=LAND, edgecolor=LAND_EDGE, linewidth=0.4, zorder=1)

    band = _elevation_band(regions, elev_min, elev_max)
    if band is not None:
        inside, extent = band
        ax.imshow(np.ma.masked_where(~inside, inside.astype(float)), extent=extent,
                  origin="upper", cmap=matplotlib.colors.ListedColormap([BAND]),
                  alpha=0.45, zorder=2, interpolation="nearest")
        shown["band_drawn"] = True

    if place_code:
        pa = _places()
        row = pa[pa["codigo"] == place_code]
        if not row.empty:
            row.plot(ax=ax, facecolor=PARK, edgecolor=PARK, linewidth=2.2,
                     alpha=0.30, zorder=3)
            row.boundary.plot(ax=ax, color=PARK, linewidth=2.2, zorder=4)
            # Above the polygon's bounding box, not over it: the label has to be
            # readable without hiding the boundary the question turns on.
            minx, miny, maxx, maxy = row.total_bounds
            label = str(row["label"].iloc[0])
            ax.annotate(label, xy=((minx + maxx) / 2, maxy), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="bottom",
                        color="#1b1f27", fontsize=10.5, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.35", facecolor=PARK,
                                  edgecolor="none", alpha=0.95))
            shown["park_drawn"] = True
            shown["park_label"] = label

    if pts is not None and not pts.empty:
        pts.to_crs(4326).plot(ax=ax, color=POINTS, markersize=34, zorder=5,
                              edgecolor="white", linewidth=0.6)

    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=POINTS,
                      markeredgecolor="white", markersize=9,
                      label=f"Registros de la especie (n={shown['n_points']})")]
    if shown["park_drawn"]:
        handles.append(Patch(facecolor=PARK, edgecolor=PARK, alpha=0.5,
                             label="Área protegida consultada"))
    if shown["band_drawn"]:
        handles.append(Patch(facecolor=BAND, edgecolor=BAND, alpha=0.5,
                             label=f"Terreno entre {int(elev_min)} y {int(elev_max)} m"))
    leg = ax.legend(handles=handles, loc="lower right", facecolor="#1f242e",
                    edgecolor="#39404d", labelcolor="#e6e8ec", fontsize=10)
    leg.set_zorder(7)

    ax.set_title(species, color="#f2f4f7", fontsize=15, fontstyle="italic", pad=12)
    ax.tick_params(colors="#9aa1ad", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#39404d")
    fig.text(0.5, 0.045,
             "Fuentes: Regiones botánicas digitalizadas de M. V. Castro · "
             "Áreas Silvestres Protegidas (SINAC) · Altitud (SRTM, INEC) · "
             "Presencias GBIF 10.15468/dl.8yhee8",
             ha="center", color="#6f7782", fontsize=7.5)

    out_path = out_path or (MAPS / f"{species.replace(' ', '_')}__{place_code or 'nopark'}.png")
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path, shown


def main() -> None:
    ap = argparse.ArgumentParser(description="Render one question-specific map")
    ap.add_argument("--species", required=True)
    ap.add_argument("--place-code", default=None)
    ap.add_argument("--elev-min", type=float, default=None)
    ap.add_argument("--elev-max", type=float, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    p, shown = render(args.species, args.place_code, elev_min=args.elev_min,
                      elev_max=args.elev_max,
                      out_path=Path(args.out) if args.out else None)
    print(f"wrote {p}")
    print(shown)


if __name__ == "__main__":
    main()
