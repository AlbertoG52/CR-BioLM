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

# Colour carries identity, and identity goes in the legend -- never as text over the
# polygon, which stops scaling the moment a second area is drawn.
#
# The number of colours is not a taste decision, it is computed. Validating candidate
# sets against this background (OKLab Delta E, Machado severity-1.0 CVD simulation):
# NO four-colour subset of the reference palette clears the all-pairs floors in dark
# mode -- normal-vision Delta E >= 15 and CVD >= 8. The first attempt, amber areas plus
# red records, failed badly: orange vs red is Delta E 7.1 for normal vision, so even
# full-colour readers could confuse them.
#
# So three identities is the hard ceiling, and one is spent on the records. Areas get
# two before colour stops working; beyond that they share one colour and the legend
# letters them, which is the usual cartographic answer and stays readable at any count.
POINTS = "#d55181"        # magenta, the focal data
AREA_COLORS = ["#3987e5", "#008300"]   # blue, green: worst pair normal 26.5 / CVD 11.3
AREA_FALLBACK = "#3987e5"
BAND = "#9aa1ad"          # achromatic on purpose (C=0.02): terrain, not a category


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


def render(species: str, place_code: str | list[str] | None = None, *,
           elev_min: float | None = None, elev_max: float | None = None,
           out_path: Path | None = None) -> tuple[Path, dict]:
    """Render the stimulus. Returns the path and what the image actually shows, so a
    run can record whether the park and the band made it onto the picture."""
    regions = load_regiones_botanicas().to_crs(4326)
    pts = get_points(species)
    codes = ([place_code] if isinstance(place_code, str)
             else list(place_code or []))
    shown = {"species": species, "place_code": codes,
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

    drawn_areas: list[tuple[str, str, str | None]] = []   # (colour, name, letter)
    if codes:
        pa = _places()
        lettered = len(codes) > len(AREA_COLORS)
        for i, code in enumerate(codes):
            row = pa[pa["codigo"] == code]
            if row.empty:
                continue
            colour = AREA_FALLBACK if lettered else AREA_COLORS[i]
            row.plot(ax=ax, facecolor=colour, edgecolor=colour, linewidth=2.2,
                     alpha=0.28, zorder=3)
            row.boundary.plot(ax=ax, color=colour, linewidth=2.2, zorder=4)
            letter = None
            if lettered:
                # Past two areas colour stops separating them, so a short chip on the
                # polygon carries identity and the legend spells it out. One or two
                # characters never collides the way a full park name does.
                letter = chr(ord("A") + len(drawn_areas))
                c = row.geometry.iloc[0].representative_point()
                ax.annotate(letter, xy=(c.x, c.y), ha="center", va="center",
                            color="#11151c", fontsize=10, fontweight="bold", zorder=7,
                            bbox=dict(boxstyle="circle,pad=0.28", facecolor=colour,
                                      edgecolor="none", alpha=0.95))
            drawn_areas.append((colour, str(row["label"].iloc[0]), letter))
        shown["park_drawn"] = bool(drawn_areas)
        shown["park_label"] = [n for _, n, _ in drawn_areas]

    if pts is not None and not pts.empty:
        pts.to_crs(4326).plot(ax=ax, color=POINTS, markersize=34, zorder=5,
                              edgecolor="white", linewidth=0.6)

    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=POINTS,
                      markeredgecolor="white", markersize=9,
                      label=f"Registros de la especie (n={shown['n_points']})")]
    for colour, name, letter in drawn_areas:
        handles.append(Patch(facecolor=colour, edgecolor=colour, alpha=0.6,
                             label=(f"{letter} · {name}" if letter else name)))
    if shown["band_drawn"]:
        handles.append(Patch(facecolor=BAND, edgecolor=BAND, alpha=0.55,
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

    tag = "-".join(codes) if codes else "nopark"
    out_path = out_path or (MAPS / f"{species.replace(' ', '_')}__{tag}.png")
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path, shown


def main() -> None:
    ap = argparse.ArgumentParser(description="Render one question-specific map")
    ap.add_argument("--species", required=True)
    ap.add_argument("--place-code", default=None,
                    help="one code, or several comma separated")
    ap.add_argument("--elev-min", type=float, default=None)
    ap.add_argument("--elev-max", type=float, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    codes = [c.strip() for c in args.place_code.split(",")] if args.place_code else None
    p, shown = render(args.species, codes, elev_min=args.elev_min,
                      elev_max=args.elev_max,
                      out_path=Path(args.out) if args.out else None)
    print(f"wrote {p}")
    print(shown)


if __name__ == "__main__":
    main()
