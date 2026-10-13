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
import json
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
AREA_FALLBACK = "#3987e5"              # blue, the asked-about protected area
AREA_COLORS = ["#3987e5"]
REGIONS = "#008300"       # green, the botanical regions the Manual reports
OTHER_AREA = "#c9cfd8"   # neutral outline: an annotation, not a category
BAND = "#9aa1ad"          # achromatic on purpose (C=0.02): terrain, not a category
# magenta / blue / green validated all-pairs on this surface: worst normal-vision
# Delta E 26.5, worst CVD 11.3. That is the whole budget -- three identities.


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


def _snapshot_points(species: str) -> gpd.GeoDataFrame:
    """The species' records from the frozen download, the same ones the text pack and
    the keys are built from.

    The map used to draw `gbif_map.get_points`, a live lookup with its own, stricter
    cleaning, and so showed fewer records than the evidence it was being compared with:
    under half of them for 7 of the 27 species of tranche 1 and 13 of the 51 of tranche
    2 (Chusquea tomentosa, 38 drawn of 84), and none of 9 for Begonia nelumbonifolia. A
    park keyed "sí" on records the picture did not carry was asking the map arm to see
    what was not drawn. Found by checking the maps ahead of tranche 2.
    """
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            # By taxon key, not by name: the download files each record under GBIF's
            # accepted name, and for 881 of the 6,664 catalogue species that have
            # records that is a different name from the Manual's (Calathea
            # donnell-smithii is Goeppertia there). Asked for by name, those species
            # come back with no records at all.
            cur.execute("""SELECT o.lon, o.lat FROM mpcr.occurrences o
                           JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                           WHERE e.species = %s AND o.lon IS NOT NULL
                             AND o.lat IS NOT NULL""", (species,))
            rows = cur.fetchall()
    finally:
        conn.close()
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([r[0] for r in rows], [r[1] for r in rows]), crs=4326)


def render(species: str, place_code: str | list[str] | None = None, *,
           elev_min: float | None = None, elev_max: float | None = None,
           highlight_regions: list[str] | None = None,
           out_path: Path | None = None, reuse: bool = False) -> tuple[Path, dict]:
    """Render the stimulus. Returns the path and what the image actually shows, so a
    run can record whether the park and the band made it onto the picture.

    With ``reuse`` an image already on disk is returned as it is, together with the
    record of what it shows that was saved beside it. Drawing is deterministic, so this
    changes nothing a model sees; it lets the maps be drawn ahead of a run, in parallel,
    and looked over before any call is paid for (benchmark/prerender.py).
    """
    zone = species_zone(species)
    if highlight_regions is None and zone['regions']:
        highlight_regions = zone['regions']
    _codes = [place_code] if isinstance(place_code, str) else list(place_code or [])
    _tag = ("-".join(_codes) if _codes else "nopark") + ("__reg" if highlight_regions else "")
    _cached = out_path or (MAPS / f"{species.replace(' ', '_')}__{_tag}.png")
    _side = Path(str(_cached) + ".json")
    if reuse and Path(_cached).exists() and _side.exists():
        return Path(_cached), json.loads(_side.read_text(encoding="utf-8"))
    if elev_min is None:
        elev_min, elev_max = zone['elev_min'], zone['elev_max']
    inferred = inferred_regions(species, zone['regions'])
    # Park selection follows the UNION now that the keys do, so the picture
    # and the answer it is scored against cannot disagree.
    zone_parks = parks_in_zone(sorted(set(zone['regions']) | set(inferred)),
                               zone['elev_min'], zone['elev_max'])
    regions = load_regiones_botanicas().to_crs(4326)
    pts = _snapshot_points(species)
    codes = ([place_code] if isinstance(place_code, str)
             else list(place_code or []))
    shown = {"species": species, "place_code": codes,
             "n_points": 0 if pts is None or pts.empty else len(pts),
             "park_drawn": False, "park_label": None, "band_drawn": False,
             "other_areas": [], "zone_source": None, "regions_inferred": []}

    fig, ax = plt.subplots(figsize=(9.5, 10), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    regions.plot(ax=ax, facecolor=LAND, edgecolor=LAND_EDGE, linewidth=0.4, zorder=1)

    # The regions the Manual reports for this species. Half of what a protected-area
    # key turns on is region overlap, so a map without this layer carries less evidence
    # than the text pack and the comparison stops being about the interface.
    #
    # The layer is drawn but never NAMED: the legend says that these are the reported
    # regions, not which ones they are. For a question about slope or region the caller
    # passes nothing, because there the identity of the region IS the answer.
    drawn_regions, missing_regions = [], []
    band_clip = None
    if highlight_regions:
        have = {str(n): n for n in regions["Nombre"].dropna().unique()}
        for want in highlight_regions:
            hit = have.get(want)
            if hit is None:                     # "Fila Costeña (Norte)" -> "Fila Costeña"
                base = want.split("(")[0].strip()
                hit = have.get(base) or next((k for k in have if k.startswith(base)), None)
            (drawn_regions if hit else missing_regions).append(hit or want)
        if drawn_regions:
            sel = regions[regions["Nombre"].isin(drawn_regions)]
            sel.plot(ax=ax, facecolor=REGIONS, edgecolor=REGIONS, linewidth=1.0,
                     alpha=0.26, zorder=2)
            # The elevation band is clipped to these regions, not drawn across the
            # country. Terrain at 2,200 m in Guanacaste says nothing about a species
            # reported only from the Cordillera Central, and showing it invites a model
            # to read habitat where there is none. Region INTERSECTED with band is the
            # potential habitat, which is what the validated expert map shows.
            band_clip = sel
            # Regions the records argue for but the Manual does not state. Same green,
            # hatched instead of filled: the palette budget on this surface is three
            # identities and all three are spent, and texture separates without a
            # fourth hue that would fail the colourblind floors.
            if inferred:
                inf_sel = regions[regions["Nombre"].isin(inferred)]
                if not inf_sel.empty:
                    inf_sel.plot(ax=ax, facecolor="none", edgecolor=REGIONS,
                                 linewidth=1.0, hatch="///", alpha=0.75, zorder=2)
                    shown["regions_inferred"] = sorted(inferred)
    shown["zone_source"] = zone["source"]
    shown["regions_drawn"] = drawn_regions
    shown["regions_missing"] = missing_regions

    band = _elevation_band(band_clip if band_clip is not None else regions,
                           elev_min, elev_max)
    if band is not None:
        inside, extent = band
        ax.imshow(np.ma.masked_where(~inside, inside.astype(float)), extent=extent,
                  origin="upper", cmap=matplotlib.colors.ListedColormap([BAND]),
                  alpha=0.45, zorder=2, interpolation="nearest")
        shown["band_drawn"] = True

    # Every protected area whose regions and elevation profile meet the species' zone,
    # drawn as context. The question names one park, but a map showing only that one
    # tells a reader nothing about whether it is a likely place among many or the only
    # candidate in the country.
    #
    # These use a neutral outline rather than a colour: the palette budget on this
    # surface is three identities and all three are spent (records, the asked area, the
    # reported regions). An outline is an annotation, not a category.
    #
    # The asked-about area is ALWAYS drawn, in its own colour, whether or not it meets
    # the zone. Drawing only the overlapping ones would make an unlikely park vanish
    # from the picture, and its absence would hand over the answer.
    if zone_parks:
        asked = set(codes or [])
        others = [p for p in zone_parks if p["code"] not in asked]
        if others:
            pa_all = _places()
            by_code = {c: g for c, g in zip(pa_all["codigo"], pa_all.geometry)}
            named = []
            for p in others:
                geom = by_code.get(p["code"])
                if geom is None:
                    continue
                import geopandas as _gpd
                _gpd.GeoSeries([geom], crs=4326).boundary.plot(
                    ax=ax, color=OTHER_AREA, linewidth=1.0, linestyle="--",
                    alpha=0.8, zorder=3)
                # A letter on the polygon, spelled out in the legend. An unnamed
                # outline is useless to a vision model: it has to identify the area to
                # reason about it, and would otherwise guess from position or mistake
                # it for the one the question asks about. A one-character chip names it
                # at any count, where the full park name collides past a single area.
                letter = chr(ord("A") + len(named))
                c = geom.representative_point()
                ax.annotate(letter, xy=(c.x, c.y), ha="center", va="center",
                            color="#11151c", fontsize=8.5, fontweight="bold", zorder=6,
                            bbox=dict(boxstyle="circle,pad=0.22", facecolor=OTHER_AREA,
                                      edgecolor="none", alpha=0.9))
                named.append((letter, p["label"]))
            shown["other_areas"] = [f"{l} · {n}" for l, n in named]
            shown["_other_named"] = named

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
            row.boundary.plot(ax=ax, color=colour, linewidth=3.0, zorder=5)
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
    if drawn_regions:
        handles.append(Patch(facecolor=REGIONS, edgecolor=REGIONS, alpha=0.6,
                             label="Regiones botánicas donde se reporta la especie"))
    if shown["regions_inferred"]:
        handles.append(Patch(facecolor="none", edgecolor=REGIONS, hatch="///",
                             label=f"Regiones inferidas por registros GBIF "
                                   f"(n={len(shown['regions_inferred'])})"))
    for letter, name in shown.get("_other_named", []):
        handles.append(Line2D([], [], color=OTHER_AREA, linestyle="--", linewidth=1.2,
                              label=f"{letter} · {name}"))
    for colour, name, letter in drawn_areas:
        handles.append(Patch(facecolor=colour, edgecolor=colour, alpha=0.6,
                             label=(f"{letter} · {name}" if letter else name)))
    if shown["band_drawn"]:
        label = (f"Hábitat potencial: {int(elev_min)}-{int(elev_max)} m dentro de esas "
                 f"regiones" if drawn_regions
                 else f"Terreno entre {int(elev_min)} y {int(elev_max)} m")
        handles.append(Patch(facecolor=BAND, edgecolor=BAND, alpha=0.55, label=label))
    # Outside the axes. Inside, a nineteen-area legend covered the Osa peninsula and
    # hid occurrence records behind itself -- the stimulus must never occlude its own
    # evidence.
    ncol = 2 if len(handles) > 9 else 1
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                    facecolor="#1f242e", edgecolor="#39404d", labelcolor="#e6e8ec",
                    fontsize=9 if ncol == 2 else 10, ncol=ncol, frameon=True)
    leg.set_zorder(7)

    ax.set_title(species, color="#f2f4f7", fontsize=15, fontstyle="italic", pad=12)
    ax.tick_params(colors="#9aa1ad", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#39404d")
    fig.text(0.5, -0.02,
             "Fuentes: Regiones botánicas digitalizadas de M. V. Castro · "
             "Áreas Silvestres Protegidas (SINAC) · Altitud (SRTM, INEC) · "
             "Presencias GBIF 10.15468/dl.8yhee8",
             ha="center", color="#6f7782", fontsize=7.5)

    tag = ("-".join(codes) if codes else "nopark") + ("__reg" if highlight_regions else "")
    shown.pop("_other_named", None)
    out_path = out_path or (MAPS / f"{species.replace(' ', '_')}__{tag}.png")
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    Path(str(out_path) + ".json").write_text(
        json.dumps(shown, ensure_ascii=False, default=str), encoding="utf-8")
    return out_path, shown


def species_zone(species: str, ficha: dict | None = None) -> dict:
    """The species' optimal zone: botanical regions and elevation band.

    Where the Manual is silent the snapshot fills the gap -- 184 species have no stated
    region and 34 no elevation -- which is the same rule the evidence pack follows:
    occurrence data fills gaps, it never competes with a stated fact. The source of
    each half is returned, because a zone derived from records is a weaker claim than
    one the Manual states and the caption should be able to say so.
    """
    import sqlite3
    from mpcr_rag import config as _cfg
    from mpcr_rag.store import pg_store

    if ficha is None:
        con = sqlite3.connect(_cfg.SQLITE_PATH)
        row = con.execute("SELECT ficha_json FROM fichas WHERE species=?", (species,)).fetchone()
        con.close()
        ficha = json.loads(row[0]) if row else {}

    regions = list(ficha.get("regions") or [])
    lo, hi = ficha.get("elev_min"), ficha.get("elev_max")
    src = {"regions": "Manual" if regions else None,
           "elevation": "Manual" if lo is not None else None}

    if not regions or lo is None:
        conn = pg_store.connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""SELECT regions, elev_dem_p05, elev_dem_p95
                               FROM mpcr.species_evidence WHERE species = %s""", (species,))
                r = cur.fetchone()
        finally:
            conn.close()
        if r:
            if not regions and r[0]:
                regions, src["regions"] = list(r[0]), "GBIF"
            if lo is None and r[1] is not None:
                lo, hi, src["elevation"] = round(r[1]), round(r[2]), "GBIF"
    return {"regions": regions, "elev_min": lo, "elev_max": hi, "source": src}


def inferred_regions(species: str, stated: list[str]) -> list[str]:
    """Regions holding at least five occurrence records that the Manual does not list.

    The same rule the validated renderer applies (_GBIF_CLUSTER_MIN = 5, "minimum GBIF
    points in a region to flip it"), reused rather than reimplemented so the stimulus
    and the expert map cannot drift apart. For Anthurium ravenii it recovers the seven
    regions that map labels "Region inferida por GBIF": the Manual names four Caribbean
    lowland regions, while 103 records sit in Osa-Golfito alone.

    These are drawn as information, but they do NOT widen the zone that selects parks.
    The protected-area keys compute overlap against the Manual's regions, so a map whose
    park selection followed the inferred set would disagree with the key it is scored
    against, and the map arm would be marked wrong for reading its own evidence.
    """
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT region, count(*) FROM mpcr.occurrences
                           WHERE species = %s AND region IS NOT NULL
                           GROUP BY region HAVING count(*) >= 5""", (species,))
            hits = {r[0] for r in cur.fetchall()}
    finally:
        conn.close()
    return sorted(hits - set(stated or []))


def parks_in_zone(regions: list[str], lo, hi) -> list[dict]:
    """Protected areas whose own regions and elevation profile meet the zone.

    This is the same overlap the protected-area keys compute, so a map showing these
    areas is showing the evidence rather than hinting at an answer.
    """
    if not regions or lo is None:
        return []
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT code, label, elev_p05, elev_p95, regions
                           FROM mpcr.places
                           WHERE category IN ('PN','RB') AND elev_p05 IS NOT NULL
                             AND name !~* 'marin|isla'
                           ORDER BY code""")
            rows = cur.fetchall()
    finally:
        conn.close()
    want = set(regions)
    return [{"code": c, "label": lb} for c, lb, p05, p95, regs in rows
            if (want & set(regs or [])) and p05 <= hi and p95 >= lo]


def main() -> None:
    ap = argparse.ArgumentParser(description="Render one question-specific map")
    ap.add_argument("--species", required=True)
    ap.add_argument("--place-code", default=None,
                    help="one code, or several comma separated")
    ap.add_argument("--elev-min", type=float, default=None)
    ap.add_argument("--elev-max", type=float, default=None)
    ap.add_argument("--regions", default="", help="comma separated region names to shade")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    codes = [c.strip() for c in args.place_code.split(",")] if args.place_code else None
    regs = [r.strip() for r in args.regions.split(",")] if args.regions else None
    p, shown = render(args.species, codes, elev_min=args.elev_min,
                      elev_max=args.elev_max, highlight_regions=regs,
                      out_path=Path(args.out) if args.out else None)
    print(f"wrote {p}")
    print(shown)


if __name__ == "__main__":
    main()
