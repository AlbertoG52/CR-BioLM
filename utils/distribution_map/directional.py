"""Read "desde X al S" in a distribution paragraph as the regions it names.

The Manual often gives a range by an anchor and a direction: "vert. Pac. desde P.N.
Carara al S", "vert. Pac. desde la región de Puriscal al S", "Baja Talamanca (desde
Puerto Limón al S)". 271 entries say something of this kind. The gazetteer resolves the
anchor and nothing else, so a species "from Carara southwards" was given at most the
anchor's own region, and for a park anchor no region at all (Heliconia stilesii,
Anthurium eximium). An expert reviewing the maps pointed it out for Streptochaeta
sodiroana, whose map lacked the Valle del Diquís and the south of the Valle del General.

The rule, deliberately simple and stated so it can be checked:

  anchor     resolved by the gazetteer to a botanical region, a protected area or a
             canton, and placed at the latitude of that polygon's centroid
  slope      the slope named closest before the phrase in the same clause ("vert.
             Pac."); without one, the slope the anchor lies on
  direction  S keeps every region on that slope whose centroid lies south of the anchor,
             N every one to the north. Other directions are left alone
  bounded    "al S hasta Y" stops at Y when Y resolves; when it does not, the phrase is
             left unexpanded rather than run to the end of the country

Regions are taken from the polygons of the region layer, so the names are the layer's
own. Nothing is removed from what the gazetteer found; this only adds.
"""
from __future__ import annotations

import re
from functools import lru_cache

# " al S" / " al N", optionally bounded by "hasta Y". The anchor is read backwards from
# the direction to the nearest clause boundary, so "llanuras de Guanacaste al S" anchors
# on the plains, not on the word after "de", and "P.N. Carara" keeps its abbreviation.
_DIRECTION = re.compile(r"\s+al\s+(?P<dir>S|N)\b(?:\s+hasta\s+(?:la\s+)?(?:vecindad\s+de\s+)?"
                        r"(?P<bound>[^;,()]{2,40}?)(?=[;,()]|\.\s|$))?")
_BOUNDARY = re.compile(r".*(?:[;,(]|\bdesde\b)\s*", re.S)
_SLOPE = re.compile(r"vert(?:iente)?s?\.?\s*(Pac|Carib)", re.I)


@lru_cache(maxsize=1)
def _layers():
    from .geodata import load_regiones_botanicas, load_protected_areas, load_cantones
    regions = load_regiones_botanicas().to_crs(4326).reset_index(drop=True)
    regions["lat"] = regions.geometry.to_crs(5367).centroid.to_crs(4326).y
    regions["slope"] = ["Caribe" if "arib" in str(v) else "Pacífico"
                        for v in regions["Vertiente"]]
    return {"regiones_botanicas": regions, "protected_areas": load_protected_areas().to_crs(4326),
            "cantones": load_cantones().to_crs(4326)}


def _locate(text: str):
    """(latitude, slope or None) of the first thing in `text` the gazetteer can place."""
    import geopandas as gpd
    from .gazetteer import lookup
    layers = _layers()
    for _level, entities in (lookup(text) or {}).items():
        for e in entities:
            layer = layers.get(e.get("target_shapefile"))
            attr, value = e.get("target_attribute"), e.get("target_value")
            if layer is None or attr not in getattr(layer, "columns", []):
                continue
            hit = layer[layer[attr] == value]
            if hit.empty:
                continue
            centre = gpd.GeoSeries([hit.geometry.to_crs(5367).unary_union.centroid],
                                   crs=5367).to_crs(4326).iloc[0]
            regions = layers["regiones_botanicas"]
            under = regions[regions.contains(centre)]
            return float(centre.y), (under["slope"].iloc[0] if not under.empty else None)
    return None


def expand(geographic_notes: str) -> list[tuple[str, str]]:
    """Regions implied by directional phrases, as (region name, the phrase it came from)."""
    if not geographic_notes or " al " not in geographic_notes:
        return []
    regions = _layers()["regiones_botanicas"]
    out: list[tuple[str, str]] = []
    for m in _DIRECTION.finditer(geographic_notes):
        before = geographic_notes[:m.start()]
        anchor_text = _BOUNDARY.sub("", before).strip()
        anchor = _locate(anchor_text) if anchor_text else None
        if anchor is None:
            continue
        lat, anchor_slope = anchor
        clause = re.split(r"[;]", before)[-1]
        named = _SLOPE.findall(clause)
        if re.search(r"ambas\s+verts?\.?\s*$|ambas\s+verts?\.?\s+[^,;]*$", clause):
            slopes = {"Pacífico", "Caribe"}          # "ambas verts. Cord. de X al S"
        elif named:
            slopes = {"Pacífico" if named[-1].lower().startswith("pac") else "Caribe"}
        elif anchor_slope:
            slopes = {anchor_slope}
        else:
            continue
        lo, hi = (-90.0, lat) if m.group("dir") == "S" else (lat, 90.0)
        if m.group("bound"):
            bound = _locate(m.group("bound"))
            if bound is None:
                continue                  # an unresolved end: do not run to the coast
            if m.group("dir") == "S":
                lo = bound[0]
            else:
                hi = bound[0]
        sel = regions[(regions["slope"].isin(slopes)) & (regions["lat"] <= hi + 0.02)
                      & (regions["lat"] >= lo - 0.02)]
        for name in dict.fromkeys(sel["Nombre"]):
            out.append((name, m.group(0)))
    return out
