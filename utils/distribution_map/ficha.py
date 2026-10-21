"""
ficha.py — DistributionFicha: structured habitat information for one species.

The Ficha is the output of the parser and the input to the renderer.
All fields are plain Python types so the whole object is JSON-serializable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# Margin drawn around a single approximate elevation. The same 200 m the answer checker
# allows on a stated elevation and the cultivation rule uses for "near the range".
SINGLE_VALUE_MARGIN_M = 200


@dataclass
class ElevationRange:
    min_m: Optional[float] = None
    max_m: Optional[float] = None
    outlier_min_m: Optional[float] = None
    outlier_max_m: Optional[float] = None

    def has_data(self) -> bool:
        return self.min_m is not None and self.max_m is not None

    def is_single_value(self) -> bool:
        """True when the Manual gives one approximate elevation ("ca. 1400 m") and no
        outlier bounds, which it does for 277 species, usually known from one
        collection."""
        return (self.has_data() and self.min_m == self.max_m
                and self.outlier_min_m is None and self.outlier_max_m is None)

    def band(self) -> tuple[float, float]:
        """The elevation interval to DRAW. A range is drawn as stated. A single
        approximate value is drawn as a band of SINGLE_VALUE_MARGIN_M either side:
        "ca." means about, and drawn as exactly that elevation the band matches almost
        no terrain and the map shows nothing (Cucurbita pepo at "ca. 1400 m" captured
        none of its 31 records; with the band, 9). The stored values are not changed --
        this is how the map reads them, not what the Manual says."""
        if self.is_single_value():
            return (max(0.0, self.min_m - SINGLE_VALUE_MARGIN_M),
                    self.max_m + SINGLE_VALUE_MARGIN_M)
        return (self.min_m, self.max_m)

    def label(self) -> str:
        """Human-readable elevation string for map labels."""
        if not self.has_data():
            return "elevación no disponible"
        if self.is_single_value():
            return f"ca. {int(self.min_m)} m (±{SINGLE_VALUE_MARGIN_M} m)"
        s = f"{int(self.min_m)}–{int(self.max_m)} m"
        if self.outlier_min_m is not None or self.outlier_max_m is not None:
            parts = []
            if self.outlier_min_m is not None:
                parts.append(f"inf: {int(self.outlier_min_m)}–{int(self.min_m)} m")
            if self.outlier_max_m is not None:
                parts.append(f"sup: {int(self.max_m)}–{int(self.outlier_max_m)} m")
            s += f"  (atípicos — {', '.join(parts)})"
        return s

    def __repr__(self) -> str:
        if not self.has_data():
            return "ElevationRange(none)"
        s = f"{int(self.min_m)}–{int(self.max_m)} m"
        if self.outlier_min_m is not None:
            s += f" (outlier-lo: {int(self.outlier_min_m)} m)"
        if self.outlier_max_m is not None:
            s += f" (outlier-hi: {int(self.outlier_max_m)} m)"
        return s


@dataclass
class EntityRef:
    """Reference to one geographic entity matched in the gazetteer."""
    canonical_name: str
    hierarchy_level: str         # cordillera, llanura, valle, park, canton, district, …
    source_span: str = ""        # verbatim text fragment that triggered this match

    def __hash__(self) -> int:
        return hash((self.canonical_name, self.hierarchy_level))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EntityRef):
            return NotImplemented
        return (self.canonical_name, self.hierarchy_level) == (
            other.canonical_name, other.hierarchy_level
        )


@dataclass
class DistributionFicha:
    """
    Structured habitat description for one species, built from the raw
    Manual de Plantas text by parser.build_ficha().

    Fields are intentionally plain-typed so asdict() → JSON works with no
    custom encoder. Save/load helpers are provided.
    """
    species: str
    habitat_raw: str = ""

    # Geographic scope — ordered from coarsest to finest
    vertientes: list[str] = field(default_factory=list)       # "Caribe" | "Pacífico"
    regions: list[EntityRef] = field(default_factory=list)    # cordilleras, llanuras, valles, …
    parks: list[EntityRef] = field(default_factory=list)
    cantons: list[EntityRef] = field(default_factory=list)
    districts: list[EntityRef] = field(default_factory=list)

    elevation: ElevationRange = field(default_factory=ElevationRange)

    # Forest type descriptors from habitat_raw (e.g. "muy húmedo", "pluvial")
    forest_types: list[str] = field(default_factory=list)

    # Structured occurrences from geo_parser (list of dicts, each with
    # vertiente, qualifier, feature_type, feature_name, sub_qualifier,
    # embedded_protected_areas, raw_span, mobot_row_indices)
    # Used by the renderer for locality buffer overlay.
    locality_occurrences: list[dict] = field(default_factory=list)

    # Parser diagnostics
    confidence: dict[str, float] = field(default_factory=dict)
    unresolved_tokens: list[str] = field(default_factory=list)

    # ── Scope helpers ────────────────────────────────────────────────────────

    def effective_scope(self) -> str:
        """
        Return the name of the deepest non-empty entity group.
        Used by the renderer to decide clip extent.

        Canton/district scope is only used when cantons are the PRIMARY
        geographic reference (no substantial region matches). When ≥2 regions
        are matched, canton matches are incidental (e.g. "Puriscal" triggers
        both region and canton) and the scope stays at "park" or "region".
        """
        if self.districts and len(self.regions) <= 1 and not self.parks:
            return "district"
        if self.cantons and len(self.regions) <= 1 and not self.parks:
            return "canton"
        if self.parks:
            return "park"
        if self.regions:
            return "region"
        if self.vertientes:
            return "vertiente"
        return "country"

    def summary(self) -> str:
        scope = self.effective_scope()
        parts = [f"scope={scope}"]
        if self.vertientes:
            parts.append("verts=" + "+".join(self.vertientes))
        if self.regions:
            parts.append(f"regions={len(self.regions)}")
        if self.parks:
            parts.append(f"parks={len(self.parks)}")
        if self.cantons:
            parts.append(f"cantons={len(self.cantons)}")
        if self.districts:
            parts.append(f"districts={len(self.districts)}")
        if self.elevation.has_data():
            parts.append(repr(self.elevation))
        return f"Ficha({', '.join(parts)})"

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "DistributionFicha":
        data = dict(data)
        data["elevation"] = ElevationRange(**data.get("elevation", {}))
        data["regions"]   = [EntityRef(**e) for e in data.get("regions", [])]
        data["parks"]     = [EntityRef(**e) for e in data.get("parks", [])]
        data["cantons"]   = [EntityRef(**e) for e in data.get("cantons", [])]
        data["districts"] = [EntityRef(**e) for e in data.get("districts", [])]
        # locality_occurrences are plain dicts — keep as-is
        data.setdefault("locality_occurrences", [])
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "DistributionFicha":
        return cls.from_dict(json.loads(text))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DistributionFicha":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
