# Distribution Map Module — public API
from .ficha import DistributionFicha, EntityRef, ElevationRange
from .parser import build_ficha
from .gazetteer import normalize_text, lookup


def __getattr__(name: str):
    if name == "generate_distribution_map":
        from .renderer import generate_distribution_map

        return generate_distribution_map
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DistributionFicha",
    "EntityRef",
    "ElevationRange",
    "build_ficha",
    "generate_distribution_map",
    "normalize_text",
    "lookup",
]
