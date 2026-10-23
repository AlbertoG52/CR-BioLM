"""What counts as a species' botanical regions, in one place.

The Manual states a range. The snapshot sometimes shows the species somewhere the
Manual never mentions -- Anthurium ravenii is listed from four Caribbean lowland
regions while 103 of its records sit in Osa-Golfito. A region holding at least five
cleaned records is treated as part of the species' range whether or not the Manual
names it, which is the rule the validated expert map already applies
(`renderer._GBIF_CLUSTER_MIN = 5`, "minimum GBIF points in a region to flip it").

This EXTENDS the earlier principle rather than breaking it. Occurrence evidence still
never overrules a stated fact: for elevation the records give a competing range -- 423
to 1248 m against the Manual's 950 to 1050 -- and two ranges cannot both be the answer,
so the Manual wins and the records are withheld. Regions are additive. "Also here" does
not contradict "here", so the union is coherent and strictly better evidence than
either half.

Every consumer reads from this module so a key and the evidence it is scored against
cannot drift apart: the question keys, the evidence pack, the map stimulus and the
co-occurrence compatibility check.
"""
from __future__ import annotations

from functools import lru_cache

MIN_RECORDS = 5          # same threshold as the validated renderer


@lru_cache(maxsize=1)
def _record_regions() -> dict[str, set]:
    """species -> regions holding at least MIN_RECORDS cleaned records.

    One query for the whole catalogue. Per-species lookups would make regenerating
    1,324 questions, or building a 6,946-species compatibility index, unusably slow.
    """
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            # By taxon key: the download files records under GBIF's accepted name,
            # which differs from the Manual's for 881 species with records.
            cur.execute("""SELECT e.species, o.region, count(*) FROM mpcr.occurrences o
                           JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                           WHERE o.region IS NOT NULL
                           GROUP BY e.species, o.region HAVING count(*) >= %s""",
                        (MIN_RECORDS,))
            out: dict[str, set] = {}
            for sp, reg, _n in cur.fetchall():
                out.setdefault(sp, set()).add(reg)
            return out
    finally:
        conn.close()


def effective(species: str, stated) -> dict:
    """{stated, inferred, union} for one species.

    `inferred` is what the records add beyond the Manual, kept separate so evidence can
    say which half it came from and a reader can weigh them differently.
    """
    stated = list(stated or [])
    inferred = sorted(_record_regions().get(species, set()) - set(stated))
    return {"stated": stated, "inferred": inferred,
            "union": sorted(set(stated) | set(inferred))}


def union(species: str, stated) -> list[str]:
    return effective(species, stated)["union"]


@lru_cache(maxsize=1)
def _record_slopes() -> dict[str, set]:
    """species -> slopes holding at least MIN_RECORDS cleaned records."""
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT e.species, o.vertiente, count(*) FROM mpcr.occurrences o
                           JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                           WHERE o.vertiente IS NOT NULL
                           GROUP BY e.species, o.vertiente HAVING count(*) >= %s""",
                        (MIN_RECORDS,))
            out: dict[str, set] = {}
            for sp, v, _n in cur.fetchall():
                out.setdefault(sp, set()).add("Caribe" if "arib" in v else "Pacífico")
            return out
    finally:
        conn.close()


def effective_slopes(species: str, stated) -> dict:
    """{stated, inferred, union} for the slopes, by the same rule as the regions.

    A slope is additive in the way a region is: records on the Caribbean side do not
    contradict a Manual that says Pacific, they add to it. But the slope question asks
    for an exact answer -- Caribbean, Pacific or both -- so the key cannot simply become
    the union, or an answer faithful to the Manual would be marked incomplete. Both
    answers are therefore accepted: what the Manual states, and what the Manual states
    plus what at least five records show. Raised in the hand review of the checker
    (rows V07 and V09), where the map plainly showed records on the other slope.
    """
    stated = list(stated or [])
    inferred = sorted(_record_slopes().get(species, set()) - set(stated))
    return {"stated": stated, "inferred": inferred,
            "union": sorted(set(stated) | set(inferred))}
