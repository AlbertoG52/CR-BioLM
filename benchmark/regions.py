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
            cur.execute("""SELECT species, region, count(*) FROM mpcr.occurrences
                           WHERE species IS NOT NULL AND region IS NOT NULL
                           GROUP BY species, region HAVING count(*) >= %s""",
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
