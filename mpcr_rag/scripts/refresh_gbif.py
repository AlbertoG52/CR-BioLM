"""Refresh the cached GBIF points of the species already in the cache (RR_56).

The MCP server refreshes a species' points on its own when they are older than
MPCR_GBIF_CACHE_MAX_DAYS, but only when someone asks for that species, and that
person waits for the GBIF call. This script does it ahead of time for every cached
species, so it can run on a schedule (cron, Windows Task Scheduler) at night.

  python -m mpcr_rag.scripts.refresh_gbif                     # older than MPCR_GBIF_CACHE_MAX_DAYS
  python -m mpcr_rag.scripts.refresh_gbif --older-than 30     # older than 30 days
  python -m mpcr_rag.scripts.refresh_gbif --all               # every cached species
  python -m mpcr_rag.scripts.refresh_gbif --species "Quercus costaricensis"
  python -m mpcr_rag.scripts.refresh_gbif --status            # last run and the oldest caches

A species whose refresh fails keeps its previous points. Every attempt is appended
to data/gbif_cache/_refresh_log.jsonl and the run's summary is written to
data/gbif_cache/_last_refresh.json. Exit code 0 if every refresh worked, 1 if some
failed, so a scheduler can alert.
"""
from __future__ import annotations

import argparse
import sys
import time

from .. import config
from ..query import gbif_map
from ..query.gbif_cache import is_due, utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mpcr_rag.scripts.refresh_gbif",
                                     description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--older-than", type=float, metavar="DAYS",
                       help="refresh the species whose points are older than DAYS")
    which.add_argument("--all", action="store_true", help="refresh every cached species")
    which.add_argument("--species", action="append", metavar="NAME",
                       help="refresh this species (repeatable); it need not be cached yet")
    which.add_argument("--status", action="store_true",
                       help="show the last run and the oldest caches; refresh nothing")
    parser.add_argument("--pause", type=float, default=1.0, metavar="SECONDS",
                        help="pause between species, to be gentle with the GBIF API (default 1)")
    args = parser.parse_args(argv)

    cache = gbif_map.points_cache()
    if args.status:
        return _status(cache)

    if args.species:
        targets = args.species
    else:
        days = 0.0 if args.all else (args.older_than if args.older_than is not None
                                     else config.GBIF_CACHE_MAX_DAYS)
        if not args.all and days <= 0:
            parser.error("MPCR_GBIF_CACHE_MAX_DAYS is 0 (the cache never expires): "
                         "pass --older-than DAYS or --all")
        now = utc_now()
        targets = [sp for sp, at in cache.entries() if args.all or is_due(at, days, now)]

    started = utc_now()
    refreshed: list[str] = []
    kept: list[str] = []
    failed: list[str] = []
    for i, species in enumerate(targets, 1):
        if i > 1 and args.pause > 0:
            time.sleep(args.pause)
        try:
            points, info = gbif_map.get_points_info(species, force=True)
        except Exception as exc:  # no previous points to fall back on
            failed.append(species)
            print(f"[{i}/{len(targets)}] {species}: FAILED, nothing cached ({exc!r})", flush=True)
            continue
        if info.stale:
            kept.append(species)
            print(f"[{i}/{len(targets)}] {species}: FAILED, kept the points of "
                  f"{info.updated_at}", flush=True)
        else:
            refreshed.append(species)
            print(f"[{i}/{len(targets)}] {species}: {len(points)} points", flush=True)

    summary = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": utc_now().isoformat(timespec="seconds"),
        "requested": len(targets),
        "refreshed": len(refreshed),
        "kept_previous": kept,
        "failed": failed,
    }
    cache.write_run_summary(summary)
    print(f"Refreshed {len(refreshed)} of {len(targets)}; kept previous points for "
          f"{len(kept)}; failed with nothing cached: {len(failed)}.")
    return 1 if kept or failed else 0


def _status(cache) -> int:
    entries = cache.entries()
    last = cache.read_run_summary()
    print(f"Cache: {cache.directory}")
    print(f"Species cached: {len(entries)}")
    max_days = config.GBIF_CACHE_MAX_DAYS
    if max_days > 0:
        now = utc_now()
        due = sum(1 for _sp, at in entries if is_due(at, max_days, now))
        print(f"Expiry: {max_days:g} days ({due} due now)")
    else:
        print("Expiry: never (MPCR_GBIF_CACHE_MAX_DAYS=0)")
    if last:
        print(f"Last run: {last.get('finished_at')} — refreshed {last.get('refreshed')} of "
              f"{last.get('requested')}, kept previous {len(last.get('kept_previous') or [])}, "
              f"failed {len(last.get('failed') or [])}")
    else:
        print("Last run: never")
    if entries:
        print("Oldest:")
        for species, at in entries[:10]:
            print(f"  {at or 'unknown':<26} {species}")
    return 0


if __name__ == "__main__":
    sys.exit(main())