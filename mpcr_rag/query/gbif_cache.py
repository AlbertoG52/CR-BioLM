"""Per-species cache of cleaned GBIF points, with an update date and an expiry.

``gbif_map.get_points`` used to keep a species' GBIF points forever once they were
fetched. This module adds three things to that cache:

1. A date. Next to each ``<Species>.geojson`` there is a ``<Species>.meta.json``
   with the moment of the GBIF fetch that produced it (``updated_at``, UTC,
   ISO-8601). Caches written before this module have no meta file; their date is
   the file's modification time.
2. An expiry. With ``max_days > 0`` (``MPCR_GBIF_CACHE_MAX_DAYS``), a cache older
   than that is fetched again the next time it is read. ``0`` keeps the previous
   behaviour: the cache never expires, so experiments stay reproducible.
3. A safe refresh. New points replace the old ones only once the fetch succeeded:
   they are written to a temporary file and moved over the old one. If GBIF fails,
   or returns no points for a species that had some, the previous points are kept
   and returned marked ``stale``. After a failed refresh, GBIF is not asked again
   for that species for ``retry_failed_after``, so an outage does not add a slow
   failing call to every query.

Every refresh attempt is appended to ``_refresh_log.jsonl`` in the cache directory.

The frozen download snapshot (DOI 10.15468/dl.8yhee8) is not touched: it backs the
paper's counts and stays pinned.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd

log = logging.getLogger(__name__)

REFRESH_LOG = "_refresh_log.jsonl"
LAST_RUN = "_last_refresh.json"

Fetch = Callable[[str], gpd.GeoDataFrame]


@dataclass(frozen=True)
class CacheInfo:
    """Where a species' points came from."""

    updated_at: str | None  # when they were fetched from GBIF (UTC ISO-8601); None if unknown
    stale: bool = False  # a refresh was due and failed: these are the previous points
    refreshed: bool = False  # fetched from GBIF in this call


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str | None) -> datetime | None:
    """ISO-8601 -> aware datetime (naive values are taken as UTC); None if invalid."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def is_due(updated_at: str | None, max_days: float, now: datetime | None = None) -> bool:
    """True if points fetched at ``updated_at`` are older than ``max_days``.

    ``max_days <= 0`` means the cache never expires. An unknown date is due.
    """
    if max_days <= 0:
        return False
    moment = parse_time(updated_at)
    if moment is None:
        return True
    return (now or utc_now()) - moment > timedelta(days=max_days)


class PointsCache:
    """The ``<Species>.geojson`` files of one directory, with their dates."""

    def __init__(
        self,
        directory: Path,
        *,
        max_days: float = 0,
        retry_failed_after: timedelta = timedelta(hours=1),
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_days = max_days
        self.retry_failed_after = retry_failed_after
        self._clock = clock
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._log_guard = threading.Lock()

    # --- reading ----------------------------------------------------------------

    def get(
        self,
        species: str,
        fetch: Fetch,
        *,
        max_days: float | None = None,
        force: bool = False,
    ) -> tuple[gpd.GeoDataFrame, CacheInfo]:
        """The species' points and their date, fetching them with ``fetch`` if needed.

        They are fetched when there is no cache, when it is older than ``max_days``
        (default: the cache's own) or when ``force`` is set. If that fetch fails and
        there is a previous cache, the previous points come back marked ``stale``;
        with no previous cache the error propagates, as before this module.
        """
        max_days = self.max_days if max_days is None else max_days
        with self._lock_for(species):  # one fetch per species at a time
            geo = self._geo_path(species)
            meta = self._read_meta(species)
            has_cache = geo.exists()
            updated_at = self._updated_at(geo, meta) if has_cache else None
            now = self._clock()

            if has_cache and not force:
                if not is_due(updated_at, max_days, now):
                    return gpd.read_file(geo), CacheInfo(updated_at)
                if self._failed_recently(meta, now):
                    return gpd.read_file(geo), CacheInfo(updated_at, stale=True)

            try:
                points = fetch(species)
            except Exception as exc:
                self._append_log(species, ok=False, error=repr(exc))
                if not has_cache:
                    raise
                log.warning("GBIF refresh of %s failed; keeping points of %s: %r",
                            species, updated_at, exc)
                self._mark_failed(species, meta, updated_at, now, repr(exc))
                return gpd.read_file(geo), CacheInfo(updated_at, stale=True)

            if points.empty and has_cache:
                previous = gpd.read_file(geo)
                if not previous.empty:
                    # A species that had points does not lose all of them at once:
                    # more likely a transient API or taxonomy problem. Keep them; to
                    # accept the empty result, delete the cache file and fetch again.
                    reason = "GBIF returned 0 points; previous points kept"
                    log.warning("GBIF refresh of %s: %s (%s)", species, reason, updated_at)
                    self._append_log(species, ok=False, error=reason)
                    self._mark_failed(species, meta, updated_at, now, reason)
                    return previous, CacheInfo(updated_at, stale=True)

            new_at = now.isoformat(timespec="seconds")
            try:
                self._write(species, points, new_at)
            except OSError as exc:
                # The points are fresh, they just could not be saved (e.g. Windows
                # locking the old file). Serve them; the next read fetches again.
                log.warning("Could not save the GBIF points of %s: %r", species, exc)
            self._append_log(species, ok=True, n=len(points))
            return points, CacheInfo(new_at, refreshed=True)

    def entries(self) -> list[tuple[str, str | None]]:
        """(species, updated_at) of every cached species, oldest first (unknown first)."""
        out = []
        for geo in self.directory.glob("*.geojson"):
            if geo.name.startswith("."):
                continue  # a temporary file of a write in progress
            species = geo.stem.replace("_", " ")
            out.append((species, self._updated_at(geo, self._read_meta(species))))
        return sorted(out, key=lambda e: (e[1] is not None, parse_time(e[1]) or 0, e[0]))

    # --- refresh runs (scripts/refresh_gbif.py) ----------------------------------

    def write_run_summary(self, summary: dict) -> None:
        _write_json(self.directory / LAST_RUN, summary)

    def read_run_summary(self) -> dict | None:
        try:
            return json.loads((self.directory / LAST_RUN).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # --- files ------------------------------------------------------------------

    def _geo_path(self, species: str) -> Path:
        # Same name gbif_map has always used, so existing caches are reused.
        return self.directory / f"{species.replace(' ', '_')}.geojson"

    def _meta_path(self, species: str) -> Path:
        return self.directory / f"{species.replace(' ', '_')}.meta.json"

    def _read_meta(self, species: str) -> dict:
        path = self._meta_path(species)
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("Unreadable %s (%r); treated as missing", path.name, exc)
            return {}
        return meta if isinstance(meta, dict) else {}

    @staticmethod
    def _updated_at(geo: Path, meta: dict) -> str | None:
        if parse_time(meta.get("updated_at")):
            return meta["updated_at"]
        try:  # cache written before this module: its file date
            mtime = geo.stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(mtime, timezone.utc).isoformat(timespec="seconds")

    def _failed_recently(self, meta: dict, now: datetime) -> bool:
        failed = parse_time(meta.get("last_failed_at"))
        return failed is not None and now - failed < self.retry_failed_after

    def _mark_failed(self, species: str, meta: dict, updated_at: str | None,
                     now: datetime, error: str) -> None:
        meta = {**meta, "species": species, "updated_at": updated_at,
                "last_failed_at": now.isoformat(timespec="seconds"), "last_error": error}
        try:
            _write_json(self._meta_path(species), meta)
        except OSError as exc:
            log.warning("Could not record the failed refresh of %s: %r", species, exc)

    def _write(self, species: str, points: gpd.GeoDataFrame, updated_at: str) -> None:
        geo = self._geo_path(species)
        tmp = geo.with_name(f".{geo.stem}.{os.getpid()}.{threading.get_ident()}.tmp.geojson")
        try:
            points.to_file(tmp, driver="GeoJSON")
            _replace(tmp, geo)
        finally:
            tmp.unlink(missing_ok=True)
        # After the points: if this fails, the next read sees an old date and
        # fetches again, which is harmless.
        _write_json(self._meta_path(species), {
            "species": species,
            "updated_at": updated_at,
            "n": len(points),
            "source": "GBIF occurrence API",
        })

    def _append_log(self, species: str, **fields) -> None:
        line = {"at": self._clock().isoformat(timespec="seconds"), "species": species, **fields}
        try:
            with self._log_guard, (self.directory / REFRESH_LOG).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Could not write %s: %r", REFRESH_LOG, exc)

    def _lock_for(self, species: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(species, threading.Lock())


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _replace(src: Path, dst: Path, attempts: int = 5) -> None:
    """os.replace, retried: on Windows it fails while another process reads ``dst``."""
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(0.2 * attempt)