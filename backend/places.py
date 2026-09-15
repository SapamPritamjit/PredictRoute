"""Place-search provider abstraction.

    User text query
      → PlaceSearchProvider.search(query, lat?, lon?, limit)
      → list[PlaceResult]

Photon is the default provider.  The abstraction is thin enough that swapping
to Nominatim, LocationIQ, or any other REST geocoder requires only a new
``PlaceSearchProvider`` subclass — no changes to the API layer or frontend.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Limits ──────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 200
MIN_LIMIT = 1
MAX_LIMIT = 10
DEFAULT_LIMIT = 5
UPSTREAM_TIMEOUT_S = 5.0
REQUEST_INTERVAL_S = 1.1  # respect 1 req/s public server policy
CACHE_TTL_S = 3600.0  # 1 hour
MAX_CACHE_ENTRIES = 512


# ── Normalised result ───────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PlaceResult:
    name: str
    latitude: float
    longitude: float
    display_name: str
    place_type: str
    osm_type: str


# ── Provider abstraction ────────────────────────────────────────
class PlaceSearchProvider(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        *,
        lat: float | None = None,
        lon: float | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[PlaceResult]:
        ...


# ── Rate limiter (thread-safe) ─────────────────────────────────
class _RateLimiter:
    def __init__(self, interval: float):
        self._interval = interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            gap = self._interval - (now - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()


# ── Simple TTL cache ───────────────────────────────────────────
class _TTLCache:
    """Dict-based cache with per-entry TTL and max-size eviction."""

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES, ttl: float = CACHE_TTL_S):
        self._max = max_entries
        self._ttl = ttl
        self._store: dict[str, tuple[float, list[PlaceResult]]] = {}
        self._lock = threading.Lock()

    def _key(self, query: str, lat: float | None, lon: float | None, limit: int) -> str:
        raw = f"{query}|{lat}|{lon}|{limit}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, lat: float | None, lon: float | None, limit: int):
        k = self._key(query, lat, lon, limit)
        with self._lock:
            entry = self._store.get(k)
            if entry is None:
                return None
            ts, results = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[k]
                return None
            return results

    def put(self, query: str, lat: float | None, lon: float | None, limit: int,
            results: list[PlaceResult]):
        k = self._key(query, lat, lon, limit)
        with self._lock:
            if len(self._store) >= self._max:
                oldest_key = min(self._store, key=lambda k2: self._store[k2][0])
                del self._store[oldest_key]
            self._store[k] = (time.monotonic(), results)

    def clear(self):
        with self._lock:
            self._store.clear()


# ── Photon provider ────────────────────────────────────────────
class PhotonProvider(PlaceSearchProvider):
    """Photon (komoot) geocoder — free, no API key, search-as-you-type."""

    def __init__(
        self,
        base_url: str = "https://photon.komoot.io",
        timeout: float = UPSTREAM_TIMEOUT_S,
        rate_limit_interval: float = REQUEST_INTERVAL_S,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._limiter = _RateLimiter(rate_limit_interval)
        self._cache = _TTLCache()
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "PredictRoute/0.1 (student-project; github.com/predict-route)"},
        )

    # ── public ──
    def search(
        self,
        query: str,
        *,
        lat: float | None = None,
        lon: float | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[PlaceResult]:
        query = query.strip()
        if not query or len(query) > MAX_QUERY_LENGTH:
            return []

        limit = max(MIN_LIMIT, min(limit, MAX_LIMIT))

        cached = self._cache.get(query, lat, lon, limit)
        if cached is not None:
            return cached

        self._limiter.wait()

        params: dict = {"q": query, "limit": limit, "lang": "en"}
        if lat is not None and lon is not None:
            params["lat"] = round(lat, 4)
            params["lon"] = round(lon, 4)

        try:
            resp = self._client.get(f"{self._base_url}/api/", params=params)
            resp.raise_for_status()
        except httpx.TimeoutException:
            logger.warning("Photon timeout for query=%s", query)
            return []
        except httpx.HTTPStatusError as exc:
            logger.warning("Photon HTTP %s for query=%s", exc.response.status_code, query)
            return []
        except httpx.RequestError as exc:
            logger.warning("Photon request error for query=%s: %s", query, exc)
            return []

        try:
            data = resp.json()
        except Exception:
            logger.warning("Photon invalid JSON for query=%s", query)
            return []

        results = self._normalise(data)
        self._cache.put(query, lat, lon, limit, results)
        return results

    def close(self):
        self._client.close()

    # ── internal ──
    @staticmethod
    def _normalise(data: dict) -> list[PlaceResult]:
        """Convert Photon GeoJSON features to PlaceResult list."""
        features = data.get("features", [])
        results: list[PlaceResult] = []
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue

            lon_c, lat_c = coords[0], coords[1]
            name = props.get("name") or props.get("city") or ""
            display_parts = [
                props.get("name", ""),
                props.get("street", ""),
                props.get("city", ""),
                props.get("state", ""),
                props.get("country", ""),
            ]
            display_name = ", ".join(p for p in display_parts if p)

            place_type = props.get("type") or props.get("osm_value") or ""
            osm_type = props.get("osm_type") or ""

            results.append(
                PlaceResult(
                    name=name,
                    latitude=float(lat_c),
                    longitude=float(lon_c),
                    display_name=display_name,
                    place_type=place_type,
                    osm_type=osm_type,
                )
            )
        return results
