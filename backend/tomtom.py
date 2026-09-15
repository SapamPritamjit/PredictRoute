"""TomTom Traffic Flow API client.

Provides live traffic flow data for Delhi road segments.
Falls back gracefully on errors, timeouts, or rate limits.

Architecture:
    TomTomClient.get_flow_for_point(lat, lon) -> FlowSegment
    TomTomClient.get_flow_for_segments(segment_ids, midpoints) -> dict
    TomTomClient.get_flow_for_segments_concurrent(...) -> dict

    Flow ratio = freeFlowSpeed / currentSpeed
    Clamped to [R_MIN, R_MAX] to match CatBoost output range.

No API key is ever exposed to the frontend or logged.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
R_MIN = 0.5
R_MAX = 3.721
DEFAULT_CONCURRENCY = 8
DEFAULT_BATCH_TIMEOUT = 8.0
SEGMENT_CACHE_TTL = 60.0


@dataclass
class FlowSegment:
    """Traffic flow data for a single road segment."""

    segment_id: int
    current_speed: float
    free_flow_speed: float
    ratio: float  # freeFlowSpeed / currentSpeed, clamped
    confidence: float
    road_closure: bool


class TomTomClient:
    """TomTom Traffic Flow API client with rate limiting and caching."""

    FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive failures before disabling

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
        cache_ttl: float = 300.0,  # 5 minutes for point cache
        concurrency: int = DEFAULT_CONCURRENCY,
        batch_timeout: float = DEFAULT_BATCH_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("TOMTOM_API_KEY", "")
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.concurrency = concurrency
        self.batch_timeout = batch_timeout
        self._client: httpx.Client | None = None
        self._cache: dict[str, tuple[float, FlowSegment]] = {}
        self._segment_cache: dict[int, tuple[float, FlowSegment]] = {}
        self._last_query_time: float = 0.0
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """Check if TomTom API key is configured."""
        return bool(self.api_key)

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def _cache_key(self, lat: float, lon: float) -> str:
        """Generate cache key for a point."""
        return f"{lat:.5f}:{lon:.5f}"

    def _get_cached(self, key: str) -> Optional[FlowSegment]:
        """Get cached result if still valid."""
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return result
            del self._cache[key]
        return None

    def _set_cached(self, key: str, result: FlowSegment):
        """Cache a result."""
        self._cache[key] = (time.time(), result)

    def _get_segment_cached(self, seg_id: int) -> Optional[FlowSegment]:
        """Get cached segment result with short TTL."""
        if seg_id in self._segment_cache:
            ts, result = self._segment_cache[seg_id]
            if time.time() - ts < SEGMENT_CACHE_TTL:
                return result
            del self._segment_cache[seg_id]
        return None

    def _set_segment_cached(self, seg_id: int, result: FlowSegment):
        """Cache a segment result with short TTL."""
        self._segment_cache[seg_id] = (time.time(), result)

    def _parse_flow_response(self, data: dict) -> Optional[FlowSegment]:
        """Parse TomTom flow response into FlowSegment."""
        flow = data.get("flowSegmentData", {})
        current_speed = flow.get("currentSpeed")
        free_flow_speed = flow.get("freeFlowSpeed")

        if current_speed is None or free_flow_speed is None:
            return None
        if not isinstance(current_speed, (int, float)) or current_speed <= 0:
            return None
        if not isinstance(free_flow_speed, (int, float)) or free_flow_speed <= 0:
            return None

        ratio = free_flow_speed / current_speed
        ratio = max(R_MIN, min(R_MAX, ratio))

        return FlowSegment(
            segment_id=0,
            current_speed=float(current_speed),
            free_flow_speed=float(free_flow_speed),
            ratio=float(ratio),
            confidence=float(flow.get("confidence", 0.0)),
            road_closure=bool(flow.get("roadClosure", False)),
        )

    def _record_failure(self):
        """Record a failure and open circuit breaker if threshold exceeded."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open = True
            logger.warning(
                "TomTom circuit breaker opened after %d consecutive failures",
                self._consecutive_failures,
            )

    def _record_success(self):
        """Reset failure counter on success."""
        self._consecutive_failures = 0

    def get_flow_for_point(self, lat: float, lon: float) -> Optional[FlowSegment]:
        """Query TomTom Traffic Flow API for a single point.

        Returns FlowSegment or None on error.
        Never logs or exposes the API key.
        """
        if not self.available:
            return None

        if self._circuit_open:
            return None

        cache_key = self._cache_key(lat, lon)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Rate limiting: minimum 200ms between queries
        now = time.time()
        elapsed = now - self._last_query_time
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)

        try:
            client = self._get_client()
            resp = client.get(
                self.FLOW_URL,
                params={"key": self.api_key, "point": f"{lat},{lon}"},
            )
            self._last_query_time = time.time()

            if resp.status_code == 429:
                logger.warning("TomTom rate limit exceeded (429)")
                self._record_failure()
                return None

            if resp.status_code != 200:
                logger.warning("TomTom API error: HTTP %d", resp.status_code)
                self._record_failure()
                return None

            self._record_success()
            result = self._parse_flow_response(resp.json())
            if result is not None:
                self._set_cached(cache_key, result)
            return result

        except httpx.TimeoutException:
            logger.warning("TomTom API timeout for (%.4f, %.4f)", lat, lon)
            self._record_failure()
            return None
        except Exception as e:
            logger.warning("TomTom API error for (%.4f, %.4f): %s", lat, lon, type(e).__name__)
            self._record_failure()
            return None

    def _query_single_segment(
        self, seg_id: int, lat: float, lon: float
    ) -> Optional[FlowSegment]:
        """Query a single segment, checking cache first."""
        cached = self._get_segment_cached(seg_id)
        if cached is not None:
            return cached

        # Check point cache too
        cache_key = self._cache_key(lat, lon)
        cached = self._get_cached(cache_key)
        if cached is not None:
            cached.segment_id = seg_id
            self._set_segment_cached(seg_id, cached)
            return cached

        if not self.available or self._circuit_open:
            return None

        try:
            client = self._get_client()
            resp = client.get(
                self.FLOW_URL,
                params={"key": self.api_key, "point": f"{lat},{lon}"},
            )

            if resp.status_code == 429:
                with self._lock:
                    self._record_failure()
                return None

            if resp.status_code != 200:
                with self._lock:
                    self._record_failure()
                return None

            with self._lock:
                self._record_success()

            result = self._parse_flow_response(resp.json())
            if result is not None:
                result.segment_id = seg_id
                self._set_cached(cache_key, result)
                self._set_segment_cached(seg_id, result)
            return result

        except httpx.TimeoutException:
            with self._lock:
                self._record_failure()
            return None
        except Exception:
            with self._lock:
                self._record_failure()
            return None

    def get_flow_for_segments_concurrent(
        self,
        segment_ids: list[int],
        midpoints: dict[int, tuple[float, float]],
        max_queries: int = 100,
    ) -> dict[int, FlowSegment]:
        """Query TomTom for multiple segments using bounded concurrency.

        Uses a thread pool to make parallel HTTP requests, respecting
        TomTom QPS limits via the semaphore. Falls back gracefully on
        errors, timeouts, or rate limits.

        Args:
            segment_ids: List of segment IDs to query
            midpoints: Dict mapping segment_id -> (lat, lon) midpoint
            max_queries: Maximum number of API queries

        Returns:
            Dict mapping segment_id -> FlowSegment for successful queries
        """
        results: dict[int, FlowSegment] = {}

        if not self.available or self._circuit_open:
            return results

        # Filter to segments with valid midpoints
        query_ids = [sid for sid in segment_ids if sid in midpoints]

        # Sample if too many segments
        if len(query_ids) > max_queries:
            step = max(1, len(query_ids) // max_queries)
            query_ids = query_ids[::step][:max_queries]

        if not query_ids:
            return results

        # Check cache first — skip uncached segments
        uncached_ids = []
        for seg_id in query_ids:
            cached = self._get_segment_cached(seg_id)
            if cached is not None:
                results[seg_id] = cached
            else:
                uncached_ids.append(seg_id)

        if not uncached_ids:
            return results

        # Query uncached segments concurrently
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {}
            for seg_id in uncached_ids:
                lat, lon = midpoints[seg_id]
                future = executor.submit(self._query_single_segment, seg_id, lat, lon)
                futures[future] = seg_id

            for future in as_completed(futures, timeout=self.batch_timeout):
                seg_id = futures[future]
                try:
                    flow = future.result()
                    if flow is not None:
                        results[seg_id] = flow
                except Exception:
                    pass

        elapsed = time.perf_counter() - t_start
        logger.debug(
            "TomTom concurrent: %d/%d segments in %.2fs",
            len(results), len(query_ids), elapsed,
        )

        return results

    def get_flow_for_segments(
        self,
        segment_ids: list[int],
        midpoints: dict[int, tuple[float, float]],
        max_queries: int = 100,
    ) -> dict[int, FlowSegment]:
        """Query TomTom for multiple segments (sequential, backward-compatible).

        Args:
            segment_ids: List of segment IDs to query
            midpoints: Dict mapping segment_id -> (lat, lon) midpoint
            max_queries: Maximum number of API queries (rate limit protection)

        Returns:
            Dict mapping segment_id -> FlowSegment for successful queries
        """
        results: dict[int, FlowSegment] = {}

        if not self.available:
            return results

        # Sample if too many segments
        query_ids = segment_ids
        if len(query_ids) > max_queries:
            step = max(1, len(query_ids) // max_queries)
            query_ids = query_ids[::step][:max_queries]

        for seg_id in query_ids:
            if seg_id not in midpoints:
                continue

            lat, lon = midpoints[seg_id]
            flow = self.get_flow_for_point(lat, lon)

            if flow is not None:
                flow.segment_id = seg_id
                results[seg_id] = flow

        return results

    def get_flow_for_route_segments(
        self,
        route_segment_ids: list[int],
        all_segment_midpoints: dict[int, tuple[float, float]],
        max_queries: int = 50,
    ) -> dict[int, FlowSegment]:
        """Query TomTom specifically for route-relevant segments.

        Prioritizes segments along the route, then samples remaining segments
        for area-wide traffic context.
        """
        results: dict[int, FlowSegment] = {}

        if not self.available:
            return results

        # Query route segments first (up to max_queries)
        route_results = self.get_flow_for_segments(
            route_segment_ids, all_segment_midpoints, max_queries=max_queries
        )
        results.update(route_results)

        return results
