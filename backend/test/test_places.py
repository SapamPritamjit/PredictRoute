"""Unit tests for backend.places — all mocked, no live network required."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.places import (
    MAX_CACHE_ENTRIES,
    MAX_LIMIT,
    MAX_QUERY_LENGTH,
    DEFAULT_LIMIT,
    PhotonProvider,
    PlaceResult,
    _RateLimiter,
    _TTLCache,
)


# ═══════════════════════════════════════════════════════════════
# FIXTURES — sample Photon API responses
# ═══════════════════════════════════════════════════════════════

PHOTON_SINGLE_RESULT = {
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [77.2167, 28.6315]},
            "properties": {
                "name": "Connaught Place",
                "street": "Rajiv Chowk",
                "city": "New Delhi",
                "state": "Delhi",
                "country": "India",
                "type": "city",
                "osm_type": "relation",
                "osm_value": "city",
            },
        }
    ]
}

PHOTON_MULTI_RESULTS = {
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [77.2295, 28.6129]},
            "properties": {
                "name": "India Gate",
                "city": "New Delhi",
                "state": "Delhi",
                "country": "India",
                "type": "attraction",
                "osm_type": "node",
                "osm_value": "attraction",
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [77.1700, 28.5800]},
            "properties": {
                "name": "Indirapuram",
                "city": "Ghaziabad",
                "state": "Uttar Pradesh",
                "country": "India",
                "type": "city",
                "osm_type": "relation",
                "osm_value": "city",
            },
        },
    ]
}

PHOTON_EMPTY = {"features": []}

PHOTON_MALFORMED = {"no_features_key": True}

PHOTON_MISSING_COORDS = {
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": []},
            "properties": {"name": "Empty"},
        }
    ]
}


# ═══════════════════════════════════════════════════════════════
# 1. Normalisation
# ═══════════════════════════════════════════════════════════════

class TestNormalisation:
    def test_single_result(self):
        results = PhotonProvider._normalise(PHOTON_SINGLE_RESULT)
        assert len(results) == 1
        r = results[0]
        assert r.name == "Connaught Place"
        assert r.latitude == pytest.approx(28.6315)
        assert r.longitude == pytest.approx(77.2167)
        assert "New Delhi" in r.display_name
        assert r.place_type == "city"
        assert r.osm_type == "relation"

    def test_multiple_results(self):
        results = PhotonProvider._normalise(PHOTON_MULTI_RESULTS)
        assert len(results) == 2
        assert results[0].name == "India Gate"
        assert results[1].name == "Indirapuram"

    def test_empty_results(self):
        results = PhotonProvider._normalise(PHOTON_EMPTY)
        assert results == []

    def test_malformed_response(self):
        results = PhotonProvider._normalise(PHOTON_MALFORMED)
        assert results == []

    def test_missing_coordinates(self):
        results = PhotonProvider._normalise(PHOTON_MISSING_COORDS)
        assert len(results) == 0

    def test_place_result_is_frozen(self):
        r = PlaceResult(
            name="X", latitude=0.0, longitude=0.0,
            display_name="X", place_type="city", osm_type="node",
        )
        with pytest.raises(AttributeError):
            r.name = "Y"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════
# 2. Cache
# ═══════════════════════════════════════════════════════════════

class TestCache:
    def test_put_and_get(self):
        cache = _TTLCache(max_entries=10, ttl=60.0)
        results = [
            PlaceResult("A", 1.0, 2.0, "A display", "city", "node"),
        ]
        cache.put("delhi", None, None, 5, results)
        got = cache.get("delhi", None, None, 5)
        assert got is not None
        assert got[0].name == "A"

    def test_cache_miss(self):
        cache = _TTLCache(max_entries=10, ttl=60.0)
        assert cache.get("missing", None, None, 5) is None

    def test_different_params_different_key(self):
        cache = _TTLCache(max_entries=10, ttl=60.0)
        r1 = [PlaceResult("A", 1.0, 2.0, "A", "city", "node")]
        cache.put("q", None, None, 5, r1)
        assert cache.get("q", None, None, 5) is not None
        assert cache.get("q", 28.0, 77.0, 5) is None

    def test_eviction_at_max(self):
        cache = _TTLCache(max_entries=3, ttl=60.0)
        for i in range(5):
            cache.put(f"q{i}", None, None, 5, [])
        assert len(cache._store) <= 3

    def test_clear(self):
        cache = _TTLCache(max_entries=10, ttl=60.0)
        cache.put("q", None, None, 5, [])
        cache.clear()
        assert cache.get("q", None, None, 5) is None


# ═══════════════════════════════════════════════════════════════
# 3. Rate limiter
# ═══════════════════════════════════════════════════════════════

class TestRateLimiter:
    def test_first_call_no_delay(self):
        limiter = _RateLimiter(interval=10.0)
        import time
        t0 = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1


# ═══════════════════════════════════════════════════════════════
# 4. PhotonProvider — mocked HTTP
# ═══════════════════════════════════════════════════════════════

def _make_provider() -> PhotonProvider:
    return PhotonProvider(rate_limit_interval=0.0)


class TestPhotonProviderSearch:
    def test_successful_search(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = PHOTON_SINGLE_RESULT

        with patch.object(provider._client, "get", return_value=mock_resp):
            results = provider.search("Connaught Place")

        assert len(results) == 1
        assert results[0].name == "Connaught Place"

    def test_empty_query(self):
        provider = _make_provider()
        assert provider.search("") == []
        assert provider.search("   ") == []

    def test_query_too_long(self):
        provider = _make_provider()
        assert provider.search("x" * (MAX_QUERY_LENGTH + 1)) == []

    def test_limit_capping(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = PHOTON_MULTI_RESULTS

        with patch.object(provider._client, "get", return_value=mock_resp) as m:
            provider.search("Delhi", limit=100)
            call_params = m.call_args[1]["params"]
            assert call_params["limit"] == MAX_LIMIT

    def test_limit_floor(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = PHOTON_SINGLE_RESULT

        with patch.object(provider._client, "get", return_value=mock_resp) as m:
            provider.search("Delhi", limit=0)
            call_params = m.call_args[1]["params"]
            assert call_params["limit"] == 1

    def test_lat_lon_passed_through(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = PHOTON_SINGLE_RESULT

        with patch.object(provider._client, "get", return_value=mock_resp) as m:
            provider.search("Delhi", lat=28.6, lon=77.2)
            call_params = m.call_args[1]["params"]
            assert call_params["lat"] == 28.6
            assert call_params["lon"] == 77.2

    def test_no_lat_lon_when_none(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = PHOTON_SINGLE_RESULT

        with patch.object(provider._client, "get", return_value=mock_resp) as m:
            provider.search("Delhi")
            call_params = m.call_args[1]["params"]
            assert "lat" not in call_params
            assert "lon" not in call_params

    def test_timeout_returns_empty(self):
        provider = _make_provider()
        with patch.object(
            provider._client, "get", side_effect=httpx.TimeoutException("timeout")
        ):
            results = provider.search("Delhi")
        assert results == []

    def test_http_error_returns_empty(self):
        provider = _make_provider()
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500)
        )
        with patch.object(provider._client, "get", return_value=resp):
            results = provider.search("Delhi")
        assert results == []

    def test_network_error_returns_empty(self):
        provider = _make_provider()
        with patch.object(
            provider._client, "get",
            side_effect=httpx.RequestError("network down"),
        ):
            results = provider.search("Delhi")
        assert results == []

    def test_malformed_json_returns_empty(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("bad json")

        with patch.object(provider._client, "get", return_value=mock_resp):
            results = provider.search("Delhi")
        assert results == []


# ═══════════════════════════════════════════════════════════════
# 5. Cache integration
# ═══════════════════════════════════════════════════════════════

class TestPhotonProviderCache:
    def test_second_call_uses_cache(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = PHOTON_SINGLE_RESULT

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch.object(provider._client, "get", side_effect=mock_get):
            r1 = provider.search("Connaught Place")
            r2 = provider.search("Connaught Place")

        assert call_count == 1
        assert len(r1) == 1
        assert len(r2) == 1
        assert r1[0].name == r2[0].name

    def test_different_query_makes_separate_call(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = PHOTON_SINGLE_RESULT

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch.object(provider._client, "get", side_effect=mock_get):
            provider.search("Connaught Place")
            provider.search("India Gate")

        assert call_count == 2


# ═══════════════════════════════════════════════════════════════
# 6. API-level integration (TestClient, mocked provider)
# ═══════════════════════════════════════════════════════════════

class TestPlacesEndpoint:
    @pytest.fixture(autouse=True)
    def _setup_app(self):
        from fastapi.testclient import TestClient

        from backend.api import app
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_missing_query_param(self):
        r = self.client.get("/api/places")
        assert r.status_code == 422

    def test_empty_query(self):
        r = self.client.get("/api/places?q=")
        assert r.status_code == 422

    def test_whitespace_only_query(self):
        r = self.client.get("/api/places?q=%20%20%20")
        assert r.status_code == 422

    def test_query_too_long(self):
        r = self.client.get(f"/api/places?q={'x' * 250}")
        assert r.status_code == 422

    def test_limit_zero_rejected(self):
        r = self.client.get("/api/places?q=Delhi&limit=0")
        assert r.status_code == 422

    def test_limit_exceeds_max(self):
        r = self.client.get("/api/places?q=Delhi&limit=100")
        assert r.status_code == 422

    def test_valid_query_mocked(self):
        from backend import api as api_mod
        original_provider = api_mod._provider

        mock_provider = MagicMock()
        mock_provider.search.return_value = [
            PlaceResult(
                name="Connaught Place",
                latitude=28.6315,
                longitude=77.2167,
                display_name="Connaught Place, New Delhi, India",
                place_type="city",
                osm_type="relation",
            )
        ]

        api_mod._provider = mock_provider
        try:
            r = self.client.get("/api/places?q=Connaught+Place")
            assert r.status_code == 200
            data = r.json()
            assert data["count"] == 1
            assert data["results"][0]["name"] == "Connaught Place"
            mock_provider.search.assert_called_once()
        finally:
            api_mod._provider = original_provider

    def test_provider_error_returns_502(self):
        from backend import api as api_mod
        original_provider = api_mod._provider

        mock_provider = MagicMock()
        mock_provider.search.side_effect = RuntimeError("provider exploded")

        api_mod._provider = mock_provider
        try:
            r = self.client.get("/api/places?q=Delhi")
            assert r.status_code == 502
            assert "provider" in r.json()["detail"].lower()
        finally:
            api_mod._provider = original_provider
