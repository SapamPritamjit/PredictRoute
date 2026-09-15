"""Tests for TomTom Traffic Flow API client.

Covers: parse/normalize, cache, rate limiting, error handling (429/timeout/invalid),
fallback behaviour, and API key security (never exposed in logs or exceptions).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from backend.tomtom import TomTomClient, FlowSegment, R_MIN, R_MAX


# ── Fixtures ────────────────────────────────────────────────────
@pytest.fixture
def client():
    """TomTomClient with a test API key."""
    return TomTomClient(api_key="test-key-12345", timeout=5.0, cache_ttl=60.0)


@pytest.fixture
def client_no_key():
    """TomTomClient with no API key."""
    import os
    old = os.environ.pop("TOMTOM_API_KEY", None)
    try:
        c = TomTomClient(api_key="", timeout=5.0)
        yield c
    finally:
        if old is not None:
            os.environ["TOMTOM_API_KEY"] = old


@pytest.fixture
def sample_flow_response():
    """Sample TomTom flow API JSON response."""
    return {
        "flowSegmentData": {
            "currentSpeed": 30.0,
            "freeFlowSpeed": 50.0,
            "confidence": 0.95,
            "roadClosure": False,
        }
    }


@pytest.fixture
def sample_midpoints():
    """Sample segment midpoints dict."""
    return {
        1001: (28.6139, 77.2090),  # Connaught Place
        1002: (28.6280, 77.2195),  # Rajiv Chowk
        1003: (28.6420, 77.2330),  # somewhere else
    }


# ── Basic availability ──────────────────────────────────────────
class TestAvailability:
    def test_available_with_key(self, client):
        assert client.available is True

    def test_not_available_without_key(self, client_no_key):
        assert client_no_key.available is False

    def test_available_from_env(self, monkeypatch):
        monkeypatch.setenv("TOMTOM_API_KEY", "env-key")
        c = TomTomClient()
        assert c.available is True

    def test_not_available_no_env(self, monkeypatch):
        monkeypatch.delenv("TOMTOM_API_KEY", raising=False)
        c = TomTomClient(api_key="")
        assert c.available is False


# ── Parse and normalize ─────────────────────────────────────────
class TestParseAndNormalize:
    def test_parse_valid_response(self, client, sample_flow_response):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)

            assert result is not None
            assert isinstance(result, FlowSegment)
            assert result.current_speed == 30.0
            assert result.free_flow_speed == 50.0
            assert result.confidence == 0.95
            assert result.road_closure is False

    def test_ratio_computed_correctly(self, client, sample_flow_response):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)

            expected_ratio = 50.0 / 30.0  # ~1.667
            assert abs(result.ratio - expected_ratio) < 0.001

    def test_ratio_clamped_to_max(self, client):
        """When freeFlowSpeed / currentSpeed > R_MAX, ratio should be clamped."""
        response = {
            "flowSegmentData": {
                "currentSpeed": 5.0,
                "freeFlowSpeed": 50.0,  # ratio = 10.0, way above R_MAX
                "confidence": 0.9,
                "roadClosure": False,
            }
        }
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result.ratio == R_MAX

    def test_ratio_clamped_to_min(self, client):
        """When freeFlowSpeed / currentSpeed < R_MIN, ratio should be clamped."""
        response = {
            "flowSegmentData": {
                "currentSpeed": 100.0,
                "freeFlowSpeed": 10.0,  # ratio = 0.1, below R_MIN
                "confidence": 0.9,
                "roadClosure": False,
            }
        }
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result.ratio == R_MIN

    def test_road_closure_detected(self, client):
        response = {
            "flowSegmentData": {
                "currentSpeed": 2.0,
                "freeFlowSpeed": 50.0,
                "confidence": 0.9,
                "roadClosure": True,
            }
        }
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is not None
            assert result.road_closure is True


# ── Missing / invalid data ──────────────────────────────────────
class TestMissingData:
    def test_missing_current_speed(self, client):
        response = {"flowSegmentData": {"freeFlowSpeed": 50.0, "confidence": 0.9}}
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None

    def test_missing_free_flow_speed(self, client):
        response = {"flowSegmentData": {"currentSpeed": 30.0, "confidence": 0.9}}
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None

    def test_empty_flow_segment_data(self, client):
        response = {"flowSegmentData": {}}
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None

    def test_invalid_current_speed_negative(self, client):
        response = {
            "flowSegmentData": {
                "currentSpeed": -10.0,
                "freeFlowSpeed": 50.0,
                "confidence": 0.9,
            }
        }
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None

    def test_invalid_free_flow_speed_string(self, client):
        response = {
            "flowSegmentData": {
                "currentSpeed": 30.0,
                "freeFlowSpeed": "fast",
                "confidence": 0.9,
            }
        }
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None

    def test_no_flow_segment_data_key(self, client):
        response = {}
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None


# ── Error handling ──────────────────────────────────────────────
class TestErrorHandling:
    def test_rate_limit_429(self, client):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None

    def test_server_error_500(self, client):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None

    def test_timeout_returns_none(self, client):
        import httpx

        with patch.object(client._get_client(), "get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("timeout")
            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None

    def test_network_error_returns_none(self, client):
        import httpx

        with patch.object(client._get_client(), "get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("connection refused")
            result = client.get_flow_for_point(28.6139, 77.2090)
            assert result is None


# ── Cache ────────────────────────────────────────────────────────
class TestCache:
    def test_second_call_uses_cache(self, client, sample_flow_response):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result1 = client.get_flow_for_point(28.6139, 77.2090)
            result2 = client.get_flow_for_point(28.6139, 77.2090)

            assert result1 is result2
            assert mock_get.call_count == 1  # only one HTTP call

    def test_different_point_makes_separate_call(self, client, sample_flow_response):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            client.get_flow_for_point(28.6139, 77.2090)
            client.get_flow_for_point(28.6280, 77.2195)

            assert mock_get.call_count == 2

    def test_cache_expired(self, client, sample_flow_response):
        """Cache entry should be ignored after TTL."""
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            client.get_flow_for_point(28.6139, 77.2090)

            # Manually expire cache entry
            key = client._cache_key(28.6139, 77.2090)
            ts, _ = client._cache[key]
            client._cache[key] = (ts - 120.0, client._cache[key][1])

            # Next call should hit API again
            client.get_flow_for_point(28.6139, 77.2090)
            assert mock_get.call_count == 2


# ── Fallback: no API key ────────────────────────────────────────
class TestFallback:
    def test_no_key_returns_none(self, client_no_key):
        result = client_no_key.get_flow_for_point(28.6139, 77.2090)
        assert result is None

    def test_segments_no_key_returns_empty(self, client_no_key, sample_midpoints):
        result = client_no_key.get_flow_for_segments(
            [1001, 1002, 1003], sample_midpoints
        )
        assert result == {}

    def test_segments_with_key_queries_api(self, client, sample_flow_response, sample_midpoints):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_segments(
                [1001, 1002], sample_midpoints, max_queries=10
            )

            assert len(result) == 2
            assert 1001 in result
            assert 1002 in result

    def test_segments_missing_midpoint_skipped(self, client, sample_flow_response, sample_midpoints):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_segments(
                [1001, 9999], sample_midpoints, max_queries=10
            )

            assert len(result) == 1
            assert 1001 in result
            assert 9999 not in result

    def test_segments_sampling(self, client, sample_flow_response, sample_midpoints):
        """When segment_ids > max_queries, should sample."""
        # Add more midpoints
        midpoints = dict(sample_midpoints)
        for i in range(200):
            midpoints[2000 + i] = (28.6 + i * 0.001, 77.2 + i * 0.001)

        seg_ids = list(range(1001, 1001 + 200))

        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_segments(seg_ids, midpoints, max_queries=10)

            # Should not query more than max_queries
            assert mock_get.call_count <= 10


# ── API key security ────────────────────────────────────────────
class TestApiKeySecurity:
    def test_api_key_not_in_exception_message(self, client_no_key):
        """API key should never appear in exception messages."""
        import io
        import logging

        handler = logging.StreamHandler(io.StringIO())
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("backend.tomtom")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            client_no_key.get_flow_for_point(28.6139, 77.2090)
        finally:
            logger.removeHandler(handler)

        log_output = handler.stream.getvalue()
        assert "test-key-12345" not in log_output
        assert "api_key" not in log_output.lower()

    def test_api_key_not_in_flow_segment(self, client, sample_flow_response):
        """FlowSegment dataclass should not contain API key."""
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_point(28.6139, 77.2090)

            # FlowSegment has no api_key field
            assert not hasattr(result, "api_key")
            assert not hasattr(result, "key")

    def test_url_params_use_key_not_body(self, client, sample_flow_response):
        """API key should be in URL params, not request body."""
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            client.get_flow_for_point(28.6139, 77.2090)

            call_kwargs = mock_get.call_args
            params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
            assert params["key"] == "test-key-12345"
            # Point must use comma separator per TomTom docs (latitude,longitude)
            assert "28.6139,77.209" in params["point"]
            # No data/body kwarg
            assert "data" not in (call_kwargs.kwargs or {})
            assert "content" not in (call_kwargs.kwargs or {})


# ── Client lifecycle ────────────────────────────────────────────
class TestClientLifecycle:
    def test_close(self, client):
        client._get_client()  # force creation
        client.close()
        assert client._client.is_closed

    def test_close_when_not_created(self, client):
        # Should not raise
        client.close()

    def test_reopen_after_close(self, client):
        client._get_client()
        client.close()
        c = client._get_client()
        assert not c.is_closed


# ── Concurrent segment queries ──────────────────────────────────
class TestConcurrentSegments:
    def test_concurrent_returns_same_results(self, client, sample_flow_response, sample_midpoints):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_segments_concurrent(
                [1001, 1002, 1003], sample_midpoints, max_queries=10
            )

            assert len(result) == 3
            for seg_id in [1001, 1002, 1003]:
                assert seg_id in result
                assert isinstance(result[seg_id], FlowSegment)

    def test_concurrent_respects_max_queries(self, client, sample_flow_response):
        midpoints = {i: (28.6 + i * 0.001, 77.2 + i * 0.001) for i in range(200)}
        seg_ids = list(range(200))

        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_segments_concurrent(
                seg_ids, midpoints, max_queries=10
            )

            assert len(result) <= 10

    def test_concurrent_skips_missing_midpoints(self, client, sample_flow_response, sample_midpoints):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            result = client.get_flow_for_segments_concurrent(
                [1001, 9999], sample_midpoints, max_queries=10
            )

            assert len(result) == 1
            assert 1001 in result
            assert 9999 not in result

    def test_concurrent_circuit_breaker(self, client, sample_midpoints):
        client._circuit_open = True
        result = client.get_flow_for_segments_concurrent(
            [1001, 1002], sample_midpoints, max_queries=10
        )
        assert result == {}

    def test_concurrent_no_key(self, client_no_key, sample_midpoints):
        result = client_no_key.get_flow_for_segments_concurrent(
            [1001, 1002], sample_midpoints, max_queries=10
        )
        assert result == {}


# ── Segment cache ───────────────────────────────────────────────
class TestSegmentCache:
    def test_segment_cache_hit(self, client, sample_flow_response, sample_midpoints):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            # First call populates cache
            result1 = client.get_flow_for_segments_concurrent(
                [1001], sample_midpoints, max_queries=10
            )
            # Second call should use segment cache (no HTTP)
            mock_get.reset_mock()
            result2 = client.get_flow_for_segments_concurrent(
                [1001], sample_midpoints, max_queries=10
            )

            assert result1 == result2
            assert mock_get.call_count == 0  # cache hit, no HTTP

    def test_segment_cache_expiry(self, client, sample_flow_response, sample_midpoints):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            client.get_flow_for_segments_concurrent(
                [1001], sample_midpoints, max_queries=10
            )

            # Manually expire both segment and point caches
            client._segment_cache[1001] = (
                time.time() - 120,
                client._segment_cache[1001][1],
            )
            # Also clear point cache so segment cache can't refill from it
            lat, lon = sample_midpoints[1001]
            pkey = client._cache_key(lat, lon)
            if pkey in client._cache:
                del client._cache[pkey]

            # Next call should hit API again
            mock_get.reset_mock()
            client.get_flow_for_segments_concurrent(
                [1001], sample_midpoints, max_queries=10
            )
            assert mock_get.call_count == 1


# ── Batch timeout ───────────────────────────────────────────────
class TestBatchTimeout:
    def test_concurrent_completes_within_timeout(self, client, sample_flow_response, sample_midpoints):
        with patch.object(client._get_client(), "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = sample_flow_response
            mock_get.return_value = mock_resp

            import time
            t0 = time.perf_counter()
            result = client.get_flow_for_segments_concurrent(
                [1001, 1002, 1003], sample_midpoints, max_queries=10
            )
            elapsed = time.perf_counter() - t0

            assert len(result) == 3
            assert elapsed < client.batch_timeout


# ── Historical fallback ─────────────────────────────────────────
class TestHistoricalFallback:
    def test_concurrent_fallback_on_all_failures(self, sample_midpoints):
        """When TomTom fails entirely, caller gets empty dict → uses historical."""
        c = TomTomClient(api_key="test-key", timeout=1.0, batch_timeout=2.0)
        with patch.object(c._get_client(), "get") as mock_get:
            mock_get.side_effect = Exception("connection refused")

            result = c.get_flow_for_segments_concurrent(
                [1001, 1002], sample_midpoints, max_queries=10
            )

            assert result == {}

    def test_concurrent_partial_success(self, client, sample_midpoints):
        """Some segments succeed, others fail → partial results."""
        success_response = {
            "flowSegmentData": {
                "currentSpeed": 30.0,
                "freeFlowSpeed": 50.0,
                "confidence": 0.95,
                "roadClosure": False,
            }
        }
        error_response = {"flowSegmentData": {}}

        call_count = [0]

        def mock_get_handler(url, **kwargs):
            point = kwargs.get("params", {}).get("point", "")
            mock_resp = MagicMock()
            # Segment 1001 is at (28.6139, 77.209)
            if "28.6139" in point and "77.209" in point:
                mock_resp.status_code = 200
                mock_resp.json.return_value = success_response
            else:
                mock_resp.status_code = 200
                mock_resp.json.return_value = error_response
            call_count[0] += 1
            return mock_resp

        client_http = client._get_client()
        original_get = client_http.get
        client_http.get = mock_get_handler
        try:
            result = client.get_flow_for_segments_concurrent(
                [1001, 1002], sample_midpoints, max_queries=10
            )
            assert 1001 in result
            assert 1002 not in result
        finally:
            client_http.get = original_get
