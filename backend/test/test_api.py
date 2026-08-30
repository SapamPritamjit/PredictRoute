"""Tests for POST /route API endpoint.

Covers all six required categories:
  1. Successful route request
  2. Invalid coordinates
  3. Out-of-range snap
  4. Unreachable route
  5. Invalid departure date / mode
  6. Prediction / routing failure handling

Requires model + graph artifacts on disk (same as R4/R7 tests).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.api import app

# ── Shared test client (loads graph + model once ~11s) ────────
@pytest.fixture(scope="module")
def client():
    from unittest.mock import MagicMock
    import backend.api as api_mod

    # Mock TomTom to prevent real API calls during tests
    mock_tomtom = MagicMock()
    mock_tomtom.available = False  # simulate no API access
    mock_tomtom.get_flow_for_segments.return_value = {}
    original_tomtom = api_mod._tomtom_client
    api_mod._tomtom_client = mock_tomtom

    with TestClient(app) as c:
        yield c

    api_mod._tomtom_client = original_tomtom


# ── Test data ────────────────────────────────────────────────
DELHI_ORIGIN = {"lat": 28.6139, "lon": 77.2090}
DELHI_DEST = {"lat": 28.6280, "lon": 77.2195}
DEMO_ORIGIN = {"lat": 28.5692, "lon": 77.2090}  # Delhi demo location
AIIMS_DEST = {"lat": 28.5643, "lon": 77.2155}   # ~491m from nearest node
LONDON = {"lat": 51.5074, "lon": -0.1278}
REPLAY_DT = "2024-08-27 08:00"
FORECAST_DT = "2024-09-01 08:00"


def _base_payload(**overrides):
    p = {
        "origin_lat": DELHI_ORIGIN["lat"],
        "origin_lon": DELHI_ORIGIN["lon"],
        "dest_lat": DELHI_DEST["lat"],
        "dest_lon": DELHI_DEST["lon"],
        "departure_datetime": REPLAY_DT,
        "mode": "replay",
    }
    p.update(overrides)
    return p


# ═══════════════════════════════════════════════════════════════
# 1. Successful route request
# ═══════════════════════════════════════════════════════════════
class TestSuccessfulRoute:
    def test_replay_route(self, client):
        resp = client.post("/route", json=_base_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["distance_km"] >= 0
        assert data["eta_minutes"] >= 0
        assert data["search_time_seconds"] >= 0
        assert isinstance(data["route"], list)
        assert len(data["route"]) > 0

    def test_forecast_route(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime=FORECAST_DT, mode="forecast",
        ))
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_auto_mode_replay(self, client):
        resp = client.post("/route", json=_base_payload(mode=None))
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_auto_mode_forecast(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime=FORECAST_DT, mode=None,
        ))
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_response_fields_complete(self, client):
        resp = client.post("/route", json=_base_payload())
        data = resp.json()
        required = [
            "status", "origin", "destination",
            "distance_km", "eta_minutes", "search_time_seconds",
            "mean_ratio", "max_ratio", "synthetic_reverse_edges",
            "origin_snap_distance_m", "destination_snap_distance_m",
            "destinationSnapFallback",
            "warnings", "route",
            # Traffic metadata
            "trafficSource", "trafficDataAvailable", "trafficIncidentCount",
            "trafficSegmentsQueried", "trafficSegmentsMatched",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

        # Traffic fields should have expected types
        assert isinstance(data["trafficSource"], str)
        assert isinstance(data["trafficDataAvailable"], bool)
        assert isinstance(data["trafficIncidentCount"], int)
        assert isinstance(data["destinationSnapFallback"], bool)

        for key in ("latitude", "longitude", "node_id"):
            assert key in data["origin"], f"Missing origin.{key}"
            assert key in data["destination"], f"Missing destination.{key}"
        # Actual destination coordinates should always be present
        assert "actual_latitude" in data["destination"]
        assert "actual_longitude" in data["destination"]

    def test_same_origin_destination(self, client):
        resp = client.post("/route", json=_base_payload(
            dest_lat=DELHI_ORIGIN["lat"],
            dest_lon=DELHI_ORIGIN["lon"],
        ))
        assert resp.status_code == 200
        data = resp.json()
        assert data["distance_km"] == 0.0
        assert data["eta_minutes"] == 0.0

    def test_ratios_in_range(self, client):
        resp = client.post("/route", json=_base_payload())
        data = resp.json()
        assert 0.5 <= data["mean_ratio"] <= 3.721
        assert 0.5 <= data["max_ratio"] <= 3.721

    def test_warnings_present(self, client):
        resp = client.post("/route", json=_base_payload())
        data = resp.json()
        assert isinstance(data["warnings"], list)
        assert len(data["warnings"]) >= 1

    def test_route_edges_have_expected_fields(self, client):
        resp = client.post("/route", json=_base_payload())
        data = resp.json()
        if data["route"]:
            edge = data["route"][0]
            for key in ("segmentId", "edge_type", "distance_m",
                        "speedLimit_kmh", "ratio", "weight_seconds"):
                assert key in edge, f"Missing route edge field: {key}"


# ═══════════════════════════════════════════════════════════════
# 2. Invalid coordinates
# ═══════════════════════════════════════════════════════════════
class TestInvalidCoordinates:
    @pytest.mark.parametrize("field,value", [
        ("origin_lat", 91),
        ("origin_lat", -91),
        ("origin_lon", 181),
        ("origin_lon", -181),
        ("dest_lat", 91),
        ("dest_lat", -91),
        ("dest_lon", 181),
        ("dest_lon", -181),
    ])
    def test_coordinate_out_of_range(self, client, field, value):
        resp = client.post("/route", json=_base_payload(**{field: value}))
        assert resp.status_code == 422

    def test_missing_required_field(self, client):
        resp = client.post("/route", json={
            "origin_lat": 28.6,
            "departure_datetime": REPLAY_DT,
        })
        assert resp.status_code == 422

    def test_non_numeric_coordinate(self, client):
        resp = client.post("/route", json=_base_payload(origin_lat="abc"))
        assert resp.status_code == 422

    def test_empty_body(self, client):
        resp = client.post("/route", json={})
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════
# 3. Out-of-range snap (beyond 400 m)
# ═══════════════════════════════════════════════════════════════
class TestOutOfRangeSnap:
    def test_origin_too_far(self, client):
        resp = client.post("/route", json=_base_payload(
            origin_lat=LONDON["lat"], origin_lon=LONDON["lon"],
        ))
        assert resp.status_code == 400
        assert "snap radius" in resp.json()["detail"].lower()

    def test_destination_too_far(self, client):
        resp = client.post("/route", json=_base_payload(
            dest_lat=LONDON["lat"], dest_lon=LONDON["lon"],
        ))
        assert resp.status_code == 400
        assert "snap radius" in resp.json()["detail"].lower()

    def test_both_too_far(self, client):
        resp = client.post("/route", json=_base_payload(
            origin_lat=LONDON["lat"], origin_lon=LONDON["lon"],
            dest_lat=LONDON["lat"], dest_lon=LONDON["lon"],
        ))
        assert resp.status_code == 400
        assert "Origin" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════
# 3b. Destination snap fallback (400-600 m)
# ═══════════════════════════════════════════════════════════════
class TestDestinationSnapFallback:
    def test_dest_within_400m_normal_snap(self, client):
        resp = client.post("/route", json=_base_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["destinationSnapFallback"] is False
        assert data["destination_snap_distance_m"] <= 400.0
        # For normal snap, actual coords should match snapped coords (within tolerance)
        assert abs(data["destination"]["actual_latitude"] - data["destination"]["latitude"]) < 0.001
        assert abs(data["destination"]["actual_longitude"] - data["destination"]["longitude"]) < 0.001

    def test_dest_between_400_600m_fallback_snap(self, client):
        resp = client.post("/route", json=_base_payload(
            dest_lat=AIIMS_DEST["lat"], dest_lon=AIIMS_DEST["lon"],
        ))
        assert resp.status_code == 200
        data = resp.json()
        assert data["destinationSnapFallback"] is True
        assert data["destination_snap_distance_m"] > 400.0
        assert data["destination_snap_distance_m"] <= 600.0
        assert data["distance_km"] > 0
        assert len(data["route"]) > 0

    def test_dest_beyond_600m_rejected(self, client):
        resp = client.post("/route", json=_base_payload(
            dest_lat=LONDON["lat"], dest_lon=LONDON["lon"],
        ))
        assert resp.status_code == 400
        assert "snap radius" in resp.json()["detail"].lower()

    def test_origin_still_400m_limit(self, client):
        resp = client.post("/route", json=_base_payload(
            origin_lat=AIIMS_DEST["lat"], origin_lon=AIIMS_DEST["lon"],
        ))
        assert resp.status_code == 400
        assert "Origin" in resp.json()["detail"]
        assert "snap radius" in resp.json()["detail"].lower()

    def test_demo_origin_with_aiims_dest(self, client):
        resp = client.post("/route", json=_base_payload(
            origin_lat=DEMO_ORIGIN["lat"], origin_lon=DEMO_ORIGIN["lon"],
            dest_lat=AIIMS_DEST["lat"], dest_lon=AIIMS_DEST["lon"],
        ))
        assert resp.status_code == 200
        data = resp.json()
        assert data["destinationSnapFallback"] is True
        assert data["destination_snap_distance_m"] > 400.0
        assert data["destination_snap_distance_m"] <= 600.0
        # Actual destination coordinates should match the input, not the snapped node
        assert data["destination"]["actual_latitude"] == AIIMS_DEST["lat"]
        assert data["destination"]["actual_longitude"] == AIIMS_DEST["lon"]
        # Route should extend to actual destination
        route_coords = data.get("route_coordinates", [])
        if route_coords:
            last_coord = route_coords[-1]
            assert abs(last_coord[0] - AIIMS_DEST["lat"]) < 0.001
            assert abs(last_coord[1] - AIIMS_DEST["lon"]) < 0.001

    def test_origin_no_fallback_flag(self, client):
        resp = client.post("/route", json=_base_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert "destinationSnapFallback" in data


# ═══════════════════════════════════════════════════════════════
# 4. Unreachable route
# ═══════════════════════════════════════════════════════════════
class TestUnreachableRoute:
    def test_unreachable_returns_400(self, client):
        with patch("backend.route.dijkstra", side_effect=ValueError("UNREACHABLE")):
            resp = client.post("/route", json=_base_payload())
            assert resp.status_code == 400
            assert "Routing error" in resp.json()["detail"]
            assert "UNREACHABLE" in resp.json()["detail"]

    def test_origin_outside_scc_returns_400(self, client):
        with patch("backend.route.dijkstra",
                   side_effect=ValueError("ORIGIN_OUTSIDE_SCC")):
            resp = client.post("/route", json=_base_payload())
            assert resp.status_code == 400
            assert "ORIGIN_OUTSIDE_SCC" in resp.json()["detail"]

    def test_dest_outside_scc_returns_400(self, client):
        with patch("backend.route.dijkstra",
                   side_effect=ValueError("DESTINATION_OUTSIDE_SCC")):
            resp = client.post("/route", json=_base_payload())
            assert resp.status_code == 400
            assert "DESTINATION_OUTSIDE_SCC" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════
# 5. Invalid departure date / mode
# ═══════════════════════════════════════════════════════════════
class TestInvalidDepartureDate:
    def test_date_before_replay_start(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime="2024-08-01 08:00", mode="replay",
        ))
        assert resp.status_code == 422

    def test_has_timezone(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime="2024-08-27 08:00+05:30",
        ))
        assert resp.status_code == 422

    def test_not_on_hour(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime="2024-08-27 08:30",
        ))
        assert resp.status_code == 422

    def test_garbage_datetime(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime="not-a-date",
        ))
        assert resp.status_code == 422


class TestInvalidMode:
    def test_invalid_mode_string(self, client):
        resp = client.post("/route", json=_base_payload(mode="walking"))
        assert resp.status_code == 422
        assert "Invalid mode" in resp.json()["detail"]

    def test_replay_mode_on_forecast_date(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime=FORECAST_DT, mode="replay",
        ))
        assert resp.status_code == 422
        assert "replay" in resp.json()["detail"].lower()

    def test_forecast_mode_on_replay_date(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime=REPLAY_DT, mode="forecast",
        ))
        assert resp.status_code == 422
        assert "forecast" in resp.json()["detail"].lower()

    def test_auto_mode_rejects_before_replay(self, client):
        resp = client.post("/route", json=_base_payload(
            departure_datetime="2024-08-01 08:00", mode=None,
        ))
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════
# 6. Prediction / routing failure handling
# ═══════════════════════════════════════════════════════════════
class TestPredictionFailure:
    def test_prediction_error_returns_422(self, client):
        from backend.predict_snapshot import SnapshotError
        with patch(
            "backend.predict_snapshot.predict_snapshot",
            side_effect=SnapshotError("test prediction failure"),
        ):
            resp = client.post("/route", json=_base_payload())
            assert resp.status_code == 422
            assert "Prediction error" in resp.json()["detail"]

    def test_graph_weighting_error_returns_500(self, client):
        from backend.route import GraphLoadError
        with patch(
            "backend.route.join_snapshot_and_weight",
            side_effect=GraphLoadError("test weighting failure"),
        ):
            resp = client.post("/route", json=_base_payload())
            assert resp.status_code == 500
            assert "Graph weighting error" in resp.json()["detail"]

    def test_model_not_loaded_returns_503(self, client):
        import backend.api as api_mod
        old_stack = api_mod._stack
        try:
            api_mod._stack = None
            resp = client.post("/route", json=_base_payload())
            assert resp.status_code == 503
            assert "Model artifacts" in resp.json()["detail"]
        finally:
            api_mod._stack = old_stack

    def test_graph_not_loaded_returns_503(self, client):
        import backend.api as api_mod
        old_graph = api_mod._graph
        try:
            api_mod._graph = None
            resp = client.post("/route", json=_base_payload())
            assert resp.status_code == 503
            assert "Graph artifacts" in resp.json()["detail"]
        finally:
            api_mod._graph = old_graph
