"""PredictRoute v1 — FastAPI backend.

Thin orchestration layer over validated core components:

    API request  (lat/lon + departure datetime)
      → snap to graph nodes (400 m radius, SCC check)
      → predict_snapshot   (CatBoost congestion ratios)
      → TomTom live traffic (hybrid override where available)
      → join + weight      (edge weight_seconds)
      → dijkstra           (shortest weighted path)
      → RouteResult        (metrics + path)
      → JSON response

No ML or routing logic is reimplemented here.
TomTom integration is an adapter layer — CatBoost remains the primary
prediction source; TomTom provides supplementary live ratios where available.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env from backend/ directory (must happen before TomTomClient init)
_backend_dir = Path(__file__).resolve().parent
load_dotenv(_backend_dir / ".env")

# ── Locked constants (must match route.py / predict_snapshot.py) ──
SNAP_RADIUS_M = 400.0
DEST_FALLBACK_RADIUS_M = 600.0

# ── Lazy-loaded singletons ───────────────────────────────────────
_graph = None
_stack = None
_tomtom_client = None
_segment_midpoints: dict[int, tuple[float, float]] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load graph + model + TomTom client once at server startup."""
    global _graph, _stack, _tomtom_client, _segment_midpoints
    from backend.route import load_graph
    from backend.predict_snapshot import ServingStack
    from backend.tomtom import TomTomClient

    _graph = load_graph()
    _stack = ServingStack()
    _tomtom_client = TomTomClient()

    # Precompute segment midpoints for TomTom queries
    _segment_midpoints = _build_segment_midpoints(_graph)

    if _tomtom_client.available:
        print("TomTom live traffic: ENABLED (API key configured)")
    else:
        print("TomTom live traffic: DISABLED (no API key — using historical fallback)")

    yield

    if _tomtom_client is not None:
        _tomtom_client.close()


def _build_segment_midpoints(graph) -> dict[int, tuple[float, float]]:
    """Precompute midpoints for each unique segment.

    Each midpoint is (lat, lon) computed from the first edge's endpoints.
    Used for TomTom Flow API queries.
    """
    import pandas as pd

    edges = graph.edges_df
    # Get first edge per unique segment_id
    first_edges = edges.groupby("segment_id").first().reset_index()

    midpoints = {}
    for _, row in first_edges.iterrows():
        seg_id = int(row["segment_id"])
        from_node = int(row["from_node"])
        to_node = int(row["to_node"])

        from_coord = graph.nodes.get(from_node)
        to_coord = graph.nodes.get(to_node)

        if from_coord and to_coord:
            # from_coord and to_coord are (lon, lat)
            mid_lon = (from_coord[0] + to_coord[0]) / 2
            mid_lat = (from_coord[1] + to_coord[1]) / 2
            midpoints[seg_id] = (mid_lat, mid_lon)

    return midpoints


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="PredictRoute API",
    version="0.1.0",
    description="Delhi congestion-aware routing API (v1).",
    lifespan=_lifespan,
)

# ── CORS (configurable origins) ──────────────────────────────────
import os as _os

_CORS_ORIGINS = _os.environ.get(
    "PREDICTROUTE_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Place search provider (lazy singleton) ───────────────────────
from backend.places import PhotonProvider

_provider: PhotonProvider | None = None


def _get_provider() -> PhotonProvider:
    global _provider
    if _provider is None:
        _provider = PhotonProvider()
    return _provider


@app.on_event("shutdown")
def _shutdown_provider():
    global _provider
    if _provider is not None:
        _provider.close()
        _provider = None


# ── Static files (frontend) ─────────────────────────────────────
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _get_graph():
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph artifacts not loaded")
    return _graph


def _get_stack():
    if _stack is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return _stack


# ── GET /api/places ──────────────────────────────────────────────
@app.get("/api/places")
def search_places(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    lat: float | None = Query(None, ge=-90, le=90, description="Latitude bias"),
    lon: float | None = Query(None, ge=-180, le=180, description="Longitude bias"),
    limit: int = Query(5, ge=1, le=10, description="Max results"),
):
    q = q.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Query must not be empty or whitespace")

    provider = _get_provider()
    try:
        results = provider.search(q, lat=lat, lon=lon, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Provider error: {exc}")

    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "name": r.name,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "display_name": r.display_name,
                "place_type": r.place_type,
                "osm_type": r.osm_type,
            }
            for r in results
        ],
    }


# ── Request / Response models ────────────────────────────────────
class RouteRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90, le=90, description="Origin latitude")
    origin_lon: float = Field(..., ge=-180, le=180, description="Origin longitude")
    dest_lat: float = Field(..., ge=-90, le=90, description="Destination latitude")
    dest_lon: float = Field(..., ge=-180, le=180, description="Destination longitude")
    departure_datetime: str = Field(
        ...,
        description="Naive IST datetime on the hour, e.g. '2024-08-27 08:00'",
    )
    mode: Optional[str] = Field(
        None,
        description="Prediction mode: 'replay' (2024-08-11..30) or 'forecast' (>=2024-08-31). "
                    "Auto-resolved from date if omitted.",
    )


# ── Coordinate snapping ─────────────────────────────────────────
def _find_nearest_node(lat: float, lon: float, graph) -> tuple[int, float]:
    """Vectorised haversine snap — returns (node_id, distance_m).

    The nodes dict stores ``node_id -> (lon, lat)``.
    """
    nodes = graph.nodes
    node_ids = np.array(list(nodes.keys()), dtype=np.int64)
    coords = np.array([nodes[nid] for nid in node_ids])  # (N, 2): lon, lat

    lon1 = np.radians(lon)
    lat1 = np.radians(lat)
    lat2r = np.radians(coords[:, 1])
    lon2r = np.radians(coords[:, 0])

    dlat = lat2r - lat1
    dlon = lon2r - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    dists = 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    idx = int(np.argmin(dists))
    return int(node_ids[idx]), float(dists[idx])


def _snap_coordinate(
    lat: float, lon: float, label: str, graph,
    fallback_radius: float | None = None,
):
    """Snap user coordinate to nearest graph node.

    Enforces SNAP_RADIUS_M radius and SCC membership.
    If *fallback_radius* is provided and the nearest node exceeds
    SNAP_RADIUS_M but is within *fallback_radius*, the snap is allowed
    (used for destination-only relaxed snapping).
    Returns (node_id, snap_distance_m, snapped_lat, snapped_lon, used_fallback).
    """
    node_id, dist_m = _find_nearest_node(lat, lon, graph)
    used_fallback = False

    if dist_m > SNAP_RADIUS_M:
        if fallback_radius is not None and dist_m <= fallback_radius:
            used_fallback = True
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{label}: nearest graph node is {dist_m:.0f} m away, "
                    f"exceeds the {SNAP_RADIUS_M:.0f} m snap radius. "
                    f"This location may be outside the supported Delhi routing "
                    f"coverage, or the road network may be too sparse nearby. "
                    f"Try a point closer to a major road."
                ),
            )

    if node_id not in graph.scc_nodes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}: nearest reachable road node is in a disconnected "
                f"section of the road network (outside the largest strongly "
                f"connected component). The road graph cannot route through "
                f"this area."
            ),
        )

    node_lon, node_lat = graph.nodes[node_id]
    return node_id, dist_m, node_lat, node_lon, used_fallback


# ── Mode auto-resolution ────────────────────────────────────────
def _resolve_mode(departure_datetime: str, mode: str | None) -> str:
    """Resolve mode from date when user omits it; validate when provided."""
    import pandas as pd
    from backend.predict_snapshot import FORECAST_START, REPLAY_END, REPLAY_START

    try:
        dstr = pd.Timestamp(departure_datetime).strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid departure datetime: '{departure_datetime}'",
        )

    if mode is not None:
        if mode not in ("replay", "forecast"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid mode '{mode}'; must be 'replay' or 'forecast'",
            )
        # Validate mode against date window
        if mode == "replay" and dstr > REPLAY_END:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"mode='replay' is only valid for {REPLAY_START}..{REPLAY_END}, "
                    f"got {dstr}"
                ),
            )
        if mode == "forecast" and dstr < FORECAST_START:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"mode='forecast' is only valid from {FORECAST_START}, got {dstr}"
                ),
            )
        return mode

    # Auto-resolve from date
    if dstr < REPLAY_START:
        raise HTTPException(
            status_code=422,
            detail=f"Date {dstr} precedes first observed date {REPLAY_START}",
        )
    if dstr <= REPLAY_END:
        return "replay"
    return "forecast"


# ── POST /route ──────────────────────────────────────────────────
@app.post("/route")
def post_route(req: RouteRequest):
    t_start = time.perf_counter()

    graph = _get_graph()
    stack = _get_stack()
    tomtom = _tomtom_client

    # ── Snap coordinates to graph nodes ──
    t_snap = time.perf_counter()
    origin_id, origin_snap_m, origin_lat, origin_lon, _origin_fb = _snap_coordinate(
        req.origin_lat, req.origin_lon, "Origin", graph,
    )
    dest_id, dest_snap_m, dest_lat, dest_lon, dest_used_fallback = _snap_coordinate(
        req.dest_lat, req.dest_lon, "Destination", graph,
        fallback_radius=DEST_FALLBACK_RADIUS_M,
    )
    # Preserve user's actual destination for UI marker
    dest_actual_lat = req.dest_lat
    dest_actual_lon = req.dest_lon
    t_snap_done = time.perf_counter()

    # ── Resolve mode ──
    mode = _resolve_mode(req.departure_datetime, req.mode)

    # ── Prediction snapshot (CatBoost) ──
    t_pred = time.perf_counter()
    try:
        from backend.predict_snapshot import SnapshotError, predict_snapshot

        snapshot, snap_info = predict_snapshot(req.departure_datetime, mode, stack=stack)
    except SnapshotError as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {e}")
    t_pred_done = time.perf_counter()

    # ── TomTom live traffic (hybrid override) ──
    traffic_info = {
        "source": "historical_fallback",
        "data_available": False,
        "incident_count": 0,
        "tomtom_segments_queried": 0,
        "tomtom_segments_matched": 0,
    }

    if tomtom is not None and tomtom.available:
        try:
            import pandas as pd

            # Get unique segment IDs from snapshot
            all_seg_ids = snapshot["segmentId"].tolist()

            # Query TomTom for sampled segments (concurrent)
            t_tomtom = time.perf_counter()
            tomtom_results = tomtom.get_flow_for_segments_concurrent(
                all_seg_ids,
                _segment_midpoints,
                max_queries=80,
            )
            tomtom_time = time.perf_counter() - t_tomtom

            traffic_info["tomtom_segments_queried"] = min(len(all_seg_ids), 80)
            traffic_info["tomtom_segments_matched"] = len(tomtom_results)
            traffic_info["tomtom_query_seconds"] = round(tomtom_time, 3)
            traffic_info["tomtom_segments_with_midpoint"] = sum(
                1 for seg_id in all_seg_ids[:80] if seg_id in _segment_midpoints
            )

            if tomtom_results:
                # Merge: use TomTom ratio where available, CatBoost elsewhere
                override = {seg_id: flow.ratio for seg_id, flow in tomtom_results.items()}
                override_series = snapshot["segmentId"].map(override)
                mask = override_series.notna()
                snapshot.loc[mask, "clipped_ratio"] = override_series[mask].values

                traffic_info["source"] = "tomtom_live"
                traffic_info["data_available"] = True

                # Count road closures as "incidents"
                traffic_info["incident_count"] = sum(
                    1 for f in tomtom_results.values() if f.road_closure
                )

        except Exception as e:
            # TomTom failure — fall back to historical
            traffic_info["source"] = "historical_fallback"
            traffic_info["error"] = type(e).__name__

    # ── Edge weighting ──
    t_weight = time.perf_counter()
    try:
        from backend.route import (
            GraphLoadError,
            build_weighted_adjacency,
            join_snapshot_and_weight,
        )

        weighted_edges = join_snapshot_and_weight(graph, snapshot)
        weighted_adj = build_weighted_adjacency(weighted_edges)
    except GraphLoadError as e:
        raise HTTPException(status_code=500, detail=f"Graph weighting error: {e}")
    t_weight_done = time.perf_counter()

    # ── Dijkstra routing ──
    t_dijk = time.perf_counter()
    try:
        from backend.route import dijkstra

        search_time, path_edges = dijkstra(
            origin_id, dest_id, weighted_adj, graph.scc_nodes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Routing error: {e}")
    t_dijk_done = time.perf_counter()

    # ── Metrics + path reconstruction ──
    from backend.route import compute_route_metrics, reconstruct_path

    result = compute_route_metrics(
        path_edges,
        search_time,
        origin_id,
        dest_id,
        req.departure_datetime,
        mode,
    )
    result.warnings.append(
        "R4 v1: all edges use departure-hour predicted ratios "
        "(no hour-crossing update)"
    )

    if traffic_info["source"] == "tomtom_live":
        result.warnings.append(
            f"Live TomTom traffic: {traffic_info['tomtom_segments_matched']} "
            f"segments overridden out of {traffic_info['tomtom_segments_queried']} queried"
        )

    route_path = reconstruct_path(path_edges)
    elapsed = time.perf_counter() - t_start

    # ── Extract ordered [lat, lon] coordinate sequence for map display ──
    route_coords: list[list[float]] = []
    if route_path:
        seen_last = graph.nodes.get(route_path[0]["from_node"])
        if seen_last is not None:
            route_coords.append([seen_last[1], seen_last[0]])
        for edge in route_path:
            node_coord = graph.nodes.get(edge["to_node"])
            if node_coord is not None:
                route_coords.append([node_coord[1], node_coord[0]])
        # Extend route to actual destination (may differ from snapped node)
        if dest_used_fallback:
            route_coords.append([dest_actual_lat, dest_actual_lon])

    return {
        "status": "success",
        "origin": {
            "latitude": origin_lat,
            "longitude": origin_lon,
            "node_id": origin_id,
        },
        "destination": {
            "latitude": dest_lat,
            "longitude": dest_lon,
            "node_id": dest_id,
            "actual_latitude": dest_actual_lat,
            "actual_longitude": dest_actual_lon,
        },
        "distance_km": result.total_distance_km,
        "eta_minutes": result.reported_eta_minutes,
        "search_time_seconds": result.search_time_seconds,
        "mean_ratio": result.weighted_mean_ratio,
        "max_ratio": result.max_ratio,
        "synthetic_reverse_edges": result.synthetic_reverse_edge_count,
        "origin_snap_distance_m": origin_snap_m,
        "destination_snap_distance_m": dest_snap_m,
        "destinationSnapFallback": dest_used_fallback,
        "warnings": result.warnings,
        "route": route_path,
        "route_coordinates": route_coords,
        "request_time_seconds": round(elapsed, 3),
        # ── Timing breakdown ──
        "timing_snap_seconds": round(t_snap_done - t_snap, 3),
        "timing_prediction_seconds": round(t_pred_done - t_pred, 3),
        "timing_tomtom_seconds": traffic_info.get("tomtom_query_seconds", 0),
        "timing_weight_seconds": round(t_weight_done - t_weight, 3),
        "timing_dijkstra_seconds": round(t_dijk_done - t_dijk, 3),
        # ── Traffic metadata ──
        "trafficSource": traffic_info["source"],
        "trafficDataAvailable": traffic_info["data_available"],
        "trafficIncidentCount": traffic_info["incident_count"],
        "trafficSegmentsQueried": traffic_info["tomtom_segments_queried"],
        "trafficSegmentsMatched": traffic_info["tomtom_segments_matched"],
    }


# ── Static file serving (frontend) ──────────────────────────────
from starlette.responses import FileResponse


@app.get("/", include_in_schema=False)
def serve_index():
    index = _FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not deployed")
    return FileResponse(str(index))


if _FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
