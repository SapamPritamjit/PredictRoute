"""R4/R7 — ROUTING / GRAPH WEIGHTING MODULE.

Loads R1 graph artifacts + R3 prediction snapshot, builds weighted
adjacency, runs Dijkstra or A* shortest-path, reconstructs paths with
full edge-level metadata, and computes route metrics.

Weight formula:
    weight_seconds = (distance_m / 1000) / speedLimit_kmh * 3600 * clipped_ratio
                   = distance_m / (speedLimit_kmh / 3.6) * clipped_ratio

C0 = 1.60 applied ONLY to reported ETA, NOT to search weights.

R3 clipped_ratio is the sole source of congestion ratios.

R7: A* added with haversine heuristic h(n) = haversine(n,goal)*R_MIN/v_max_mps.
    Heuristic is admissible and consistent (non-negative weights).
"""
from __future__ import annotations

import heapq
import json
import pickle
import time
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd

# ── Project root (backend/ is one level below root) ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Locked constants ──────────────────────────────────────────────
R_MIN = 0.5
C0 = 1.60

GRAPH_DIR = str(PROJECT_ROOT / "cleaned" / "graph")

EXPECTED_NODE_COUNT = 18916
EXPECTED_EDGE_COUNT = 42700
EXPECTED_SCC_COUNT = 18865


# ── R4.1  Load verified graph ─────────────────────────────────────
class GraphLoadError(RuntimeError):
    pass


@dataclass
class GraphArtifacts:
    nodes: dict          # node_id -> (lon, lat)
    edges_df: pd.DataFrame  # all 42,700 directed edges
    adjacency: dict      # node_id -> [(neighbor, edge_key), ...]
    scc_nodes: set       # largest SCC node set
    metadata: dict       # graph_metadata.json


def load_graph(graph_dir: str = GRAPH_DIR) -> GraphArtifacts:
    """Load R1 artifacts and verify counts against metadata."""
    t0 = time.perf_counter()

    with open(f"{graph_dir}/nodes.pkl", "rb") as f:
        nodes = pickle.load(f)
    with open(f"{graph_dir}/adjacency.pkl", "rb") as f:
        adjacency = pickle.load(f)
    with open(f"{graph_dir}/largest_scc_nodes.pkl", "rb") as f:
        scc_list = pickle.load(f)
    with open(f"{graph_dir}/graph_metadata.json") as f:
        metadata = json.load(f)

    edges_df = pd.read_csv(f"{graph_dir}/edges.csv")

    scc_nodes = set(scc_list)

    # ── Verification ──
    errs = []
    if len(nodes) != EXPECTED_NODE_COUNT:
        errs.append(f"node count {len(nodes)} != {EXPECTED_NODE_COUNT}")
    if len(edges_df) != EXPECTED_EDGE_COUNT:
        errs.append(f"edge count {len(edges_df)} != {EXPECTED_EDGE_COUNT}")
    if len(scc_nodes) != EXPECTED_SCC_COUNT:
        errs.append(f"SCC count {len(scc_nodes)} != {EXPECTED_SCC_COUNT}")
    if metadata["node_count"] != EXPECTED_NODE_COUNT:
        errs.append(f"metadata node_count {metadata['node_count']} != {EXPECTED_NODE_COUNT}")
    if metadata["total_directed_edges"] != EXPECTED_EDGE_COUNT:
        errs.append(f"metadata total_directed_edges != {EXPECTED_EDGE_COUNT}")
    if metadata["largest_scc_size"] != EXPECTED_SCC_COUNT:
        errs.append(f"metadata largest_scc_size != {EXPECTED_SCC_COUNT}")
    if len(adjacency) != EXPECTED_NODE_COUNT:
        errs.append(f"adjacency node count {len(adjacency)} != {EXPECTED_NODE_COUNT}")
    if errs:
        raise GraphLoadError("R4.1 graph load verification FAILED: " + "; ".join(errs))

    load_s = time.perf_counter() - t0
    print(f"R4.1 graph loaded: {len(nodes)} nodes, {len(edges_df)} edges, "
          f"{len(scc_nodes)} SCC nodes in {load_s:.2f}s")

    return GraphArtifacts(
        nodes=nodes,
        edges_df=edges_df,
        adjacency=adjacency,
        scc_nodes=scc_nodes,
        metadata=metadata,
    )


# ── R4.3–R4.5  Join snapshot + compute edge weights ──────────────
@dataclass
class WeightedEdge:
    """Single edge with full metadata + weight."""
    edge_key: str
    segment_id: int
    parent_segment_id: Any      # int for observed (self), int for synthetic (parent)
    from_node: int
    to_node: int
    distance_m: float
    speedLimit_kmh: float
    frc: int
    streetName: str
    edge_type: str              # "observed" | "synthetic_reverse"
    clipped_ratio: float
    weight_seconds: float


def join_snapshot_and_weight(
    graph: GraphArtifacts,
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Join R3 snapshot onto edges, compute weight_seconds.

    - Observed edges: use their own segment_id's clipped_ratio
    - Synthetic reverse edges: use their parent segment_id's clipped_ratio
    - Real reverse twins: each uses its own segment_id's ratio

    Returns the edges DataFrame with clipped_ratio and weight_seconds columns.
    """
    edges = graph.edges_df.copy()

    # Build a ratio lookup from snapshot
    ratio_lookup = dict(zip(snapshot["segment_id"] if "segment_id" in snapshot.columns
                           else snapshot["segmentId"],
                           snapshot["clipped_ratio"]))
    snap_seg_col = "segmentId" if "segmentId" in snapshot.columns else "segment_id"

    # For observed edges: lookup by own segment_id
    # For synthetic reverse edges: lookup by parent_segment_id (the original segment)
    observed_mask = edges["edge_type"] == "observed"
    synthetic_mask = edges["edge_type"] == "synthetic_reverse"

    # Observed edges use their own segment_id
    edges.loc[observed_mask, "clipped_ratio"] = (
        edges.loc[observed_mask, "segment_id"].map(ratio_lookup)
    )

    # Synthetic reverse edges use their parent_segment_id
    # parent_segment_id is the original segment's ID (float due to CSV round-trip)
    edges.loc[synthetic_mask, "clipped_ratio"] = (
        edges.loc[synthetic_mask, "parent_segment_id"].astype(np.int64).map(ratio_lookup)
    )

    # ── Validation: no missing ratios ──
    missing = edges["clipped_ratio"].isna().sum()
    if missing > 0:
        raise GraphLoadError(f"R4.3 join: {missing} edges have no matching ratio")

    # ── R4.4: weight formula ──
    # CSV columns: distance (meters), speedLimit (km/h)
    edges["clipped_ratio"] = edges["clipped_ratio"].astype(np.float64)
    edges["weight_seconds"] = (
        (edges["distance"] / 1000.0)
        / edges["speedLimit"]
        * 3600.0
        * edges["clipped_ratio"]
    )

    # ── R4.5: positive weight guarantee ──
    ws = edges["weight_seconds"]
    n_bad = int((ws <= 0).sum())
    n_nan = int(ws.isna().sum())
    n_inf = int(np.isinf(ws).sum())
    if n_bad > 0 or n_nan > 0 or n_inf > 0:
        raise GraphLoadError(
            f"R4.5 FAILED: {n_bad} <= 0, {n_nan} NaN, {n_inf} inf weights"
        )

    print(f"R4.3–R4.5 weights computed: min={ws.min():.4f}, max={ws.max():.4f}, "
          f"median={ws.median():.4f}, p95={ws.quantile(0.95):.4f}, "
          f"p99={ws.quantile(0.99):.4f}")
    print(f"  observed edges: {observed_mask.sum()}, synthetic: {synthetic_mask.sum()}")

    return edges


# ── R4.7  Build query-time weighted adjacency ─────────────────────
def build_weighted_adjacency(
    weighted_edges: pd.DataFrame,
) -> dict[int, list[WeightedEdge]]:
    """Build weighted adjacency from the full edge DataFrame.

    Preserves parallel edges. Each adjacency entry is a WeightedEdge.
    Vectorized for speed on 42.7k rows.
    """
    # Pre-extract arrays for speed
    from_nodes = weighted_edges["from_node"].to_numpy()
    edge_keys = weighted_edges["edge_key"].to_numpy()
    seg_ids = weighted_edges["segment_id"].to_numpy()
    par_ids = weighted_edges["parent_segment_id"].to_numpy()
    to_nodes = weighted_edges["to_node"].to_numpy()
    distances = weighted_edges["distance"].to_numpy()
    speeds = weighted_edges["speedLimit"].to_numpy()
    frcs = weighted_edges["frc"].to_numpy()
    streets = weighted_edges["streetName"].to_numpy()
    edge_types = weighted_edges["edge_type"].to_numpy()
    ratios = weighted_edges["clipped_ratio"].to_numpy()
    weights = weighted_edges["weight_seconds"].to_numpy()

    adj: dict[int, list[WeightedEdge]] = {}
    for i in range(len(weighted_edges)):
        pid = int(par_ids[i]) if not np.isnan(par_ids[i]) else int(seg_ids[i])
        street = "" if pd.isna(streets[i]) else str(streets[i])
        we = WeightedEdge(
            edge_key=str(edge_keys[i]),
            segment_id=int(seg_ids[i]),
            parent_segment_id=pid,
            from_node=int(from_nodes[i]),
            to_node=int(to_nodes[i]),
            distance_m=float(distances[i]),
            speedLimit_kmh=float(speeds[i]),
            frc=int(frcs[i]),
            streetName=street,
            edge_type=str(edge_types[i]),
            clipped_ratio=float(ratios[i]),
            weight_seconds=float(weights[i]),
        )
        fn = int(from_nodes[i])
        if fn in adj:
            adj[fn].append(we)
        else:
            adj[fn] = [we]
    return adj


# ── R4.8–R4.10  Dijkstra + path reconstruction ───────────────────
@dataclass
class RouteResult:
    success: bool
    reason: str = ""
    departure_datetime: str = ""
    mode: str = ""
    origin_node: int = 0
    destination_node: int = 0
    path: list[dict] = field(default_factory=list)
    total_distance_m: float = 0.0
    total_distance_km: float = 0.0
    search_time_seconds: float = 0.0
    reported_eta_seconds: float = 0.0
    reported_eta_minutes: float = 0.0
    weighted_mean_ratio: float = 0.0
    max_ratio: float = 0.0
    edge_count: int = 0
    observed_edge_count: int = 0
    synthetic_reverse_edge_count: int = 0
    warnings: list[str] = field(default_factory=list)


def dijkstra(
    start_node: int,
    dest_node: int,
    weighted_adj: dict[int, list[WeightedEdge]],
    scc_nodes: set[int],
) -> tuple[float, list[WeightedEdge]]:
    """Standard weighted Dijkstra with priority queue.

    Returns (total_weight, path_edges) or raises if unreachable.
    SCC restriction enforced: both endpoints must be in scc_nodes.

    Raises ValueError for invalid nodes or unreachable destinations.
    """
    # R4.8 SCC restriction
    if start_node not in scc_nodes:
        raise ValueError("ORIGIN_OUTSIDE_SCC")
    if dest_node not in scc_nodes:
        raise ValueError("DESTINATION_OUTSIDE_SCC")

    if start_node == dest_node:
        return (0.0, [])

    # Dijkstra state
    dist: dict[int, float] = {start_node: 0.0}
    # predecessor: node -> (prev_node, WeightedEdge)
    pred: dict[int, tuple[int, WeightedEdge]] = {}
    visited: set[int] = set()
    heap: list[tuple[float, int]] = [(0.0, start_node)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if u == dest_node:
            break

        for we in weighted_adj.get(u, []):
            v = we.to_node
            # R4.8: all edges must have endpoints in SCC
            if v not in scc_nodes:
                continue
            nd = d + we.weight_seconds
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                pred[v] = (u, we)
                heapq.heappush(heap, (nd, v))

    if dest_node not in pred and dest_node != start_node:
        raise ValueError("UNREACHABLE")

    # Path reconstruction
    path_edges: list[WeightedEdge] = []
    cur = dest_node
    while cur != start_node:
        prev, we = pred[cur]
        path_edges.append(we)
        cur = prev
    path_edges.reverse()

    return (dist[dest_node], path_edges)


# ── R7.1–R7.5  A* with haversine heuristic ────────────────────────
# Earth radius for haversine (meters)
_EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine straight-line distance in meters."""
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return float(2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a)))


def _build_haversine_index(nodes: dict) -> dict[int, tuple[float, float]]:
    """Pre-compute lon/lat arrays for haversine heuristic."""
    return {nid: (lon, lat) for nid, (lon, lat) in nodes.items()}


def _compute_v_max(graph: GraphArtifacts) -> float:
    """Maximum speedLimit in graph (m/s). Used as upper bound for heuristic."""
    v_max_kmh = float(graph.edges_df["speedLimit"].max())
    return v_max_kmh / 3.6  # convert km/h -> m/s


def astar(
    start_node: int,
    dest_node: int,
    weighted_adj: dict[int, list[WeightedEdge]],
    scc_nodes: set[int],
    nodes: dict,
    v_max_mps: float,
) -> tuple[float, list[WeightedEdge]]:
    """A* shortest path with haversine heuristic.

    f(n) = g(n) + h(n)
    h(n) = haversine(n, goal) * R_MIN / v_max_mps

    Admissibility proof:
        h(n) = straight_line * R_MIN / v_max
        actual remaining cost >= network_distance * R_MIN / v_max
        haversine <= network_distance (straight line <= any path)
        => h(n) <= actual remaining cost.  QED.

    Consistency (monotonicity):
        For edge (u, v) with cost c(u,v):
            c(u,v) >= edge_distance * R_MIN / v_max >= haversine(u,v) * R_MIN / v_max
            = |h(u) - h(v)| (by triangle inequality of haversine)
        => h is consistent => nodes never need reopening.

    Tie-breaking: (f_score, -g_score, node_id)
        - prefers larger g (= smaller h) when f values tie
        - deterministic node ordering as final tie-breaker

    Returns (total_weight, path_edges) or raises ValueError.
    """
    if start_node not in scc_nodes:
        raise ValueError("ORIGIN_OUTSIDE_SCC")
    if dest_node not in scc_nodes:
        raise ValueError("DESTINATION_OUTSIDE_SCC")
    if start_node == dest_node:
        return (0.0, [])

    h_index = _build_haversine_index(nodes)
    goal_lon, goal_lat = h_index[dest_node]
    h_cache: dict[int, float] = {}

    def _h(node: int) -> float:
        if node in h_cache:
            return h_cache[node]
        if node in h_index:
            nl, na = h_index[node]
            h = _haversine_m(nl, na, goal_lon, goal_lat) * R_MIN / v_max_mps
        else:
            h = 0.0
        h_cache[node] = h
        return h

    dist: dict[int, float] = {start_node: 0.0}
    pred: dict[int, tuple[int, WeightedEdge]] = {}
    visited: set[int] = set()
    counter = 0  # tie-breaker for identical (f, -g)
    h0 = _h(start_node)
    heap: list[tuple[float, float, int, int]] = [(h0, 0.0, counter, start_node)]

    while heap:
        f, _g_neg, _cnt, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        g_u = dist[u]  # use authoritative g-value

        if u == dest_node:
            break

        for we in weighted_adj.get(u, []):
            v = we.to_node
            if v not in scc_nodes:
                continue
            nd = g_u + we.weight_seconds  # g(v) = g(u) + cost(u,v)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                pred[v] = (u, we)
                counter += 1
                fv = nd + _h(v)
                heapq.heappush(heap, (fv, -nd, counter, v))

    if dest_node not in pred and dest_node != start_node:
        raise ValueError("UNREACHABLE")

    path_edges: list[WeightedEdge] = []
    cur = dest_node
    while cur != start_node:
        prev, we = pred[cur]
        path_edges.append(we)
        cur = prev
    path_edges.reverse()

    return (dist[dest_node], path_edges)


def reconstruct_path(path_edges: list[WeightedEdge]) -> list[dict]:
    """Convert WeightedEdge list to route path dicts."""
    path = []
    for we in path_edges:
        path.append({
            "segmentId": we.segment_id,
            "edge_type": we.edge_type,
            "parent_segment_id": we.parent_segment_id,
            "from_node": we.from_node,
            "to_node": we.to_node,
            "distance_m": we.distance_m,
            "speedLimit_kmh": we.speedLimit_kmh,
            "frc": we.frc,
            "streetName": we.streetName,
            "ratio": we.clipped_ratio,
            "weight_seconds": we.weight_seconds,
        })
    return path


# ── R4.11–R4.13  Route metrics + contract ────────────────────────
def compute_route_metrics(
    path_edges: list[WeightedEdge],
    search_time: float,
    origin_node: int,
    dest_node: int,
    departure_datetime: str = "",
    mode: str = "",
) -> RouteResult:
    """Compute all route metrics per R4.11–R4.13."""
    if not path_edges:
        return RouteResult(
            success=True,
            departure_datetime=departure_datetime,
            mode=mode,
            origin_node=origin_node,
            destination_node=dest_node,
            path=[],
            total_distance_m=0.0,
            total_distance_km=0.0,
            search_time_seconds=0.0,
            reported_eta_seconds=0.0,
            reported_eta_minutes=0.0,
            weighted_mean_ratio=0.0,
            max_ratio=0.0,
            edge_count=0,
            observed_edge_count=0,
            synthetic_reverse_edge_count=0,
            warnings=["zero-length route (same node)"],
        )

    total_dist = sum(e.distance_m for e in path_edges)
    ratios = np.array([e.clipped_ratio for e in path_edges])

    # Weighted mean ratio (weighted by base free-flow time)
    base_times = np.array([e.distance_m / e.speedLimit_kmh for e in path_edges])
    weighted_mean = float(np.sum(base_times * ratios) / np.sum(base_times))
    max_ratio = float(np.max(ratios))

    n_observed = sum(1 for e in path_edges if e.edge_type == "observed")
    n_synthetic = sum(1 for e in path_edges if e.edge_type == "synthetic_reverse")

    reported_eta = search_time * C0

    return RouteResult(
        success=True,
        departure_datetime=departure_datetime,
        mode=mode,
        origin_node=origin_node,
        destination_node=dest_node,
        path=reconstruct_path(path_edges),
        total_distance_m=total_dist,
        total_distance_km=total_dist / 1000.0,
        search_time_seconds=search_time,
        reported_eta_seconds=reported_eta,
        reported_eta_minutes=reported_eta / 60.0,
        weighted_mean_ratio=weighted_mean,
        max_ratio=max_ratio,
        edge_count=len(path_edges),
        observed_edge_count=n_observed,
        synthetic_reverse_edge_count=n_synthetic,
        warnings=[],
    )


def route(
    start_node: int,
    dest_node: int,
    departure_datetime: str,
    mode: str,
    graph: GraphArtifacts,
    weighted_adj: dict[int, list[WeightedEdge]],
    method: str = "dijkstra",
    v_max_mps: float | None = None,
) -> RouteResult:
    """Main route query function per R4.13 contract.

    start_node and dest_node are graph node IDs.
    method: "dijkstra" (default) or "astar"
    v_max_mps: required when method="astar"
    Returns RouteResult with success=True/False.
    """
    warnings = []
    try:
        if method == "astar":
            if v_max_mps is None:
                raise ValueError("v_max_mps required for A*")
            search_time, path_edges = astar(
                start_node, dest_node, weighted_adj, graph.scc_nodes,
                graph.nodes, v_max_mps,
            )
        else:
            search_time, path_edges = dijkstra(
                start_node, dest_node, weighted_adj, graph.scc_nodes
            )
        result = compute_route_metrics(
            path_edges, search_time, start_node, dest_node,
            departure_datetime, mode,
        )
        # Static weight snapshot warning
        warnings.append(
            "R4 v1: all edges use departure-hour predicted ratios "
            "(no hour-crossing update)"
        )
        result.warnings = warnings
        return result

    except ValueError as e:
        return RouteResult(
            success=False,
            reason=str(e),
            departure_datetime=departure_datetime,
            mode=mode,
            origin_node=start_node,
            destination_node=dest_node,
        )


# ── Convenience: load everything + one query ──────────────────────
def load_and_prepare(departure_datetime: str, mode: str):
    """Load graph + snapshot + build weighted adjacency. Returns (graph, weighted_edges, weighted_adj, snapshot_info)."""
    from predict_snapshot import predict_snapshot, ServingStack

    graph = load_graph()
    stack = ServingStack()
    snapshot, snap_info = predict_snapshot(departure_datetime, mode, stack=stack)

    weighted_edges = join_snapshot_and_weight(graph, snapshot)
    weighted_adj = build_weighted_adjacency(weighted_edges)

    return graph, weighted_edges, weighted_adj, snap_info
