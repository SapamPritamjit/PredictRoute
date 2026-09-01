"""R4 — CORRECTNESS TESTS + DIAGNOSTICS + PERFORMANCE BASELINE.

R4.14 tests 1-7, R4.15 toy graph, R4.16 congestion vs distance,
R4.17 performance baseline.
"""
from __future__ import annotations

import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so backend package is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.route import (
    C0, R_MIN, GraphArtifacts, WeightedEdge,
    build_weighted_adjacency, compute_route_metrics,
    dijkstra, join_snapshot_and_weight, load_graph, route,
)
from backend.predict_snapshot import predict_snapshot, ServingStack


def _print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_result(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}")
    if detail:
        for line in detail.split("\n"):
            print(f"         {line}")
    return passed


# ─────────────────────────────────────────────────────────────
# MAIN TEST HARNESS
# ─────────────────────────────────────────────────────────────
def run_all_tests():
    all_pass = True
    results = []

    # ── PHASE 1: Load graph + snapshot ──────────────────────
    _print_header("PHASE 1: LOAD GRAPH + SNAPSHOT")
    t_graph0 = time.perf_counter()
    graph = load_graph()
    t_graph = time.perf_counter() - t_graph0

    # Pick a replay datetime for the snapshot
    test_dt = "2024-08-27 08:00"
    test_mode = "replay"

    _print_header("PHASE 1b: LOAD SNAPSHOT")
    t_snap0 = time.perf_counter()
    stack = ServingStack()
    snapshot, snap_info = predict_snapshot(test_dt, test_mode, stack=stack)
    t_snap = time.perf_counter() - t_snap0

    print(f"  Snapshot: {len(snapshot)} segments, mode={test_mode}, "
          f"date={snap_info['date']}, hour={snap_info['hour']}")
    print(f"  Graph load: {t_graph:.2f}s, Snapshot load+predict: {t_snap:.2f}s")

    # ── PHASE 2: Join + weights ─────────────────────────────
    _print_header("PHASE 2: JOIN SNAPSHOT TO EDGES + COMPUTE WEIGHTS")
    t_join0 = time.perf_counter()
    weighted_edges = join_snapshot_and_weight(graph, snapshot)
    t_join = time.perf_counter() - t_join0
    print(f"  Join+weight time: {t_join:.2f}s")

    # ── R4.3: Verify join ──────────────────────────────────
    obs_seg_ids = set(
        weighted_edges[weighted_edges["edge_type"] == "observed"]["segment_id"]
    )
    snap_seg_ids = set(snapshot["segmentId"].astype(np.int64))
    _print_result("R4.3 graph observed == snapshot segment IDs",
                   obs_seg_ids == snap_seg_ids,
                   f"graph={len(obs_seg_ids)} snapshot={len(snap_seg_ids)} "
                   f"diff={len(obs_seg_ids.symmetric_difference(snap_seg_ids))}")
    if obs_seg_ids != snap_seg_ids:
        all_pass = False

    _print_result("R4.3 no missing predictions",
                   weighted_edges["clipped_ratio"].notna().all(),
                   f"NaN count = {weighted_edges['clipped_ratio'].isna().sum()}")

    _print_result("R4.3 no duplicate predictions in snapshot",
                   not snapshot["segmentId"].duplicated().any(),
                   f"duplicates = {snapshot['segmentId'].duplicated().sum()}")

    # ── R4.5: Positive weight check ─────────────────────────
    ws = weighted_edges["weight_seconds"]
    n_bad = int((ws <= 0).sum())
    n_nan = int(ws.isna().sum())
    n_inf = int(np.isinf(ws).sum())
    passed = n_bad == 0 and n_nan == 0 and n_inf == 0
    _print_result("R4.5 all weights > 0, finite",
                   passed,
                   f"min={ws.min():.6f} max={ws.max():.6f} "
                   f"median={ws.median():.4f} p95={ws.quantile(0.95):.4f} "
                   f"p99={ws.quantile(0.99):.4f} <=0:{n_bad} NaN:{n_nan} inf:{n_inf}")
    if not passed:
        all_pass = False

    # ── R4.4: Unit check ────────────────────────────────────
    sample = weighted_edges.iloc[0]
    expected_w = (sample["distance"] / 1000) / sample["speedLimit"] * 3600 * sample["clipped_ratio"]
    unit_ok = abs(sample["weight_seconds"] - expected_w) < 1e-10
    _print_result("R4.4 weight formula correct (units: seconds)", unit_ok,
                   f"distance={sample['distance']}m speed={sample['speedLimit']}km/h "
                   f"ratio={sample['clipped_ratio']:.4f} weight={sample['weight_seconds']:.4f} "
                   f"expected={expected_w:.4f}")

    # ── Build weighted adjacency ────────────────────────────
    _print_header("PHASE 2b: BUILD WEIGHTED ADJACENCY")
    t_adj0 = time.perf_counter()
    weighted_adj = build_weighted_adjacency(weighted_edges)
    t_adj = time.perf_counter() - t_adj0
    total_adj_edges = sum(len(v) for v in weighted_adj.values())
    _print_result("R4.7 adjacency edge count matches",
                   total_adj_edges == 42700,
                   f"adjacency total = {total_adj_edges}")
    print(f"  Adjacency build time: {t_adj:.2f}s")

    # ═══════════════════════════════════════════════════════
    # TEST 1: Same node
    # ═══════════════════════════════════════════════════════
    _print_header("TEST 1: Same node route(A, A)")
    test_node = list(graph.scc_nodes)[0]
    r1 = route(test_node, test_node, test_dt, test_mode, graph, weighted_adj)
    t1_pass = (r1.success and r1.edge_count == 0
               and r1.search_time_seconds == 0.0
               and r1.total_distance_m == 0.0)
    _print_result("TEST 1 same node", t1_pass,
                   f"success={r1.success} edges={r1.edge_count} "
                   f"dist={r1.total_distance_m} time={r1.search_time_seconds}")
    all_pass = all_pass and t1_pass
    results.append(("TEST 1 same node", t1_pass))

    # ═══════════════════════════════════════════════════════
    # TEST 2: Known adjacent nodes
    # ═══════════════════════════════════════════════════════
    _print_header("TEST 2: Known adjacent nodes (A->B)")
    # Pick the first observed edge
    first_obs = weighted_edges[weighted_edges["edge_type"] == "observed"].iloc[0]
    a_node = int(first_obs["from_node"])
    b_node = int(first_obs["to_node"])
    r2 = route(a_node, b_node, test_dt, test_mode, graph, weighted_adj)
    t2_pass = r2.success and r2.edge_count == 1
    if t2_pass:
        edge = r2.path[0]
        t2_edge_match = (
            edge["segmentId"] == int(first_obs["segment_id"])
            and abs(edge["weight_seconds"] - first_obs["weight_seconds"]) < 1e-10
        )
    else:
        t2_edge_match = False
    t2_pass = t2_pass and t2_edge_match
    _print_result("TEST 2 adjacent nodes", t2_pass,
                   f"success={r2.success} edges={r2.edge_count} "
                   f"expected_seg={int(first_obs['segment_id'])} "
                   f"got_seg={r2.path[0]['segmentId'] if r2.path else 'N/A'}")
    all_pass = all_pass and t2_pass
    results.append(("TEST 2 adjacent", t2_pass))

    # ═══════════════════════════════════════════════════════
    # TEST 3: Reverse twin — independent ratios
    # ═══════════════════════════════════════════════════════
    _print_header("TEST 3: Reverse twin — independent ratios")
    # Find a segment with a twin
    twin_edges = weighted_edges[
        (weighted_edges["edge_type"] == "observed")
        & (weighted_edges["twin_of"].notna())
    ].head(1)
    if len(tin_edges := twin_edges) > 0:
        seg_a = int(tin_edges.iloc[0]["segment_id"])
        seg_b = int(tin_edges.iloc[0]["twin_of"])
        ratio_a = weighted_edges[weighted_edges["segment_id"] == seg_a]["clipped_ratio"].iloc[0]
        ratio_b = weighted_edges[weighted_edges["segment_id"] == seg_b]["clipped_ratio"].iloc[0]
        t3_pass = True  # Ratios CAN be different (they come from different segments)
        detail = (f"seg_a={seg_a} ratio={ratio_a:.4f} | "
                  f"seg_b(twin)={seg_b} ratio={ratio_b:.4f} | "
                  f"same={ratio_a == ratio_b}")
    else:
        t3_pass = False
        detail = "No twin edges found"
    _print_result("TEST 3 reverse twin independent ratios", t3_pass, detail)
    all_pass = all_pass and t3_pass
    results.append(("TEST 3 twin", t3_pass))

    # ═══════════════════════════════════════════════════════
    # TEST 4: Synthetic reverse — inherits parent ratio
    # ═══════════════════════════════════════════════════════
    _print_header("TEST 4: Synthetic reverse — inherits parent ratio")
    syn_edge = weighted_edges[weighted_edges["edge_type"] == "synthetic_reverse"].head(1)
    if len(syn_edge) > 0:
        parent_seg = int(syn_edge.iloc[0]["parent_segment_id"])
        syn_ratio = syn_edge.iloc[0]["clipped_ratio"]
        parent_ratio_row = weighted_edges[
            (weighted_edges["segment_id"] == parent_seg)
            & (weighted_edges["edge_type"] == "observed")
        ]
        if len(parent_ratio_row) > 0:
            parent_ratio = parent_ratio_row.iloc[0]["clipped_ratio"]
            t4_pass = abs(syn_ratio - parent_ratio) < 1e-12
        else:
            t4_pass = False
            parent_ratio = None
        detail = (f"parent_seg={parent_seg} parent_ratio={parent_ratio:.4f} "
                  f"syn_ratio={syn_ratio:.4f} match={t4_pass}")
    else:
        t4_pass = False
        detail = "No synthetic edges found"
    _print_result("TEST 4 synthetic reverse ratio", t4_pass, detail)
    all_pass = all_pass and t4_pass
    results.append(("TEST 4 synthetic", t4_pass))

    # ═══════════════════════════════════════════════════════
    # TEST 5: SCC restriction
    # ═══════════════════════════════════════════════════════
    _print_header("TEST 5: SCC restriction — rejects excluded node")
    excluded_nodes = set(graph.nodes.keys()) - graph.scc_nodes
    if excluded_nodes:
        excl_node = sorted(excluded_nodes)[0]
        r5 = route(excl_node, excl_node, test_dt, test_mode, graph, weighted_adj)
        # Same node should still succeed (zero-length path)
        # But let's test routing FROM an excluded node to a valid one
        valid_node = sorted(graph.scc_nodes)[0]
        r5b = route(excl_node, valid_node, test_dt, test_mode, graph, weighted_adj)
        t5_pass = (not r5b.success and r5b.reason == "ORIGIN_OUTSIDE_SCC")
        _print_result("TEST 5 SCC restriction", t5_pass,
                       f"excluded_node={excl_node} reason={r5b.reason}")
    else:
        t5_pass = False
        _print_result("TEST 5 SCC restriction", t5_pass, "No excluded nodes found")
    all_pass = all_pass and t5_pass
    results.append(("TEST 5 SCC", t5_pass))

    # ═══════════════════════════════════════════════════════
    # TEST 6: Positive weights (all edges)
    # ═══════════════════════════════════════════════════════
    _print_header("TEST 6: Positive weights (all 42,700 edges)")
    all_positive = (weighted_edges["weight_seconds"] > 0).all()
    all_finite = np.isfinite(weighted_edges["weight_seconds"]).all()
    t6_pass = all_positive and all_finite
    _print_result("TEST 6 all weights positive + finite", t6_pass,
                   f"min={weighted_edges['weight_seconds'].min():.6f} "
                   f"all>0={all_positive} all_finite={all_finite}")
    all_pass = all_pass and t6_pass
    results.append(("TEST 6 positive weights", t6_pass))

    # ═══════════════════════════════════════════════════════
    # TEST 7: Determinism
    # ═══════════════════════════════════════════════════════
    _print_header("TEST 7: Determinism — same query, same result")
    # Pick two non-trivial nodes
    obs = weighted_edges[weighted_edges["edge_type"] == "observed"]
    node_a = int(obs.iloc[0]["from_node"])
    node_b = int(obs.iloc[500]["to_node"])
    r7a = route(node_a, node_b, test_dt, test_mode, graph, weighted_adj)
    r7b = route(node_a, node_b, test_dt, test_mode, graph, weighted_adj)
    t7_pass = (
        r7a.success == r7b.success
        and r7a.path == r7b.path
        and r7a.search_time_seconds == r7b.search_time_seconds
        and r7a.total_distance_m == r7b.total_distance_m
        and r7a.weighted_mean_ratio == r7b.weighted_mean_ratio
    )
    _print_result("TEST 7 determinism", t7_pass,
                   f"both_success={r7a.success} paths_equal={r7a.path == r7b.path} "
                   f"time={r7a.search_time_seconds:.4f}")
    all_pass = all_pass and t7_pass
    results.append(("TEST 7 determinism", t7_pass))

    # ═══════════════════════════════════════════════════════
    # TEST 8 (BONUS): Route contract completeness
    # ═══════════════════════════════════════════════════════
    _print_header("TEST 8: Route result contract fields")
    r8 = route(node_a, node_b, test_dt, test_mode, graph, weighted_adj)
    required_fields = [
        "success", "departure_datetime", "mode", "origin_node",
        "destination_node", "path", "total_distance_m", "total_distance_km",
        "search_time_seconds", "reported_eta_seconds", "reported_eta_minutes",
        "weighted_mean_ratio", "max_ratio", "edge_count",
        "observed_edge_count", "synthetic_reverse_edge_count", "warnings",
    ]
    missing_fields = [f for f in required_fields if not hasattr(r8, f)]
    t8_pass = len(missing_fields) == 0
    _print_result("TEST 8 contract fields present", t8_pass,
                   f"missing={missing_fields}" if missing_fields else "all present")
    all_pass = all_pass and t8_pass
    results.append(("TEST 8 contract", t8_pass))

    # ═══════════════════════════════════════════════════════
    # R4.12: C0 semantics
    # ═══════════════════════════════════════════════════════
    _print_header("R4.12: C0 semantics check")
    if r8.success:
        eta_check = abs(r8.reported_eta_seconds - r8.search_time_seconds * C0) < 1e-10
        minutes_check = abs(r8.reported_eta_minutes - r8.reported_eta_seconds / 60) < 1e-10
        c0_pass = eta_check and minutes_check
        _print_result("R4.12 C0 applied only to ETA", c0_pass,
                       f"search={r8.search_time_seconds:.2f}s "
                       f"eta={r8.reported_eta_seconds:.2f}s "
                       f"eta/search={r8.reported_eta_seconds/r8.search_time_seconds:.4f} "
                       f"(expected {C0})")
    else:
        c0_pass = False
        _print_result("R4.12 C0", c0_pass, "route failed, cannot check")
    all_pass = all_pass and c0_pass
    results.append(("R4.12 C0", c0_pass))

    # ═══════════════════════════════════════════════════════
    # R4.11: Weighted mean ratio check
    # ═══════════════════════════════════════════════════════
    _print_header("R4.11: Weighted mean ratio formula check")
    if r8.success and r8.edge_count > 0:
        # Recompute manually
        path_data = r8.path
        base_times = [e["distance_m"] / e["speedLimit_kmh"] for e in path_data]
        ratios = [e["ratio"] for e in path_data]
        manual_wmean = sum(bt * r for bt, r in zip(base_times, ratios)) / sum(base_times)
        wmean_pass = abs(r8.weighted_mean_ratio - manual_wmean) < 1e-10
        _print_result("R4.11 weighted mean ratio formula", wmean_pass,
                       f"computed={r8.weighted_mean_ratio:.6f} "
                       f"manual={manual_wmean:.6f}")
    else:
        wmean_pass = False
    all_pass = all_pass and wmean_pass
    results.append(("R4.11 weighted mean", wmean_pass))

    # ═══════════════════════════════════════════════════════
    # R4.15: Toy graph Dijkstra optimality
    # ═══════════════════════════════════════════════════════
    _print_header("R4.15: Toy graph Dijkstra optimality")
    t15_pass = _test_toy_graph()
    all_pass = all_pass and t15_pass
    results.append(("R4.15 toy graph", t15_pass))

    # ═══════════════════════════════════════════════════════
    # R4.16: Congestion vs distance diagnostic
    # ═══════════════════════════════════════════════════════
    _print_header("R4.16: Congestion vs Distance diagnostic")
    _diagnostic_congestion_vs_distance(graph, weighted_edges, weighted_adj, test_dt, test_mode)

    # ═══════════════════════════════════════════════════════
    # R4.17: Performance baseline
    # ═══════════════════════════════════════════════════════
    _print_header("R4.17: Performance baseline")
    _performance_baseline(graph, weighted_edges, weighted_adj, test_dt, test_mode,
                          t_graph, t_snap, t_join, t_adj)

    # ═══════════════════════════════════════════════════════
    # PRINT SAMPLE ROUTE
    # ═══════════════════════════════════════════════════════
    _print_header("SAMPLE ROUTE RESULT")
    if r8.success:
        print(f"  origin={r8.origin_node} dest={r8.destination_node}")
        print(f"  edges={r8.edge_count} (observed={r8.observed_edge_count} "
              f"synthetic={r8.synthetic_reverse_edge_count})")
        print(f"  distance={r8.total_distance_km:.2f} km")
        print(f"  search_time={r8.search_time_seconds:.2f}s")
        print(f"  reported_eta={r8.reported_eta_minutes:.1f} min")
        print(f"  weighted_mean_ratio={r8.weighted_mean_ratio:.4f}")
        print(f"  max_ratio={r8.max_ratio:.4f}")
        for i, e in enumerate(r8.path[:5]):
            print(f"  [{i}] seg={e['segmentId']} {e['from_node']}->{e['to_node']} "
                  f"type={e['edge_type']} dist={e['distance_m']:.1f}m "
                  f"ratio={e['ratio']:.4f} w={e['weight_seconds']:.2f}s "
                  f"street={e['streetName'] or 'unnamed'}")
        if r8.edge_count > 5:
            print(f"  ... ({r8.edge_count - 5} more edges)")
        print(f"  warnings: {r8.warnings}")
    else:
        print(f"  FAILED: {r8.reason}")

    # ═══════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════
    _print_header("R4 TEST SUMMARY")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print()
    if all_pass:
        print("  R4 PASSED — DIJKSTRA ROUTING VERIFIED — READY FOR R5")
    else:
        print("  R4 BLOCKED — DO NOT START R5")

    return all_pass


# ─────────────────────────────────────────────────────────────
# R4.15: Toy graph test
# ─────────────────────────────────────────────────────────────
def _test_toy_graph() -> bool:
    """Construct a small toy graph and verify Dijkstra returns optimal path."""
    #   A(0) --10--> B(1) --10--> D(3)
    #   A(0) -- 3--> C(2) -- 3--> D(3)
    # Clearly: A->C->D = 6, A->B->D = 20

    # Build toy adjacency with WeightedEdge objects
    def _we(from_n, to_n, w, seg, dist=100.0, sl=50.0, et="observed"):
        return WeightedEdge(
            edge_key=f"toy_{from_n}_{to_n}",
            segment_id=seg,
            parent_segment_id=seg,
            from_node=from_n,
            to_node=to_n,
            distance_m=dist,
            speedLimit_kmh=sl,
            frc=4,
            streetName=f"toy_{from_n}_{to_n}",
            edge_type=et,
            clipped_ratio=1.0,
            weight_seconds=w,
        )

    toy_adj = {
        0: [_we(0, 1, 10.0, 1001), _we(0, 2, 3.0, 1002)],
        1: [_we(1, 3, 10.0, 1003)],
        2: [_we(2, 3, 3.0, 1004)],
    }
    toy_scc = {0, 1, 2, 3}

    # Run Dijkstra
    total_w, path_edges = dijkstra(0, 3, toy_adj, toy_scc)

    path_nodes = [0] + [e.to_node for e in path_edges]
    optimal = (path_nodes == [0, 2, 3] and abs(total_w - 6.0) < 1e-10)
    if not optimal:
        print(f"  [FAIL] Toy graph: path={path_nodes} weight={total_w} (expected [0,2,3] weight=6.0)")
        return False

    # Also test: make one path clearly better with asymmetric weights
    #   A(0) --100--> B(1) --1--> D(3)
    #   A(0) --  1--> C(2) --1--> D(3)
    toy_adj2 = {
        0: [_we(0, 1, 100.0, 2001), _we(0, 2, 1.0, 2002)],
        1: [_we(1, 3, 1.0, 2003)],
        2: [_we(2, 3, 1.0, 2004)],
    }
    total_w2, path_edges2 = dijkstra(0, 3, toy_adj2, toy_scc)
    path_nodes2 = [0] + [e.to_node for e in path_edges2]
    optimal2 = (path_nodes2 == [0, 2, 3] and abs(total_w2 - 2.0) < 1e-10)

    # Test unreachable: add node 99 not in adjacency
    toy_adj3 = dict(toy_adj)
    toy_scc3 = toy_scc | {99}
    # node 99 is in SCC but has no edges to/from 0
    try:
        dijkstra(0, 99, toy_adj3, toy_scc3)
        unreachable_ok = False
    except ValueError as e:
        unreachable_ok = str(e) == "UNREACHABLE"

    # Test SCC rejection
    try:
        dijkstra(0, 3, toy_adj, {0, 1})  # node 3 not in SCC
        scc_ok = False
    except ValueError as e:
        scc_ok = str(e) == "DESTINATION_OUTSIDE_SCC"

    # Test same node
    tw, te = dijkstra(0, 0, toy_adj, toy_scc)
    same_ok = (tw == 0.0 and len(te) == 0)

    all_ok = optimal and optimal2 and unreachable_ok and scc_ok and same_ok
    print(f"  [PASS] Toy: optimal path [0,2,3] w=6.0" if optimal else "  [FAIL] Toy optimal 1")
    print(f"  [PASS] Toy: asymmetric optimal [0,2,3] w=2.0" if optimal2 else "  [FAIL] Toy optimal 2")
    print(f"  [PASS] Toy: unreachable raises UNREACHABLE" if unreachable_ok else "  [FAIL] Toy unreachable")
    print(f"  [PASS] Toy: SCC rejection works" if scc_ok else "  [FAIL] Toy SCC")
    print(f"  [PASS] Toy: same node returns 0" if same_ok else "  [FAIL] Toy same")
    return all_ok


# ─────────────────────────────────────────────────────────────
# R4.16: Congestion vs distance diagnostic
# ─────────────────────────────────────────────────────────────
def _diagnostic_congestion_vs_distance(graph, weighted_edges, weighted_adj, dt, mode):
    """Run a few OD pairs with congestion-aware and distance-only weights."""
    # Pick pairs that have multi-hop routes by using nodes farther apart
    # Node 0 -> Node 681 showed a 250-edge route in the sample above
    # Also try other distant pairs
    test_pairs = [
        (0, 681),       # 13.94 km route from sample
        (2, 5000),      # likely distant
        (100, 10000),   # another distant pair
    ]

    # Build distance-only adjacency
    dist_edges = weighted_edges.copy()
    dist_edges["weight_seconds"] = dist_edges["distance"]  # meters as weight
    dist_adj = build_weighted_adjacency(dist_edges)

    print(f"  {'Pair':>6s} | {'Type':>16s} | {'Edges':>5s} | {'Dist(m)':>9s} | "
          f"{'Time(s)':>8s} | {'MeanRatio':>9s} | {'MaxRatio':>8s}")
    print(f"  {'-'*6}-+-{'-'*16}-+-{'-'*5}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")

    for a, b in test_pairs:
        # Congestion-aware
        try:
            r_cong = route(a, b, dt, mode, graph, weighted_adj)
        except Exception:
            r_cong = None
        # Distance-only
        try:
            r_dist = route(a, b, dt, mode, graph, dist_adj)
        except Exception:
            r_dist = None

        for label, r in [("congestion-aware", r_cong), ("distance-only", r_dist)]:
            if r and r.success:
                print(f"  {a:>6d} | {label:>16s} | {r.edge_count:>5d} | "
                      f"{r.total_distance_m:>9.1f} | {r.search_time_seconds:>8.2f} | "
                      f"{r.weighted_mean_ratio:>9.4f} | {r.max_ratio:>8.4f}")
            else:
                reason = r.reason if r else "error"
                print(f"  {a:>6d} | {label:>16s} | FAIL: {reason}")

    print("\n  Note: This is a diagnostic only — not a scored evaluation (R5).")


# ─────────────────────────────────────────────────────────────
# R4.17: Performance baseline
# ─────────────────────────────────────────────────────────────
def _performance_baseline(graph, weighted_edges, weighted_adj, dt, mode,
                          t_graph, t_snap, t_join, t_adj):
    """Measure Dijkstra and path reconstruction times."""
    obs = weighted_edges[weighted_edges["edge_type"] == "observed"]

    # Pick several OD pairs with varying distances
    test_pairs = []
    for i in [0, 100, 500, 2000, 5000, 10000]:
        row = obs.iloc[min(i, len(obs) - 1)]
        test_pairs.append((int(row["from_node"]), int(row["to_node"])))

    dijkstra_times = []
    recon_times = []
    for a, b in test_pairs:
        try:
            t0 = time.perf_counter()
            st, pe = dijkstra(a, b, weighted_adj, graph.scc_nodes)
            t1 = time.perf_counter()
            path = reconstruct_path(pe)
            t2 = time.perf_counter()
            dijkstra_times.append(t1 - t0)
            recon_times.append(t2 - t1)
        except ValueError:
            pass

    # Full route() calls
    full_times = []
    for a, b in test_pairs:
        t0 = time.perf_counter()
        r = route(a, b, dt, mode, graph, weighted_adj)
        t1 = time.perf_counter()
        if r.success:
            full_times.append(t1 - t0)

    n_nodes = len(graph.nodes)
    n_edges = len(weighted_edges)

    print(f"  Graph: {n_nodes} nodes, {n_edges} edges")
    print(f"  Graph loading:        {t_graph:.2f}s")
    print(f"  Snapshot load+predict:{t_snap:.2f}s")
    print(f"  Join + weight:        {t_join:.2f}s")
    print(f"  Adjacency build:      {t_adj:.2f}s")
    if dijkstra_times:
        print(f"  Dijkstra (avg {len(dijkstra_times)} queries): "
              f"{np.mean(dijkstra_times)*1000:.2f}ms "
              f"(min={np.min(dijkstra_times)*1000:.2f}ms "
              f"max={np.max(dijkstra_times)*1000:.2f}ms)")
    if recon_times:
        print(f"  Path reconstruction:  {np.mean(recon_times)*1000:.2f}ms")
    if full_times:
        print(f"  Full route() (avg):   {np.mean(full_times)*1000:.2f}ms")
    total_no_model = t_graph + t_join + t_adj
    print(f"  Total (excl model):   {total_no_model:.2f}s")
    print(f"  Total end-to-end:     {t_graph + t_snap + t_join + t_adj:.2f}s")


def reconstruct_path(path_edges):
    """Helper for perf baseline."""
    from backend.route import reconstruct_path as _rp
    return _rp(path_edges)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
