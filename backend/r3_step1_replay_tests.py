"""R3 replay gates: R3.4 verbatim snapshots, R3.6 validation, R3.7 stats,
R3.8 reproduction vs direct pipeline on engineered rows, R3.3 boundary
rejections, R3.11 performance baseline."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


import predict_snapshot as ps

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        fails.append(name)


SRC = "cleaned/engineered_causal.parquet"
print("=" * 70)
print("R3.1 LOAD SERVING STACK (fresh)")
print("=" * 70)
stack = ps.ServingStack()
print(f"  features ({len(stack.feature_names)}): {stack.feature_names}")
print(f"  stack load time: {stack.load_seconds:.2f}s")

print()
print("=" * 70)
print("R3.3 BOUNDARY REJECTION TESTS")
print("=" * 70)
for bad_dt, bad_mode, why in [
    ("2024-08-10 12:00", "replay", "before first observed date"),
    ("2024-08-10 12:00", "forecast", "before first observed date"),
    ("2024-08-31 08:00", "replay", "replay after window end"),
    ("2024-08-15 17:00", "forecast", "forecast inside replay window"),
    ("2024-08-27 08:30", "replay", "sub-hour granularity"),
]:
    try:
        ps.predict_snapshot(bad_dt, bad_mode, stack)
        check(f"reject {bad_dt} [{bad_mode}] ({why})", False, "NO exception raised!")
    except ps.SnapshotError as e:
        check(f"reject {bad_dt} [{bad_mode}] ({why})", True, str(e)[:80])

print()
print("=" * 70)
print("R3.4/R3.7 REPLAY SNAPSHOTS + PREDICTION STATS (one batch each)")
print("=" * 70)
QUERIES = ["2024-08-26 17:00",   # festival rush hour
           "2024-08-26 03:00",   # festival night
           "2024-08-27 08:00",   # ordinary morning
           "2024-08-27 22:00"]   # ordinary night
snaps = {}
for q in QUERIES:
    snap, info = ps.predict_snapshot(q, "replay", stack)
    snaps[q] = snap
    s = info["stats"]
    print(f"  {q} (festival={q.startswith('2024-08-26')})")
    print(f"    raw : min {s['raw_min']:+.4f} p1 {s['raw_p1']:.4f} p5 {s['raw_p5']:.4f} "
          f"med {s['raw_median']:.4f} mean {s['raw_mean']:.4f} "
          f"p95 {s['raw_p95']:.4f} p99 {s['raw_p99']:.4f} max {s['raw_max']:+.4f}")
    print(f"    clip: min {s['clip_min']:.4f} max {s['clip_max']:.4f} mean {s['clip_mean']:.4f} "
          f"med {s['clip_median']:.4f} clipped_fraction {s['clipped_fraction']:.5f}")
    print(f"    counts: <0.5 {s['n_below_R_MIN']} | >3.721 {s['n_above_cap']} | "
          f"<0 {s['n_negative']} | nonfinite {s['n_nonfinite_raw']}")
    ok = (info["n_segments"] == 24938 and s["n_nonfinite_raw"] == 0
          and s["clip_min"] >= 0.5 and s["clip_max"] <= 3.721)
    check(f"snapshot valid {q}", ok)

print()
print("=" * 70)
print("R3.8 REPLAY REPRODUCTION vs DIRECT PIPELINE (fresh 2nd deserialize)")
print("=" * 70)
pipe_direct = joblib.load(ps.MODEL_PKL)
direct_feature_order = [str(x) for x in pipe_direct.steps[-1][1].feature_names_]
assert direct_feature_order == stack.feature_names
REPRO = [("2024-08-26", 17), ("2024-08-26", 3), ("2024-08-27", 8), ("2024-08-27", 22)]
for dstr, hour in REPRO:
    eng = pq.read_table(
        SRC,
        columns=list(direct_feature_order),   # includes segmentId itself
        filters=[("date", "=", dstr), ("hour", "=", int(hour))]).to_pandas()
    eng = eng[direct_feature_order]
    direct_raw = np.asarray(pipe_direct.predict(eng), dtype=np.float64)
    served_raw = snaps[f"{dstr} {hour:02d}:00"]["raw_ratio"].to_numpy()
    order_ok = True  # both frames come from the same stored row order? verify via segment ids
    # align by segmentId to be safe
    seg_served = snaps[f"{dstr} {hour:02d}:00"]["segmentId"].to_numpy()
    seg_direct = eng["segmentId"].to_numpy()
    if not np.array_equal(np.sort(seg_served), np.sort(seg_direct)):
        check(f"repro {dstr} {hour:02d}:00 segment sets equal", False)
        continue
    idx = pd.Index(seg_direct).get_indexer(seg_served)
    diff = np.abs(served_raw - direct_raw[idx])
    n_diff = int((diff > 0).sum())
    print(f"  {dstr} {hour:02d}:00 -> max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} "
          f"differing={n_diff}/{len(diff)} exact={(diff == 0).all()}")
    check(f"repro {dstr} {hour:02d}:00 bit-exact", bool((diff == 0).all()))

print()
print("=" * 70)
print("R3.11 PERFORMANCE BASELINE")
print("=" * 70)
for q, m in [("2024-08-26 17:00", "replay"), ("2024-08-31 08:00", "forecast")]:
    _, info = ps.predict_snapshot(q, m, stack)
    t = info["timing"]
    print(f"  {m:8s} {q}: assembly {t['assembly_s']*1e3:7.1f} ms | predict "
          f"{t['predict_s']*1e3:7.1f} ms | mapping {t['mapping_s']*1e3:6.1f} ms | "
          f"total {t['total_s']*1e3:7.1f} ms | segments {info['n_segments']}")

if fails:
    print("\nFAILURES:", fails)
    sys.exit(1)
print("\nR3 REPLAY GATES ALL PASSED")
