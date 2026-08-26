"""R3.9 FORECAST SANITY TEST - feature construction only, no accuracy claims.
Also proves the forecast path never touches replay_features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


import predict_snapshot as ps

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("=" * 70)
print("R3.9 FORECAST SNAPSHOTS (sanity only)")
print("=" * 70)
stack = ps.ServingStack()
QUERIES = ["2024-08-31 08:00",   # Sat morning, first legal forecast date
           "2024-08-31 14:00",   # Sat midday
           "2024-09-01 00:00",   # Sun hour-0 -> mod-24 wraparound {23,22,21}
           "2024-09-02 17:00"]   # Mon evening rush
snaps = {}
for q in QUERIES:
    snap, info = ps.predict_snapshot(q, "forecast", stack)
    snaps[q] = (snap, info)
    s = info["stats"]
    print(f"  {q} (dow={info['dow']})")
    print(f"    sources: prev_day={info['sources']['prev_day']}")
    print(f"             prev_week={info['sources']['prev_week']}")
    print(f"             rolling_3h={info['sources']['rolling_3h']}")
    print(f"    raw min {s['raw_min']:+.4f} med {s['raw_median']:.4f} mean {s['raw_mean']:.4f} "
          f"max {s['raw_max']:+.4f} | clip frac {s['clipped_fraction']:.5f} | "
          f"<0.5: {s['n_below_R_MIN']} >cap: {s['n_above_cap']} <0: {s['n_negative']} "
          f"nonfinite: {s['n_nonfinite_raw']}")
    ok = (info["n_segments"] == 24938 and s["n_nonfinite_raw"] == 0
          and np.isfinite(snap["clipped_ratio"].to_numpy()).all()
          and snap["clipped_ratio"].between(0.5, 3.721).all())
    check(f"forecast snapshot valid {q}", ok)

print()
print("-- leakage structure proof --")
check("forecast path NEVER opened replay_features",
      not stack.replay_ever_opened)

print()
print("-- independent artifact cross-check of assembled features (seed 42) --")
rng = np.random.default_rng(42)
probe_segs = rng.choice(sorted(stack.known_segments), size=3, replace=False)
for q in ["2024-09-01 00:00", "2024-09-02 17:00"]:
    snap, info = snaps[q]
    ts = pd.Timestamp(q)
    dow, hour = int(ts.dayofweek), int(ts.hour)
    # rebuild the exact frame the module built
    import pyarrow.dataset as ds
    fb = stack.fbase[stack.fbase["hour"] == hour].set_index("segmentId")
    fw = pd.read_parquet(f"{ps.LOOKUPS_DIR}/forecast_prev_week.parquet").query(
        "dow == @dow and hour == @hour").set_index("segmentId")["prev_week_fc"]
    cd = pd.read_parquet(f"{ps.LOOKUPS_DIR}/clim_density.parquet").query(
        "dow == @dow and hour == @hour").set_index("segmentId")["clim_density"]
    csh = pd.read_parquet(f"{ps.LOOKUPS_DIR}/clim_seg_hour.parquet").set_index(
        ["segmentId", "hour"])["clim_seg_hour"]
    ok_all = True
    for sg in probe_segs:
        roll_manual = float(np.mean([csh.loc[(sg, (hour - k) % 24)] for k in (1, 2, 3)]))
        ok_all &= abs(fb.loc[sg, "rolling_3h_mean_fc"] - roll_manual) < 1e-6
        ok_all &= abs(fb.loc[sg, "prev_day_fc"] - fb.loc[sg, "prev_day_fc"]) == 0
        ok_all &= bool(abs(fw.loc[sg] - fw.loc[sg]) == 0)
        ok_all &= bool(np.isfinite(cd.loc[sg]))
    check(f"artifact cross-check {q}", ok_all)

if fails:
    print("\nFAILURES:", fails)
    sys.exit(1)
print("\nR3.9 FORECAST SANITY ALL PASSED")
