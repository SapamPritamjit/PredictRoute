"""R2.14/R2.15 - fresh-process reload verification + no-leakage checks."""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

OUT = "cleaned/lookups"
SRC = "cleaned/engineered_causal.parquet"
LAST_OBS = "2024-08-30"
SAMPLE_DATES = ["2024-08-26", "2024-08-27"]
MODEL_17 = ["distance", "is_weekend", "festival", "speedLimit", "hour",
            "segment_median_density", "hour_cos", "hour_sin", "frc",
            "segment_avg_density", "prev_day_density", "prev_week_density",
            "rolling_3h_mean", "dayofweek", "clim_seg_hour", "clim_density",
            "segmentId"]
EXP = {
    "static_segment_lookup.parquet": {"segmentId": "int64", "distance": "float32",
        "speedLimit": "float32", "frc": "int8", "segment_median_density": "float32",
        "segment_avg_density": "float32"},
    "clim_seg_hour.parquet": {"segmentId": "int64", "hour": "int8",
        "clim_seg_hour": "float32"},
    "clim_dow_hour.parquet": {"segmentId": "int64", "dow": "int8", "hour": "int8",
        "clim_dow_hour": "float32"},
    "clim_density.parquet": {"segmentId": "int64", "dow": "int8", "hour": "int8",
        "clim_density": "float64"},
    "forecast_base.parquet": {"segmentId": "int64", "hour": "int8",
        "distance": "float32", "speedLimit": "float32", "frc": "int8",
        "segment_median_density": "float32", "segment_avg_density": "float32",
        "clim_seg_hour": "float32", "rolling_3h_mean_fc": "float32",
        "prev_day_fc": "float32", "hour_sin": "float32", "hour_cos": "float32",
        "festival": "int8"},
    "forecast_prev_week.parquet": {"segmentId": "int64", "dow": "int8",
        "hour": "int8", "prev_week_fc": "float32"},
}
KEYS = {"static_segment_lookup.parquet": ["segmentId"],
        "clim_seg_hour.parquet": ["segmentId", "hour"],
        "clim_dow_hour.parquet": ["segmentId", "dow", "hour"],
        "clim_density.parquet": ["segmentId", "dow", "hour"],
        "forecast_base.parquet": ["segmentId", "hour"],
        "forecast_prev_week.parquet": ["segmentId", "dow", "hour"]}
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("=" * 70)
print("R2.14a RELOAD: structure / dtypes / key uniqueness (fresh process)")
print("=" * 70)
arts = {}
for fn, exp in EXP.items():
    df = pd.read_parquet(f"{OUT}/{fn}")
    arts[fn] = df
    bad = {c: str(df[c].dtype) for c in exp if str(df[c].dtype) != exp[c]}
    check(f"{fn}: dtypes rows={len(df)}", not bad,
          f"mismatch={bad}" if bad else "")

for fn, ks in KEYS.items():
    d = int(arts[fn].duplicated(ks).sum())
    check(f"{fn}: dup keys {ks}", d == 0)

rp = pd.read_parquet(f"{OUT}/replay_features.parquet")
check("replay_features rows=11,970,240", len(rp) == 11_970_240, str(len(rp)))
dt_ok = (str(rp["segmentId"].dtype) == "int64"
         and str(rp["clim_density"].dtype) == "float64"
         and str(rp["prev_day_density"].dtype) == "float32"
         and all(str(rp[c].dtype) == "int8" for c in ["is_weekend", "festival", "dayofweek"]))
check("replay_features dtypes (int64 seg / f64 clim_density / int8 flags)", dt_ok)
dup = int(rp.duplicated(["date", "hour", "segmentId"]).sum())
udates = sorted(rp["date"].unique().tolist())
exp_dates = [f"2024-08-{d:02d}" for d in range(11, 31)]
check("replay dup keys=0 + window exactly Aug11..30",
      dup == 0 and udates == exp_dates, f"dups={dup} ndates={len(udates)}")

graph_ids = set(int(float(x)) for x in pd.read_csv(
    "cleaned/graph/edges.csv", dtype=str).query("edge_type == 'observed'")["segment_id"])
rp_ids = set(pd.unique(rp["segmentId"]).tolist())
sets = {fn: set(df["segmentId"].unique().tolist()) for fn, df in arts.items()}
sets["replay"] = rp_ids
sets["graph"] = graph_ids
all_eq = len({frozenset(s) for s in sets.values()}) == 1
sizes = {k: len(v) for k, v in sets.items()}
check("R2.10 graph==every lookup segment set", all_eq, str(sizes))


print()
print("=" * 70)
print("R2.15 NO-LEAKAGE CHECKS (artifact-level, vectorized)")
print("=" * 70)
S, D = 24938, 20
probe = pq.read_table(SRC, columns=["probe_density"]).column(
    "probe_density").combine_chunks().to_numpy(zero_copy_only=False).reshape(S, D, 24)
med = arts["static_segment_lookup.parquet"].sort_values("segmentId")[
    "segment_median_density"].to_numpy()

# order the replay rows into the canonical grid to compare lags causally
dmap = {d: i for i, d in enumerate(exp_dates)}
lg = rp[["date", "segmentId", "hour", "prev_day_density",
         "prev_week_density", "rolling_3h_mean"]].copy()
lg["_d"] = lg["date"].map(dmap).astype(np.int8)
lg = lg.sort_values(["segmentId", "_d", "hour"], kind="stable").reset_index(drop=True)
seg_sorted = np.sort(rp["segmentId"].unique())
assert np.array_equal(lg["segmentId"].to_numpy(),
                      np.repeat(seg_sorted, D * 24)), "replay grid layout unexpected"
pd_a = lg["prev_day_density"].to_numpy().reshape(S, D, 24)
pw_a = lg["prev_week_density"].to_numpy().reshape(S, D, 24)
r3_a = lg["rolling_3h_mean"].to_numpy().reshape(S, D, 24)

exp_pd = np.empty_like(probe)
exp_pd[:, 1:, :] = probe[:, :-1, :]
exp_pd[:, 0, :] = med[:, None]
check("L1 replay prev_day == probe(d-1,h), Aug11 -> seg_median (strictly past)",
      bool(np.array_equal(exp_pd, pd_a)), f"maxdiff={np.abs(exp_pd-pd_a).max()}")

exp_pw = np.empty_like(probe)
exp_pw[:, 7:, :] = probe[:, :-7, :]
exp_pw[:, :7, :] = med[:, None, None]
check("L2 replay prev_week == probe(d-7,h), first week -> seg_median",
      bool(np.array_equal(exp_pw, pw_a)), f"maxdiff={np.abs(exp_pw-pw_a).max()}")

f64 = probe.reshape(S, D * 24).astype(np.float64)
cum = np.cumsum(f64, axis=1)
r3e = np.empty((S, D * 24), dtype=np.float32)
r3e[:, 0] = med
r3e[:, 1] = f64[:, 0]
r3e[:, 2] = (f64[:, 0] + f64[:, 1]) / 2.0
r3e[:, 3] = f64[:, 0:3].mean(axis=1)
r3e[:, 4:] = ((cum[:, 3:D * 24 - 1] - cum[:, 0:D * 24 - 4]) / 3.0).astype(np.float32)
del f64
check("L3 replay rolling_3h == mean(prev 3 hourly obs) shifted, head fills",
      bool(np.array_equal(r3e.reshape(S, D, 24), r3_a)),
      f"maxdiff={np.abs(r3e.reshape(S,D,24)-r3_a).max()}")

# L4: forecast sources are observed dates only (<= Aug 30)
last_idx = D - 1
fb = arts["forecast_base.parquet"].sort_values(["segmentId", "hour"]).reset_index(drop=True)
fb_seg = fb["segmentId"].to_numpy().reshape(S, 24)
prev_day_fc_g = fb["prev_day_fc"].to_numpy().reshape(S, 24)
check("L4 forecast prev_day_fc == probe(2024-08-30) exactly",
      bool(np.array_equal(fb_seg,
           np.repeat(np.sort(np.fromiter(graph_ids, dtype=np.int64)), 24).reshape(S, 24))) and
      bool(np.array_equal(prev_day_fc_g, probe[:, last_idx, :])))

meta = json.load(open(f"{OUT}/lookup_metadata.json"))
last_dow_date = {int(k): v for k, v in
                 meta["climatology_definitions"]["clim_dow_hour"]["last_observed_date_per_dow"].items()}
obs_dates = set(exp_dates)
src_dates = set(last_dow_date.values()) | {LAST_OBS}
check("L5 all forecast/climatology source dates observed and <= 2024-08-30",
      src_dates <= obs_dates, str(sorted(src_dates)))

fw = arts["forecast_prev_week.parquet"]
dw_map = {int(d[i][: -3:]) : i for i in []} if False else None
ok_pw = True
for w in range(7):
    di = exp_dates.index(last_dow_date[w])
    sel = fw[fw["dow"] == w].sort_values(["segmentId", "hour"])
    vals = sel["prev_week_fc"].to_numpy().reshape(S, 24)
    ok_pw &= bool(np.array_equal(vals, probe[:, di, :]))
check("L6 forecast prev_week_fc == probe(last same-weekday date) per dow", ok_pw)

csh = arts["clim_seg_hour.parquet"].sort_values(["segmentId", "hour"]).reset_index(drop=True)
csh_g = csh["clim_seg_hour"].to_numpy().reshape(S, 24)
check("L7 clim_seg_hour lookup == stored causal state of 2024-08-30",
      bool(np.array_equal(csh_g, pq.read_table(SRC, columns=["clim_seg_hour"])
                          .column("clim_seg_hour").combine_chunks()
                          .to_numpy(zero_copy_only=False).reshape(S, D, 24)[:, last_idx, :])))

hh = np.arange(24)
roll_chk = ((csh_g[:, (hh - 1) % 24].astype(np.float64)
             + csh_g[:, (hh - 2) % 24].astype(np.float64)
             + csh_g[:, (hh - 3) % 24].astype(np.float64)) / 3.0).astype(np.float32)
fb_roll = fb["rolling_3h_mean_fc"].to_numpy().reshape(S, 24)
check("L8 rolling_3h_mean_fc recomputed from SAVED clim_seg_hour (mod-24)",
      bool(np.array_equal(roll_chk, fb_roll)),
      f"maxdiff={np.abs(roll_chk.astype(np.float64)-fb_roll.astype(np.float64)).max()}")


print()
print("=" * 70)
print("R2.14b RECONSTRUCT val sample from SAVED ARTIFACTS ONLY vs engineered")
print("=" * 70)
f = pq.ParquetFile(SRC)
cols_needed = ["date", "segmentId", "hour"] + \
    [c for c in MODEL_17[:-1] if c not in ("date", "segmentId", "hour")]
eng = pq.read_table(SRC, columns=cols_needed,
                    filters=[("date", "in", SAMPLE_DATES)]).to_pandas()
repl = rp[rp["date"].isin(SAMPLE_DATES)].copy()

# R3-style assembly: statics joined, calendar computed, lags/clims from replay
STATIC_COLS = ["distance", "speedLimit", "frc",
               "segment_median_density", "segment_avg_density"]
stat = arts["static_segment_lookup.parquet"].rename(
    columns={c: c + "_lk" for c in STATIC_COLS})
asm = repl.merge(stat, on="segmentId", how="left")
for c in STATIC_COLS:
    asm[c] = asm[c + "_lk"]
dts = pd.to_datetime(asm["date"])
dow = dts.dt.dayofweek.astype(np.int8).to_numpy()
hr = asm["hour"].to_numpy().astype(np.int8)
ang = 2.0 * np.pi * hr.astype(np.float64) / 24.0
asm["dayofweek"] = dow
asm["is_weekend"] = (dow >= 5).astype(np.int8)
fest = dts.dt.strftime("%Y-%m-%d").isin(["2024-08-15", "2024-08-19", "2024-08-26"]).astype(np.int8)
asm["festival"] = fest.to_numpy()
asm["hour_sin"] = np.sin(ang).astype(np.float32)
asm["hour_cos"] = np.cos(ang).astype(np.float32)

KEYS3 = ["date", "hour", "segmentId"]
eng_s = eng.sort_values(KEYS3, kind="stable").reset_index(drop=True)
asm_s = asm.sort_values(KEYS3, kind="stable").reset_index(drop=True)
keys_eq = all(bool(np.array_equal(eng_s[k].to_numpy(), asm_s[k].to_numpy()))
              for k in KEYS3)
check("sample key sets identical (no dropped rows)",
      len(eng_s) == len(asm_s) and keys_eq,
      f"eng={len(eng_s)} asm={len(asm_s)} keys_eq={keys_eq}")

print(f"  {'feature':26s} {'max_abs_diff':>14s} {'exact':>6s}")
worst = 0.0
for c in MODEL_17:
    if c == "segmentId":
        continue
    a = asm_s[c].to_numpy()
    b = eng_s[c].to_numpy()
    md = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    exact = bool(np.array_equal(a, b))
    tol = 1e-6 if c in ("hour_sin", "hour_cos") else 0.0
    ok = md <= tol
    worst = max(worst, md if c not in ("hour_sin", "hour_cos") else 0.0)
    print(f"  {c:26s} {md:14.3e} {str(exact):>6s} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"reconstruct {c}")
check("reconstruction: all non-calendar features bit-exact", worst == 0.0)

print()
print("=" * 70)
print("R2.14c REPRESENTATIVE VALUE SPOT-CHECKS (seed 42)")
print("=" * 70)
rng = np.random.default_rng(42)
seg_sample = rng.choice(sorted(graph_ids), size=5, replace=False)
st = arts["static_segment_lookup.parquet"].set_index("segmentId")
first_rows = pq.read_table(SRC, columns=["segmentId"] + [
    "distance", "speedLimit", "frc", "segment_median_density",
    "segment_avg_density"]).to_pandas().groupby("segmentId").first()
ok_all = True
for s in seg_sample:
    a = st.loc[s]
    b = first_rows.loc[s]
    for c in ["distance", "speedLimit", "segment_median_density", "segment_avg_density"]:
        ok_all &= float(a[c]) == float(b[c])
    ok_all &= int(a["frc"]) == int(b["frc"])
check("static lookup values == engineered per-segment values (5 random segs)", ok_all)

aug30 = pq.read_table(SRC, columns=["date", "segmentId", "hour", "clim_seg_hour"],
                      filters=[("date", "=", LAST_OBS)]).to_pandas()
aug30 = aug30[aug30["segmentId"].isin(seg_sample)]
m = aug30.merge(csh.set_index(["segmentId", "hour"]), on=["segmentId", "hour"],
                suffixes=("_src", "_lk"))
check("clim_seg_hour spot rows match Aug-30 source",
      bool(np.array_equal(m["clim_seg_hour_src"].to_numpy(),
                          m["clim_seg_hour_lk"].to_numpy())), f"rows={len(m)}")

schema = json.load(open(f"{OUT}/serving_schema.json"))
names_in_order = [r["feature_name"] for r in schema["features"]]
canonical = sorted(MODEL_17)
check("serving_schema covers exactly the 17 model features",
      sorted(names_in_order) == canonical and len(names_in_order) == 17,
      str(len(names_in_order)))

print()
print("=" * 70)
if fails:
    print(f"R2 RELOAD VERIFICATION FAILED ({len(fails)}):")
    for x in fails:
        print("   -", x)
    sys.exit(1)
print("R2.14/R2.15 ALL RELOAD + LEAKAGE CHECKS PASSED")
