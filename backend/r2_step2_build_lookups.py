"""R2.3-R2.13 - Build all R2 lookup artifacts from engineered_causal.parquet.

Design decisions (documented into lookup_metadata.json):
- STATIC lookups: single value per segment, verified constant across all
  20 dates x 24 h. Any variation -> hard STOP.
- CLIMATOLOGY: stored causal (expanding, past-only) columns are
  date-dependent BY DESIGN. Lookups store the TERMINAL causal state:
    clim_seg_hour  <- rows of the last observed date (2024-08-30)
    clim_dow_hour  <- rows of the LAST OBSERVED DATE PER WEEKDAY
                      (dow-compatible history, still strictly past)
    clim_density   <- same per-weekday terminal rows as clim_dow_hour
  Nothing is recomputed; only stored values are re-keyed.
- REPLAY: verbatim projection of the 17 model features + keys.
- FORECAST: locked D4 definitions, modulo-24 rolling mean, festival=0.
No model is touched; no routing logic here.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "cleaned/engineered_causal.parquet"
OUT = "cleaned/lookups"
os.makedirs(OUT, exist_ok=True)

MODEL_17 = [
    "distance", "is_weekend", "festival", "speedLimit", "hour",
    "segment_median_density", "hour_cos", "hour_sin", "frc",
    "segment_avg_density", "prev_day_density", "prev_week_density",
    "rolling_3h_mean", "dayofweek", "clim_seg_hour", "clim_density",
    "segmentId",
]
FESTIVALS = ["2024-08-15", "2024-08-19", "2024-08-26"]
LAST_OBS_DATE = "2024-08-30"
FORECAST_LEGAL_FROM = "2024-08-31"

fails = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        fails.append(f"{name}: {detail}")
    return cond


def jdump(obj):
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return super().default(o)
    return json.dumps(obj, indent=2, cls=NpEnc)


# ===========================================================================
# KEYS / GRID ORDER VERIFICATION
# ===========================================================================
print("=" * 70)
print("STEP A - key columns, grid order verification")
print("=" * 70)
kdf = pq.read_table(SRC, columns=["segmentId", "date", "hour"]).to_pandas()
S = kdf["segmentId"].nunique()
n_rows = len(kdf)
seg_arr = kdf["segmentId"].to_numpy()
check("segmentId sorted non-decreasing", bool(np.all(np.diff(seg_arr) >= 0)))
first_rows = np.flatnonzero(np.diff(seg_arr) != 0) + 1
block_sizes = np.diff(np.concatenate(([0], first_rows, [n_rows])))
check("all segment blocks == 480 rows", bool((block_sizes == 480).all()),
      f"segments={S}, unique sizes={sorted(set(block_sizes.tolist()))}")

hours = kdf["hour"].to_numpy().astype(np.int8)
exp_hours = np.tile(np.tile(np.arange(24, dtype=np.int8), 20), S)
check("within-block layout = [date-major, hour 0..23]",
      bool(np.array_equal(hours, exp_hours)))

dates_u = sorted(kdf["date"].unique().tolist())
D = len(dates_u)
check("20 unique ISO dates, Aug 11..30",
      D == 20 and dates_u[0] == "2024-08-11" and dates_u[-1] == LAST_OBS_DATE,
      f"{D} dates {dates_u[0]}..{dates_u[-1]}")
dser = pd.to_datetime(pd.Series(dates_u))
last_dow_date = {int(dw): d.strftime("%Y-%m-%d") for dw, d in dser.groupby(dser.dt.dayofweek).max().items()}
last_dow_idx = {dw: dates_u.index(ds) for dw, ds in last_dow_date.items()}
print(f"  last observed date per dow: {last_dow_date}")
del kdf, hours, exp_hours

grid_shape = (S, D, 24)


def load_grid(col):
    arr = pq.read_table(SRC, columns=[col]).column(col).combine_chunks()
    npa = arr.to_numpy(zero_copy_only=False)
    return npa.reshape(grid_shape)


def audit_lookup(df, key_cols, val_cols, name):
    """R2.11 audit for one lookup artifact."""
    n = len(df)
    dup = int(df.duplicated(subset=key_cols).sum())
    nulls = {c: int(df[c].isna().sum()) for c in df.columns}
    inf = {c: int(np.isinf(df[c].to_numpy()).sum()) if np.issubdtype(df[c].dtype, np.number) else 0
           for c in val_cols}
    useg = int(df["segmentId"].nunique())
    print(f"  AUDIT {name}: rows={n} unique_seg={useg} dup_keys={dup} "
          f"nulls={ {k: v for k, v in nulls.items() if v} or '{}' } "
          f"inf={ {k: v for k, v in inf.items() if v} or '{}' }")
    check(f"{name}: no duplicate keys", dup == 0)
    check(f"{name}: zero nulls", all(v == 0 for v in nulls.values()))
    check(f"{name}: zero infinities", all(v == 0 for v in inf.values()))
    return {"rows": n, "unique_segments": useg, "duplicate_keys": dup,
            "nulls": nulls, "infinite": inf,
            "ranges": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                           "median": float(df[c].median())} for c in val_cols}}


# ===========================================================================
# R2.3 STATIC SEGMENT FEATURES
# ===========================================================================
print("\n" + "=" * 70)
print("R2.3 STATIC SEGMENT FEATURE AUDIT")
print("=" * 70)
static_cols = ["distance", "speedLimit", "frc",
               "segment_median_density", "segment_avg_density"]
static_meta = {}
static_grids = {}
for c in static_cols:
    g = load_grid(c)
    static_grids[c] = g
    lo = g.min(axis=(1, 2))
    hi = g.max(axis=(1, 2))
    varying = int((hi != lo).sum())
    vals = g[:, 0, 0]
    nulls = int(pd.isna(vals).sum())
    static_meta[c] = {
        "unique_values_per_segment_expected": 1,
        "segments_with_variation": varying,
        "missing": nulls,
        "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)),
        "median": float(np.median(vals)),
        "dtype": str(vals.dtype),
    }
    print(f"  {c:26s} dtype={vals.dtype} varying_segs={varying} missing={nulls} "
          f"min={static_meta[c]['min']:.6g} max={static_meta[c]['max']:.6g} "
          f"median={static_meta[c]['median']:.6g}")
    check(f"static '{c}' constant across time for ALL segments", varying == 0)
if any(static_meta[c]["segments_with_variation"] > 0 for c in static_cols):
    print("STOP: supposedly-static feature varies across time. Investigate before R3.")
    sys.exit(2)

static_df = pd.DataFrame({
    "segmentId": np.sort(np.unique(seg_arr)).astype(np.int64),
})
for c in static_cols:
    static_df[c] = static_grids[c][:, 0, 0].astype(
        np.float32 if static_grids[c].dtype == np.float32 else static_grids[c].dtype)
static_df.to_parquet(f"{OUT}/static_segment_lookup.parquet", index=False)
am = audit_lookup(static_df, ["segmentId"], static_cols, "static_segment_lookup")
static_meta["_audit"] = am
print(f"  saved static_segment_lookup.parquet rows={len(static_df)}")

# ===========================================================================
# R2.4 / R2.5 / R2.6 CLIMATOLOGY TERMINAL LOOKUPS
# ===========================================================================
print("\n" + "=" * 70)
print("R2.4-R2.6 CLIMATOLOGY LOOKUPS (terminal causal states, verbatim)")
print("=" * 70)
clim_sh = load_grid("clim_seg_hour")
clim_dh = load_grid("clim_dow_hour")
clim_dn = load_grid("clim_density")
probe = load_grid("probe_density")

last_idx = D - 1
sh_term = clim_sh[:, last_idx, :]
drift_mean = float(np.abs(sh_term - clim_sh[:, last_idx - 1, :]).mean())
drift_max = float(np.abs(sh_term - clim_sh[:, last_idx - 1, :]).max())
nan_hist_sh = [int(np.isnan(clim_sh[:, i, :]).sum()) for i in range(D)]
print(f"  clim_seg_hour NaN count by date index: {nan_hist_sh}")
print(f"  terminal(Aug30) vs Aug29 drift: mean={drift_mean:.6g} max={drift_max:.6g}")

cs = pd.DataFrame({
    "segmentId": np.repeat(static_df["segmentId"].to_numpy(), 24).astype(np.int64),
    "hour": np.tile(np.arange(24, dtype=np.int8), S),
    "clim_seg_hour": sh_term.reshape(-1),
})
cs.to_parquet(f"{OUT}/clim_seg_hour.parquet", index=False)
meta_sh = audit_lookup(cs, ["segmentId", "hour"], ["clim_seg_hour"], "clim_seg_hour")
hours_repr = sorted(cs["hour"].unique().tolist())
check("clim_seg_hour hours 0..23 fully represented", hours_repr == list(range(24)))
combos_missing = S * 24 - len(cs.drop_duplicates(["segmentId", "hour"]))
check("clim_seg_hour full (seg,hour) grid", combos_missing == 0)

rows_dw = []
for dw in range(7):
    di = last_dow_idx[dw]
    blk_dh = clim_dh[:, di, :]
    blk_dn = clim_dn[:, di, :]
    nan_dh = int(np.isnan(blk_dh).sum())
    nan_dn = int(np.isnan(blk_dn).sum())
    print(f"  dow={dw} terminal date={last_dow_date[dw]} NaN clim_dow_hour={nan_dh} "
          f"NaN clim_density={nan_dn}")
    rows_dw.append(pd.DataFrame({
        "segmentId": np.repeat(static_df["segmentId"].to_numpy(), 24).astype(np.int64),
        "dow": np.full(S * 24, dw, dtype=np.int8),
        "hour": np.tile(np.arange(24, dtype=np.int8), S),
        "clim_dow_hour": blk_dh.reshape(-1),
        "clim_density": blk_dn.reshape(-1),
    }))
cd = pd.concat(rows_dw, ignore_index=True)
cd = cd.sort_values(["segmentId", "dow", "hour"], kind="stable").reset_index(drop=True)
cd["segmentId"] = cd["segmentId"].astype(np.int64)
cd["dow"] = cd["dow"].astype(np.int8)
cd["hour"] = cd["hour"].astype(np.int8)
# split into two artifacts to preserve each value column's own schema focus
cd_out = cd[["segmentId", "dow", "hour", "clim_dow_hour"]]
cd_out.to_parquet(f"{OUT}/clim_dow_hour.parquet", index=False)
meta_dh = audit_lookup(cd_out, ["segmentId", "dow", "hour"], ["clim_dow_hour"], "clim_dow_hour")
check("clim_dow_hour full (seg,dow,hour) grid",
      len(cd_out.drop_duplicates(["segmentId", "dow", "hour"])) == S * 168)
check("clim_dow_hour dow domain 0..6", sorted(cd_out["dow"].unique().tolist()) == list(range(7)))

cdn = cd[["segmentId", "dow", "hour", "clim_density"]]
cdn.to_parquet(f"{OUT}/clim_density.parquet", index=False)
meta_dn = audit_lookup(cdn, ["segmentId", "dow", "hour"], ["clim_density"], "clim_density")

# ===========================================================================
# R2.15(replay side) LAG CONSTRUCTION VERIFICATION (vectorized on the grid)
# ===========================================================================
print("\n" + "=" * 70)
print("R2.7-PREP REPLAY LAG VERIFICATION vs causal construction rules")
print("=" * 70)
med = static_grids["segment_median_density"][:, 0, 0]

exp_prev_day = np.empty_like(probe)
exp_prev_day[:, 1:, :] = probe[:, :-1, :]
exp_prev_day[:, 0, :] = med[:, None]
diff_pd = np.abs(exp_prev_day - load_grid("prev_day_density"))
print(f"  prev_day: max|stored-expected|={diff_pd.max():.6g}  "
      f"nonzero_rows={(diff_pd > 0).sum()}")

exp_prev_week = np.empty_like(probe)
exp_prev_week[:, 7:, :] = probe[:, :-7, :]
exp_prev_week[:, :7, :] = med[:, None, None]
diff_pw = np.abs(exp_prev_week - load_grid("prev_week_density"))
print(f"  prev_week: max|stored-expected|={diff_pw.max():.6g}  "
      f"nonzero_rows={(diff_pw > 0).sum()}")

# Training implementation (main.ipynb cell 51):
#   grp.rolling(3, min_periods=1).mean().groupby(level=0).shift(1)
# => stored[t] = mean(probe[t-3 .. t-1]) for t>=3 (per segment, hourly
#    continuity across midnight); head rows per segment:
#    t=0 -> NaN -> segment_median fill; t=1 -> probe[0]; t=2 -> mean(probe[0..1])
flat64 = probe.reshape(S, D * 24).astype(np.float64)
cum = np.cumsum(flat64, axis=1)                          # cum[:, i] = sum(flat[:i+1])
r3_exp_f = np.empty((S, D * 24), dtype=np.float32)
r3_exp_f[:, 0] = med                                     # fill rule (NaN head)
r3_exp_f[:, 1] = flat64[:, 0]
r3_exp_f[:, 2] = (flat64[:, 0] + flat64[:, 1]) / 2.0
r3_exp_f[:, 3] = flat64[:, 0:3].mean(axis=1)
r3_exp_f[:, 4:] = ((cum[:, 3:D * 24 - 1] - cum[:, 0:D * 24 - 4]) / 3.0).astype(np.float32)
del flat64
r3_stored = load_grid("rolling_3h_mean").reshape(S, D * 24)
diff_r3 = np.abs(r3_exp_f - r3_stored)
exact_r3 = float((diff_r3 == 0).mean())
print(f"  rolling_3h: max|stored-expected|={diff_r3.max():.6g}  "
      f"exact-match fraction={exact_r3:.6f}  mismatches>1e-3: {(diff_r3 > 1e-3).sum()}")
check("prev_day reproduces shift(24)+fill rule", float(diff_pd.max()) <= 1e-3)
check("prev_week reproduces shift(168)+fill rule", float(diff_pw.max()) <= 1e-3)
check("rolling_3h reproduces shift(1).rolling(3)+fill rule",
      float(diff_r3.max()) <= 1e-3, f"exact={exact_r3:.4f}")
lag_meta = {
    "prev_day_max_abs_diff": float(diff_pd.max()),
    "prev_week_max_abs_diff": float(diff_pw.max()),
    "rolling_3h_max_abs_diff": float(diff_r3.max()),
    "rolling_3h_exact_match_fraction": exact_r3,
}

# ===========================================================================
# R2.7 REPLAY FEATURES ARTIFACT (verbatim streaming projection)
# ===========================================================================
print("\n-- writing replay_features.parquet (verbatim projection) --")
replay_cols = ["date", "segmentId", "hour"] + \
    [c for c in MODEL_17[:-1] if c != "hour"]
src_schema = pq.read_schema(SRC)
schema = pa.schema([src_schema.field(c) for c in replay_cols])
writer = pq.ParquetWriter(f"{OUT}/replay_features.parquet", schema)
n_written = 0
for batch in pq.ParquetFile(SRC).iter_batches(batch_size=2_000_000, columns=replay_cols):
    tbl = pa.Table.from_batches([batch]).select(replay_cols)
    writer.write_table(tbl)
    n_written += batch.num_rows
writer.close()
print(f"  written rows={n_written}")
check("replay_features row count matches source", n_written == n_rows)

# ===========================================================================
# R2.8 FORECAST ARTIFACTS (locked definitions)
# ===========================================================================
print("\n" + "=" * 70)
print("R2.8 FORECAST BASE (definitions LOCKED per D4)")
print("=" * 70)
roll_fc = np.empty((S, 24), dtype=np.float32)
hh = np.arange(24)
srcs = ((hh - 1) % 24, (hh - 2) % 24, (hh - 3) % 24)
roll_fc = ((sh_term[:, srcs[0]].astype(np.float64)
            + sh_term[:, srcs[1]].astype(np.float64)
            + sh_term[:, srcs[2]].astype(np.float64)) / 3.0).astype(np.float32)
roll_fc64 = ((sh_term[:, srcs[0]].astype(np.float64)
              + sh_term[:, srcs[1]].astype(np.float64)
              + sh_term[:, srcs[2]].astype(np.float64)) / 3.0)
ang = 2.0 * np.pi * hh / 24.0
sin_h = np.sin(ang).astype(np.float32)
cos_h = np.cos(ang).astype(np.float32)

fb = pd.DataFrame({
    "segmentId": np.repeat(static_df["segmentId"].to_numpy(), 24).astype(np.int64),
    "hour": np.tile(hh.astype(np.int8), S),
    "distance": static_grids["distance"][:, 0, 0][np.repeat(np.arange(S), 24)].astype(np.float32),
    "speedLimit": static_grids["speedLimit"][:, 0, 0][np.repeat(np.arange(S), 24)].astype(np.float32),
    "frc": static_grids["frc"][:, 0, 0][np.repeat(np.arange(S), 24)],
    "segment_median_density": med[np.repeat(np.arange(S), 24)].astype(np.float32),
    "segment_avg_density": static_grids["segment_avg_density"][:, 0, 0][np.repeat(np.arange(S), 24)].astype(np.float32),
    "clim_seg_hour": sh_term.reshape(-1),
    "rolling_3h_mean_fc": roll_fc.reshape(-1),
    "prev_day_fc": probe[:, last_idx, :].reshape(-1),
    "hour_sin": np.tile(sin_h, S),
    "hour_cos": np.tile(cos_h, S),
    "festival": np.zeros(S * 24, dtype=np.int8),
})
fb["frc"] = fb["frc"].astype(np.int8)
fb["hour_sin"] = fb["hour_sin"].astype(np.float32)
fb["hour_cos"] = fb["hour_cos"].astype(np.float32)
fb.to_parquet(f"{OUT}/forecast_base.parquet", index=False)
meta_fb = audit_lookup(fb, ["segmentId", "hour"],
                       ["clim_seg_hour", "rolling_3h_mean_fc", "prev_day_fc",
                        "hour_sin", "hour_cos"], "forecast_base")

pw_fc = np.stack([probe[:, last_dow_idx[w], :] for w in range(7)], axis=1)
fw = pd.DataFrame({
    "segmentId": np.repeat(static_df["segmentId"].to_numpy(), 7 * 24).astype(np.int64),
    "dow": np.tile(np.repeat(np.arange(7, dtype=np.int8), 24), S),
    "hour": np.tile(np.arange(24, dtype=np.int8), 7 * S),
    "prev_week_fc": pw_fc.reshape(-1).astype(np.float32),
})
fw = fw.sort_values(["segmentId", "dow", "hour"], kind="stable").reset_index(drop=True)
fw.to_parquet(f"{OUT}/forecast_prev_week.parquet", index=False)
meta_fw = audit_lookup(fw, ["segmentId", "dow", "hour"], ["prev_week_fc"], "forecast_prev_week")
check("forecast_prev_week full grid", len(fw.drop_duplicates(["segmentId", "dow", "hour"])) == S * 168)

# hour-wraparound spot proof (the hour-0 trap): recompute fc for hour 0 manually
manual_h0 = (sh_term[:, 23].astype(np.float64)
             + sh_term[:, 22].astype(np.float64) + sh_term[:, 21].astype(np.float64)) / 3.0
check("rolling_fc hour0 uses hours 23,22,21 (exact at float64)",
      bool(np.array_equal(manual_h0, roll_fc64[:, 0])))

# ===========================================================================
# R2.12 SERVING SCHEMA
# ===========================================================================
STATIC_SRC = "cleaned/lookups/static_segment_lookup.parquet"
REPLAY_SRC = "cleaned/lookups/replay_features.parquet"
FB_SRC = "cleaned/lookups/forecast_base.parquet"
FW_SRC = "cleaned/lookups/forecast_prev_week.parquet"
CSH_SRC = "cleaned/lookups/clim_seg_hour.parquet"
CDN_SRC = "cleaned/lookups/clim_density.parquet"
CAL = ("calendar helper at query time (formulas verified against training "
       "encodings: dayofweek=pandas Mon=0; is_weekend=(dow>=5); "
       "festival in {2024-08-15,19,26}; hour_sin=sin(2pi*hour/24); "
       "hour_cos=cos(2pi*hour/24), computed float64->float32)")

feat_rows = []
def add(name, dtype, replay, forecast, transformation, required=True):
    feat_rows.append({
        "feature_name": name, "source": None, "dtype": dtype,
        "replay_source": replay, "forecast_source": forecast,
        "required_for_model": required, "transformation": transformation,
    })

add("distance", "float32", f"{REPLAY_SRC}:distance", f"{FB_SRC}:distance",
    "verbatim static per segment")
add("speedLimit", "float32", f"{REPLAY_SRC}:speedLimit", f"{FB_SRC}:speedLimit",
    "verbatim static per segment (km/h)")
add("frc", "int8", f"{REPLAY_SRC}:frc", f"{FB_SRC}:frc",
    "verbatim static per segment")
add("segment_median_density", "float32", f"{REPLAY_SRC}:segment_median_density",
    f"{FB_SRC}:segment_median_density", "verbatim static per segment")
add("segment_avg_density", "float32", f"{REPLAY_SRC}:segment_avg_density",
    f"{FB_SRC}:segment_avg_density", "verbatim static per segment")
add("hour", "int8", f"{REPLAY_SRC}:hour (=query hour)", "calendar helper (query hour)",
    "identity 0..23")
add("dayofweek", "int8", f"{REPLAY_SRC}:dayofweek", CAL,
    "pandas Monday=0 convention (verified 0 mismatches vs date)")
add("is_weekend", "int8", f"{REPLAY_SRC}:is_weekend", CAL,
    "(dayofweek>=5) (verified 0 mismatches)")
add("festival", "int8", f"{REPLAY_SRC}:festival", f"{FB_SRC}:festival (=0)",
    "True only for 2024-08-15/19/26; impossible for legal forecast dates")
add("hour_sin", "float32", f"{REPLAY_SRC}:hour_sin", f"{FB_SRC}:hour_sin",
    "sin(2*pi*hour/24) float64->float32 (max dev vs stored 1.55e-08)")
add("hour_cos", "float32", f"{REPLAY_SRC}:hour_cos", f"{FB_SRC}:hour_cos",
    "cos(2*pi*hour/24) float64->float32")
add("prev_day_density", "float32", f"{REPLAY_SRC}:prev_day_density",
    f"{FB_SRC}:prev_day_fc (probe_density of 2024-08-30 same hour)",
    "replay: stored causal lag; forecast: last observed date's same-hour density")
add("prev_week_density", "float32", f"{REPLAY_SRC}:prev_week_density",
    f"{FW_SRC}:prev_week_fc keyed (segmentId,dow,hour)",
    "replay: stored causal lag; forecast: probe_density of last observed SAME-WEEKDAY date")
add("rolling_3h_mean", "float32", f"{REPLAY_SRC}:rolling_3h_mean",
    f"{FB_SRC}:rolling_3h_mean_fc",
    "replay: stored causal lag; forecast: mean(clim_seg_hour[(h-1)%24,(h-2)%24,(h-3)%24]) "
    "- KNOWN train/serve skew (observed vs climatological history), accepted v1")
add("clim_seg_hour", "float32", f"{REPLAY_SRC}:clim_seg_hour",
    f"{CSH_SRC}:clim_seg_hour (terminal state of 2024-08-30)",
    "stored expanding past-only median, never recomputed")
add("clim_density", "float64", f"{REPLAY_SRC}:clim_density",
    f"{CDN_SRC}:clim_density (terminal state of last same-weekday observed date)",
    "stored causal blend, never recomputed")
add("segmentId", "int64", "query parameter", "query parameter",
    "MUST be int64 (CatBoost native categorical idx 16 rejects float)")

serving_schema = {
    "model": "catboost_model.pkl",
    "feature_order_from_feature_names_": MODEL_17,
    "notes": [
        "R3 assembles features strictly via this schema; do not reimplement logic.",
        "segmentId must stay int64 end-to-end.",
        "Replay legal window: 2024-08-11..2024-08-30 (hard assert).",
        "Forecast legal only for date >= 2024-08-31 (hard assert).",
        "Queries before 2024-08-11 are rejected outright.",
    ],
    "features": feat_rows,
}
with open(f"{OUT}/serving_schema.json", "w") as f:
    f.write(jdump(serving_schema))
print(f"  serving_schema.json written ({len(feat_rows)} features)")

# ===========================================================================
# R2.13 lookup_metadata.json
# ===========================================================================
graph_ids = set(int(float(x)) for x in
                pd.read_csv("cleaned/graph/edges.csv", dtype=str)
                .query("edge_type == 'observed'")["segment_id"])
lookup_ids = set(static_df["segmentId"].tolist())
metadata = {
    "round": "R2 - LOOKUP TABLES",
    "source": {
        "parquet": SRC,
        "rows": int(n_rows),
        "columns": 24,
        "segments": int(S),
        "dates": dates_u,
        "perfect_grid": "24938 segs x 20 dates x 24 h, 480 rows/segment, 0 duplicate keys",
        "segmentId_dtype": "int64 (preserved everywhere)",
    },
    "model_contract": {
        "artifact": "catboost_model.pkl",
        "feature_names_": MODEL_17,
        "categorical_indices": [16],
        "note": "clim_dow_hour is NOT a model feature; built for completeness/traceability",
    },
    "static_features": static_meta,
    "climatology_definitions": {
        "principle": "STORED causal columns only; terminal causal state re-keyed; never recomputed",
        "clim_seg_hour": {"key": ["segmentId", "hour"],
                          "source_rows": f"date == {LAST_OBS_DATE}",
                          "value_dtype": "float32"},
        "clim_dow_hour": {"key": ["segmentId", "dow", "hour"],
                          "source_rows": "last observed date PER WEEKDAY",
                          "last_observed_date_per_dow": {str(k): v for k, v in last_dow_date.items()},
                          "value_dtype": "float32"},
        "clim_density": {"key": ["segmentId", "dow", "hour"],
                         "source_rows": "same per-weekday terminal rows as clim_dow_hour",
                         "value_dtype": "float64 (source dtype preserved)"},
        "terminal_drift_seg_hour_Aug29_to_Aug30": {"mean_abs": drift_mean, "max_abs": drift_max},
    },
    "replay": {
        "window": "2024-08-11..2024-08-30",
        "keys": ["date", "hour", "segmentId"],
        "features": MODEL_17,
        "values": "verbatim stored engineered rows; lags NOT recomputed, NOT replaced by climatology",
        "lag_verification_vs_causal_rules": lag_meta,
    },
    "forecast": {
        "legal_only_on_or_after": FORECAST_LEGAL_FROM,
        "prev_day_density": "probe_density of last observed date (2024-08-30), same hour",
        "prev_week_density": {str(w): last_dow_date[w] for w in range(7)},
        "rolling_3h_mean": "mean(clim_seg_hour[(h-1)%24,(h-2)%24,(h-3)%24]); hour0 -> 23,22,21",
        "festival_dates": FESTIVALS,
        "known_skew": ("training rolling_3h_mean uses observed causal history; forecast uses "
                       "climatological history - accepted v1 approximation"),
        "leakage_note": "all sources are observation dates <= 2024-08-30; no future data touched",
    },
    "calendar": {
        "dayofweek": "pandas Monday=0 (verified equal to stored column on all 11,970,240 rows)",
        "is_weekend": "(dayofweek >= 5) (verified 0 mismatches)",
        "festival": "1 iff date in [2024-08-15, 2024-08-19, 2024-08-26] (verified 0 mismatches)",
        "hour_sin_cos": "sin/cos(2*pi*hour/24); max deviation vs stored 1.554e-08 (float32 rounding)",
    },
    "artifacts_audit": {
        "static_segment_lookup": static_meta["_audit"],
        "clim_seg_hour": meta_sh,
        "clim_dow_hour": meta_dh,
        "clim_density": meta_dn,
        "forecast_base": meta_fb,
        "forecast_prev_week": meta_fw,
        "replay_features": {"rows": int(n_written), "keys": ["date", "hour", "segmentId"],
                            "duplicate_keys_source_verified": 0},
    },
    "coverage_graph_vs_lookups": {
        "graph_segments": len(graph_ids),
        "lookup_segments": len(lookup_ids),
        "intersection": len(graph_ids & lookup_ids),
        "graph_only": sorted(graph_ids - lookup_ids)[:5],
        "lookup_only": sorted(lookup_ids - graph_ids)[:5],
    },
}
check("graph vs lookup segment sets identical",
      graph_ids == lookup_ids and len(graph_ids) == S)
with open(f"{OUT}/lookup_metadata.json", "w") as f:
    f.write(jdump(metadata))

print("\n" + "=" * 70)
print("R2 BUILD SUMMARY")
print("=" * 70)
for fn in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, fn))
    print(f"  {fn:36s} {sz/1e6:9.2f} MB")
if fails:
    print("\nFAILURES:")
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("\nALL BUILD CHECKS PASSED")
