"""R2.1 / R2.2 - Load and inspect engineered_causal.parquet.

Reports schema, dtypes, ranges, rows-per-segment, missing values for all
columns (batched to bound memory), segment-set equality vs the R1 graph,
and calendar-encoding consistency checks. READ-ONLY: writes no artifacts.
"""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PQ = "cleaned/engineered_causal.parquet"
EDGES = "cleaned/graph/edges.csv"

MODEL_17 = [
    "distance", "is_weekend", "festival", "speedLimit", "hour",
    "segment_median_density", "hour_cos", "hour_sin", "frc",
    "segment_avg_density", "prev_day_density", "prev_week_density",
    "rolling_3h_mean", "dayofweek", "clim_seg_hour", "clim_density",
    "segmentId",
]

pf = pq.ParquetFile(PQ)
sch = pf.schema_arrow
print("=" * 70)
print("R2.1 SCHEMA")
print("=" * 70)
print(f"rows={pf.metadata.num_rows}  cols={pf.metadata.num_columns}  "
      f"row_groups={pf.metadata.num_row_groups}")
for f in sch:
    print(f"  {f.name:32s} {str(f.type):16s} nullable={f.nullable}")

col_names = [f.name for f in sch]

# ---- full-column reads for keys -------------------------------------------
print("\n-- reading key columns (segmentId, date, hour, dayofweek) --")
key_cols = [c for c in ["segmentId", "date", "hour", "dayofweek"] if c in col_names]
kdf = pf.read(columns=key_cols).to_pandas()
n = len(kdf)
seg = kdf["segmentId"]
print(f"segmentId dtype={seg.dtype}  nulls={seg.isna().sum()}")
u_seg = seg.nunique()
print(f"unique segmentId = {u_seg}")
rpc = seg.value_counts()
print(f"rows/segment: min={rpc.min()} max={rpc.max()} n_unique_counts={rpc.nunique()} "
      f"value_counts_of_counts={dict(rpc.value_counts())}")

date = kdf["date"]
if not np.issubdtype(date.dtype, np.datetime64):
    date = pd.to_datetime(date)
    print(f"(date column was {kdf['date'].dtype}, parsed -> datetime64)")
print(f"date dtype(after)={date.dtype} min={date.min()} max={date.max()} "
      f"unique_dates={date.nunique()}")
print(f"hour: min={kdf['hour'].min()} max={kdf['hour'].max()} "
      f"unique={sorted(kdf['hour'].unique())}")
print(f"dayofweek unique={sorted(kdf['dayofweek'].unique())}")

# perfect-grid check
grid = kdf.groupby(["segmentId", "date"], observed=True).size()
print(f"(segmentId,date) groups={len(grid)} all==24: {(grid == 24).all()}")

dup_full = kdf.duplicated(subset=["segmentId", "date", "hour"]).sum()
print(f"duplicate (segmentId,date,hour) keys = {dup_full}")
del grid

# dow encoding vs real weekday
dow_calc = date.dt.dayofweek.astype(kdf["dayofweek"].dtype)
mism = int((dow_calc != kdf["dayofweek"]).sum())
print(f"dayofweek vs pandas Monday=0 mismatches = {mism}")
dates_by_dow = (
    pd.DataFrame({"d": date, "dow": kdf["dayofweek"]})
    .drop_duplicates().sort_values(["dow", "d"])
)
print("dates by dow:")
for dw, grp in dates_by_dow.groupby("dow"):
    print(f"  dow={dw}: {[str(x.date()) for x in grp['d']]}")
last_dow_date = dates_by_dow.groupby("dow")["d"].max()
print("LAST observed date per dow:")
for dw, d in last_dow_date.items():
    print(f"  dow={dw}: {d.date()}")
del kdf, dow_calc, dates_by_dow

# ---- graph segment set -----------------------------------------------------
print("\n" + "=" * 70)
print("R2.10-PREP SEGMENT SET vs R1 GRAPH")
print("=" * 70)
e = pd.read_csv(EDGES, dtype=str)
print(f"edges.csv columns={list(e.columns)}")
obs = e[e["edge_type"] == "observed"]
graph_ids = set(int(float(x)) for x in obs["segment_id"])
parq_ids = set(int(x) for x in pd.read_parquet(PQ, columns=["segmentId"])["segmentId"].unique())
print(f"graph observed edges={len(obs)}  graph unique seg ids={len(graph_ids)}")
print(f"parquet unique seg ids={len(parq_ids)}")
print(f"intersection={len(graph_ids & parq_ids)}  graph_only={len(graph_ids - parq_ids)}  "
      f"parquet_only={len(parq_ids - graph_ids)}")
if graph_ids - parq_ids:
    print("  graph_only sample:", sorted(graph_ids - parq_ids)[:10])
if parq_ids - graph_ids:
    print("  parquet_only sample:", sorted(parq_ids - graph_ids)[:10])
del e, obs

# ---- batched scan of ALL columns: nulls / min / max ------------------------
print("\n" + "=" * 70)
print("R2.1 MISSING / RANGE SCAN (batched over all columns)")
print("=" * 70)
BATCH_COLS = 6
stats = {}
for i in range(0, len(col_names), BATCH_COLS):
    cols = col_names[i:i + BATCH_COLS]
    tbl = pf.read(columns=cols)
    bdf = tbl.to_pandas()
    for c in cols:
        s = bdf[c]
        st = {"dtype": str(s.dtype), "nulls": int(s.isna().sum())}
        if np.issubdtype(s.dtype, np.number):
            fin = s.dropna()
            if len(fin):
                st.update(min=float(np.nanmin(fin.values)), max=float(np.nanmax(fin.values)),
                          nonfinite=int((~np.isfinite(fin.values)).sum()))
            else:
                st.update(min=None, max=None, nonfinite=0)
        stats[c] = st
    del tbl, bdf
hdr = f"{'column':32s} {'dtype':14s} {'nulls':>9s} {'min':>14s} {'max':>14s} {'nonfin':>7s}"
print(hdr)
for c, st in stats.items():
    mn = "-" if st.get("min") is None else f"{st['min']:.6g}"
    mx = "-" if st.get("max") is None else f"{st['max']:.6g}"
    print(f"{c:32s} {st['dtype']:14s} {st['nulls']:9d} {mn:>14s} {mx:>14s} {st.get('nonfinite','-'):>7}")

# ---- calendar consistency (batched) ----------------------------------------
print("\n-- calendar encoding consistency (batched) --")
cal_cols = [c for c in ["date", "segmentId", "hour", "dayofweek", "is_weekend",
                        "festival", "hour_sin", "hour_cos"] if c in col_names]
acc = {
    "sin_err": 0.0, "cos_err": 0.0, "we_mism": 0, "fest_rows_true": set(),
    "fest_mism": 0, "rows": 0,
}
for batch in pf.iter_batches(batch_size=2_000_000, columns=cal_cols):
    b = batch.to_pandas()
    d = b["date"]
    if not np.issubdtype(d.dtype, np.datetime64):
        d = pd.to_datetime(d)
    h = b["hour"].to_numpy()
    acc["rows"] += len(b)
    if "hour_sin" in b and "hour_cos" in b:
        ang = 2.0 * np.pi * h / 24.0
        acc["sin_err"] = max(acc["sin_err"], float(np.abs(b["hour_sin"].to_numpy() - np.sin(ang)).max()))
        acc["cos_err"] = max(acc["cos_err"], float(np.abs(b["hour_cos"].to_numpy() - np.cos(ang)).max()))
    if "is_weekend" in b:
        we_calc = (b["dayofweek"].to_numpy() >= 5)
        acc["we_mism"] += int((we_calc != b["is_weekend"].to_numpy().astype(bool)).sum())
    if "festival" in b:
        fmask = d.dt.strftime("%Y-%m-%d").isin(["2024-08-15", "2024-08-19", "2024-08-26"])
        acc["fest_mism"] += int((fmask.to_numpy() != b["festival"].to_numpy().astype(bool)).sum())
        acc["fest_rows_true"] |= set(d[fmask].dt.strftime("%Y-%m-%d").unique())
print(f"rows scanned={acc['rows']}")
print(f"max|hour_sin - sin(2pi h/24)| = {acc['sin_err']:.3e}")
print(f"max|hour_cos - cos(2pi h/24)| = {acc['cos_err']:.3e}")
print(f"is_weekend vs (dow>=5) mismatches = {acc['we_mism']}")
print(f"festival flag mismatches vs {{15,19,26}} = {acc['fest_mism']}; "
      f"true-dates seen={sorted(acc['fest_rows_true'])}")

# ---- R2.2 mapping draft ------------------------------------------------------
print("\n" + "=" * 70)
print("R2.2 CONCEPT -> COLUMN MAPPING DRAFT (vs parquet schema)")
print("=" * 70)
concepts = MODEL_17 + ["clim_dow_hour", "probe_density", "probeCount",
                       "congestion_ratio", "congestion_ratio_clipped"]
missing_in_pq = []
for c in concepts:
    ok = c in col_names
    if not ok:
        missing_in_pq.append(c)
    print(f"  {c:28s} {'OK ' if ok else 'MISSING'} {stats.get(c, {}).get('dtype', '')}")
print(f"missing from parquet: {missing_in_pq}")
present_17 = [c for c in MODEL_17 if c in col_names]
print(f"\nmodel-17 present in parquet: {len(present_17)}/17")
print("NOTE: confirm exact order against catboost feature_names_ (r2_model_features.py)")
sys.exit(0)
