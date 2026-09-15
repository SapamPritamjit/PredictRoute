"""R3 - PREDICTION SNAPSHOT LAYER (reusable module).

predict_snapshot(departure_datetime, mode) -> DataFrame
    [segmentId int64, raw_ratio float64, clipped_ratio float64]

- Feature names/order come from the saved CatBoost pipeline
  (feature_names_), cross-checked against serving_schema.json.
  No hardcoded feature list.
- REPLAY (2024-08-11..30): features read VERBATIM from
  cleaned/lookups/replay_features.parquet for the exact (date, hour).
- FORECAST (>= 2024-08-31): features assembled from forecast_base +
  forecast_prev_week using the locked D4 definitions. This code path
  never opens replay_features (structural no-leakage guarantee).
- Dates before 2024-08-11 are rejected outright.
- ONE batch predict over all segments. segmentId stays int64.
- clipped_ratio = clip(raw_ratio, 0.5, 3.721); raw kept for diagnostics.

No routing / snapping / travel-time logic lives here (that is R4).
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path

# ── Project root (backend/ is one level below root) ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PKL = str(PROJECT_ROOT / "catboost_model.pkl")
LOOKUPS_DIR = str(PROJECT_ROOT / "cleaned" / "lookups")
REPLAY_START = "2024-08-11"
REPLAY_END = "2024-08-30"
FORECAST_START = "2024-08-31"
FESTIVAL_DATES = {"2024-08-15", "2024-08-19", "2024-08-26"}
R_MIN = 0.5
R_MAX = 3.721
N_SEGMENTS = 24938

_DTYPE_MAP = {"float32": np.float32, "float64": np.float64,
              "int8": np.int8, "int16": np.int16, "int32": np.int32,
              "int64": np.int64}


class SnapshotError(RuntimeError):
    """Raised on any contract/validation violation (hard STOP)."""


class ServingStack:
    """Fresh-load of model + R2 artifacts + schemas."""

    def __init__(self, lookups_dir: str = LOOKUPS_DIR,
                 model_path: str = MODEL_PKL):
        t0 = time.perf_counter()
        self.lookups_dir = lookups_dir
        with open(f"{lookups_dir}/serving_schema.json") as f:
            self.schema = json.load(f)
        with open(f"{lookups_dir}/lookup_metadata.json") as f:
            self.meta = json.load(f)
        self.pipeline = joblib.load(model_path)
        est = self.pipeline.steps[-1][1]
        self.feature_names = [str(x) for x in est.feature_names_]
        schema_names = sorted(fr["feature_name"] for fr in self.schema["features"])
        if schema_names != sorted(self.feature_names) or len(self.feature_names) != 17:
            raise SnapshotError("serving_schema.json does not match model.feature_names_")
        self.feature_dtypes = {fr["feature_name"]: fr["dtype"]
                               for fr in self.schema["features"]}
        self.static = pd.read_parquet(f"{lookups_dir}/static_segment_lookup.parquet")
        self.fbase = pd.read_parquet(f"{lookups_dir}/forecast_base.parquet")
        self.fweek = pd.read_parquet(f"{lookups_dir}/forecast_prev_week.parquet")
        self.clim_density_ds = ds.dataset(
            f"{lookups_dir}/clim_density.parquet", format="parquet")
        self.known_segments = frozenset(self.static["segmentId"].tolist())
        self._replay_ds = None          # opened lazily, ONLY by replay path
        self.replay_ever_opened = False
        self.load_seconds = time.perf_counter() - t0

    @property
    def replay_ds(self):
        self.replay_ever_opened = True
        if self._replay_ds is None:
            self._replay_ds = ds.dataset(
                f"{self.lookups_dir}/replay_features.parquet", format="parquet")
        return self._replay_ds


def parse_query(departure_datetime) -> tuple[pd.Timestamp, str, int]:
    ts = pd.Timestamp(departure_datetime)
    if ts.tzinfo is not None:
        raise SnapshotError("naive IST datetimes only (no tz)")
    if ts.minute or ts.second or ts.microsecond:
        raise SnapshotError("hourly model: departure must be exactly on the hour")
    return ts, ts.strftime("%Y-%m-%d"), int(ts.hour)


def resolve_mode(departure_datetime, mode: str):
    """R3.3 hard date/mode boundary rules."""
    if mode not in ("replay", "forecast"):
        raise SnapshotError(f"mode must be 'replay'|'forecast', got {mode!r}")
    ts, dstr, hour = parse_query(departure_datetime)
    if dstr < REPLAY_START:
        raise SnapshotError(
            f"rejected: {dstr} precedes first observed date {REPLAY_START}")
    if mode == "replay" and dstr > REPLAY_END:
        raise SnapshotError(
            f"rejected: replay mode is legal only {REPLAY_START}..{REPLAY_END}, got {dstr}")
    if mode == "forecast" and dstr < FORECAST_START:
        raise SnapshotError(
            f"rejected: forecast mode is legal only from {FORECAST_START}, got {dstr}")
    return ts, dstr, hour


def _enforce_dtypes(df: pd.DataFrame, stack: ServingStack) -> pd.DataFrame:
    for c in df.columns:
        want = _DTYPE_MAP.get(stack.feature_dtypes.get(c, ""), None)
        if want is not None and str(df[c].dtype) != stack.feature_dtypes[c]:
            df[c] = df[c].astype(want)
    return df


def build_replay_frame(stack: ServingStack, dstr: str, hour: int) -> pd.DataFrame:
    tbl = stack.replay_ds.to_table(
        filter=(ds.field("date") == dstr) & (ds.field("hour") == hour))
    df = tbl.to_pandas()
    missing = [c for c in stack.feature_names if c not in df.columns]
    if missing:
        raise SnapshotError(f"replay rows missing columns: {missing}")
    return _enforce_dtypes(df[stack.feature_names], stack)


def build_forecast_frame(stack: ServingStack, ts: pd.Timestamp,
                         dstr: str, hour: int) -> pd.DataFrame:
    if dstr in FESTIVAL_DATES:      # structurally impossible for legal dates
        raise SnapshotError("festival date reached forecast path - bug")
    dow = int(ts.dayofweek)
    fb = stack.fbase[stack.fbase["hour"] == hour]
    fw = stack.fweek[(stack.fweek["dow"] == dow)
                     & (stack.fweek["hour"] == hour)][["segmentId", "prev_week_fc"]]
    df = fb.merge(fw, on="segmentId", how="inner", validate="one_to_one")
    cd = stack.clim_density_ds.to_table(
        filter=(ds.field("dow") == dow) & (ds.field("hour") == hour)).to_pandas()
    df = df.merge(cd[["segmentId", "clim_density"]], on="segmentId",
                  how="inner", validate="one_to_one")
    df = df.rename(columns={"rolling_3h_mean_fc": "rolling_3h_mean",
                            "prev_day_fc": "prev_day_density",
                            "prev_week_fc": "prev_week_density"})
    df["dayofweek"] = dow
    df["is_weekend"] = int(dow >= 5)
    df["festival"] = 0
    missing = [c for c in stack.feature_names if c not in df.columns]
    if missing:
        raise SnapshotError(f"forecast assembly missing columns: {missing}")
    return _enforce_dtypes(df[stack.feature_names], stack)


def validate_features(X: pd.DataFrame, stack: ServingStack) -> list[str]:
    """R3.6 twelve hard checks; returns [] or raises via caller."""
    errs = []
    if len(X) != N_SEGMENTS:
        errs.append(f"row count {len(X)} != {N_SEGMENTS}")
    if X.shape[1] != 17:
        errs.append(f"column count {X.shape[1]} != 17")
    if list(X.columns) != stack.feature_names:
        errs.append("feature names/order != model.feature_names_")
    if str(X["segmentId"].dtype) != "int64":
        errs.append(f"segmentId dtype {X['segmentId'].dtype} != int64")
    n_na = int(X.isna().to_numpy().sum())
    if n_na:
        errs.append(f"{n_na} NaN cells")
    arrs = [X[c].to_numpy() for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    pos_inf = sum(int(np.isinf(a).sum()) for a in arrs)
    neg_inf = sum(int((a == -np.inf).sum()) for a in arrs)
    if pos_inf:
        errs.append(f"{pos_inf} +inf cells")
    if neg_inf:
        errs.append(f"{neg_inf} -inf cells")
    dup = int(X["segmentId"].duplicated().sum())
    if dup:
        errs.append(f"{dup} duplicate segmentIds")
    unknown = set(X["segmentId"].tolist()) - stack.known_segments
    if unknown:
        errs.append(f"{len(unknown)} unknown segmentIds e.g. {sorted(unknown)[:5]}")
    return errs


def prediction_stats(raw: np.ndarray, clipped: np.ndarray) -> dict:
    finite = np.isfinite(raw)
    return {
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()), "raw_median": float(np.median(raw)),
        "raw_p1": float(np.percentile(raw, 1)), "raw_p5": float(np.percentile(raw, 5)),
        "raw_p95": float(np.percentile(raw, 95)), "raw_p99": float(np.percentile(raw, 99)),
        "clip_min": float(clipped.min()), "clip_max": float(clipped.max()),
        "clip_mean": float(clipped.mean()), "clip_median": float(np.median(clipped)),
        "clipped_fraction": float((clipped != raw).mean()),
        "n_below_R_MIN": int((raw < R_MIN).sum()),
        "n_above_cap": int((raw > R_MAX).sum()),
        "n_negative": int((raw < 0).sum()),
        "n_nonfinite_raw": int((~finite).sum()),
    }


def predict_snapshot(departure_datetime, mode: str, stack: ServingStack | None = None
                     ) -> tuple[pd.DataFrame, dict]:
    """One-batch congestion-ratio snapshot for every graph segment.

    Returns (snapshot_df, info). snapshot_df columns:
        segmentId int64 | raw_ratio float64 | clipped_ratio float64
    info carries mode/date/hour, lag source dates, timings, stats.
    """
    own = stack is None
    if own:
        stack = ServingStack()
    t0 = time.perf_counter()
    ts, dstr, hour = resolve_mode(departure_datetime, mode)

    if mode == "replay":
        X = build_replay_frame(stack, dstr, hour)
        sources = {"lag_source": f"stored causal rows of {dstr}",
                   "prev_day": f"observed {dstr}-1d same-hour (stored)",
                   "prev_week": f"observed {dstr}-7d same-hour (stored)",
                   "rolling_3h": f"observed hours {(hour-1)%24},{(hour-2)%24},{(hour-3)%24} (stored)"}
    else:
        last_meta = stack.meta["climatology_definitions"]["clim_dow_hour"]["last_observed_date_per_dow"]
        X = build_forecast_frame(stack, ts, dstr, hour)
        sources = {
            "prev_day": "last observed date 2024-08-30 same-hour density (forecast_base.prev_day_fc)",
            "prev_week": f"last observed same-weekday date {last_meta[str(int(ts.dayofweek))]} "
                         "(forecast_prev_week.prev_week_fc)",
            "rolling_3h": f"terminal clim_seg_hour at hours {(hour-1)%24},{(hour-2)%24},{(hour-3)%24} "
                          "(mod-24 wraparound; terminal state of 2024-08-30)",
            "festival": "0 (all festival dates <= 2024-08-30)",
        }

    errs = validate_features(X, stack)
    if errs:
        raise SnapshotError("feature validation failed: " + "; ".join(errs))

    t_pred = time.perf_counter()
    raw = np.asarray(stack.pipeline.predict(X), dtype=np.float64)
    pred_s = time.perf_counter() - t_pred

    t_map = time.perf_counter()
    clipped = np.clip(raw, R_MIN, R_MAX)
    snap = pd.DataFrame({
        "segmentId": X["segmentId"].to_numpy().astype(np.int64),
        "raw_ratio": raw,
        "clipped_ratio": clipped,
    })
    map_s = time.perf_counter() - t_map
    total_s = time.perf_counter() - t0

    info = {
        "mode": mode, "date": dstr, "hour": hour, "dow": int(ts.dayofweek),
        "n_segments": int(len(snap)), "sources": sources,
        "timing": {"assembly_s": t_pred - t0, "predict_s": pred_s,
                   "mapping_s": map_s, "total_s": total_s},
        "stats": prediction_stats(raw, clipped),
        "model_load_s_own_process": round(stack.load_seconds, 3) if own else None,
    }
    return snap, info


def attach_static_metadata(snapshot: pd.DataFrame, stack: ServingStack) -> pd.DataFrame:
    """Optional helper for R4: join light static attrs (no big tables duplicated)."""
    return snapshot.merge(
        stack.static[["segmentId", "distance", "speedLimit", "frc"]],
        on="segmentId", how="left", validate="one_to_one")
