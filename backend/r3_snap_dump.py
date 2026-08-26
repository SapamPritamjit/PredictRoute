"""R3.10 determinism harness: dump one snapshot to parquet.
Usage: python r3_snap_dump.py <datetime> <mode> <out.parquet>"""
import sys
from pathlib import Path


import predict_snapshot as ps

dt, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
stack = ps.ServingStack()
snap, info = ps.predict_snapshot(dt, mode, stack)
snap.to_parquet(out, index=False)
print(f"{dt} [{mode}] -> {out} rows={len(snap)} "
      f"raw_sum={snap['raw_ratio'].sum():.9f} clip_sum={snap['clipped_ratio'].sum():.9f}")
