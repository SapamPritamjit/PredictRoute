"""R3.10 determinism verdict: EXACT comparison of snapshot dumps from
separate fresh processes (replay + forecast)."""
import sys

import numpy as np
import pandas as pd

PAIRS = [
    (r"C:\Users\prita\AppData\Local\Temp\opencode\r3_det_rp_a.parquet",
     r"C:\Users\prita\AppData\Local\Temp\opencode\r3_det_rp_b.parquet", "replay 2024-08-27 08:00"),
    (r"C:\Users\prita\AppData\Local\Temp\opencode\r3_det_fc_a.parquet",
     r"C:\Users\prita\AppData\Local\Temp\opencode\r3_det_fc_b.parquet", "forecast 2024-08-31 08:00"),
]
fails = []
for pa, pb, label in PAIRS:
    a = pd.read_parquet(pa)
    b = pd.read_parquet(pb)
    ok = (a.shape == b.shape
          and list(a.columns) == list(b.columns)
          and all(str(a[c].dtype) == str(b[c].dtype) for c in a.columns)
          and all(np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in a.columns))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: shapes {a.shape}=={b.shape}, "
          f"dtypes equal, EXACT bit-equality all columns")
    if not ok:
        fails.append(label)

if fails:
    print("FAILURES:", fails)
    sys.exit(1)
print("\nR3.10 DETERMINISM PASSED - snapshots identical across fresh processes")
