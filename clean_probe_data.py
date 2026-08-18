import json
import os
import sys
import pandas as pd

SRC_DIR = "new_delhi_traffic_dataset/probe_counts/geojson"
OUT_DIR = "cleaned"
COLS = ["date", "segmentId", "streetName", "frc", "speedLimit", "distance",
        "timeSet", "hour", "probeCount"]

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".geojson"))
print(f"{len(files)} files found")

for fname in files:
    date = fname.split("__")[1].split("_to_")[0]
    path = os.path.join(SRC_DIR, fname)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for feature in data["features"]:
        if feature.get("geometry") is None:
            continue
        props = feature["properties"]
        for ticket in props["segmentProbeCounts"]:
            rows.append([date, props["segmentId"], props.get("streetName", ""),
                         props["frc"], props["speedLimit"], props["distance"],
                         ticket["timeSet"], ticket["timeSet"] - 2,
                         ticket["probeCount"]])

    out = os.path.join(OUT_DIR, f"{date}.csv")
    pd.DataFrame(rows, columns=COLS).to_csv(out, index=False)
    print(f"{date}: {len(rows)} rows -> {out}")

df = pd.concat([pd.read_csv(os.path.join(OUT_DIR, f"{f.split('__')[1].split('_to_')[0]}.csv"))
                for f in files], ignore_index=True)
df.to_parquet(os.path.join(OUT_DIR, "all_days.parquet"))
print(f"combined: {df.shape} -> cleaned/all_days.parquet")

