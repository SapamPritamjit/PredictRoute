"""End-to-end validation of the FastAPI /route endpoint.

Runs the real uvicorn server in a background thread, makes actual HTTP
requests, records all results.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import requests
import uvicorn

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
BASE = "http://127.0.0.1:8765"

_results: list[dict] = []
_server_thread: threading.Thread | None = None
_server_ready = threading.Event()


def record(name: str, passed: bool, detail: str = "", status_code: int = None,
           request_payload: dict = None):
    entry = {"name": name, "passed": passed, "status_code": status_code, "detail": detail}
    if request_payload:
        entry["request"] = {k: v for k, v in request_payload.items()}
    _results.append(entry)
    tag = "PASS" if passed else "FAIL"
    print(f"\n  [{tag}] {name}")
    if status_code is not None:
        print(f"         HTTP {status_code}")
    if detail:
        for line in detail.split("\n")[:8]:
            print(f"         {line}")


def _post(payload: dict) -> tuple[int, dict]:
    try:
        r = requests.post(f"{BASE}/route", json=payload, timeout=180)
        return r.status_code, r.json()
    except requests.ConnectionError:
        return 0, {"detail": "Connection refused"}
    except Exception as e:
        return 0, {"detail": f"Request error: {e}"}


def _server_alive() -> bool:
    try:
        r = requests.get(f"{BASE}/openapi.json", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Server lifecycle ──────────────────────────────────────────
def _run_server():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    config = uvicorn.Config(
        "backend.api:app",
        host="127.0.0.1",
        port=8765,
        log_level="error",
    )
    server = uvicorn.Server(config)
    server.run()


def start_server():
    global _server_thread
    print("Starting FastAPI server on :8765 ...")
    _server_thread = threading.Thread(target=_run_server, daemon=True)
    _server_thread.start()
    for i in range(90):
        time.sleep(1)
        if _server_alive():
            print(f"  Server ready after {i+1}s")
            return True
    print("  ERROR: Server did not start in 90s")
    return False


# ── DELHI TEST COORDINATES ───────────────────────────────────
CONNAUGHT_PLACE = {"lat": 28.6315, "lon": 77.2167}
INDIA_GATE      = {"lat": 28.6129, "lon": 77.2295}
CHANDNI_CHOWK  = {"lat": 28.6506, "lon": 77.2303}
KAROL_BAGH     = {"lat": 28.6514, "lon": 77.1908}
SAFDARJUNG     = {"lat": 28.5687, "lon": 77.2062}


# ═══════════════════════════════════════════════════════════════
# SCENARIO 1: Normal daytime route (replay)
# ═══════════════════════════════════════════════════════════════
def test_01_normal_daytime():
    name = "1. Normal daytime route (replay, 10:00)"
    payload = {
        "origin_lat": CONNAUGHT_PLACE["lat"], "origin_lon": CONNAUGHT_PLACE["lon"],
        "dest_lat": INDIA_GATE["lat"], "dest_lon": INDIA_GATE["lon"],
        "departure_datetime": "2024-08-27 10:00", "mode": "replay",
    }
    code, body = _post(payload)
    ok = (code == 200 and body.get("status") == "success"
          and body.get("distance_km", 0) > 0
          and body.get("eta_minutes", 0) > 0)
    detail = (f"distance={body.get('distance_km',0):.3f}km | "
              f"eta={body.get('eta_minutes',0):.1f}min | "
              f"mean_ratio={body.get('mean_ratio',0):.4f} | "
              f"edges={len(body.get('route', []))}")
    record(name, ok, detail, code, payload)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 2: Morning rush (08:00)
# ═══════════════════════════════════════════════════════════════
def test_02_morning_rush():
    name = "2. Morning rush (08:00)"
    payload = {
        "origin_lat": SAFDARJUNG["lat"], "origin_lon": SAFDARJUNG["lon"],
        "dest_lat": CONNAUGHT_PLACE["lat"], "dest_lon": CONNAUGHT_PLACE["lon"],
        "departure_datetime": "2024-08-27 08:00", "mode": "replay",
    }
    code, body = _post(payload)
    ok = (code == 200 and body.get("status") == "success"
          and body.get("distance_km", 0) > 0)
    if code == 200:
        detail = (f"distance={body.get('distance_km',0):.3f}km | "
                  f"eta={body.get('eta_minutes',0):.1f}min | "
                  f"mean_ratio={body.get('mean_ratio',0):.4f} | "
                  f"max_ratio={body.get('max_ratio',0):.4f} | "
                  f"synthetic={body.get('synthetic_reverse_edges')}")
    else:
        detail = f"error={body.get('detail', body)}"
    record(name, ok, detail, code, payload)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 3: Evening rush (18:00)
# ═══════════════════════════════════════════════════════════════
def test_03_evening_rush():
    name = "3. Evening rush (18:00)"
    payload = {
        "origin_lat": INDIA_GATE["lat"], "origin_lon": INDIA_GATE["lon"],
        "dest_lat": SAFDARJUNG["lat"], "dest_lon": SAFDARJUNG["lon"],
        "departure_datetime": "2024-08-28 18:00", "mode": "replay",
    }
    code, body = _post(payload)
    ok = (code == 200 and body.get("status") == "success"
          and body.get("distance_km", 0) > 0)
    if code == 200:
        detail = (f"distance={body.get('distance_km',0):.3f}km | "
                  f"eta={body.get('eta_minutes',0):.1f}min | "
                  f"mean_ratio={body.get('mean_ratio',0):.4f} | "
                  f"max_ratio={body.get('max_ratio',0):.4f}")
    else:
        detail = f"error={body.get('detail', body)}"
    record(name, ok, detail, code, payload)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 4: Night (23:00)
# ═══════════════════════════════════════════════════════════════
def test_04_night():
    name = "4. Night (23:00)"
    payload = {
        "origin_lat": SAFDARJUNG["lat"], "origin_lon": SAFDARJUNG["lon"],
        "dest_lat": CONNAUGHT_PLACE["lat"], "dest_lon": CONNAUGHT_PLACE["lon"],
        "departure_datetime": "2024-08-29 23:00", "mode": "replay",
    }
    code, body = _post(payload)
    ok = (code == 200 and body.get("status") == "success"
          and body.get("distance_km", 0) > 0)
    detail = (f"distance={body.get('distance_km',0):.3f}km | "
              f"eta={body.get('eta_minutes',0):.1f}min | "
              f"mean_ratio={body.get('mean_ratio',0):.4f}")
    record(name, ok, detail, code, payload)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 5: Forecast mode (2024-09-01)
# ═══════════════════════════════════════════════════════════════
def test_05_forecast():
    name = "5. Forecast mode (2024-09-01 14:00)"
    payload = {
        "origin_lat": CONNAUGHT_PLACE["lat"], "origin_lon": CONNAUGHT_PLACE["lon"],
        "dest_lat": SAFDARJUNG["lat"], "dest_lon": SAFDARJUNG["lon"],
        "departure_datetime": "2024-09-01 14:00", "mode": "forecast",
    }
    code, body = _post(payload)
    ok = (code == 200 and body.get("status") == "success"
          and body.get("distance_km", 0) > 0)
    detail = (f"distance={body.get('distance_km',0):.3f}km | "
              f"eta={body.get('eta_minutes',0):.1f}min | "
              f"mean_ratio={body.get('mean_ratio',0):.4f}")
    record(name, ok, detail, code, payload)

    # 5b. Auto-resolved forecast (no mode specified)
    payload2 = {
        "origin_lat": CONNAUGHT_PLACE["lat"], "origin_lon": CONNAUGHT_PLACE["lon"],
        "dest_lat": SAFDARJUNG["lat"], "dest_lon": SAFDARJUNG["lon"],
        "departure_datetime": "2024-09-01 14:00",
    }
    code2, body2 = _post(payload2)
    ok2 = (code2 == 200 and body2.get("status") == "success")
    record("5b. Forecast auto-resolve (no mode)", ok2,
           f"status={body2.get('status')} distance={body2.get('distance_km',0):.3f}km",
           code2, payload2)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 6: Invalid coordinates
# ═══════════════════════════════════════════════════════════════
def test_06_invalid_coords():
    base = {
        "origin_lat": CONNAUGHT_PLACE["lat"], "origin_lon": CONNAUGHT_PLACE["lon"],
        "dest_lat": INDIA_GATE["lat"], "dest_lon": INDIA_GATE["lon"],
        "departure_datetime": "2024-08-27 10:00", "mode": "replay",
    }
    cases = [
        ("6a. Lat > 90",        {**base, "origin_lat": 91.0}),
        ("6b. Lat < -90",       {**base, "origin_lat": -91.0}),
        ("6c. Lon > 180",       {**base, "origin_lon": 181.0}),
        ("6d. Lon < -180",      {**base, "origin_lon": -181.0}),
        ("6e. Dest lat > 90",   {**base, "dest_lat": 95.0}),
        ("6f. Dest lon < -180", {**base, "dest_lon": -200.0}),
    ]
    for label, payload in cases:
        code, body = _post(payload)
        ok = code == 422
        record(label, ok, f"detail={body.get('detail', '')[:80]}", code, payload)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 7: Snap-radius violation
# ═══════════════════════════════════════════════════════════════
def test_07_snap_violation():
    payload = {
        "origin_lat": 51.5074, "origin_lon": -0.1278,
        "dest_lat": CONNAUGHT_PLACE["lat"], "dest_lon": CONNAUGHT_PLACE["lon"],
        "departure_datetime": "2024-08-27 10:00", "mode": "replay",
    }
    code, body = _post(payload)
    ok = code == 400 and "snap radius" in body.get("detail", "").lower()
    record("7a. Origin snap-radius (London)", ok,
           f"detail={body.get('detail', '')[:100]}", code, payload)

    payload2 = {
        "origin_lat": CONNAUGHT_PLACE["lat"], "origin_lon": CONNAUGHT_PLACE["lon"],
        "dest_lat": 51.5074, "dest_lon": -0.1278,
        "departure_datetime": "2024-08-27 10:00", "mode": "replay",
    }
    code2, body2 = _post(payload2)
    ok2 = code2 == 400 and "snap radius" in body2.get("detail", "").lower()
    record("7b. Dest snap-radius (London)", ok2,
           f"detail={body2.get('detail', '')[:100]}", code2, payload2)

    payload3 = {
        "origin_lat": 0.0, "origin_lon": 70.0,
        "dest_lat": CONNAUGHT_PLACE["lat"], "dest_lon": CONNAUGHT_PLACE["lon"],
        "departure_datetime": "2024-08-27 10:00", "mode": "replay",
    }
    code3, body3 = _post(payload3)
    ok3 = code3 == 400
    record("7c. Origin snap-radius (ocean)", ok3,
           f"detail={body3.get('detail', '')[:100]}", code3, payload3)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 8: Invalid datetime / mode
# ═══════════════════════════════════════════════════════════════
def test_08_invalid_datetime():
    base = {
        "origin_lat": CONNAUGHT_PLACE["lat"], "origin_lon": CONNAUGHT_PLACE["lon"],
        "dest_lat": INDIA_GATE["lat"], "dest_lon": INDIA_GATE["lon"],
    }
    cases = [
        ("8a. Garbage datetime",
         {**base, "departure_datetime": "not-a-date", "mode": "replay"}),
        ("8b. Timezone-aware datetime",
         {**base, "departure_datetime": "2024-08-27 10:00+05:30", "mode": "replay"}),
        ("8c. Non-hourly datetime",
         {**base, "departure_datetime": "2024-08-27 10:30", "mode": "replay"}),
        ("8d. Date before replay start",
         {**base, "departure_datetime": "2024-07-01 10:00", "mode": "replay"}),
        ("8e. Invalid mode string",
         {**base, "departure_datetime": "2024-08-27 10:00", "mode": "walking"}),
        ("8f. Replay on forecast date",
         {**base, "departure_datetime": "2024-09-05 10:00", "mode": "replay"}),
        ("8g. Forecast on replay date",
         {**base, "departure_datetime": "2024-08-27 10:00", "mode": "forecast"}),
    ]
    for label, payload in cases:
        code, body = _post(payload)
        ok = code in (400, 422)
        record(label, ok, f"detail={body.get('detail', '')[:100]}", code, payload)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 9: Response contract
# ═══════════════════════════════════════════════════════════════
def test_09_response_contract():
    payload = {
        "origin_lat": CONNAUGHT_PLACE["lat"], "origin_lon": CONNAUGHT_PLACE["lon"],
        "dest_lat": INDIA_GATE["lat"], "dest_lon": INDIA_GATE["lon"],
        "departure_datetime": "2024-08-27 10:00", "mode": "replay",
    }
    code, body = _post(payload)
    required = [
        "status", "origin", "destination",
        "distance_km", "eta_minutes", "search_time_seconds",
        "mean_ratio", "max_ratio", "synthetic_reverse_edges",
        "origin_snap_distance_m", "destination_snap_distance_m",
        "warnings", "route",
    ]
    missing = [f for f in required if f not in body]
    ok = code == 200 and len(missing) == 0

    sub_checks = []
    if not missing:
        for key in ("latitude", "longitude", "node_id"):
            if key not in body["origin"]:
                sub_checks.append(f"origin.{key}")
            if key not in body["destination"]:
                sub_checks.append(f"destination.{key}")
        if not isinstance(body["route"], list):
            sub_checks.append("route is not list")
        if body["route"]:
            edge = body["route"][0]
            for ek in ("segmentId", "edge_type", "distance_m",
                       "speedLimit_kmh", "ratio", "weight_seconds"):
                if ek not in edge:
                    sub_checks.append(f"route[0].{ek}")

    ok = ok and len(sub_checks) == 0
    detail = ("all fields present" if not missing and not sub_checks
              else f"missing={missing}" if missing
              else f"sub_issues={sub_checks}")
    record("9. Response contract", ok, detail, code, payload)


# ═══════════════════════════════════════════════════════════════
# SCENARIO 10: Determinism
# ═══════════════════════════════════════════════════════════════
def test_10_determinism():
    payload = {
        "origin_lat": INDIA_GATE["lat"], "origin_lon": INDIA_GATE["lon"],
        "dest_lat": CONNAUGHT_PLACE["lat"], "dest_lon": CONNAUGHT_PLACE["lon"],
        "departure_datetime": "2024-08-27 08:00", "mode": "replay",
    }
    runs = []
    for _ in range(3):
        code, body = _post(payload)
        if code == 200:
            runs.append(body)

    if len(runs) < 3:
        record("10. Determinism (3 runs)", False,
               f"Only {len(runs)}/3 succeeded")
        return

    fields = ["distance_km", "eta_minutes", "search_time_seconds",
              "mean_ratio", "max_ratio", "synthetic_reverse_edges"]
    diffs = []
    for f in fields:
        vals = [r[f] for r in runs]
        if not all(abs(v - vals[0]) < 1e-10 for v in vals):
            diffs.append(f"{f}: {vals}")

    seg_ids = [[e["segmentId"] for e in r["route"]] for r in runs]
    if not all(s == seg_ids[0] for s in seg_ids):
        diffs.append("route segment IDs differ")

    ok = len(diffs) == 0
    detail = "all 3 runs identical" if ok else "DIFFS: " + "; ".join(diffs)
    record("10. Determinism (3 runs)", ok, detail, code, payload)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  PredictRoute API — End-to-End Validation")
    print("=" * 70)

    if not start_server():
        return 1

    try:
        print("\n--- Running test scenarios ---")
        test_01_normal_daytime()
        test_02_morning_rush()
        test_03_evening_rush()
        test_04_night()
        test_05_forecast()
        test_06_invalid_coords()
        test_07_snap_violation()
        test_08_invalid_datetime()
        test_09_response_contract()
        test_10_determinism()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for r in _results if r["passed"])
    n_fail = sum(1 for r in _results if not r["passed"])
    for r in _results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{tag}] {r['name']}")
    print(f"\n  Total: {len(_results)}  |  Pass: {n_pass}  |  Fail: {n_fail}")
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    record_path = Path(__file__).resolve().parent / "e2e_results.json"
    with open(record_path, "w") as f:
        json.dump(_results, f, indent=2)
    print(f"\n  Full results written to {record_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
