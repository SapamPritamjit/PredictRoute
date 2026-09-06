# PredictRoute v1 — Field-Testing Checklist

Delhi/NCR only. All locked constants unchanged.

## Prerequisites

- Application running: `python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000`
- Open `http://localhost:8000/` in a modern browser (Chrome/Edge/Firefox)
- Tester physically located in Delhi/NCR for real-GPS tests

## Test Scenarios

### A. Browser Geolocation

| # | Scenario | Expected Result | Pass/Fail |
|---|----------|----------------|-----------|
| A1 | Click "Use My Location" — permission granted | Origin shows `[REAL]` tag; coordinates match physical location | |
| A2 | Click "Use My Location" — permission denied | Error message: "Location permission denied…" | |
| A3 | Click "Use My Location" — device offline/unavailable | Error: "Location information is unavailable." | |
| A4 | Click "Use My Location" — slow GPS (>10 s) | Error: "Location request timed out." | |

### B. Delhi Demo Origin

| # | Scenario | Expected Result | Pass/Fail |
|---|----------|----------------|-----------|
| B1 | Click "Use Delhi Demo Location" | Origin shows "Safdarjung, Delhi (demo) [DEMO]"; info status appears | |
| B2 | Verify demo origin works with India Gate destination | Route found; origin snap < 100 m; destination snap < 200 m | |
| B3 | Verify demo origin works with Connaught Place destination | Route found; both snaps < 400 m | |
| B4 | Verify demo origin works with AIIMS Delhi destination | Route found; both snaps < 400 m | |

### C. Destination Search (Photon)

| # | Scenario | Expected Result | Pass/Fail |
|---|----------|----------------|-----------|
| C1 | Search "India Gate" | Multiple results displayed with name + address | |
| C2 | Search "Connaught Place" | Multiple results displayed | |
| C3 | Search "AIIMS Delhi" | Results include AIIMS | |
| C4 | Select a result | Destination shows selected name + coordinates | |
| C5 | Search for nonsense string (e.g. "xyzabc123") | "No results found" message | |
| C6 | Clear search input | Results list disappears | |

### D. Routing — Time-of-Day

Use **Delhi Demo Location** as origin for all routing tests.

| # | Scenario | Departure | Expected Result | Pass/Fail |
|---|----------|-----------|----------------|-----------|
| D1 | Normal daytime | 2024-08-27 10:00 | Route found; ETA ~5–15 min | |
| D2 | Morning rush | 2024-08-27 08:00 | Route found; higher mean congestion ratio than D1 | |
| D3 | Evening rush | 2024-08-27 18:00 | Route found; highest mean congestion ratio | |
| D4 | Night | 2024-08-27 23:00 | Route found; lowest mean congestion ratio | |
| D5 | Forecast date | 2024-09-01 14:00 | Route found; mode auto-resolves to forecast | |

### E. Error Handling

| # | Scenario | Expected Result | Pass/Fail |
|---|----------|----------------|-----------|
| E1 | Origin outside Delhi (real GPS from outside Delhi) | "Use My Location" shows snap-radius error mentioning "outside Delhi routing coverage"; suggestion to use demo location | |
| E2 | Destination outside Delhi (search "Imphal, Manipur") | Backend returns 400; frontend shows user-friendly error | |
| E3 | Both origin and destination valid but unreachable | "No route could be found" error (not a snap error) | |
| E4 | Repeated same request | Same distance, ETA, snap distances — deterministic | |

### F. Diagnostic Visibility

| # | Scenario | Expected Result | Pass/Fail |
|---|----------|----------------|-----------|
| F1 | After successful route | Result card shows: Origin type, Distance, ETA, Origin snap, Destination snap, Mean/Max congestion, Warnings | |
| F2 | After failed route | Error message is human-readable; no raw JSON or node IDs shown | |

### G. Field-Test Log

| # | Scenario | Expected Result | Pass/Fail |
|---|----------|----------------|-----------|
| G1 | Complete a route (success or failure) | Entry appears in localStorage key `predictroute_field_test_log` | |
| G2 | Click "Export test log" | JSON file downloads with all entries | |
| G3 | Click "Clear log" | Confirm dialog; log emptied | |
| G4 | Verify no PII in log | Entries contain: timestamp, originType, destLabel, success, snap distances, distance, ETA, warnings, error — no coordinates, no user identity | |

## Notes

- Do NOT change the 400 m snap radius during testing
- Record any Delhi location that fails the 400 m snap check — this is field data for future radius evaluation
- All route results are deterministic — if a repeated request gives different results, file a bug
- Real-GPS tests require physical presence in Delhi; demo-origin tests can be done from anywhere
