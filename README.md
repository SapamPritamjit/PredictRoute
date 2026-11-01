# PredictRoute

A congestion-aware routing application for **Delhi/NCR** that takes the time of day into account when finding a route.

> **Status: work in progress.** PredictRoute is a college mini project under active development. Features and structure may change.

## What it does

Standard routing treats roads the same at 3 AM and at 6 PM. PredictRoute factors in expected traffic congestion for the departure time you choose, so the route and ETA reflect rush-hour vs. off-peak conditions.

- **Time-of-day routing**: compare routes for normal daytime, morning rush, evening rush, and night departures
- **Live traffic data**: uses the [TomTom](https://developer.tomtom.com/) API to bring in real-time traffic information
- **Forecast mode**: for dates beyond the available historical data, the mode switches to forecast automatically
- **Destination search**: search for places in Delhi/NCR using [Photon](https://photon.komoot.io/) geocoding
- **Location options**: use your real GPS location or a built-in Delhi demo location (Safdarjung) so you can try the app from anywhere
- **Input validation**: origins and destinations are snapped to the road network, and locations outside Delhi/NCR return a clear error instead of a wrong route
- **Deterministic results**: the same request always returns the same route, distance, and ETA
- **Diagnostics**: each result shows origin type, distance, ETA, snap distances, mean/max congestion, and warnings

## Project structure

```
PredictRoute/
├── backend/                  # FastAPI backend (routing + API)
├── frontend/                 # Web frontend served by the backend
├── main.ipynb                # Notebook for data exploration and modelling
├── FIELD_TEST_CHECKLIST.md   # Manual field-testing scenarios
└── .gitignore
```

## Getting started

### Prerequisites

- Python 3.9 or newer
- A modern browser (Chrome, Edge, or Firefox)

### Installation

```bash
git clone https://github.com/SapamPritamjit/PredictRoute.git
cd PredictRoute
pip install -r requirements.txt   # TODO: add a requirements.txt if not present
```

### Configuration

PredictRoute uses the TomTom API for live traffic data, so you need your own API key:

1. Create a free account at [developer.tomtom.com](https://developer.tomtom.com/) and generate an API key.
2. Provide the key to the backend as an environment variable (TODO: match the variable name used in your code):

   ```bash
   export TOMTOM_API_KEY="your_key_here"      # macOS / Linux
   set TOMTOM_API_KEY=your_key_here           # Windows (cmd)
   ```

> **Never commit your API key to GitHub.** Keep it in an environment variable or a `.env` file that is listed in `.gitignore`.

### Run the app

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Then open <http://localhost:8000/> in your browser.

### Try it quickly

1. Click **Use Delhi Demo Location** to set the origin to Safdarjung, Delhi.
2. Search for a destination such as "India Gate", "Connaught Place", or "AIIMS Delhi".
3. Pick a departure time and compare a rush-hour result with a night result.

## Testing

Manual field-testing scenarios are listed in [FIELD_TEST_CHECKLIST.md](FIELD_TEST_CHECKLIST.md). They cover geolocation, the demo origin, destination search, time-of-day routing, error handling, and diagnostics.

The in-app field-test log:

- is stored locally in your browser (`localStorage`)
- can be exported as JSON
- does **not** record coordinates or user identity

## Limitations

- Coverage is **Delhi/NCR only**
- Routes are limited by the road network snapping radius (400 m)
- Live traffic depends on the TomTom API (requires an API key and is subject to its rate limits and quotas)
- Estimates for other times and future dates rely on historical data and a forecasting approach
- Still a prototype: expect rough edges

## Roadmap

- [ ] Add a `requirements.txt` and setup instructions for all dependencies
- [ ] Document the data source and modelling approach (see `main.ipynb`)
- [ ] Add automated tests for the backend API
- [ ] Improve forecast accuracy and evaluate it against held-out data
- [ ] Extend coverage beyond Delhi/NCR
- [ ] Add screenshots or a demo GIF

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Geocoding:** Photon
- **Live traffic data:** TomTom API
- **Modelling and analysis:** Jupyter Notebook

## Contributing

This is a student project, but feedback and suggestions are welcome. Feel free to open an issue.

## Author

**Sapam Pritamjit**: [GitHub](https://github.com/SapamPritamjit)

