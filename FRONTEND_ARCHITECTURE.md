# SkyGuard AI — Architecture Blueprint

Shared reference so backend/ML and frontend can be built **in parallel**.
Frontend does not need to wait for the real backend — every endpoint
below has an example response your teammates can hardcode as mock data
in Lovable *today*, then swap for the real URL later with zero UI changes.

---

## 1. System Overview

```
┌─────────────────────┐      HTTPS (JSON)      ┌──────────────────────┐
│   Frontend (Lovable) │ ─────────────────────► │   Backend (FastAPI)  │
│   React dashboard    │ ◄───────────────────── │   /api/*  endpoints  │
└─────────────────────┘      polling every      └───────────┬──────────┘
                              3–5 sec                        │
                                                              ▼
                                                  ┌──────────────────────┐
                                                  │  Model layer          │
                                                  │  - Isolation Forest   │
                                                  │  - Rule checks        │
                                                  │  - SHAP explainer     │
                                                  └───────────┬──────────┘
                                                              │
                                                              ▼
                                                  ┌──────────────────────┐
                                                  │  Data layer            │
                                                  │  - Open-Meteo history  │
                                                  │  - Synthetic injector  │
                                                  └──────────────────────┘
```

**Why polling, not WebSockets:** a plain `GET` every few seconds is far
easier for a Lovable-generated frontend to get right on the first try
than managing a WebSocket connection, reconnect logic, etc. It's
"real-time enough" for a dashboard refreshing every few seconds, and it's
what the judges will see as "Live" regardless. We can upgrade to
WebSockets later if there's time — not a v1 requirement.

---

## 2. Repo folder structure

```
skyguard-ai/
├── backend/
│   ├── main.py              # FastAPI app, all routes
│   ├── model/
│   │   ├── train.py          # trains Isolation Forest on real data
│   │   ├── detect.py         # runs detection on new readings
│   │   └── explain.py        # SHAP wiring
│   ├── data/
│   │   ├── data_fetch.py     # Open-Meteo pull (already built)
│   │   └── anomaly_injector.py
│   └── requirements.txt
├── frontend/                 # Lovable-exported React app lives here
├── docs/
│   └── ARCHITECTURE.md       # this file
└── README.md
```

---

## 3. Shared vocabulary (both sides must use these exact strings)

**Severity levels:** `"low"` | `"medium"` | `"high"` | `"critical"`
(maps to green / yellow / orange / red — same as your reference image)

**Sensor health status:** `"HEALTHY"` | `"WARNING"` | `"CRITICAL"` | `"OFFLINE"`

**System status:** `"NORMAL"` | `"WARNING"` | `"CRITICAL"` | `"OFFLINE"`
(used for the map dots)

**Anomaly types:** `"spike"` | `"frozen_value"` | `"drift"` | `"dropout"` | `"multivariate_inconsistency"`

---

## 4. API Contract

### `GET /api/stations`
Powers: station dropdown (sidebar).

```json
[
  { "station_id": "AWS-CHN-024", "name": "Chennai", "lat": 13.0827, "lon": 80.2707, "status": "CRITICAL" },
  { "station_id": "AWS-DEL-011", "name": "Delhi", "lat": 28.6139, "lon": 77.2090, "status": "NORMAL" },
  { "station_id": "AWS-MUM-007", "name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "status": "WARNING" }
]
```

---

### `GET /api/current-reading?station_id=AWS-CHN-024`
Powers: the 5 top metric cards (Temperature, Pressure, Humidity, Anomaly Score, Sensor Health).

```json
{
  "station_id": "AWS-CHN-024",
  "timestamp": "2026-09-06T10:24:35Z",
  "temperature_c": { "value": 28.6, "normal_min": 22.0, "normal_max": 32.0 },
  "pressure_hpa": { "value": 1008.7, "normal_min": 995, "normal_max": 1015 },
  "humidity_pct": { "value": 68.3, "normal_min": 30, "normal_max": 90 },
  "anomaly_score_pct": 12,
  "risk_level": "low",
  "sensor_health_pct": 98,
  "sensor_health_status": "HEALTHY"
}
```

Frontend polls this every 3–5 sec for the "Live" cards.

---

### `GET /api/trends?station_id=AWS-CHN-024&hours=6`
Powers: the "Real-time Sensor Trends" line chart.

```json
{
  "station_id": "AWS-CHN-024",
  "points": [
    { "timestamp": "2026-09-06T04:00:00Z", "temperature_c": 21.5, "pressure_hpa": 1010.2, "humidity_pct": 55.0 },
    { "timestamp": "2026-09-06T04:05:00Z", "temperature_c": 21.7, "pressure_hpa": 1010.0, "humidity_pct": 54.8 }
  ],
  "anomaly_windows": [
    { "start": "2026-09-06T07:15:00Z", "end": "2026-09-06T07:45:00Z", "label": "Anomaly Detected" }
  ]
}
```

Frontend: use Recharts `LineChart` with 3 lines (temp/pressure/humidity,
different Y-axes as in the reference image) and shade `anomaly_windows`
as a highlighted region — exactly the red band in your screenshot.

---

### `GET /api/anomalies/latest?station_id=AWS-CHN-024`
Powers: the "Latest Anomaly Alert" panel.

```json
{
  "anomaly_id": "anom_00231",
  "timestamp": "2026-09-06T10:21:43Z",
  "station_id": "AWS-CHN-024",
  "anomaly_score_pct": 94,
  "severity": "critical",
  "type": "multivariate_inconsistency",
  "root_cause": "Temperature sensor malfunction",
  "description": "Sudden increase in temperature value is not consistent with pressure and humidity trends."
}
```

---

### `GET /api/anomalies/recent?station_id=AWS-CHN-024&limit=5`
Powers: "Recent Anomalies" list.

```json
[
  { "anomaly_id": "anom_00231", "type": "spike", "label": "Temperature Spike", "station_id": "AWS-CHN-024", "timestamp": "2026-09-06T10:21:00Z", "score_pct": 94, "severity": "critical" },
  { "anomaly_id": "anom_00229", "type": "drift", "label": "Pressure Drop", "station_id": "AWS-DEL-011", "timestamp": "2026-09-06T09:47:00Z", "score_pct": 71, "severity": "high" },
  { "anomaly_id": "anom_00227", "type": "drift", "label": "Humidity Sensor Drift", "station_id": "AWS-MUM-007", "timestamp": "2026-09-06T08:15:00Z", "score_pct": 58, "severity": "medium" }
]
```

---

### `GET /api/explain/{anomaly_id}`
Powers: the SHAP "Explainability" bar chart.

```json
{
  "anomaly_id": "anom_00231",
  "features": [
    { "name": "Temp Deviation", "impact": 0.68 },
    { "name": "Pressure Inconsistency", "impact": 0.32 },
    { "name": "Humidity Deviation", "impact": 0.18 },
    { "name": "Rate of Change (Temp)", "impact": -0.12 },
    { "name": "Time of Day Pattern", "impact": -0.08 },
    { "name": "Seasonal Pattern", "impact": -0.05 }
  ]
}
```

Frontend: horizontal bar chart, positive impact = red bar to the right,
negative = blue bar to the left — matches the reference image exactly.

---

### `GET /api/sensor-health?station_id=AWS-CHN-024`
Powers: Sensor Health card + Station Network dot colors.

```json
{ "station_id": "AWS-CHN-024", "health_pct": 98, "status": "HEALTHY" }
```

---

### `POST /api/inject-anomaly`
Powers: "Inject Anomaly (Test)" button.

Request:
```json
{ "station_id": "AWS-CHN-024", "type": "spike" }
```
Response:
```json
{ "success": true, "anomaly_id": "anom_00232", "message": "Anomaly injected and detected in next reading cycle." }
```

---

### `POST /api/maintenance-ticket`
Powers: "Generate Maintenance Ticket" button.

Request:
```json
{ "anomaly_id": "anom_00231" }
```
Response:
```json
{ "ticket_id": "TCK-1042", "station_id": "AWS-CHN-024", "issue": "Temperature sensor malfunction", "priority": "high", "created_at": "2026-09-06T10:25:00Z" }
```

---

## 5. Build order for the frontend team (independent of backend)

1. Hardcode the JSON examples above as local mock objects in Lovable.
2. Build every panel against the mocks: top 5 cards → trend chart →
   latest alert panel → recent anomalies list → SHAP bar chart → station
   map → buttons.
3. Once backend is live, replace mock objects with real `fetch()` calls
   to the same shapes — no component logic should need to change if the
   contract is followed exactly.
4. Use a single `const API_BASE = "http://localhost:8000"` (later the
   deployed URL) so switching environments is a one-line change.

## 6. Build order for backend/ML (what I'm doing)

1. ✅ `data_fetch.py` — real historical data
2. `anomaly_injector.py` — synthetic fault injection (waiting on Day 41–43)
3. `train.py` — Isolation Forest training
4. `detect.py` + `explain.py` — live detection + SHAP
5. `main.py` — FastAPI wrapping all of the above into the exact endpoints above
