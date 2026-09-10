"""
SkyGuard AI — main.py: FastAPI app, all route handlers.

Lives at repo root alongside config.py/data_fetch.py (see the actual
directory structure -- model/, data/, model_artifacts/ are subpackages;
this file and config.py are the two root-level pieces that tie them
together). Wires simulator.py's SimulatorState into the exact 9
endpoints the frontend is already built against
(DEVELOPMENT_PROGRESS.md's "Approved Contract Endpoints" list) -- no
endpoint here was invented; every route matches FRONTEND_ARCHITECTURE.md
exactly, including reusing POST /api/inject-anomaly as the
replay-mode trigger (see simulator.py's start_replay() docstring for
why, and the earlier confirmation flag on that design choice).
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio

sys.path.append(str(Path(__file__).parent))
from model.simulator import create_simulator_state, run_simulation_loop
from config import score_to_severity


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sim = create_simulator_state()
    app.state.sim_task = asyncio.create_task(run_simulation_loop(app.state.sim))
    yield
    app.state.sim_task.cancel()


app = FastAPI(title="SkyGuard AI", lifespan=lifespan)

# allow_origins=["*"] -- fine for hackathon per BACKEND_BLUEPRINT.md
# section 6; tighten to the deployed frontend URL once known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_sim(request_app: FastAPI = None):
    return app.state.sim


# ---------------- GET /api/stations ----------------

@app.get("/api/stations")
def get_stations():
    sim = app.state.sim
    result = []
    for _, row in sim.metadata.iterrows():
        sid = row["station_id"]
        status = sim.manager.get_station_status(sid)["status"]
        # Contract wants NORMAL/WARNING/CRITICAL/OFFLINE, not health's
        # own HEALTHY/WARNING/OFFLINE vocabulary -- translate.
        mapped_status = {"HEALTHY": "NORMAL", "WARNING": "WARNING", "OFFLINE": "OFFLINE"}.get(status, "NORMAL")
        result.append({
            "station_id": sid,
            "name": row["name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "status": mapped_status,
        })
    return result


# ---------------- GET /api/current-reading ----------------

@app.get("/api/current-reading")
def get_current_reading(station_id: str):
    sim = app.state.sim
    entry = sim.latest.get(station_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No reading yet for {station_id}")

    raw = entry["raw_reading"]
    verdict = entry["verdict"]

    # normal_min/max: static fallback ranges since config.py's exact
    # constant names aren't in view here -- if config.py already
    # defines per-parameter normal ranges, swap these literals for
    # that import instead of duplicating the values.
    return {
        "station_id": station_id,
        "timestamp": entry["timestamp"].isoformat(),
        "temperature_c": {"value": raw.get("temperature_c"), "normal_min": 10.0, "normal_max": 45.0},
        "pressure_hpa": {"value": raw.get("pressure_hpa"), "normal_min": 950.0, "normal_max": 1050.0},
        "humidity_pct": {"value": raw.get("humidity_pct"), "normal_min": 10.0, "normal_max": 100.0},
        "anomaly_score_pct": verdict["anomaly_score_pct"],
        "risk_level": verdict["severity"],
        "sensor_health_pct": 100 if not verdict["is_anomaly"] else max(0, 100 - int(verdict["anomaly_score_pct"])),
        "sensor_health_status": sim.manager.get_station_status(station_id)["status"],
    }


# ---------------- GET /api/trends ----------------

@app.get("/api/trends")
def get_trends(station_id: str, hours: int = 6):
    sim = app.state.sim
    history = sim.trend_history.get(station_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Unknown station {station_id}")

    points = list(history)
    # Points-per-hour is mode-dependent (live: 1/min via
    # LIVE_FETCH_INTERVAL_SECONDS refresh cadence; replay: 1 every
    # TICK_SECONDS=2s), so slicing by a fixed row-count-per-hour
    # assumption would be wrong in one mode or the other. Simple,
    # honest approach: just return everything currently buffered,
    # capped by TREND_HISTORY_MAXLEN in simulator.py -- frontend's
    # chart already handles arbitrary point counts.
    trend_points = [
        {
            "timestamp": p["timestamp"].isoformat(),
            "temperature_c": p["temperature_c"],
            "pressure_hpa": p["pressure_hpa"],
            "humidity_pct": p["humidity_pct"],
        }
        for p in points
    ]

    anomaly_windows = []
    in_window = False
    for p in points:
        if p["is_anomaly"] and not in_window:
            window_start = p["timestamp"]
            in_window = True
        elif not p["is_anomaly"] and in_window:
            anomaly_windows.append({
                "start": window_start.isoformat(),
                "end": p["timestamp"].isoformat(),
                "label": "Anomaly Detected",
            })
            in_window = False
    if in_window:
        anomaly_windows.append({
            "start": window_start.isoformat(),
            "end": points[-1]["timestamp"].isoformat(),
            "label": "Anomaly Detected",
        })

    return {"station_id": station_id, "points": trend_points, "anomaly_windows": anomaly_windows}


# ---------------- GET /api/anomalies/latest ----------------

@app.get("/api/anomalies/latest")
def get_latest_anomaly(station_id: str):
    sim = app.state.sim
    for a in sim.recent_anomalies:
        if a["station_id"] == station_id:
            return {
                "anomaly_id": a["anomaly_id"],
                "timestamp": a["timestamp"].isoformat(),
                "station_id": a["station_id"],
                "anomaly_score_pct": a["anomaly_score_pct"],
                "severity": a["severity"],
                "type": a["type"],
                "root_cause": a["root_cause"],
                "description": f"{a['root_cause']} detected at {a['station_id']}.",
                "suggested_values": a.get("suggested_values"),
            }
    raise HTTPException(status_code=404, detail=f"No anomalies recorded for {station_id}")


# ---------------- GET /api/anomalies/recent ----------------

@app.get("/api/anomalies/recent")
def get_recent_anomalies(station_id: str, limit: int = 5):
    sim = app.state.sim
    matches = [a for a in sim.recent_anomalies if a["station_id"] == station_id][:limit]
    return [
        {
            "anomaly_id": a["anomaly_id"],
            "type": a["type"],
            "label": a["root_cause"],
            "station_id": a["station_id"],
            "timestamp": a["timestamp"].isoformat(),
            "score_pct": a["anomaly_score_pct"],
            "severity": a["severity"],
            "suggested_values": a.get("suggested_values"),
        }
        for a in matches
    ]


# ---------------- GET /api/explain/{anomaly_id} ----------------

@app.get("/api/explain/{anomaly_id}")
def get_explanation(anomaly_id: str):
    sim = app.state.sim

    match = next(
        (a for a in sim.recent_anomalies if a["anomaly_id"] == anomaly_id),
        None
    )

    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown anomaly_id {anomaly_id}"
        )

    return {
        "anomaly_id": anomaly_id,
        "features": match.get("shap_features", []),
        "likely_faulty_sensors": match.get("likely_faulty_sensors", []),
    }



# ---------------- GET /api/sensor-health ----------------

@app.get("/api/sensor-health")
def get_sensor_health(station_id: str):
    sim = app.state.sim
    status = sim.manager.get_station_status(station_id)
    mapped = {"HEALTHY": "HEALTHY", "WARNING": "WARNING", "OFFLINE": "OFFLINE"}.get(status["status"], "HEALTHY")
    entry = sim.latest.get(station_id)
    score = entry["verdict"]["anomaly_score_pct"] if entry else 0
    return {"station_id": station_id, "health_pct": max(0, 100 - int(score)), "status": mapped}

# ---------------- POST /api/repair-sensor ----------------

@app.post("/api/repair-sensor")
def repair_sensor(body: dict):
    sim = app.state.sim

    station_id = body.get("station_id")
    if not station_id:
        raise HTTPException(
            status_code=400,
            detail="station_id is required"
        )

    if station_id not in sim.manager.buffers:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown station {station_id}"
        )

    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc)

    sim.manager.mark_station_repaired(
        station_id,
        timestamp
    )

    return {
        "success": True,
        "station_id": station_id,
        "status": "WARNING",
        "recovery_active": True,
        "message": "Sensor marked for repair recovery. Clean readings will be evaluated before returning it to HEALTHY."
    }


# ---------------- POST /api/inject-anomaly ----------------

@app.post("/api/inject-anomaly")
def inject_anomaly(body: dict):
    """
    Repurposed as the REPLAY-MODE trigger -- see simulator.py's
    start_replay() docstring. body's station_id/type are accepted for
    contract-shape compatibility but not used to target one station;
    replay always drives all 20 simultaneously from their own
    pre-injected faults, then auto-reverts to live mode when exhausted.
    """
    sim = app.state.sim
    if sim.mode == "replay":
        raise HTTPException(status_code=409, detail="Simulator replay already running.")
    anomaly_id = sim.start_replay()
    return {
        "success": True,
        "anomaly_id": anomaly_id,
        "message": "Simulator started: replaying labeled historical data with injected faults across all stations.",
    }


# ---------------- POST /api/maintenance-ticket ----------------

_ticket_counter = 0

@app.post("/api/maintenance-ticket")
def create_maintenance_ticket(body: dict):
    global _ticket_counter
    sim = app.state.sim

    anomaly_id = body.get("anomaly_id")
    match = next((a for a in sim.recent_anomalies if a["anomaly_id"] == anomaly_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Unknown anomaly_id {anomaly_id}")

    _ticket_counter += 1
    from datetime import datetime, timezone
    return {
        "ticket_id": f"TCK-{_ticket_counter:04d}",
        "station_id": match["station_id"],
        "issue": match["root_cause"],
        "priority": "high" if match["severity"] in ("high", "critical") else "medium",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }