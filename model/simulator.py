"""
SkyGuard AI — simulator.py: drives the live/demo data stream.

TWO MODES, switched by the whole system at once (not per-station):

  LIVE (default): fetches CURRENT real weather from Open-Meteo for
  each station's real coordinates. No faults -- genuine, unsupervised
  detection against real present-day conditions. This is the actual
  production behavior.

  REPLAY (demo/simulator mode): steps through all 20 *_labeled.csv
  files in lockstep at live cadence. These already contain real
  injected faults with known ground truth (anomaly_injector.py) --
  detect.py scores each reading BLIND, exactly as it would any other
  reading, and whatever it flags is a genuine model detection, not a
  scripted/faked frontend event. This replaces an earlier design that
  reimplemented fault shapes live; that's been removed -- there's no
  reason to reinvent faults when validated labeled data already exists.

MODE SWITCH VIA EXISTING CONTRACT: the frontend's POST
/api/inject-anomaly button ({station_id, type} -> {anomaly_id,
message}) is the only trigger the frontend already has wired, and the
frontend doc explicitly prohibits inventing new endpoints. So that
route (wired in main.py, not here) calls SimulatorState.start_replay(),
which switches the WHOLE system into replay mode -- station_id/type
are accepted for contract-shape compatibility but not used to target
a single station, since replay always drives all 20 simultaneously
from their own pre-injected faults. Replay auto-reverts to LIVE once
every labeled file is exhausted. If per-station/per-type targeting is
actually wanted instead, this needs revisiting -- flagging the
assumption rather than guessing further.

LIVE-FETCH CAVEAT: the exact Open-Meteo "current conditions" call
below is written to the pattern data_fetch.py's archive-history call
likely follows (same station coordinates from stations_metadata.csv),
but I don't have data_fetch.py in context to confirm the exact
endpoint/params it uses. Verify _fetch_live_reading against your real
data_fetch.py before relying on it -- the shape may need adjusting.
"""

import asyncio
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import pandas as pd

# With (matches detect.py/train.py's established pattern exactly):
sys.path.append(str(Path(__file__).parent.parent))
from model.state import StateManager
from config import score_to_severity

DATA_DIR = Path(__file__).parent.parent / "data"
ARTIFACTS_PATH = Path(__file__).parent.parent / "model_artifacts" / "isolation_forest.pkl"

TICK_SECONDS = 2  # BACKEND_BLUEPRINT.md: ~2s per simulated reading
LIVE_FETCH_INTERVAL_SECONDS = 60  # real weather doesn't need per-2s polling; cache between fetches
TREND_HISTORY_MAXLEN = 2000
RECENT_ANOMALIES_MAXLEN = 200

OPEN_METEO_CURRENT_URL = "https://api.open-meteo.com/v1/forecast"

ROOT_CAUSE_BY_FAULT_TYPE = {
    "physical_bounds": "Reading outside physically possible range",
    "dropout": "Sensor communication failure",
    "frozen_value": "Sensor stuck / communication fault",
    "drift": "Calibration drift suspected",
    "spike": "Sudden reading spike -- possible sensor malfunction",
    "statistical_anomaly": "Unusual reading pattern flagged by model",
}


async def _fetch_live_reading(client: httpx.AsyncClient, lat: float, lon: float) -> dict | None:
    """
    Pulls CURRENT conditions for one station's coordinates. Returns
    None on any failure so a transient network hiccup degrades to
    "reuse last cached value" (handled by the caller) rather than
    crashing the tick loop.

    VERIFY against your real data_fetch.py -- written to the likely
    Open-Meteo current-weather shape, not confirmed against your
    actual archive-fetch implementation.
    """
    try:
        resp = await client.get(
            OPEN_METEO_CURRENT_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,surface_pressure,relative_humidity_2m",
                "timezone": "UTC",
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()["current"]
        return {
            "temperature_c": float(data["temperature_2m"]),
            "pressure_hpa": float(data["surface_pressure"]),
            "humidity_pct": float(data["relative_humidity_2m"]),
        }
    except Exception as e:
        print(f"[simulator] live fetch failed for ({lat},{lon}): {e!r}")
        return None


class SimulatorState:
    """The object main.py's routes read from. One instance, created at FastAPI startup."""

    def __init__(self, metadata: pd.DataFrame, artifact: dict):
        self.metadata = metadata
        self.manager = StateManager(metadata, artifact)

        self.mode: str = "live"  # "live" | "replay"
        self._replay_cursor_idx: int = 0
        self._replay_frames: dict[str, pd.DataFrame] = {}  # populated lazily on start_replay()
        self._replay_len: int = 0

        self._live_cache: dict[str, dict] = {}  # station_id -> last fetched reading
        self._live_last_fetch: datetime | None = None
        self._last_ingested: dict[str, dict] = {}
        self.latest: dict[str, dict] = {}
        self.trend_history: dict[str, deque] = {
            sid: deque(maxlen=TREND_HISTORY_MAXLEN) for sid in metadata["station_id"]
        }
        self.recent_anomalies: deque = deque(maxlen=RECENT_ANOMALIES_MAXLEN)
        self._anomaly_counter = 0

        self._http_client = httpx.AsyncClient()

    # ---------------- mode control ----------------

    def start_replay(self) -> str:
        """
        Called from main.py's POST /api/inject-anomaly handler. Loads
        every *_labeled.csv fresh (so repeated demo runs always replay
        from the start) and switches mode. Returns a queued anomaly_id
        for the contract response -- the real detections happen
        asynchronously as tick() runs, this id is just a placeholder
        acknowledging the request per the existing response shape
        ("Anomaly injected and detected in next reading cycle.").
        """
        self._replay_frames = {}
        for sid in self.metadata["station_id"]:
            path = DATA_DIR / f"{sid}_labeled.csv"
            if not path.exists():
                raise FileNotFoundError(f"No labeled data for {sid} at {path} -- run anomaly_injector.py first.")
            df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            self._replay_frames[sid] = df

        self._replay_len = min(len(df) for df in self._replay_frames.values())
        self._replay_cursor_idx = 0
        self.mode = "replay"

        # Reset all detection state so live history does not contaminate replay.
        self.manager = StateManager(self.metadata, self.manager.artifact)
        self._last_ingested = {}
        self.latest = {}
        self.trend_history = {
            sid: deque(maxlen=TREND_HISTORY_MAXLEN)
            for sid in self.metadata["station_id"]
        }
        self.recent_anomalies.clear()

        self._anomaly_counter += 1
        return f"anom_{self._anomaly_counter:05d}"

    def _stop_replay(self):
        self.mode = "live"
        self._replay_frames = {}

        # Reset replay state before returning to real live weather.
        self.manager = StateManager(self.metadata, self.manager.artifact)
        self._last_ingested = {}
        self.latest = {}
        self.trend_history = {
            sid: deque(maxlen=TREND_HISTORY_MAXLEN)
            for sid in self.metadata["station_id"]
        }
        self.recent_anomalies.clear()

        self._live_cache = {}
        self._live_last_fetch = None

    # ---------------- per-tick data sourcing ----------------

    async def _maybe_refresh_live_cache(self):
        now = datetime.now(timezone.utc)
        if (
            self._live_last_fetch is not None
            and (now - self._live_last_fetch).total_seconds() < LIVE_FETCH_INTERVAL_SECONDS
        ):
            return
        self._live_last_fetch = now

        for _, row in self.metadata.iterrows():
            sid = row["station_id"]
            reading = await _fetch_live_reading(self._http_client, row["lat"], row["lon"])
            if reading is not None:
                self._live_cache[sid] = reading
            # else: keep whatever was cached before -- degrade gracefully

    def _next_live_row(self, station_id: str) -> dict | None:
        return self._live_cache.get(station_id)  # None until first successful fetch

    def _next_replay_row(self, station_id: str) -> dict:
        df = self._replay_frames[station_id]
        row = df.iloc[self._replay_cursor_idx]
        return {
            "temperature_c": float(row["temperature_c"]) if pd.notna(row["temperature_c"]) else None,
            "pressure_hpa": float(row["pressure_hpa"]) if pd.notna(row["pressure_hpa"]) else None,
            "humidity_pct": float(row["humidity_pct"]) if pd.notna(row["humidity_pct"]) else None,
        }

    # ---------------- the tick ----------------

    async def tick(self):
        """
        One simulation step across all stations. Uses wall-clock NOW
        as every reading's timestamp regardless of mode -- state.py's
        recovery timer and the rolling-window features need real
        elapsed time; replaying a labeled file's original hourly
        timestamps at 2s/tick would make every window/timer logic
        nonsensical.
        """
        now = datetime.now(timezone.utc)

        if self.mode == "live":
            await self._maybe_refresh_live_cache()

        for station_id in self.metadata["station_id"]:
            if self.mode == "replay":
                raw_reading = self._next_replay_row(station_id)
            else:
                raw_reading = self._next_live_row(station_id)
                if raw_reading is None:
                    continue

            # Replay: every CSV row is a genuine new reading.
            #
            # Live: only ingest when Open-Meteo gives us a genuinely
            # different reading. The UI still updates every 2 seconds,
            # but the ML/state history only gets new observations.
            should_ingest = (
                self.mode == "replay"
                or self._last_ingested.get(station_id) != raw_reading
            )

            if should_ingest:
                verdict = self.manager.ingest_reading(
                    station_id,
                    raw_reading,
                    now
                )
                self._last_ingested[station_id] = dict(raw_reading)
            else:
                cached = self.latest.get(station_id)
                if cached is None:
                    continue
                verdict = cached["verdict"]

            self.latest[station_id] = {
                "raw_reading": raw_reading,
                "verdict": verdict,
                "timestamp": now,
            }

            # UI trend gets a point every 2 seconds.
            self.trend_history[station_id].append({
                "timestamp": now,
                "temperature_c": raw_reading.get("temperature_c"),
                "pressure_hpa": raw_reading.get("pressure_hpa"),
                "humidity_pct": raw_reading.get("humidity_pct"),
                "is_anomaly": verdict["is_anomaly"],
            })

            # Only create a new anomaly event when a new reading
            # was actually processed.
            if should_ingest and verdict["is_anomaly"]:
                self._anomaly_counter += 1
                self.recent_anomalies.appendleft({
                    "anomaly_id": f"anom_{self._anomaly_counter:05d}",
                    "timestamp": now,
                    "station_id": station_id,
                    "anomaly_score_pct": verdict["anomaly_score_pct"],
                    "suggested_values": verdict.get("suggested_values"),
                    "shap_features": verdict.get("shap_features", []),
                    "likely_faulty_sensors": verdict.get("likely_faulty_sensors", []),
                    "severity": verdict["severity"],
                    "type": verdict["fault_type"] or "statistical_anomaly",
                    "root_cause": ROOT_CAUSE_BY_FAULT_TYPE.get(
                        verdict["fault_type"],
                        "Unusual reading pattern flagged by model"
                    ),
                })

        if self.mode == "replay":
            self._replay_cursor_idx += 1
            if self._replay_cursor_idx >= self._replay_len:
                self._stop_replay()


async def run_simulation_loop(sim_state: SimulatorState):
    """Scheduled by main.py via asyncio.create_task() at startup. One bad tick is logged and skipped, not fatal."""
    while True:
        try:
            await sim_state.tick()
        except Exception as e:
            print(f"[simulator] tick failed: {e!r}")
        await asyncio.sleep(TICK_SECONDS)


def create_simulator_state() -> SimulatorState:
    """Called once from main.py's startup event."""
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(f"No trained model at {ARTIFACTS_PATH} -- run model/train.py first.")
    artifact = joblib.load(ARTIFACTS_PATH)
    if "rule_thresholds" not in artifact:
        raise KeyError("Artifact missing 'rule_thresholds' -- retrain with the current train.py.")

    metadata_path = DATA_DIR / "stations_metadata.csv"
    metadata = pd.read_csv(metadata_path)

    return SimulatorState(metadata, artifact)