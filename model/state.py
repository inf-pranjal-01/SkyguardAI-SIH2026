"""
SkyGuard AI — state.py: in-memory live state store.

Wires model/detect.py's score_reading() into a per-station history
buffer with CAUSAL trusted-history exclusion: a reading only enters a
station's buffer if it was judged clean AT THE TIME, so a faulty
sensor never teaches its own future baseline that its fault is normal
(BACKEND_BLUEPRINT.md's core state.py requirement).

Matches the REAL detect.py/features.py signatures from this project --
score_reading(raw_reading, history_df, artifact, cluster_neighbor_buffers=None),
SensorHealthTracker from detect.py, build_features_for_latest from
features.py. No spatial-baseline/volatility machinery that was never
actually built -- an earlier draft of this file imported constants and
functions (VOLATILITY_BASELINE_WINDOW_HOURS, compute_raw_spatial_devs,
SPATIAL_COLS, a spatial_baselines kwarg) that don't exist anywhere in
this codebase and would ImportError on load. Scoped down deliberately
given the time remaining -- see the "future work" note at bottom for
what's intentionally deferred.
"""

import sys
from collections import deque

from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from model.detect import score_reading, SensorHealthTracker
from model.features import ROLLING_WINDOW_HOURS, DRIFT_LOOKBACK_HOURS
from model.explain import ExplainerCache
from config import RECOVERY_CLEAN_STREAK_REQUIRED
# History buffer needs enough hours for the longest lookback any
# feature uses -- DRIFT_LOOKBACK_HOURS (24) vs ROLLING_WINDOW_HOURS (48)
# -- plus slack so the buffer never truncates a rolling window short.
RAW_HISTORY_MAXLEN_HOURS = max(ROLLING_WINDOW_HOURS, DRIFT_LOOKBACK_HOURS) + 12

# Recovery requires the same number of consecutive clean readings
# used by SensorHealthTracker. This works consistently in live and
# replay because replay advances readings faster than wall-clock time.


class StationBuffer:
    """
    Per-station live state: raw-reading history (feeds
    build_features_for_latest via score_reading), SensorHealthTracker,
    and repair/recovery bookkeeping.
    """

    def __init__(self, station_id: str):
        self.station_id = station_id
        self.health = SensorHealthTracker(station_id)
        self._raw_rows: deque = deque(maxlen=RAW_HISTORY_MAXLEN_HOURS)

        # Repair/recovery state -- separate from health.status so a
        # sensor can be OFFLINE (health's own circuit breaker) while
        # ALSO being mid-recovery after an operator-initiated repair.
        self.recovery_active: bool = False
        self.recovery_clean_count: int = 0

    def raw_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._raw_rows))

    def record_raw_reading(self, raw_reading: dict, timestamp):
        """
        CAUSAL EXCLUSION: only append if health.should_include_in_baseline()
        says this station isn't currently OFFLINE. This is the live
        equivalent of features.py's exclude_mask (used in batch/eval
        mode with ground truth) -- achieved here using the model's OWN
        prior verdicts instead, since live serving has no ground truth.
        A faulty sensor's own readings never enter the buffer used to
        judge its future readings.
        """
        if not self.health.should_include_in_baseline() or self.recovery_active:
            return
        row = dict(raw_reading)
        row["station_id"] = self.station_id
        row["timestamp"] = timestamp
        self._raw_rows.append(row)

    def mark_repaired(self, timestamp):
        """
        Operator action from the frontend's 'Mark Repaired' button.
        Does NOT immediately trust the sensor -- starts a clean-reading
        recovery period instead.
        """
        self.recovery_active = True
        self.recovery_clean_count = 0
        self.health.status = "WARNING"
        self.health.offline_reason = None
        self.health._clean_streak = 0

    def update_recovery(self, verdict: dict):
        """
        Reading-count based recovery. Requires consecutive clean readings
        after repair instead of wall-clock elapsed time.
        """
        if not self.recovery_active:
            return

        if self.health.status == "OFFLINE":
            # Went offline again during recovery.
            self.recovery_active = False
            self.recovery_clean_count = 0
            return

        if verdict["is_anomaly"]:
            self.recovery_clean_count = 0
            return

        self.recovery_clean_count += 1

        if self.recovery_clean_count >= RECOVERY_CLEAN_STREAK_REQUIRED:
            self.recovery_active = False
            self.recovery_clean_count = 0
            self.health.status = "HEALTHY"


class StateManager:
    """
    Owns one StationBuffer per station + cluster membership for
    spatial-feature neighbor lookups. ingest_reading() is the ONLY
    entry point both live serving (simulator.py/main.py) and any
    future replay/test harness should call -- never call
    detect.score_reading() directly elsewhere, or state.py and
    detect.py can silently drift apart the same way evaluate.py once
    diverged from train.py earlier this project.
    """

    def __init__(self, metadata: pd.DataFrame, artifact: dict):
        self.metadata = metadata
        self.artifact = artifact
        self.explainer = ExplainerCache(artifact)
        self.buffers: dict[str, StationBuffer] = {
            sid: StationBuffer(sid) for sid in metadata["station_id"]
        }
        self._cluster_of = dict(zip(metadata["station_id"], metadata["cluster_id"]))

    def _neighbor_buffers(self, station_id: str) -> dict:
        cluster = self._cluster_of.get(station_id)
        return {
            sid: self.buffers[sid].raw_history_df()
            for sid, cid in self._cluster_of.items()
            if cid == cluster and sid != station_id and len(self.buffers[sid]._raw_rows) > 0
        }

    def ingest_reading(self, station_id: str, raw_reading: dict, timestamp) -> dict:
        buf = self.buffers[station_id]
        history_df = buf.raw_history_df()

        current_row = dict(raw_reading, station_id=station_id, timestamp=timestamp)
        history_df_with_current = (
            pd.concat([history_df, pd.DataFrame([current_row])], ignore_index=True)
            if not history_df.empty else pd.DataFrame([current_row])
        )

        neighbor_buffers = self._neighbor_buffers(station_id)

        verdict = score_reading(
            raw_reading,
            history_df_with_current,
            self.artifact,
            cluster_neighbor_buffers=neighbor_buffers,
            explainer=self.explainer,
        )

        # Health updated with THIS verdict BEFORE recording, so a
        # reading that tips the station into OFFLINE is itself
        # correctly excluded from its own future baseline.
        buf.health.record(verdict)

        if buf.recovery_active:
            buf.update_recovery(verdict)

        buf.record_raw_reading(raw_reading, timestamp)

        return verdict

    def mark_station_repaired(self, station_id: str, timestamp):
        """Called by main.py's repair endpoint, once that route exists."""
        self.buffers[station_id].mark_repaired(timestamp)

    def get_station_status(self, station_id: str) -> dict:
        buf = self.buffers[station_id]
        return {
            "station_id": station_id,
            "status": buf.health.status,
            "offline_reason": buf.health.offline_reason,
            "recovery_active": buf.recovery_active,
        }


# DEFERRED (see 6-hour scoping decision): full HEALTHY -> SUSPICIOUS ->
# FAULTY -> REPAIRED -> RECOVERY state machine, retrospective cleanup
# of already-buffered contaminated readings, spatial-baseline
# volatility z-scoring. Current scope: binary HEALTHY/OFFLINE (from
# detect.py's existing SensorHealthTracker) + real-time exclusion +
# reading-count based recovery. Sufficient for main.py/simulator.py to
# work correctly now; upgrade path documented for the PPT roadmap slide.