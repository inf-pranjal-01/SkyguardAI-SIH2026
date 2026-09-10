"""
SkyGuard AI — Phase 2c: Detection engine.

Combines the trained Isolation Forest with rule-based checks into one
verdict per reading, PLUS the per-station "circuit breaker" (mark a
sensor OFFLINE and stop trusting its data).

TWO LAYERS, fused together:
  1. MODEL layer: Isolation Forest's statistical anomaly score.
  2. RULE layer: deterministic checks (physical bounds, frozen-value,
     drift, dropout).
A reading can be flagged anomalous by either layer; severity fuses both.

BUG FIX LOG (keep this in sync with evaluate.py, whose docstring
promises the two files match):

  1. IS_ANOMALY_THRESHOLD raised 40 -> 55. pct=50 IS the model's own
     zero-boundary; 40 was flagging points on the NORMAL side of that
     line. Rule-fired detections are untouched (floor is 85).

  2. _rule_checks' frozen-value prefix derivation was broken for
     temperature ("temperature_c".replace("_c","") -> "temperature",
     not "temp"). Fixed with an explicit prefix map
     (features.py's RULE_ONLY_PREFIXES) instead of string-stripping.

  3. REPLACED the 48h-rolling-std frozen check entirely. It relied on
     temp_rolling_std/roc_1h, both computed over
     ROLLING_WINDOW_HOURS=48 -- far longer than a typical injected
     freeze (observed ~3-6h), so the window was diluted with 40+ hours
     of normal data and rarely collapsed enough to trip. Replaced with
     a short, direct raw-reading comparison (features.py's
     *_consec_diff) requiring 2 CONSECUTIVE readings under a
     threshold calibrated from real clean data (train.py's
     calibrate_rule_thresholds) -- a single very-still reading alone
     doesn't reliably separate "frozen sensor" from "genuinely calm
     weather," diagnosed empirically.

  4. ADDED a drift rule that didn't exist before: sustained
     DRIFT_LOOKBACK_HOURS-delta beyond a calibrated threshold, combined
     with a SMALL short-term roc_1h -- "small steps, large cumulative
     change" is drift's actual signature, distinct from a spike (large
     delta AND large roc_1h) which the model already handles well on
     its own.

  5. Both new rule thresholds live in the model artifact
     (artifact["rule_thresholds"]), calibrated once in train.py against
     real clean training data rather than hardcoded here -- retrain
     with the updated train.py before relying on these checks; an
     artifact saved by the OLD train.py will not have this key and
     score_reading will raise KeyError rather than silently no-op.
  6. TIERED RULE FLOOR added: physical_bounds/dropout (certain facts)
     keep the 85 floor; frozen_value/drift (calibrated statistical
     guesses) now get a lower 65 floor instead of sharing 85. This is
     the structural fix for why every rule-threshold recalibration this
     project went through caused a dramatic precision swing rather than
     a mild one -- soft rules were getting hard-rule confidence. A
     miscalibrated soft rule now degrades to a medium-severity flag
     instead of a false critical.


    
    Main entry point: scores ONE new reading given its station's recent
    history buffer. Returns the verdict dict main.py will serialize
    into the /api/current-reading and /api/anomalies/* shapes.

    cluster_neighbor_buffers: optional {station_id: history_df} for this
    station's cluster-mates. REQUIRED to get a real model score for the
    spatial features; without it they're filled with a neutral 0.0.

    spatial_baselines: optional {spatial_col: (baseline_median, spread_std)}
    from state.py's StationBuffer.spatial_baseline_and_spread() -- this
    station's OWN rolling history of raw spatial deviations, used to
    z-score the raw deviation instead of using it unnormalized. Without
    it, spatial features fall back to neutral 0.0.
    
"""

import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from model.features import (
    build_features_for_latest,
    build_rule_signals_recent,
    RULE_ONLY_PREFIXES,
    DRIFT_LOOKBACK_HOURS,
    get_threshold,
)
from model.explain import likely_faulty_params

from config import score_to_severity, HEALTH_WINDOW_SIZE, OFFLINE_ANOMALY_COUNT_THRESHOLD, RECOVERY_CLEAN_STREAK_REQUIRED

ARTIFACTS_PATH = Path(__file__).parent.parent / "model_artifacts" / "isolation_forest.pkl"

# Same hard physical ceilings used in anomaly_injector.py's clip step --
# duplicated intentionally rather than imported, since these represent
# a genuinely independent check (what CAN physically be true) rather
# than shared fault-generation logic.
PHYSICAL_BOUNDS = {
    "temperature_c": (-10.0, 55.0),
    "pressure_hpa": (850.0, 1080.0),
    "humidity_pct": (0.0, 100.0),
}

# (raw_column -> features.py prefix), imported from features.py so this
# mapping has exactly one source of truth -- see BUG FIX LOG #2.
PARAM_PREFIXES = dict(RULE_ONLY_PREFIXES)


# Spatial-consistency columns: this station's own deviation vs the
# median of its cluster neighbors, at the same moment -- computed raw
# in compute_raw_spatial_devs() below, then z-scored against this
# station's OWN rolling spatial-deviation history in score_reading()
# (state.py's StationBuffer.spatial_baseline_and_spread() supplies that).
SPATIAL_COLS = {
    "spatial_temp_inconsistency": "temp_deviation",
    "spatial_pressure_inconsistency": "pressure_deviation",
    "spatial_humidity_inconsistency": "humidity_deviation",
}


FROZEN_PERSISTENCE_READINGS = 3  # was 4 -- corrected math: freeze_length=randint(3,7)
# in anomaly_injector.py means only freeze_length CONSECUTIVE tiny-diff rows exist
# (idx+1..end_idx), NOT freeze_length+1 -- row idx itself is the transition INTO
# the freeze (frozen_value assigned from idx's own original reading), so its
# consec_diff vs idx-1 is normal-sized, not tiny. Persistence=4 could never fire
# on the minimum freeze_length=3 case (only 3 tiny diffs available), which is
# exactly why recall collapsed 72.4%->43.1%. 3 is the correct ceiling.
# Drift never had a persistence requirement (frozen always did) --
# confirmed as the dominant soft_fired FP source before fixing.
# Real drift events last 20-50h; 3 consecutive qualifying readings is
# a tiny fraction of that, so this should cost little real recall.
DRIFT_PERSISTENCE_READINGS = 3

# Statistical (model-only) anomaly floor. See BUG FIX LOG #1.
IS_ANOMALY_THRESHOLD = 75

# Model-only anomalies require a much stronger Isolation Forest score.
# Rule-backed anomalies continue to use the existing rule floors.
MODEL_ONLY_THRESHOLD = 90

# TIERED RULE FLOOR (BUG FIX LOG #6): physical_bounds/dropout are
# certain, unambiguous facts -- humidity CANNOT exceed 100%, a null
# reading IS missing, full stop, no statistical judgment involved.
# frozen_value/drift are calibrated STATISTICAL GUESSES from
# empirically-tuned percentile thresholds -- much more likely to be
# wrong at the margin, as this project's own precision swings
# (0.05 -> 0.232 -> 0.030 -> 0.337 across successive threshold
# recalibrations) demonstrate. Previously both tiers shared one
# floor (85), so every rule-threshold miscalibration produced a
# dramatic precision swing instead of a mild one -- a shaky heuristic
# got the same "critical-adjacent" confidence as an ironclad physical
# fact. Splitting the floor means a miscalibrated soft rule degrades
# gracefully into a medium-severity flag instead of a false critical.
HARD_RULE_TYPES = {"physical_bounds", "dropout"}
SOFT_RULE_TYPES = {"frozen_value", "drift", "spike"}
HARD_RULE_FLOOR = 85.0
SOFT_RULE_FLOOR = 75.0  # still above IS_ANOMALY_THRESHOLD=75, so it's
                         # still flagged as an anomaly -- just not
                         # forced into "critical" territory on a
                         # rule-layer guess alone.5


def load_model():
    """Loads the trained model artifact ONCE -- call this at API startup, not per-request."""
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(f"No trained model at {ARTIFACTS_PATH} -- run model/train.py first.")
    artifact = joblib.load(ARTIFACTS_PATH)
    if "rule_thresholds" not in artifact:
        raise KeyError(
            "Loaded artifact has no 'rule_thresholds' key -- it was saved by an older "
            "train.py, before calibrate_rule_thresholds() was added. Retrain with the "
            "current train.py before running detection."
        )
    return artifact


def _model_score_to_pct(raw_reading: dict, feature_row: pd.Series, artifact: dict) -> float:
    """
    Converts Isolation Forest's raw decision_function output into a
    0-100 "anomaly score." Centered on 0 (the model's own contamination-
    calibrated outlier boundary), NOT on training_score_mean -- see BUG
    FIX LOG #1 for the false-positive-rate bug this fixed.
    """
    model = artifact["model"]
    X = feature_row[artifact["feature_columns"]].values.reshape(1, -1).astype(np.float64)

    if np.isnan(X).any():
        return None

    raw_score = model.decision_function(X)[0]
    z = (0.0 - raw_score) / (artifact["training_score_std"] + 1e-9)
    pct = 100 / (1 + np.exp(-1.5 * z))
    return float(np.clip(pct, 0, 100))


def _rule_checks(raw_reading: dict, feature_row: pd.Series, history_df: pd.DataFrame, artifact: dict) -> dict:
    """
    Deterministic checks independent of the model. Returns which rules
    fired and a best-guess fault_type if any did.

    thresholds come from the model artifact (calibrated in train.py
    against real clean data) -- see BUG FIX LOG #5.
    """
    fired = []
    thresholds = artifact["rule_thresholds"]
    station_id = raw_reading.get("station_id") or history_df["station_id"].iloc[-1]

    for param, (low, high) in PHYSICAL_BOUNDS.items():
        val = raw_reading.get(param)
        if val is not None and not pd.isna(val) and not (low <= val <= high):
            fired.append(("physical_bounds", param))

    # Frozen-value: short raw-diff comparison + calibrated threshold,
    # requiring FROZEN_PERSISTENCE_READINGS consecutive hits -- see BUG
    # FIX LOG #3.
    recent = build_rule_signals_recent(history_df, n=FROZEN_PERSISTENCE_READINGS)
    for param, prefix in PARAM_PREFIXES.items():
        consec_col = f"{prefix}_consec_diff"
        if consec_col in recent.columns and len(recent) >= FROZEN_PERSISTENCE_READINGS:
            vals = recent[consec_col]
            if vals.notna().all() and (vals < get_threshold(thresholds, "frozen", prefix, station_id)).all():
                fired.append(("frozen_value", param))

    # Drift: sustained DRIFT_LOOKBACK_HOURS displacement with a SMALL
    # short-term step, now requiring DRIFT_PERSISTENCE_READINGS
    # consecutive hits -- same pattern as frozen's persistence check,
    # reusing build_rule_signals_recent since it already computes both
    # Nh_delta and roc_1h for the tail window.
    recent_drift = build_rule_signals_recent(history_df, n=DRIFT_PERSISTENCE_READINGS)
    for param, prefix in PARAM_PREFIXES.items():
        delta_col = f"{prefix}_{DRIFT_LOOKBACK_HOURS}h_delta"
        roc_col = f"{prefix}_roc_1h"
        if (
                delta_col in recent_drift.columns
                and roc_col in recent_drift.columns
                and len(recent_drift) >= DRIFT_PERSISTENCE_READINGS
        ):
            deltas = recent_drift[delta_col]
            rocs = recent_drift[roc_col]
            if (
                deltas.notna().all()
                and rocs.notna().all()
                and (
                    deltas.abs()
                    > get_threshold(
                        thresholds, "drift", prefix, station_id
                    )
                ).all()
                and (
                    rocs.abs()
                    < get_threshold(
                        thresholds, "roc_small", prefix, station_id
                    )
                ).all()
            ):
                fired.append(("drift", param))

    # Spike: unusually large instantaneous deviation from
    # the station's rolling baseline.
    for param, prefix in PARAM_PREFIXES.items():
        dev_col = f"{prefix}_deviation"
        if dev_col in feature_row:
            dev_val = feature_row[dev_col]
            if (
                pd.notna(dev_val)
                and abs(dev_val) > get_threshold(thresholds, "spike", prefix, station_id)
            ):
                fired.append(("spike", param))



                           

    for param in PHYSICAL_BOUNDS:
        if raw_reading.get(param) is None or pd.isna(raw_reading.get(param)):
            fired.append(("dropout", param))

    return {"fired": fired, "any": len(fired) > 0}


def compute_raw_spatial_devs(own_feature_row: pd.Series, cluster_neighbor_buffers: dict) -> dict:
    """Single source of truth for the RAW spatial difference -- shared by score_reading() and state.py."""
    raw_devs = {}
    if not cluster_neighbor_buffers:
        return raw_devs
    neighbor_features = [build_features_for_latest(buf) for buf in cluster_neighbor_buffers.values()]
    for spatial_col, base_col in SPATIAL_COLS.items():
        neighbor_vals = [nf[base_col] for nf in neighbor_features if pd.notna(nf.get(base_col))]
        own_val = own_feature_row.get(base_col)
        if neighbor_vals and pd.notna(own_val):
            raw_devs[spatial_col] = own_val - float(np.median(neighbor_vals))
    return raw_devs




def score_reading(raw_reading: dict, history_df: pd.DataFrame, artifact: dict,
                   cluster_neighbor_buffers: dict = None, spatial_baselines: dict = None,
                   explainer=None) -> dict:
    """
    spatial_baselines: optional {spatial_col: (baseline_median, spread_std)}
    from state.py's StationBuffer.spatial_baseline_and_spread() -- this
    station's OWN rolling history of raw spatial deviations, used to
    z-score the raw deviation instead of using it unnormalized. Without
    it, spatial features fall back to neutral 0.0.
    """
    feature_row = build_features_for_latest(history_df)
    suggested_values = {
        param: round(float(feature_row[f"{prefix}_rolling_mean"]), 2)
        for param, prefix in PARAM_PREFIXES.items()
        if pd.notna(feature_row.get(f"{prefix}_rolling_mean"))
    }    
    raw_spatial_devs = compute_raw_spatial_devs(feature_row, cluster_neighbor_buffers)
    for spatial_col in SPATIAL_COLS:
        raw_dev = raw_spatial_devs.get(spatial_col)
        if raw_dev is None:
            feature_row[spatial_col] = 0.0
            continue
        baseline_spread = (spatial_baselines or {}).get(spatial_col)
        if baseline_spread and baseline_spread[0] is not None and baseline_spread[1]:
            baseline, spread = baseline_spread
            feature_row[spatial_col] = (raw_dev - baseline) / spread
        else:
            feature_row[spatial_col] = 0.0

    model_pct = _model_score_to_pct(raw_reading, feature_row, artifact)
    rules = _rule_checks(raw_reading, feature_row, history_df, artifact)

    if rules["any"]:
        fired_types = {f[0] for f in rules["fired"]}

        if fired_types & HARD_RULE_TYPES:
            rule_floor = HARD_RULE_FLOOR
            fault_type = next(
                f[0] for f in rules["fired"]
                if f[0] in HARD_RULE_TYPES
            )
        else:
            rule_floor = SOFT_RULE_FLOOR
            fault_type = rules["fired"][0][0]

        score_pct = max(
            rule_floor,
            model_pct if model_pct is not None else 0.0
        )
        is_anomaly = True

    elif model_pct is not None and model_pct >= MODEL_ONLY_THRESHOLD:
        score_pct = model_pct
        fault_type = "statistical_anomaly"
        is_anomaly = True

    else:
        score_pct = model_pct if model_pct is not None else 0.0
        fault_type = None
        is_anomaly = False

    severity = score_to_severity(score_pct)

    shap_features_public, likely_sensors = [], []
    if is_anomaly and explainer is not None:
        try:
            shap_features = explainer.explain(feature_row)
            likely_sensors = likely_faulty_params(shap_features)
            shap_features_public = [
                {"name": f["name"], "impact": f["impact"]}
                for f in shap_features
            ]
        except Exception as e:
            print(
                f"[detect] explanation failed for this reading, "
                f"returning verdict without it: {e!r}"
            )

    return {
        "anomaly_score_pct": round(score_pct, 1),
        "is_anomaly": bool(is_anomaly),
        "severity": severity,
        "fault_type": fault_type,
        "rules_fired": rules["fired"],
        "_raw_spatial_devs": raw_spatial_devs,
        "suggested_values": suggested_values,
        "shap_features": shap_features_public,
        "likely_faulty_sensors": likely_sensors,
    }


class SensorHealthTracker:
    """
    The circuit breaker: tracks one station's recent verdict history and
    decides when it's erratic enough to mark OFFLINE.
    """

    def __init__(self, station_id: str):
        self.station_id = station_id
        self.recent_verdicts = deque(maxlen=HEALTH_WINDOW_SIZE)
        self.status = "HEALTHY"
        self.offline_reason = None
        self._clean_streak = 0

    def record(self, verdict: dict):
        self.recent_verdicts.append(verdict["is_anomaly"])

        if self.status == "OFFLINE":
            if verdict["is_anomaly"]:
                self._clean_streak = 0
            else:
                self._clean_streak += 1
                if self._clean_streak >= RECOVERY_CLEAN_STREAK_REQUIRED:
                    self.status = "HEALTHY"
                    self.offline_reason = None
                    self._clean_streak = 0
            return

        anomaly_count = sum(self.recent_verdicts)
        if len(self.recent_verdicts) >= HEALTH_WINDOW_SIZE and anomaly_count >= OFFLINE_ANOMALY_COUNT_THRESHOLD:
            self.status = "OFFLINE"
            self.offline_reason = (
                f"{anomaly_count} of the last {HEALTH_WINDOW_SIZE} readings were anomalous -- "
                f"erratic pattern, excluded from baseline/neighbor calculations until recovered."
            )
        elif anomaly_count >= 2:
            self.status = "WARNING"
        else:
            self.status = "HEALTHY"

    def should_include_in_baseline(self) -> bool:
        return self.status != "OFFLINE"