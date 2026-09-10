"""
SkyGuard AI — Phase 2e: Explainability.

Answers "WHICH sensor is most likely at fault," not just "the score is
94%" -- a rule-fired verdict already carries this (rules_fired is
(rule_type, param) tuples), but a model-only "statistical_anomaly"
verdict is one scalar across all 22 features with no indication of
which raw sensor -- temperature/pressure/humidity -- is actually
implicated. Any one of the three can be the genuinely faulty reading
even when the other two are fine; this file localizes the station-level
verdict down to specific sensor(s). Powers GET /api/explain/{anomaly_id}
per ARCHITECTURE.md's contract.

SHAP + sklearn's IsolationForest is genuinely fragile across shap
versions (IsolationForest's decision_function isn't the same shape
TreeExplainer expects for a standard regressor/classifier -- support
varies by shap/sklearn version pairing and is known to raise on some
combinations). Rather than let a shap incompatibility take the whole
endpoint down, this degrades to a deterministic magnitude ranking (same
feature-magnitude-based ordering, no shap dependency) if TreeExplainer
construction or scoring fails -- logged loudly, not silently, so a
degraded explanation is visible if it's happening, not mistaken for a
real SHAP attribution.

FEATURE NAME MAP matches ARCHITECTURE.md's /api/explain/{id} example
shape ("Temp Deviation", "Pressure Inconsistency", "Rate of Change
(Temp)", "Time of Day Pattern", "Seasonal Pattern") so main.py returns
this file's output with zero translation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# feature_column -> display name. Every features.py FEATURE_COLUMNS
# entry MUST have one here -- a silently-missing feature in an
# explanation is worse than a loud startup error, see ExplainerCache.explain.
FEATURE_DISPLAY_NAMES = {
    "temp_deviation": "Temp Deviation",
    "pressure_deviation": "Pressure Deviation",
    "humidity_deviation": "Humidity Deviation",
    "temp_roc_1h": "Rate of Change (Temp, 1h)",
    "pressure_roc_1h": "Rate of Change (Pressure, 1h)",
    "humidity_roc_1h": "Rate of Change (Humidity, 1h)",
    "temp_roc_3h": "Rate of Change (Temp, 3h)",
    "pressure_roc_3h": "Rate of Change (Pressure, 3h)",
    "humidity_roc_3h": "Rate of Change (Humidity, 3h)",
    "temp_volatility_z": "Temp Volatility",
    "pressure_volatility_z": "Pressure Volatility",
    "humidity_volatility_z": "Humidity Volatility",
    "temp_humidity_coupling_signal": "Temp/Humidity Coupling",
    "pressure_inconsistency": "Pressure Inconsistency",
    "spatial_temp_inconsistency": "Temp vs. Neighboring Stations",
    "spatial_pressure_inconsistency": "Pressure vs. Neighboring Stations",
    "spatial_humidity_inconsistency": "Humidity vs. Neighboring Stations",
    "hour_sin": "Time of Day Pattern",
    "hour_cos": "Time of Day Pattern",
    "doy_sin": "Seasonal Pattern",
    "doy_cos": "Seasonal Pattern",
}

# Which raw sensor each feature implicates -- None means the feature
# couples multiple sensors or is a time signal, not attributable to
# one sensor alone.
FEATURE_TO_PARAM = {
    "temp_deviation": "temperature_c", "temp_roc_1h": "temperature_c", "temp_roc_3h": "temperature_c",
    "temp_volatility_z": "temperature_c", "spatial_temp_inconsistency": "temperature_c",
    "pressure_deviation": "pressure_hpa", "pressure_roc_1h": "pressure_hpa", "pressure_roc_3h": "pressure_hpa",
    "pressure_volatility_z": "pressure_hpa", "pressure_inconsistency": "pressure_hpa",
    "spatial_pressure_inconsistency": "pressure_hpa",
    "humidity_deviation": "humidity_pct", "humidity_roc_1h": "humidity_pct", "humidity_roc_3h": "humidity_pct",
    "humidity_volatility_z": "humidity_pct", "spatial_humidity_inconsistency": "humidity_pct",
    "temp_humidity_coupling_signal": None,
    "hour_sin": None, "hour_cos": None, "doy_sin": None, "doy_cos": None,
}


class ExplainerCache:
    """
    shap.TreeExplainer(model) walks every tree to build -- expensive.
    Build ONCE per loaded artifact, same "load once at startup"
    principle as detect.py's load_model(). state.py's StateManager
    owns one instance.
    """

    def __init__(self, artifact: dict):
        self.artifact = artifact
        self._explainer = None
        self._shap_broken = not _SHAP_AVAILABLE
        if _SHAP_AVAILABLE:
            try:
                self._explainer = shap.TreeExplainer(artifact["model"])
            except Exception as e:
                print(f"[explain] shap.TreeExplainer construction failed, falling back to "
                      f"magnitude ranking for all explanations: {e!r}")
                self._shap_broken = True

    def explain(self, feature_row: pd.Series) -> list[dict]:
        """Returns [{"name", "impact", "column"}, ...] sorted by |impact| descending. "column" is internal -- strip before returning over the API."""
        feature_columns = self.artifact["feature_columns"]
        missing = [c for c in feature_columns if c not in FEATURE_DISPLAY_NAMES]
        if missing:
            raise KeyError(f"FEATURE_DISPLAY_NAMES missing entries for: {missing}")

        X = feature_row[feature_columns].values.reshape(1, -1).astype(np.float64)
        if np.isnan(X).any():
            return []  # incomplete feature vector -- nothing honest to explain

        if not self._shap_broken:
            try:
                shap_values = self._explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                return self._format(feature_columns, shap_values[0])
            except Exception as e:
                print(f"[explain] shap_values() failed on this reading, falling back to "
                      f"magnitude ranking: {e!r}")

        # FALLBACK: features are already deviations/z-scores/
        # inconsistency signals centered near 0 for normal data (see
        # features.py), so raw magnitude is a reasonable proxy for
        # "how much did this feature contribute" without shap's exact
        # game-theoretic attribution.
        return self._format(feature_columns, X[0])

    def _format(self, feature_columns, impacts) -> list[dict]:
        rows = [
            {"name": FEATURE_DISPLAY_NAMES[col], "impact": round(float(val), 4), "column": col}
            for col, val in zip(feature_columns, impacts)
        ]
        rows.sort(key=lambda r: abs(r["impact"]), reverse=True)
        return rows


def likely_faulty_params(features: list[dict], top_n: int = 3) -> list[str]:
    """Collapses the ranked feature list to WHICH raw sensor(s) are implicated, using the top_n highest-magnitude features."""
    seen = []
    for row in features[:top_n]:
        param = FEATURE_TO_PARAM.get(row["column"])
        if param and param not in seen:
            seen.append(param)
    return seen