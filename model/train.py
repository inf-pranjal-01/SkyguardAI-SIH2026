"""
SkyGuard AI — Phase 2b: Model training.

Trains the Isolation Forest ONCE, offline, on RAW real historical data
only -- NEVER on a *_labeled.csv. Feeding injected/corrupted data into
training would teach the model a wrong definition of "normal" (see
BACKEND_BLUEPRINT.md section 4, and the leakage warnings throughout
anomaly_injector.py / features.py). This script actively checks for
and refuses that mistake below, rather than just warning about it.

Unlocked by: Day 80 (trees: entropy/gini/information gain), 84
(ensemble learning), 88 (bagging), 91-93 (random forest, bias-variance,
bagging vs RF). Isolation Forest reuses all of this directly: it's a
forest of randomized trees, each isolating points via random
feature/threshold splits -- but instead of voting on a prediction
(what Random Forest's trees do), a point's anomaly score comes from
HOW FEW splits it took to isolate it. Genuinely anomalous points sit
far from the bulk of the data, so random splits separate them out
quickly (few splits = anomalous); normal points are surrounded by
similar points and take many splits to isolate alone. n_estimators=100
stabilizes this score across many random trees, the same bias-variance
reasoning as Day 92 for Random Forest.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.append(str(Path(__file__).parent.parent))
from model.features import build_feature_matrix, FEATURE_COLUMNS

DATA_DIR = Path(__file__).parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).parent.parent / "model_artifacts"

N_ESTIMATORS = 100
RANDOM_STATE = 42


def load_clean_training_data():
    """
    Loads the RAW combined dataset + cluster metadata. Hard-fails if the
    input looks like a labeled/injected file -- this is the training/
    serving separation rule enforced in code, not just a comment.
    """
    stations_path = DATA_DIR / "all_stations.csv"
    metadata_path = DATA_DIR / "stations_metadata.csv"

    if not stations_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected {stations_path.name} and {metadata_path.name} in {DATA_DIR} "
            f"(outputs of data_fetch.py -> validate_data.py). Run those first."
        )

    df = pd.read_csv(stations_path, parse_dates=["timestamp"])

    if "is_anomaly" in df.columns or "fault_type" in df.columns:
        raise ValueError(
            "all_stations.csv contains is_anomaly/fault_type columns -- this looks "
            "like a labeled/injected file, not the raw combined dataset. train.py "
            "must only ever see raw data. Re-run data_fetch.py's combined output, "
            "or check you haven't accidentally pointed this at a *_labeled.csv."
        )

    metadata = pd.read_csv(metadata_path)
    return df, metadata


def train():
    df, metadata = load_clean_training_data()
    print(f"Loaded {len(df)} raw rows across {df['station_id'].nunique()} stations.\n")

    featured = build_feature_matrix(df, metadata)

    # Drop warm-up rows (first ~48h per station) that don't have a
    # complete rolling baseline yet -- an incomplete feature vector
    # isn't a real training example, it's a startup artifact.
    before = len(featured)
    featured = featured.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    after = len(featured)
    print(f"Dropped {before - after} warm-up rows with incomplete features "
          f"(expected: roughly ROLLING_MIN_PERIODS=6h x {df['station_id'].nunique()} "
          f"stations -- NOT the full 48h window, since min_periods lets rolling "
          f"features start producing values after just 6h of history).")
    print(f"{after} rows remain for training.\n")

    X = featured[FEATURE_COLUMNS].values

    # contamination='auto', deliberately NOT a fixed 0.05 -- fixed
    # earlier in the project. We're training on data we've validated as
    # genuinely normal, so asserting "5% of this is anomalous" would be
    # baking a false premise into the decision boundary. 'auto' uses
    # the original Isolation Forest paper's own offset heuristic
    # instead. Real severity thresholds (low/medium/high/critical) get
    # calibrated separately in the next phase, against the LABELED data
    # where we actually know the true anomaly rate -- not smuggled in
    # here as an assumption about clean training data.
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)

    # decision_function: higher = more normal, lower/negative = more
    # anomalous. Saving this training distribution alongside the model
    # gives detect.py (and whoever builds the eval script next) a
    # reference point for "what did normal actually look like," instead
    # of calibrating severity thresholds blind.
    scores = model.decision_function(X)
    print("Training score distribution (decision_function; LOWER = more anomalous):")
    print(pd.Series(scores).describe())

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "training_score_mean": float(scores.mean()),
        "training_score_std": float(scores.std()),
        "training_score_min": float(scores.min()),
        "training_score_max": float(scores.max()),
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "n_training_rows": after,
    }
    output_path = ARTIFACTS_DIR / "isolation_forest.pkl"
    joblib.dump(artifact, output_path)
    print(f"\nSaved trained model + metadata -> {output_path}")
    print("\nThis .pkl is what detect.py loads ONCE at API startup -- see "
          "BACKEND_BLUEPRINT.md section 4 for why it must never retrain per-request.")


if __name__ == "__main__":
    train()