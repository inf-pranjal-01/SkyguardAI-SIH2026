"""
SkyGuard AI — Phase 2a: Feature engineering.

Sits between the data layer (data_fetch.py -> validate_data.py) and the
model layer (train.py). Turns raw per-hour temp/pressure/humidity
readings into the feature vectors the Isolation Forest actually trains
and scores on.

WHERE THIS FITS IN THE PIPELINE:

    data_fetch.py -> validate_data.py -> [THIS FILE] -> train.py
                                              ^
                          also called on labeled/live data at
                          detect.py time, same feature logic both ways

CRITICAL RULE (carried over from anomaly_injector.py / validate_data.py):
this module does NOT know or care whether its input is the clean
raw CSV or an injected _labeled.csv -- it just computes features from
whatever `temperature_c` / `pressure_hpa` / `humidity_pct` columns it's
given. If `is_anomaly` / `fault_type` ground-truth columns are present
(from anomaly_injector.py) they are carried through untouched for
Phase 3 evaluation, but build_feature_matrix() NEVER includes them in
the returned FEATURE_COLUMNS list. train.py must only ever fit on
FEATURE_COLUMNS from the raw CSV; feeding ground-truth label columns
into the model, or training on the labeled/injected file at all, would
be a real leak, not a style choice.

FOUR FEATURE GROUPS, each tied to a specific PS objective / fault type:

  1. Raw values              -- the baseline signal itself.
  2. Temporal features       -- rate-of-change + deviation from each
                                 station's own recent rolling baseline.
                                 Catches spike, frozen_value (near-zero
                                 rate-of-change AND near-zero rolling
                                 std at once), drift, dropout.
  3. Cross-parameter features -- "multivariate consistency analysis",
                                 a named PS objective. Catches the
                                 multivariate_inconsistency fault and
                                 the PS's own worked example (temp up +
                                 humidity up + pressure flat).
  4. Spatial features         -- cluster-based "neighboring stations
                                 show normal conditions" check, the
                                 other half of the PS's worked example.
                                 This is the gap flagged earlier in the
                                 project: temporal-only consistency
                                 wasn't enough, the PS explicitly wants
                                 spatial consistency against real nearby
                                 stations too (Day 41-43 z-score logic,
                                 applied across stations in the same
                                 data_fetch.py cluster instead of across
                                 time).

LEAKAGE NOTE: every rolling/baseline statistic below is computed with
`.shift(1)` before the rolling window, i.e. "as of the reading before
this one." If we let a reading's own value bleed into the baseline it's
being compared against, a genuine spike would partially absorb into its
own rolling mean and understate its own z-score -- exactly the kind of
bug that looks fine on a quick eyeball check and quietly wrecks
detection accuracy. Every *_deviation feature below is safe against
this by construction.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Rolling window for each station's own recent baseline. 24h = one full
# diurnal cycle, so "recent normal" already accounts for the daily
# temperature/pressure/humidity swing instead of falsely flagging
# "it's just afternoon" as a deviation. min_periods keeps the first
# ~day of each station's data from producing all-NaN features.
ROLLING_WINDOW_HOURS = 48
ROLLING_MIN_PERIODS = 6

# Rate-of-change lookback in hours. Short (1h) catches sudden jumps
# (spike); slightly longer (3h) catches a fast ramp that no single
# 1h step looks extreme (early drift).
ROC_SHORT_HOURS = 1
ROC_LONG_HOURS = 3

RAW_COLUMNS = ["temperature_c", "pressure_hpa", "humidity_pct"]

# Names deliberately match the SHAP feature-name examples already
# promised in ARCHITECTURE.md's /api/explain/{id} contract ("Temp
# Deviation", "Pressure Inconsistency", "Rate of Change (Temp)", "Time
# of Day Pattern", "Seasonal Pattern") so explain.py can map straight
# from these column names to that response shape later without a
# separate translation table.
FEATURE_COLUMNS = [
    # raw
    "temperature_c", "pressure_hpa", "humidity_pct",
    # temporal
    "temp_deviation", "pressure_deviation", "humidity_deviation",
    "temp_roc_1h", "pressure_roc_1h", "humidity_roc_1h",
    "temp_roc_3h", "pressure_roc_3h", "humidity_roc_3h",
    "temp_rolling_std", "pressure_rolling_std", "humidity_rolling_std",
    # cross-parameter (multivariate consistency)
    "temp_humidity_coupling_signal",
    "pressure_inconsistency",
    # spatial (cluster neighbor consistency)
    "spatial_temp_inconsistency",
    "spatial_pressure_inconsistency",
    "spatial_humidity_inconsistency",
    # cyclical time
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]


def _rolling_baseline(series: pd.Series, exclude_mask: pd.Series = None):
    """
    Shifted rolling mean/std -- "normal, as of the reading before this
    one." See LEAKAGE NOTE at module top for why the shift matters.

    exclude_mask (optional, boolean Series aligned to `series`): rows
    where this is True are masked to NaN BEFORE the rolling window sees
    them. This matters most for long-running faults like drift: without
    it, a drift event's own earlier (already-anomalous) readings enter
    the rolling window used to judge its later readings, so the
    baseline drifts along with the fault and progressively understates
    how anomalous it is. Masking them out means the baseline is only
    ever built from confirmed-normal history, even mid-fault. The
    reading being SCORED is never masked, only what's used to define
    "normal" -- see add_temporal_features for how this is applied.
    """
    clean_series = series.mask(exclude_mask) if exclude_mask is not None else series
    prior = clean_series.shift(1)
    roll = prior.rolling(ROLLING_WINDOW_HOURS, min_periods=ROLLING_MIN_PERIODS)
    return roll.mean(), roll.std()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-station rolling deviation + rate-of-change. Assumes df is
    ALREADY one station, sorted by timestamp (see build_feature_matrix,
    which does this via groupby before calling in).

    frozen_value shows up here as a near-perfect signature even before
    the model sees it: rolling_std collapses toward ~0 (the noise
    jitter anomaly_injector.py adds is tiny on purpose) while roc_1h
    also collapses toward ~0 -- two features agreeing "nothing is
    moving" is a much stronger signal than either alone, which is
    exactly why both are included rather than picking one.

    If an `is_anomaly` ground-truth column is present (i.e. this is a
    _labeled.csv being scored for Phase 3 evaluation), known-anomalous
    rows are excluded from the rolling baseline -- see _rolling_baseline.
    In live serving (detect.py), the equivalent exclusion uses the
    model's OWN prior verdicts instead of ground truth, since the
    future obviously isn't known yet -- that's implemented in state.py's
    history buffer, not here.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    exclude_mask = df["is_anomaly"].fillna(False).astype(bool) if "is_anomaly" in df.columns else None

    for col, prefix in [("temperature_c", "temp"), ("pressure_hpa", "pressure"), ("humidity_pct", "humidity")]:
        mean, std = _rolling_baseline(df[col], exclude_mask)
        # std of 0 (or NaN from too little history) would divide-by-zero
        # into inf; treat as "no meaningful deviation info yet" instead.
        safe_std = std.replace(0, np.nan)
        df[f"{prefix}_deviation"] = (df[col] - mean) / safe_std
        df[f"{prefix}_rolling_std"] = std
        df[f"{prefix}_roc_1h"] = df[col].diff(ROC_SHORT_HOURS)
        df[f"{prefix}_roc_3h"] = df[col].diff(ROC_LONG_HOURS)

    return df




def add_cross_parameter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multivariate consistency -- the PS's own named objective, and the
    exact mechanism behind inject_multivariate() in anomaly_injector.py.
    Rather than hand-coding a threshold rule ("flag if temp up AND
    humidity up AND pressure flat"), we hand the model two continuous,
    physically-motivated signals and let the Isolation Forest learn
    what's actually unusual about their joint distribution -- more
    honest ML than baking the threshold in ourselves, and it's what
    lets SHAP later say *how much* each contributed instead of a flat
    yes/no.

    temp_humidity_coupling_signal:
        NOAA's own guidance is a useful caution here: relative humidity
        depends on BOTH temperature and actual atmospheric moisture
        content, not temperature alone -- so a temperature rise
        alongside a humidity rise isn't automatically a sensor fault.
        Real moisture influx (an incoming front, a storm system) can
        genuinely produce that exact combination. This is deliberately
        framed as a soft, continuous SIGNAL for the model to weigh
        alongside everything else -- not a hard rule, and not something
        we ever claim is "physically impossible." We z-score each
        param's 1h rate-of-change against its own rolling std, then
        multiply: opposite-signed changes (the more common case) give a
        negative number; same-signed changes (temp up + humidity up
        together, or both down together) give a positive number. Large
        positive = an atypical coupling worth the model's attention,
        not a certain fault -- SHAP can later show exactly how much
        weight it actually carried in any given verdict, alongside
        spatial and other evidence, rather than us asserting it alone.

    pressure_inconsistency:
        A real weather event large enough to move temperature sharply
        usually shows up in pressure too (a front, a storm system).
        Sharp temp movement with a flat pressure trace is the "sensor
        fault, not real weather" signature. |z(temp_roc)| minus
        |z(pressure_roc)|: large positive = temp moving without
        pressure responding.
    """
    temp_std = df["temperature_c"].rolling(ROLLING_WINDOW_HOURS, min_periods=ROLLING_MIN_PERIODS).std().shift(1)
    pressure_std = df["pressure_hpa"].rolling(ROLLING_WINDOW_HOURS, min_periods=ROLLING_MIN_PERIODS).std().shift(1)
    humidity_std = df["humidity_pct"].rolling(ROLLING_WINDOW_HOURS, min_periods=ROLLING_MIN_PERIODS).std().shift(1)

    z_temp_roc = df["temp_roc_1h"] / temp_std.replace(0, np.nan)
    z_pressure_roc = df["pressure_roc_1h"] / pressure_std.replace(0, np.nan)
    z_humidity_roc = df["humidity_roc_1h"] / humidity_std.replace(0, np.nan)

    df["temp_humidity_coupling_signal"] = z_temp_roc * z_humidity_roc
    df["pressure_inconsistency"] = z_temp_roc.abs() - z_pressure_roc.abs()

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical encodings so the model sees hour 23 and hour 0 as
    adjacent (not maximally far apart, which a raw integer hour would
    imply), and so seasonal position wraps around the year the same
    way. This is what lets the model tell "unusual for 3am" apart from
    "unusual for 3pm" without us hardcoding diurnal rules -- it learns
    the normal daily/seasonal shape directly from the real Open-Meteo
    data instead.
    """
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    doy = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def add_spatial_features(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster-based spatial consistency -- "neighboring stations show
    normal conditions," the half of the PS's worked example that
    temporal-only features can't answer on their own. Only ever
    compares a station against the OTHER stations in its own
    data_fetch.py cluster (e.g. Chennai vs Tambaram/Ambattur/
    Sriperumbudur), never across clusters (never Chennai vs Delhi) --
    cross-cluster comparison would be physically meaningless, which is
    the exact mistake this feature was designed to avoid.

    Mechanism: each station already has its own temp/pressure/humidity
    *_deviation (a self-relative z-score from add_temporal_features).
    That's already climate-normalized -- Chennai's humid-coastal
    "normal" and Bhopal's dry-inland "normal" are both just "0" on
    their own deviation scale. spatial_*_inconsistency is then: this
    station's own deviation minus the MEDIAN deviation of its cluster
    neighbors AT THE SAME TIMESTAMP. Near 0 = the whole neighborhood is
    moving together (a real, shared weather event). Large = this
    station alone is doing something its neighbors aren't -- exactly
    the PS's "AWS reports 55C while neighboring stations show normal
    conditions" scenario.

    Median, not mean, deliberately -- caught by testing on synthetic
    data before this shipped: with a mean, one genuinely faulty station
    drags its *neighbors'* scores too, since it's still sitting inside
    their "neighbor average." A faulty station's own huge deviation
    would pull the mean it contributes to everyone else's comparison
    upward, making the three CLEAN stations next to it look spuriously
    inconsistent as well. Median is robust to a single outlier in a
    small cluster, so one bad station distorts only its own score, not
    the neighbors it's being compared against.

    Requires df to contain ALL stations already merged with metadata's
    cluster_id (see build_feature_matrix) -- must run after temporal
    features are computed per-station but before returning the final
    combined frame.
    """
    df = df.merge(metadata[["station_id", "cluster_id"]], on="station_id", how="left")

    for param in ["temp_deviation", "pressure_deviation", "humidity_deviation"]:
        out_name = "spatial_" + param.replace("_deviation", "_inconsistency")
        # Pivot to (timestamp x station) so each row/timestamp lines
        # every station in a cluster up side by side.
        wide = df.pivot_table(index="timestamp", columns="station_id", values=param)

        result = pd.Series(index=df.index, dtype=float)
        for cluster_id, group in metadata.groupby("cluster_id"):
            cluster_stations = group["station_id"].tolist()
            present = [s for s in cluster_stations if s in wide.columns]
            if len(present) < 2:
                continue  # can't compute "neighbors" with fewer than 2 stations
            cluster_wide = wide[present]

            for station in present:
                own = cluster_wide[station]
                others = [s for s in present if s != station]
                neighbor_median = cluster_wide[others].median(axis=1, skipna=True)
                spatial_dev = own - neighbor_median

                mask = df["station_id"] == station
                # align by timestamp since wide's index is timestamp
                result.loc[mask] = df.loc[mask, "timestamp"].map(spatial_dev).values

        df[out_name] = result

    return df


def build_features_for_latest(history_df: pd.DataFrame) -> pd.Series:
    """
    LIVE-MODE entry point, for detect.py -- NOT for training/eval.

    Takes one station's recent history buffer (kept in-memory by
    state.py, no CSV involved at all) and returns just the feature
    vector for the LATEST reading in it. Reuses the exact same
    add_temporal_features/add_cross_parameter_features/add_time_features
    logic as the batch path -- same math, so training and serving can
    never quietly drift apart into two different feature definitions.

    `history_df` should NOT include an is_anomaly column in live mode
    (we don't have ground truth for readings that haven't happened
    yet); state.py is responsible for only keeping CONFIRMED-clean
    readings in this buffer (i.e. excluding whatever detect.py itself
    previously flagged), which is the live equivalent of the
    exclude_mask logic in _rolling_baseline -- same principle, applied
    causally instead of with hindsight.

    Spatial features are intentionally NOT computed here -- they need
    every station in a cluster in view simultaneously, which detect.py
    handles separately by pulling each cluster-mate's latest reading
    from state.py and calling add_spatial_features on that small
    multi-station slice instead.
    """
    df = add_temporal_features(history_df)
    df = add_cross_parameter_features(df)
    df = add_time_features(df)
    return df.iloc[-1]


def build_feature_matrix(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Takes the combined multi-station dataframe (the
    shape of all_stations.csv, optionally with is_anomaly/fault_type
    ground-truth columns already attached from anomaly_injector.py) and
    the stations_metadata.csv cluster/role table, returns a dataframe
    with every column in FEATURE_COLUMNS added, ready for train.py to
    select FEATURE_COLUMNS as X.

    Ground-truth columns (is_anomaly, fault_type), station_id, and
    timestamp all pass through untouched -- they're not features, but
    train.py / detect.py need them for evaluation and bookkeeping.
    """
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column -- did you pass the raw fetched/validated CSV?")

    missing_raw = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing expected raw columns: {missing_raw}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Temporal + time-of-day features are per-station and don't need
    # other stations in scope, so compute them one station at a time.
    per_station_frames = []
    for station_id, group in df.groupby("station_id", sort=False):
        group = add_temporal_features(group)
        group = add_cross_parameter_features(group)
        group = add_time_features(group)
        per_station_frames.append(group)
    df = pd.concat(per_station_frames, ignore_index=True)

    # Spatial features need every station in view at once (comparing
    # across a cluster at the same timestamp), so this runs on the
    # full recombined frame, not per-station.
    df = add_spatial_features(df, metadata)

    return df


def main():
    stations_path = DATA_DIR / "all_stations.csv"
    metadata_path = DATA_DIR / "stations_metadata.csv"

    if not stations_path.exists() or not metadata_path.exists():
        print(f"Expected {stations_path.name} and {metadata_path.name} in {DATA_DIR} "
              f"(outputs of data_fetch.py -> validate_data.py). Run those first.")
        return

    df = pd.read_csv(stations_path, parse_dates=["timestamp"])
    metadata = pd.read_csv(metadata_path)

    # Reminder baked into the runner itself, not just the docstring:
    # this should be the RAW validated file, not a *_labeled.csv, when
    # the output is headed into train.py.
    if "is_anomaly" in df.columns:
        print("NOTE: input already has is_anomaly/fault_type columns (looks like a "
              "_labeled.csv). That's fine for testing detect.py against known faults, "
              "but train.py must fit only on a RAW un-injected file.\n")

    featured = build_feature_matrix(df, metadata)

    n_total = len(featured)
    n_ready = featured[FEATURE_COLUMNS].notna().all(axis=1).sum()
    print(f"Built {len(FEATURE_COLUMNS)} features for {n_total} rows across "
          f"{df['station_id'].nunique()} stations.")
    print(f"{n_ready}/{n_total} rows have a complete feature vector (no NaNs from "
          f"rolling-window warm-up); the rest are the first ~{ROLLING_WINDOW_HOURS}h "
          f"of each station's history and should be dropped before training.\n")

    sample_cols = ["station_id", "timestamp", "temp_deviation", "pressure_inconsistency",
                   "spatial_temp_inconsistency"]
    print("Sample rows:")
    print(featured[sample_cols].dropna().head(5).to_string(index=False))

    output_path = DATA_DIR / "all_stations_features.csv"
    featured.to_csv(output_path, index=False)
    print(f"\nSaved full feature matrix -> {output_path}")


if __name__ == "__main__":
    main()