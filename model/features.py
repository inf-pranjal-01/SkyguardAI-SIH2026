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

CALLER CONTRACT ON `is_anomaly` -- READ BEFORE PASSING A LABELED FRAME:
add_temporal_features() masks any row where is_anomaly==True OUT of its
OWN rolling baseline window before computing rolling_std/roc/deviation
for it (see _rolling_baseline). That's intentional and correct for ONE
specific purpose: stopping an ongoing fault from dragging down the
baseline used to judge ITS OWN later readings. It is NOT something
train.py or detect.py ever exercises -- train.py fits on raw data with
no is_anomaly column at all, and detect.py's live path
(build_features_for_latest) never sees one either, since state.py's
history buffer only holds confirmed-clean readings (the causal
equivalent, applied without hindsight). If you're computing features
for EVALUATION against a _labeled.csv, strip is_anomaly/fault_type
before featurizing and reattach them afterward by a station_id+
timestamp merge (never by row position -- see build_feature_matrix's
ROW ORDER NOTE). See evaluate.py's module docstring for the bug this
caused when it wasn't done.

FOUR MODEL FEATURE GROUPS (all in FEATURE_COLUMNS, all seen by
train.py -- changing any of these requires a retrain):

  1. Raw values              -- the baseline signal itself.
  2. Temporal features       -- rate-of-change + deviation from each
                                 station's own recent rolling baseline
                                 (ROLLING_WINDOW_HOURS=48, one diurnal
                                 cycle).
  3. Cross-parameter features -- "multivariate consistency analysis",
                                 a named PS objective.
  4. Spatial features         -- cluster-based neighbor consistency.

PLUS a fifth group, RULE-ONLY signals (added below): short-horizon raw-
reading comparisons used exclusively by the deterministic rule layer in
detect.py/evaluate.py, NEVER added to FEATURE_COLUMNS and NEVER seen by
the trained model. This distinction exists because of a diagnosed real
problem: ROLLING_WINDOW_HOURS=48 is correctly sized for diurnal-cycle
normalization, but it's much longer than a typical injected fault
(observed frozen faults ~3-6h, drift faults tens of hours) -- a 48h
window dilutes a short fault with 40+ hours of surrounding normal data,
so *_rolling_std/roc_1h rarely swing far enough to trip a frozen/drift
rule built on them. Rather than shrinking ROLLING_WINDOW_HOURS (which
would silently invalidate the already-trained model, since that's a
training-time constant baked into its learned splits), these are
independent, short-horizon signals that only the rule layer reads.
Their thresholds are calibrated empirically from real clean training
data inside train.py (see calibrate_rule_thresholds) and stored in the
model artifact -- not hardcoded here, since what counts as
"suspiciously static" or "too much 24h change" is scale- and
station-dependent, not a single guessable constant.

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

# Rolling window for each station's own recent baseline. 48h = two full
# diurnal cycles, so "recent normal" already accounts for the daily
# temperature/pressure/humidity swing instead of falsely flagging
# "it's just afternoon" as a deviation. min_periods keeps the first
# ~day of each station's data from producing all-NaN features.
ROLLING_WINDOW_HOURS = 48
ROLLING_MIN_PERIODS = 6
VOLATILITY_BASELINE_WINDOW_HOURS = 24 * 30  # ~30 days: "this station's own typical variability"
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

    # temporal
    "temp_deviation", "pressure_deviation", "humidity_deviation",
    "temp_roc_1h", "pressure_roc_1h", "humidity_roc_1h",
    "temp_roc_3h", "pressure_roc_3h", "humidity_roc_3h",
  "temp_volatility_z", "pressure_volatility_z", "humidity_volatility_z",
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

# Shared (raw_column -> features.py prefix) mapping, used by BOTH the
# rule-only signals below AND by detect.py/evaluate.py's rule checks.
# Keep this as the single source of truth for the mapping -- the
# earlier bug (temperature_c's frozen check silently never firing) was
# caused by a second, ad hoc string-strip derivation of this same
# mapping drifting out of sync with the real column names.
RULE_ONLY_PREFIXES = [
    ("temperature_c", "temp"),
    ("pressure_hpa", "pressure"),
    ("humidity_pct", "humidity"),
]

# Lookback for the drift signal, in hours. Deliberately independent of
# ROLLING_WINDOW_HOURS -- see module docstring.
DRIFT_LOOKBACK_HOURS = 24


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

    If an `is_anomaly` ground-truth column is present (i.e. this is a
    _labeled.csv being scored for Phase 3 evaluation), known-anomalous
    rows are excluded from the rolling baseline -- see _rolling_baseline.
    In live serving (detect.py), the equivalent exclusion uses the
    model's OWN prior verdicts instead of ground truth, since the
    future obviously isn't known yet -- that's implemented in state.py's
    history buffer, not here.

    IMPORTANT for evaluation callers: this exclusion makes an ongoing
    fault's OWN rows use a PRE-fault baseline for their entire
    duration, since the fault's own rows never enter their own rolling
    window. Evaluation code must strip is_anomaly/fault_type before
    calling build_feature_matrix. See the module-level CALLER CONTRACT
    note and evaluate.py's docstring.
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
        df[f"{prefix}_rolling_mean"] = mean
        long_baseline = std.rolling(VOLATILITY_BASELINE_WINDOW_HOURS, min_periods=ROLLING_WINDOW_HOURS).median()
        long_spread = std.rolling(VOLATILITY_BASELINE_WINDOW_HOURS, min_periods=ROLLING_WINDOW_HOURS).std()
        df[f"{prefix}_volatility_z"] = (std - long_baseline) / long_spread.replace(0, np.nan)
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
    what's actually unusual about their joint distribution.

    temp_humidity_coupling_signal:
        NOAA's own guidance is a useful caution here: relative humidity
        depends on BOTH temperature and actual atmospheric moisture
        content, not temperature alone -- so a temperature rise
        alongside a humidity rise isn't automatically a sensor fault.
        This is deliberately a soft, continuous SIGNAL for the model to
        weigh alongside everything else. We z-score each param's 1h
        rate-of-change against its own rolling std, then multiply:
        opposite-signed changes give a negative number; same-signed
        changes give a positive number.

    pressure_inconsistency:
        A real weather event large enough to move temperature sharply
        usually shows up in pressure too. Sharp temp movement with a
        flat pressure trace is the "sensor fault, not real weather"
        signature. |z(temp_roc)| minus |z(pressure_roc)|.
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
    adjacent, and seasonal position wraps around the year the same way.
    """
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    doy = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def add_rule_only_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-station, adds two RULE-LAYER-ONLY signal families -- never
    added to FEATURE_COLUMNS, never seen by the trained Isolation
    Forest. See module docstring for why these exist alongside the
    48h-windowed temporal features instead of replacing them.

    `{prefix}_consec_diff`: absolute difference between this reading
    and the immediately-previous raw reading (NOT a rolling window).
    A genuinely frozen sensor repeats near-identical values reading-to-
    reading (the injector's jitter is tiny by design), which this
    catches directly and immediately, unlike a windowed std that needs
    the fault to occupy most of a 48h window before it moves much.

    `{prefix}_{DRIFT_LOOKBACK_HOURS}h_delta`: raw value minus its value
    DRIFT_LOOKBACK_HOURS ago. A slow drift moves this steadily away
    from 0 over the fault's duration even though each individual 1h
    step (roc_1h) looks unremarkable on its own -- the "small steps,
    large cumulative change" signature a short-horizon rate-of-change
    feature can't see by itself.

    Thresholds for both are NOT defined here -- see
    calibrate_rule_thresholds, called once in train.py against real
    clean data and stored in the model artifact.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col, prefix in RULE_ONLY_PREFIXES:
        df[f"{prefix}_consec_diff"] = df[col].diff(1).abs()
        df[f"{prefix}_{DRIFT_LOOKBACK_HOURS}h_delta"] = df[col] - df[col].shift(DRIFT_LOOKBACK_HOURS)
    return df


def get_threshold(thresholds: dict, rule_type: str, prefix: str, station_id: str) -> float:
    """
    Single source of truth for threshold lookup -- used by BOTH
    detect.py and evaluate.py so they can never drift apart on how a
    per-station threshold falls back to the pooled global value.
    """
    per_station = thresholds[rule_type][prefix]
    return per_station.get(station_id, per_station["__global__"])


def calibrate_rule_thresholds(featured_clean: pd.DataFrame) -> dict:
    """
    PER-STATION thresholds, not one pooled global value -- diagnosed
    directly against real 20-station eval output: temp_rolling_std
    ranges 2.38 (AWS-CHN-024) to 5.42 (AWS-MUM-101) across stations,
    more than 2x. A single global percentile is necessarily
    miscalibrated for whichever end of that range it doesn't fit --
    the actual cause of frozen's 302 clean-row false positives being
    spread broadly across nearly every station instead of concentrated
    in one or two.

    Each threshold is now {station_id: value, "__global__": value} --
    the global entry is the same pooled-percentile calculation as
    before, kept as a fallback for any station with too little clean
    data to calibrate its own threshold reliably (min_rows guard below)
    or any station missing entirely at calibration time. See
    get_threshold() for the lookup that applies this fallback.

    frozen[prefix]: 1st percentile of NONZERO consec_diff, per station.
    drift[prefix]: 99th percentile of Nh_delta magnitude, per station.
    roc_small[prefix]: 20th percentile of roc_1h magnitude, per station.
    spike[prefix]: 99.9th percentile of *_deviation magnitude, per station.
    (Percentile choices carried over unchanged from the pooled version --
    only the GROUPING changed here, not the statistical target each
    rule is calibrated against.)
    """
    MIN_ROWS_FOR_PER_STATION = 200  # below this, a station's own quantile is too noisy to trust

    thresholds = {"frozen": {}, "drift": {}, "roc_small": {}, "spike": {}}

    def _calibrate(
        rule_key: str,
        value_fn,
        quantile: float,
        per_station: bool = True,
    ):
        per_station_vals = {}

        if per_station:
            for station_id, group in featured_clean.groupby("station_id"):
                vals = value_fn(group)
                if len(vals) >= MIN_ROWS_FOR_PER_STATION:
                    per_station_vals[station_id] = float(
                        vals.quantile(quantile)
                    )

        global_val = float(
            value_fn(featured_clean).quantile(quantile)
        )

        thresholds[rule_key][prefix] = {
            "__global__": global_val,
            **per_station_vals,
        }

    for _, prefix in RULE_ONLY_PREFIXES:
        consec_col = f"{prefix}_consec_diff"
        delta_col = f"{prefix}_{DRIFT_LOOKBACK_HOURS}h_delta"
        roc_col = f"{prefix}_roc_1h"
        dev_col = f"{prefix}_deviation"

        # Frozen: GLOBAL ONLY.
        # The 1st percentile of nonzero consec_diff is determined
        # by Open-Meteo's reporting precision, not station climate.
        _calibrate(
            "frozen",
            lambda g: g[consec_col][g[consec_col] > 0],
            0.01,
            per_station=False,
        )

        # Drift: keep per-station calibration.
        _calibrate(
            "drift",
            lambda g: g[delta_col].abs(),
            0.99,
        )

        # Small ROC: keep per-station calibration.
        _calibrate(
            "roc_small",
            lambda g: g[roc_col].abs(),
            0.20,
        )

        # Spike: GLOBAL ONLY.
        # 99.9th percentile is too extreme to estimate reliably
        # from only ~2,000 rows per station.
        _calibrate(
            "spike",
            lambda g: g[dev_col].abs(),
            0.999,
            per_station=False,
        )

    return thresholds


def add_spatial_features(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster-based spatial consistency -- "neighboring stations show
    normal conditions." Only ever compares a station against the OTHER
    stations in its own cluster, never across clusters.

    Mechanism: spatial_*_inconsistency is this station's own
    *_deviation minus the MEDIAN deviation of its cluster neighbors AT
    THE SAME TIMESTAMP. Near 0 = the whole neighborhood is moving
    together (a real, shared weather event). Large = this station
    alone is doing something its neighbors aren't.

    Median, not mean, deliberately -- robust to a single outlier in a
    small cluster, so one bad station distorts only its own score, not
    the neighbors it's being compared against.

    REQUIRES df to contain enough of a cluster's stations in the SAME
    call to compute anything real. With fewer than 2 present stations
    in a cluster, this silently produces NaN for that cluster (see the
    `len(present) < 2` guard below), which callers then typically
    fill with a neutral 0.0. THIS IS EXACTLY THE BUG THAT WAS FOUND IN
    evaluate.py: processing one labeled file (one station) at a time
    means every cluster only ever has 1 station present, so spatial
    features were ALWAYS falling back to the neutral 0.0 for every row
    of every evaluation run -- a genuine mismatch against train.py,
    which computes real non-zero spatial values because it runs on the
    full multi-station file. See evaluate.py's module docstring.
    """
    df = df.merge(metadata[["station_id", "cluster_id"]], on="station_id", how="left")

    for param in ["temp_deviation", "pressure_deviation", "humidity_deviation"]:
        out_name = "spatial_" + param.replace("_deviation", "_inconsistency")
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

                 # Exclude known-anomalous rows from the baseline/spread
                # denominator -- same exclude_mask principle as
                # _rolling_baseline. Without this, a fault station's own
                # historical fault period inflates ITS OWN spread,
                # diluting the z-score for every future row scored
                # against it, including genuine anomalies (this is what
                # dropped drift/spike recall last round).
                if "is_anomaly" in df.columns:
                    station_rows = df["station_id"] == station
                    station_exclude = (
                        df.loc[station_rows].set_index("timestamp")["is_anomaly"]
                        .fillna(False).astype(bool)
                        .reindex(spatial_dev.index).fillna(False)
                    )
                    clean_spatial_dev = spatial_dev.mask(station_exclude)
                else:
                    clean_spatial_dev = spatial_dev

                baseline = clean_spatial_dev.shift(1).rolling(VOLATILITY_BASELINE_WINDOW_HOURS, min_periods=ROLLING_WINDOW_HOURS).median()
                spread = clean_spatial_dev.shift(1).rolling(VOLATILITY_BASELINE_WINDOW_HOURS, min_periods=ROLLING_WINDOW_HOURS).std()
                spatial_dev_z = (spatial_dev - baseline) / spread.replace(0, np.nan)

                mask = df["station_id"] == station
                result.loc[mask] = df.loc[mask, "timestamp"].map(spatial_dev_z).values

        df[out_name] = result

    return df


def build_features_for_latest(history_df: pd.DataFrame) -> pd.Series:
    """
    LIVE-MODE entry point, for detect.py -- NOT for training/eval.

    Takes one station's recent history buffer (kept in-memory by
    state.py, no CSV involved at all) and returns just the feature
    vector for the LATEST reading in it. Includes rule-only signals too
    (consec_diff / Nh_delta), since detect.py's rule layer needs them
    from the same computation path as the model features -- same math,
    so training/serving/rule-checking never quietly drift apart.

    `history_df` should NOT include an is_anomaly column in live mode.

    Spatial features are intentionally NOT computed here -- they need
    every station in a cluster in view simultaneously; detect.py
    handles that separately via cluster_neighbor_buffers.
    """
    df = add_temporal_features(history_df)
    df = add_cross_parameter_features(df)
    df = add_time_features(df)
    df = add_rule_only_signals(df)
    return df.iloc[-1]


def build_rule_signals_recent(history_df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    LIVE-MODE helper for the rule layer's PERSISTENCE requirement (see
    detect.py's frozen-value check): returns the last `n` rows with
    consec_diff/Nh_delta computed, so detect.py can check "did this
    condition hold for the last N consecutive readings of THIS
    station" without re-deriving the heavier rolling temporal features
    (which persistence doesn't need) or duplicating
    add_rule_only_signals' logic separately.

    n=2 by default: a single very-still reading alone doesn't reliably
    separate "frozen sensor" from "genuinely calm weather" -- diagnosed
    empirically (see calibrate_rule_thresholds' PROVISIONAL note) --
    but two consecutive readings both under threshold is a much
    stronger signal, at the cost of one extra reading's worth of
    detection latency.
    """
    df = history_df.sort_values("timestamp").reset_index(drop=True)
    df = add_rule_only_signals(df)
    return df.tail(n)


def build_feature_matrix(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point. Takes the combined multi-station dataframe (the
    shape of all_stations.csv, optionally with is_anomaly/fault_type
    ground-truth columns already attached from anomaly_injector.py) and
    the stations_metadata.csv cluster/role table, returns a dataframe
    with every column in FEATURE_COLUMNS added (plus the rule-only
    signal columns), ready for train.py to select FEATURE_COLUMNS as X
    and calibrate_rule_thresholds to calibrate against.

    Ground-truth columns (is_anomaly, fault_type), station_id, and
    timestamp all pass through untouched. See the module-level CALLER
    CONTRACT note before passing is_anomaly in for evaluation purposes.

    ROW ORDER NOTE: internally this groups by station_id (sort=False)
    and concatenates each station's rows back with ignore_index=True.
    The returned frame's row order does NOT match the input df's.
    Callers that need to reattach any side-channel data after the fact
    must join on (station_id, timestamp), never on row position/index.

    SPATIAL FEATURE NOTE: pass in df containing every station whose
    cluster context you actually want -- calling this once per station
    silently degrades every spatial_*_inconsistency feature to a
    neutral fallback. See add_spatial_features' docstring.
    """
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column -- did you pass the raw fetched/validated CSV?")

    missing_raw = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing expected raw columns: {missing_raw}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    per_station_frames = []
    for station_id, group in df.groupby("station_id", sort=False):
        group = add_temporal_features(group)
        group = add_cross_parameter_features(group)
        group = add_time_features(group)
        group = add_rule_only_signals(group)
        per_station_frames.append(group)
    df = pd.concat(per_station_frames, ignore_index=True)

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

    if "is_anomaly" in df.columns:
        print("NOTE: input already has is_anomaly/fault_type columns (looks like a "
              "_labeled.csv). That's fine for testing detect.py against known faults, "
              "but train.py must fit only on a RAW un-injected file.\n")

    featured = build_feature_matrix(df, metadata)

    n_total = len(featured)
    n_ready = featured[FEATURE_COLUMNS].notna().all(axis=1).sum()
    print(f"Built {len(FEATURE_COLUMNS)} model features for {n_total} rows across "
          f"{df['station_id'].nunique()} stations.")
    print(f"{n_ready}/{n_total} rows have a complete feature vector (no NaNs from "
          f"rolling-window warm-up); the rest are the first ~{ROLLING_WINDOW_HOURS}h "
          f"of each station's history and should be dropped before training.\n")

    sample_cols = ["station_id", "timestamp", "temp_deviation", "pressure_inconsistency",
                   "spatial_temp_inconsistency", "temp_consec_diff", f"temp_{DRIFT_LOOKBACK_HOURS}h_delta"]
    print("Sample rows:")
    print(featured[sample_cols].dropna().head(5).to_string(index=False))

    output_path = DATA_DIR / "all_stations_features.csv"
    featured.to_csv(output_path, index=False)
    print(f"\nSaved full feature matrix -> {output_path}")


if __name__ == "__main__":
    main()