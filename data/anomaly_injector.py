"""
SkyGuard AI — Phase 1b: Synthetic anomaly injector.

Takes real, clean historical readings (from data_fetch.py) and
deliberately corrupts a small fraction of them in known, labeled ways.
This gives us ground truth to actually measure the model against later
(precision/recall/F1 in Phase 3) -- something that doesn't exist for
real AWS anomaly data.

Fault types implemented, each tied to a real AWS failure mode named in
the problem statement:
  - spike           : sensor malfunction -> reading jumps far outside
                       physically plausible range for a single instant
  - frozen_value    : communication/sensor fault -> same value repeats
                       for several consecutive readings (Day 42/43 logic:
                       zero variance over a window is itself anomalous)
  - drift           : calibration drift -> slow, growing offset over time
  - dropout         : communication failure -> missing/null reading

Ground truth (is_injected, fault_type) is stored alongside the data so
Phase 3 can compute real accuracy metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# This script lives in the SAME folder as your fetched CSVs
# (backend/data/AWS-*.csv), unlike data_fetch.py which saves INTO a
# ./data subfolder relative to itself. If your CSVs are somewhere else,
# change this to point at that folder directly.
DATA_DIR = Path(__file__).parent

# How much of the data to corrupt. Keep this modest and realistic --
# real sensor faults are rare events, not half your dataset.
INJECTION_RATE = 0.05  # ~5% of rows per station get a fault

RANDOM_SEED = 42


def compute_bounds(series: pd.Series, z_thresh: float = 3.0):
    """
    Day 42 (z-score method): mean +/- z_thresh * std defines the
    'normal' envelope. We use this to make sure injected spikes are
    genuinely, unambiguously outside normal behavior -- not borderline.
    """
    mean = series.mean()
    std = series.std()
    return mean, std, mean + z_thresh * std, mean - z_thresh * std


# Hard physical ceilings that NO fault should cross, because they're
# not just statistically unusual -- they're physically impossible.
# Humidity is the critical one: it's a percentage, so a sensor CANNOT
# genuinely report 173% no matter how broken it is (a real malfunctioning
# sensor saturates/clips at its measurement limits, it doesn't exceed
# them). Pressure gets a generous real-world floor/ceiling too. This is
# NOT applied to inject_fail_low (which intentionally uses an even lower
# fixed sentinel to represent total sensor failure -- a different, valid
# fault archetype) or inject_dropout (NaN has no numeric bound to violate).
HARD_PHYSICAL_LIMITS = {
    "humidity_pct": (0.0, 100.0),
    "pressure_hpa": (800.0, 1100.0),
}


def clip_to_physical_limits(df: pd.DataFrame, column: str, start_idx: int, end_idx: int):
    """Clamps an injected window back within hard physical limits, if the column has any."""
    if column in HARD_PHYSICAL_LIMITS:
        low, high = HARD_PHYSICAL_LIMITS[column]
        df.loc[start_idx:end_idx, column] = df.loc[start_idx:end_idx, column].clip(low, high)


def inject_spike(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Push a single reading far outside its z-score bounds."""
    mean, std, upper, lower = compute_bounds(df[column])
    direction = rng.choice([1, -1])
    # Push 4-6 std deviations out -- unambiguous, not a borderline case.
    magnitude = rng.uniform(4.0, 6.0)
    df.loc[idx, column] = mean + direction * magnitude * std
    clip_to_physical_limits(df, column, idx, idx)
    return "spike"


def inject_frozen(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Repeat the value at idx for the next few rows -- a comms/sensor
    fault where the station keeps reporting stale data. Day 42/43:
    zero variance in a rolling window is itself a strong outlier signal,
    even if the frozen value is individually 'normal'-looking.

    Small sensor noise (+/- tiny jitter) is added on top of the frozen
    value rather than a bit-exact repeat -- real stuck sensors often
    still show a hair of electrical noise, and a perfectly identical
    float repeated N times is an easy giveaway for a model to key on
    for the wrong reason (memorizing "exact duplicate" rather than
    learning "implausibly low variance").
    """
    freeze_length = rng.integers(3, 7)
    frozen_value = df.loc[idx, column]
    end_idx = min(idx + freeze_length, len(df) - 1)
    noise = rng.normal(0, df[column].std() * 0.01, end_idx - idx + 1)
    df.loc[idx:end_idx, column] = frozen_value + noise
    return "frozen_value", idx, end_idx


def inject_drift(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Calibration drift -- a slow, growing offset starting at idx and
    continuing to the end of the window. Unlike a spike, no single
    point looks extreme; only the trend over time reveals it.

    The ramp itself is randomized rather than perfectly linear (small
    per-step noise added), and occasionally curved instead of straight
    -- real calibration drift doesn't follow a textbook-clean line.
    """
    drift_length = rng.integers(20, 50)
    end_idx = min(idx + drift_length, len(df) - 1)
    max_offset = df[column].std() * rng.uniform(1.5, 3.0)
    steps = end_idx - idx + 1

    # Randomly choose a linear or slightly curved (quadratic) ramp shape.
    if rng.choice([True, False]):
        ramp = np.linspace(0, max_offset, steps)
    else:
        ramp = max_offset * (np.linspace(0, 1, steps) ** rng.uniform(1.3, 2.0))

    jitter = rng.normal(0, df[column].std() * 0.03, steps)
    df.loc[idx:end_idx, column] = df.loc[idx:end_idx, column].values + ramp + jitter
    clip_to_physical_limits(df, column, idx, end_idx)
    return "drift", idx, end_idx


def inject_dropout(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """Communication failure -- reading goes missing entirely."""
    df.loc[idx, column] = np.nan
    return "dropout"


def inject_fail_low(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Hardware fail-low -- distinct from a spike. Real sensors often fail
    by clamping to a fixed physical floor or error sentinel rather than
    a random statistical outlier: a humidity sensor stuck reading ~0%,
    a pressure transducer that lost power reading near 0 hPa, a
    temperature probe reporting a fixed out-of-range error value. This
    is a flatline at an implausible LOW bound, held for a short window
    -- different from `frozen` (which freezes at whatever the last real
    reading happened to be) and different from `spike` (a brief
    statistical extreme in either direction).
    """
    fail_length = rng.integers(2, 6)
    end_idx = min(idx + fail_length, len(df) - 1)

    # Physically-motivated failure floors per parameter, with a little
    # jitter so it's not a bit-exact repeated constant.
    floors = {
        "temperature_c": -15.0,   # implausible cold snap for tropical/plains stations
        "pressure_hpa": 50.0,     # near-total transducer failure reading
        "humidity_pct": 0.5,      # sensor stuck near-zero
    }
    floor_value = floors[column]
    noise = rng.normal(0, abs(floor_value) * 0.02 + 0.1, end_idx - idx + 1)
    df.loc[idx:end_idx, column] = floor_value + noise
    return "sensor_fail_low", idx, end_idx


def inject_multivariate(df: pd.DataFrame, idx: int, column: str, rng: np.random.Generator):
    """
    Multivariate inconsistency -- the PS's own example scenario: a
    station reports a temperature spike while pressure/humidity move
    in directions that don't physically make sense together (e.g. temp
    sharply up but humidity ALSO up and pressure barely reacting, which
    real weather physics wouldn't produce together). `column` is ignored
    here since this fault always touches all three parameters at once --
    it's the multivariate case the single-column fault functions can't
    represent.
    """
    window = rng.integers(2, 5)
    end_idx = min(idx + window, len(df) - 1)

    temp_std = df["temperature_c"].std()
    pressure_std = df["pressure_hpa"].std()
    humidity_std = df["humidity_pct"].std()

    # Temperature spikes up sharply (sensor fault)...
    df.loc[idx:end_idx, "temperature_c"] += rng.uniform(4.0, 6.0) * temp_std
    # ...while humidity ALSO rises (physically, a real heat spike should
    # usually correlate with humidity dropping, not rising)...
    df.loc[idx:end_idx, "humidity_pct"] += rng.uniform(2.0, 3.5) * humidity_std
    # ...and pressure barely moves, when a real weather event of this
    # magnitude would typically show a pressure change too.
    df.loc[idx:end_idx, "pressure_hpa"] += rng.normal(0, pressure_std * 0.1, end_idx - idx + 1)

    clip_to_physical_limits(df, "humidity_pct", idx, end_idx)
    clip_to_physical_limits(df, "pressure_hpa", idx, end_idx)

    return "multivariate_inconsistency", idx, end_idx


def inject_anomalies(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Walks through one station's dataframe and injects labeled faults
    at random locations across temperature/pressure/humidity columns.
    Returns a new dataframe with two extra columns: is_anomaly (bool)
    and fault_type (str or None) -- this is the ground truth label set.
    """
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    df["is_anomaly"] = False
    df["fault_type"] = None

    columns = ["temperature_c", "pressure_hpa", "humidity_pct"]
    df[columns] = df[columns].astype(float)
    n_rows = len(df)

    # IMPORTANT: this is a budget on total AFFECTED ROWS, not on the
    # number of injection events. A single drift/frozen event can span
    # dozens of rows, so counting events (the earlier, buggy version)
    # let a handful of events balloon into ~30% of the dataset flagged
    # anomalous -- unrealistic for real sensor fault rates. We now keep
    # injecting one event at a time and stop once the row budget is hit.
    target_anomalous_rows = int(n_rows * INJECTION_RATE)
    fault_functions = [inject_spike, inject_frozen, inject_drift, inject_dropout, inject_multivariate, inject_fail_low]

    # Split the row budget evenly across fault types rather than picking
    # randomly until the total budget runs out -- otherwise a couple of
    # long drift/frozen events can exhaust the whole budget before the
    # other fault types ever get a turn (this happened in testing: one
    # run came out ~99% drift, 1 spike, zero frozen/dropout). Even
    # representation matters because Phase 3 evaluates precision/recall
    # per fault type, not just overall.
    per_type_budget = target_anomalous_rows // len(fault_functions)

    already_used = set()

    for fault_fn in fault_functions:
        rows_this_type = 0
        attempts = 0
        max_attempts = n_rows  # safety valve against infinite loop

        while rows_this_type < per_type_budget and attempts < max_attempts:
            attempts += 1
            idx = int(rng.integers(10, n_rows - 60))
            if idx in already_used:
                continue

            column = rng.choice(columns)
            result = fault_fn(df, idx, column, rng)

            if isinstance(result, tuple) and len(result) == 3:
                fault_type, start, end = result
                df.loc[start:end, "is_anomaly"] = True
                df.loc[start:end, "fault_type"] = fault_type
                new_rows = set(range(start, end + 1)) - already_used
                already_used.update(new_rows)
                rows_this_type += len(new_rows)
            else:
                fault_type = result
                df.loc[idx, "is_anomaly"] = True
                df.loc[idx, "fault_type"] = fault_type
                already_used.add(idx)
                rows_this_type += 1

    return df


def main():
    # Only a RANDOM SUBSET of stations get faults -- not all 20.
    # If every station were corrupted simultaneously, there'd be no
    # clean neighbor left to compare against, which defeats spatial
    # consistency before it's even built (a neighbor comparison is only
    # meaningful if the neighbor is actually trustworthy). Real sensor
    # networks also don't have every unit fail at once -- a handful of
    # faulty stations among many healthy ones is the realistic picture.
    #
    # Separate seed from RANDOM_SEED (which controls fault content/
    # placement) so "which stations fail" and "what the fault looks
    # like" are independently reproducible.
    STATION_SELECTION_SEED = 7
    N_FAULTY_STATIONS = 3

    station_files = sorted(
        p for p in DATA_DIR.glob("AWS-*.csv") if "_labeled" not in p.name
    )

    if len(station_files) < N_FAULTY_STATIONS:
        print(f"Only {len(station_files)} station CSVs found -- check DATA_DIR / that data_fetch.py has run.")

    selection_rng = np.random.default_rng(STATION_SELECTION_SEED)
    faulty_indices = selection_rng.choice(
        len(station_files), size=min(N_FAULTY_STATIONS, len(station_files)), replace=False
    )
    faulty_files = {station_files[i] for i in faulty_indices}

    print(f"Selected {len(faulty_files)} of {len(station_files)} stations to receive injected faults:")
    for f in faulty_files:
        print(f"  -> {f.stem}")
    print(f"Remaining {len(station_files) - len(faulty_files)} stations stay clean (real data only) -- these are your trustworthy spatial-consistency neighbors.\n")

    for csv_path in station_files:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])

        if csv_path in faulty_files:
            print(f"Injecting anomalies into {csv_path.name}...")
            # Each station gets its OWN seed, derived from the global
            # RANDOM_SEED plus its position in the sorted file list.
            # Without this, every faulty station picked the exact same
            # relative row pattern and fault-type mix (confirmed in
            # testing -- 3 stations, identical 60/60/60 breakdown),
            # which isn't realistic: independent sensor failures
            # shouldn't sync up like that.
            station_seed = RANDOM_SEED + station_files.index(csv_path)
            injected = inject_anomalies(df, seed=station_seed)
            n_anomalies = injected["is_anomaly"].sum()
            print(f"  -> {n_anomalies} of {len(injected)} rows flagged as ground-truth anomalies")
            print(f"  -> fault type breakdown:\n{injected['fault_type'].value_counts()}\n")
        else:
            # Clean station: still write a "_labeled" file for schema
            # consistency downstream (features.py can always expect
            # is_anomaly/fault_type columns to exist), just with
            # everything correctly labeled as non-anomalous.
            injected = df.copy()
            injected["is_anomaly"] = False
            injected["fault_type"] = None
            print(f"{csv_path.name}: left clean (0 anomalies) -- serves as a trustworthy neighbor\n")

        output_path = DATA_DIR / csv_path.name.replace(".csv", "_labeled.csv")
        injected.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()