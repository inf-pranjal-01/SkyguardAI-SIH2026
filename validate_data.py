"""
SkyGuard AI — Phase 1a.5: Data quality validation.

Runs AFTER data_fetch.py, BEFORE anomaly_injector.py.

Open-Meteo's historical archive is reanalysis data (model-blended, not
raw sensor telemetry), so it's already smoothed and quality-controlled
upstream -- genuine sensor-style glitches (spikes, frozen values,
dropouts) are very unlikely to already be in it. That's actually
convenient: we want "normal" training data to genuinely mean normal,
since we inject all fault behavior ourselves in a controlled way.

What CAN still slip in, and what this script actually checks for:
  - missing/NaN readings (gaps in the API response)
  - duplicate timestamps (known edge case in some reanalysis sources)
  - physically impossible values (interpolation/boundary artifacts)

This is a validation GATE, not a silent auto-fixer: it reports what it
finds and fixes only the narrow, safe cases (small gaps via
interpolation, exact duplicate rows via dropping). Anything larger or
ambiguous is flagged loudly rather than silently patched, because
silently "cleaning" data you're about to call ground-truth "normal" is
exactly the kind of thing that should require a human to look at it.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Physically implausible bounds for these 5 Indian city clusters.
# Deliberately generous -- these are sanity limits, not climate norms,
# so they only catch genuinely broken values, not just unusual weather.
PHYSICAL_BOUNDS = {
    "temperature_c": (-10.0, 55.0),
    "pressure_hpa": (850.0, 1080.0),
    "humidity_pct": (0.0, 100.0),
}

MAX_INTERPOLATABLE_GAP = 3  # only auto-fill gaps up to 3 consecutive hours


def validate_station(df: pd.DataFrame, station_id: str) -> tuple[pd.DataFrame, dict]:
    """
    Validates and lightly repairs one station's dataframe. Returns the
    (possibly repaired) dataframe plus a report dict describing what
    was found and what was done about it.
    """
    report = {"station_id": station_id, "issues": []}
    df = df.copy()

    # 1. Duplicate timestamps -- keep first occurrence, drop the rest.
    n_before = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        report["issues"].append(f"{n_dupes} duplicate timestamp rows dropped")

    # 2. Missing timestamps entirely (gaps in the hourly sequence).
    df = df.sort_values("timestamp").reset_index(drop=True)
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    missing_count = len(full_range) - len(df)
    if missing_count > 0:
        report["issues"].append(f"{missing_count} missing hourly timestamps in sequence (reindexed, left as NaN for step 3)")
        df = df.set_index("timestamp").reindex(full_range).rename_axis("timestamp").reset_index()

    # 3. NaN values -- interpolate only short gaps; flag long ones loudly.
    for col in ["temperature_c", "pressure_hpa", "humidity_pct"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            # Identify max consecutive-NaN run length for this column.
            is_na = df[col].isna()
            run_lengths = is_na.groupby((~is_na).cumsum()).sum()
            max_run = run_lengths.max() if len(run_lengths) else 0

            if max_run <= MAX_INTERPOLATABLE_GAP:
                df[col] = df[col].interpolate(method="linear", limit=MAX_INTERPOLATABLE_GAP)
                report["issues"].append(f"{col}: {n_nan} NaNs, longest gap {max_run}h -> linearly interpolated")
            else:
                report["issues"].append(
                    f"{col}: {n_nan} NaNs, longest gap {max_run}h -- EXCEEDS auto-fill threshold, "
                    f"left as NaN, needs manual review"
                )

    # 4. Physically impossible values -- flag, don't silently alter.
    # (We don't auto-correct these because a value outside physical
    # bounds in "clean" baseline data is unusual enough that it deserves
    # a human look, not an automatic guess at the "right" value.)
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        out_of_bounds = df[(df[col] < low) | (df[col] > high)]
        if len(out_of_bounds) > 0:
            report["issues"].append(
                f"{col}: {len(out_of_bounds)} rows outside physical bounds ({low}, {high}) -- NEEDS MANUAL REVIEW, not auto-fixed"
            )

    if not report["issues"]:
        report["issues"].append("clean -- no issues found")

    return df, report


def main():
    reports = []

    for csv_path in sorted(DATA_DIR.glob("AWS-*.csv")):
        if "_labeled" in csv_path.name:
            continue  # skip already-injected files if this is re-run later

        station_id = csv_path.stem
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        cleaned_df, report = validate_station(df, station_id)
        reports.append(report)

        cleaned_df.to_csv(csv_path, index=False)  # overwrite with validated version

    print(f"{'Station':<16} Issues")
    print("-" * 70)
    for r in reports:
        for issue in r["issues"]:
            print(f"{r['station_id']:<16} {issue}")

    needs_review = [r for r in reports if any("MANUAL REVIEW" in i for i in r["issues"])]
    if needs_review:
        print(f"\n⚠ {len(needs_review)} station(s) have issues that need your manual review before training.")
    else:
        print(f"\n✅ All {len(reports)} stations passed validation (or were safely auto-repaired).")


if __name__ == "__main__":
    main()
