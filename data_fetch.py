"""
SkyGuard AI — Phase 1a: Real historical AWS-equivalent data fetch.

Pulls real hourly temperature, pressure, and humidity from Open-Meteo's
free Historical Weather Archive API (no key required) for three Indian
stations, matching the map in the reference dashboard.

Why Open-Meteo: it's the closest free stand-in for "real AWS station
data" the PS asks for. We treat this as our "normal" ground truth,
then inject synthetic faults on top of it in Phase 1b (anomaly_injector.py)
— that script waits until you've done Day 41-43 (Outliers).

Docs: https://open-meteo.com/en/docs/historical-weather-api
"""

import requests
import pandas as pd
from pathlib import Path

# Stations matching the reference dashboard's map (Chennai, Delhi, Mumbai)
STATIONS = {
    "AWS-CHN-024": {"lat": 13.0827, "lon": 80.2707, "name": "Chennai"},
    "AWS-DEL-011": {"lat": 28.6139, "lon": 77.2090, "name": "Delhi"},
    "AWS-MUM-007": {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai"},
}

# Date range: 3 months of hourly data is plenty for training + demo,
# and keeps the download small and fast.
START_DATE = "2025-01-01"
END_DATE = "2025-03-31"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_DIR = Path(__file__).parent / "data"


def fetch_station(station_id: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch hourly temperature (2m), surface pressure, and relative humidity
    (2m) for one station over the configured date range.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,surface_pressure,relative_humidity_2m",
        "timezone": "auto",
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    hourly = payload["hourly"]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature_c": hourly["temperature_2m"],
            "pressure_hpa": hourly["surface_pressure"],
            "humidity_pct": hourly["relative_humidity_2m"],
        }
    )
    df["station_id"] = station_id
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for station_id, info in STATIONS.items():
        print(f"Fetching {station_id} ({info['name']})...")
        df = fetch_station(station_id, info["lat"], info["lon"])
        df["station_name"] = info["name"]

        # Save each station separately (useful for the dashboard's
        # per-station selector) and keep a copy for the combined file.
        station_path = OUTPUT_DIR / f"{station_id}.csv"
        df.to_csv(station_path, index=False)
        print(f"  -> saved {len(df)} rows to {station_path}")

        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = OUTPUT_DIR / "all_stations.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined dataset: {len(combined)} rows -> {combined_path}")
    print("\nSample rows:")
    print(combined.head())
    print("\nBasic stats (this is your 'normal' baseline the model will learn):")
    print(combined[["temperature_c", "pressure_hpa", "humidity_pct"]].describe())


if __name__ == "__main__":
    main()
