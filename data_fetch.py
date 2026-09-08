"""
SkyGuard AI — Phase 1a: Real historical AWS-equivalent data fetch.

Pulls real hourly temperature, pressure, and humidity from Open-Meteo's
free Historical Weather Archive API (no key required).

UPDATED: now pulls 5 CENTER stations (Delhi, Mumbai, Chennai, Kolkata,
Bhopal) each with 3 real, physically nearby NEIGHBOR towns (~15-50km
away). This is what makes genuine spatial-consistency checking possible
later: comparing Chennai against Delhi is meaningless (different
climates entirely), but comparing Chennai against Tambaram/Ambattur/
Sriperumbudur -- towns close enough to plausibly share the same local
weather event -- is exactly what the PS's example use case describes
("neighboring stations show normal conditions").

Docs: https://open-meteo.com/en/docs/historical-weather-api
"""

import requests
import pandas as pd
from pathlib import Path

# Each cluster = one center metro + 3 real nearby towns (~15-50km away).
# This is the actual "neighboring stations" set for spatial-consistency
# features later -- NOT to be confused with cross-cluster comparisons
# (Chennai vs Delhi), which stay separate for map/UI diversity only.
CLUSTERS = {
    "CHN": {
        "center": {"station_id": "AWS-CHN-024", "name": "Chennai", "lat": 13.0827, "lon": 80.2707},
        "neighbors": [
            {"station_id": "AWS-CHN-101", "name": "Tambaram", "lat": 12.9249, "lon": 80.1000},
            {"station_id": "AWS-CHN-102", "name": "Ambattur", "lat": 13.1143, "lon": 80.1548},
            {"station_id": "AWS-CHN-103", "name": "Sriperumbudur", "lat": 12.9675, "lon": 79.9430},
        ],
    },
    "DEL": {
        "center": {"station_id": "AWS-DEL-011", "name": "Delhi", "lat": 28.6139, "lon": 77.2090},
        "neighbors": [
            {"station_id": "AWS-DEL-101", "name": "Noida", "lat": 28.5355, "lon": 77.3910},
            {"station_id": "AWS-DEL-102", "name": "Gurugram", "lat": 28.4595, "lon": 77.0266},
            {"station_id": "AWS-DEL-103", "name": "Ghaziabad", "lat": 28.6692, "lon": 77.4538},
        ],
    },
    "MUM": {
        "center": {"station_id": "AWS-MUM-007", "name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        "neighbors": [
            {"station_id": "AWS-MUM-101", "name": "Thane", "lat": 19.2183, "lon": 72.9781},
            {"station_id": "AWS-MUM-102", "name": "Navi Mumbai", "lat": 19.0330, "lon": 73.0297},
            {"station_id": "AWS-MUM-103", "name": "Kalyan", "lat": 19.2403, "lon": 73.1305},
        ],
    },
    "KOL": {
        "center": {"station_id": "AWS-KOL-015", "name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
        "neighbors": [
            {"station_id": "AWS-KOL-101", "name": "Howrah", "lat": 22.5958, "lon": 88.2636},
            {"station_id": "AWS-KOL-102", "name": "Bidhannagar", "lat": 22.5697, "lon": 88.4171},
            {"station_id": "AWS-KOL-103", "name": "Barrackpore", "lat": 22.7645, "lon": 88.3792},
        ],
    },
    "BHO": {
        "center": {"station_id": "AWS-BHO-030", "name": "Bhopal", "lat": 23.2599, "lon": 77.4126},
        "neighbors": [
            {"station_id": "AWS-BHO-101", "name": "Sehore", "lat": 23.2032, "lon": 77.0844},
            {"station_id": "AWS-BHO-102", "name": "Vidisha", "lat": 23.5251, "lon": 77.8081},
            {"station_id": "AWS-BHO-103", "name": "Raisen", "lat": 23.3315, "lon": 77.7899},
        ],
    },
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
    metadata_rows = []

    for cluster_id, cluster in CLUSTERS.items():
        stations_in_cluster = [
            {**cluster["center"], "role": "center"},
        ] + [{**n, "role": "neighbor"} for n in cluster["neighbors"]]

        for info in stations_in_cluster:
            station_id = info["station_id"]
            print(f"Fetching {station_id} ({info['name']}, cluster {cluster_id}, {info['role']})...")
            df = fetch_station(station_id, info["lat"], info["lon"])
            df["station_name"] = info["name"]
            df["cluster_id"] = cluster_id
            df["role"] = info["role"]

            station_path = OUTPUT_DIR / f"{station_id}.csv"
            df.to_csv(station_path, index=False)
            print(f"  -> saved {len(df)} rows to {station_path}")

            all_frames.append(df)
            metadata_rows.append(
                {
                    "station_id": station_id,
                    "name": info["name"],
                    "lat": info["lat"],
                    "lon": info["lon"],
                    "cluster_id": cluster_id,
                    "role": info["role"],
                }
            )

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = OUTPUT_DIR / "all_stations.csv"
    combined.to_csv(combined_path, index=False)

    # Cluster/role metadata -- features.py needs this to know which
    # stations are genuine "neighbors" for spatial-consistency checks
    # (only compare within a cluster, never across clusters).
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = OUTPUT_DIR / "stations_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\n{len(metadata_rows)} stations across {len(CLUSTERS)} clusters -> {combined_path}")
    print(f"Cluster/role metadata -> {metadata_path}")
    print("\nSample rows:")
    print(combined.head())
    print("\nBasic stats per parameter (across ALL stations combined):")
    print(combined[["temperature_c", "pressure_hpa", "humidity_pct"]].describe())


if __name__ == "__main__":
    main()