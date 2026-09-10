"""
SkyGuard AI — shared config.

Single place for severity/health thresholds so detect.py and (later)
main.py never define these independently and drift out of sync.
"""

# Anomaly score (0-100 scale, higher = more anomalous) -> severity bucket.
SEVERITY_THRESHOLDS = {
    "critical": 90,
    "high": 70,
    "medium": 55,
    # anything below "medium" is "low"
}

# --- Sensor health / circuit-breaker (the "mark this sensor OFFLINE"
# mechanism) ---

# How many recent verdicts to look at per station when deciding health.
HEALTH_WINDOW_SIZE = 10

# If at least this many of the last HEALTH_WINDOW_SIZE readings were
# anomalous, the station is erratic enough to take offline -- not one
# spike (which could be a real event), but a genuinely unstable run.
OFFLINE_ANOMALY_COUNT_THRESHOLD = 5

# How many CONSECUTIVE clean readings are needed after going OFFLINE
# before a station is allowed to recover back to HEALTHY.
RECOVERY_CLEAN_STREAK_REQUIRED = 5


def score_to_severity(score_pct: float) -> str:
    """Maps a 0-100 anomaly score to a severity label using the shared thresholds."""
    if score_pct >= SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    if score_pct >= SEVERITY_THRESHOLDS["high"]:
        return "high"
    if score_pct >= SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    return "low"
