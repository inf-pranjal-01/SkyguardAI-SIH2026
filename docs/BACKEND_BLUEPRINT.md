# SkyGuard AI — Backend Blueprint

Companion to `ARCHITECTURE.md` (the API contract). This document is the
build spec for everything behind those endpoints. Map integration is
deferred — nothing below depends on it.

---

## 1. Tech stack

| Layer | Tool | Why |
|---|---|---|
| API framework | FastAPI | async, auto-generates OpenAPI docs at `/docs`, easy CORS setup |
| Server | Uvicorn | standard FastAPI ASGI server |
| ML | scikit-learn `IsolationForest` | unsupervised, fast, no GPU, explainable via SHAP |
| Explainability | `shap` (TreeExplainer) | works directly on Isolation Forest's trees |
| Data handling | pandas, numpy | CSV loading, feature engineering |
| Data source | Open-Meteo Archive API | free real historical weather data |
| Storage | flat CSV + in-memory state (v1) | no DB needed at hackathon scale — see §5 |
| Deployment | Render (free tier) | one-click FastAPI deploy from GitHub |

No database required for v1. Reasoning: your data volume (3 stations,
months of hourly readings, a rolling window of recent anomalies) fits
comfortably in memory and CSV files. Adding Postgres/SQLite is a
"nice to have if time remains," not a blocker — don't let anyone on
the team burn hours on it early.

---

## 2. Module breakdown

```
backend/
├── main.py                 # FastAPI app + all route handlers
├── config.py                # station list, thresholds, constants
├── state.py                  # in-memory "current live state" store
├── data/
│   ├── data_fetch.py         # ✅ done — pulls real Open-Meteo history
│   └── anomaly_injector.py   # injects spike/frozen/drift/dropout faults
├── model/
│   ├── features.py            # turns raw readings into model-ready features
│   ├── train.py                # trains + saves the Isolation Forest
│   ├── detect.py                # loads model, scores a new reading
│   └── explain.py                # SHAP values for a given detection
├── simulator.py               # background loop that fakes a "live" stream
├── requirements.txt
└── model_artifacts/
    └── isolation_forest.pkl   # saved trained model (generated, not hand-written)
```

### What each file is responsible for

**`config.py`** — single place for the station dict (id/lat/lon/name),
normal-range constants for fallback, severity thresholds
(e.g. score ≥ 90 → `critical`, ≥ 70 → `high`, ≥ 40 → `medium`, else `low`).
Both `detect.py` and `main.py` import from here so thresholds are never
duplicated or drift out of sync.

**`data/anomaly_injector.py`** — takes a clean row of real data and
returns a corrupted version plus a label of what was injected. Fault
types: `spike` (single reading pushed far outside normal range),
`frozen_value` (repeats the last N readings identically — a comms
fault), `drift` (slow linear offset growing over time), `dropout`
(missing/null reading). This is what Day 41–43 (z-score/IQR) directly
informs — the injector should push values past a z-score or IQR
threshold on purpose, so ground truth is known.

**`model/features.py`** — converts a raw reading (plus recent history)
into the feature vector the model actually sees: raw temp/pressure/humidity,
rate-of-change over the last N readings, deviation from rolling mean,
cross-parameter consistency (e.g. does temp↑ correlate with the
pressure/humidity direction physics would predict). This file is where
"multivariate consistency analysis," a named PS objective, actually lives.

**`model/train.py`** — loads `all_stations.csv`, builds features via
`features.py`, fits `IsolationForest(n_estimators=100, contamination=0.05, random_state=42)`
on it, saves the fitted model to `model_artifacts/isolation_forest.pkl`.
Run once (or whenever data changes), not on every API request.

**`model/detect.py`** — loads the saved model once at startup, exposes
`score_reading(reading) -> {anomaly_score, is_anomaly, severity}`.
Combines the Isolation Forest's anomaly score with the rule-based checks
(frozen-value detection, hard physical bounds) — rules catch things a
purely statistical model might miss, and give you an extra layer to
explain in Q&A.

**`model/explain.py`** — given a reading already flagged anomalous, runs
`shap.TreeExplainer(model).shap_values(features)` and returns the
per-feature contribution list in the exact shape `GET /api/explain/{id}`
promises.

**`simulator.py`** — since there's no real live AWS feed, this is a
background task (FastAPI `BackgroundTasks` or a simple loop on startup)
that steps through the historical CSV one row at a time on a timer
(e.g. every 2 seconds = 1 simulated reading), occasionally calling the
injector, and updates `state.py` with "the current reading." This is
what makes `/api/current-reading` feel live without needing a real sensor.

**`state.py`** — a simple in-memory object (a Python dict or small class)
holding: latest reading per station, rolling history buffer for the
trends chart, list of recent anomalies. `main.py` reads from this;
`simulator.py` writes to it. This is the entire "database" for v1.

**`main.py`** — wires it all together into the exact routes from
`ARCHITECTURE.md`: reads from `state.py`, calls `detect.py`/`explain.py`
as needed, handles the two `POST` actions. Also sets up CORS
(`allow_origins=["*"]` is fine for a hackathon) so the Lovable frontend
can call it from a different domain.

---

## 3. Data flow at runtime

```
simulator.py (every ~2s)
   │
   ├─► pulls next row from historical CSV
   ├─► maybe corrupts it via anomaly_injector.py
   ├─► features.py builds the feature vector
   ├─► detect.py scores it (Isolation Forest + rules)
   ├─► if anomalous: explain.py computes SHAP values
   └─► writes result into state.py
                │
                ▼
        FastAPI routes in main.py
                │
                ▼
        Frontend polls every 3-5s → sees updated cards/chart/alerts
```

---

## 4. Training vs. serving — don't confuse these

- **Training** (`train.py`) happens **once**, offline, on real historical
  "normal" data. Output: a `.pkl` file checked into the repo (or
  regenerated on deploy).
- **Serving** (`detect.py`) happens **per reading**, loads that `.pkl`
  once at API startup, and just scores — it never retrains live. Keep
  these conceptually separate; a common mistake is retraining on every
  request, which is both slow and wrong (you'd be training on
  potentially-anomalous data).

---

## 5. Why no database, precisely

A database earns its place when data must survive a server restart or
scale beyond memory. Neither applies here: if the server restarts during
a demo, `simulator.py` just starts again from the top of the CSV — no
data loss that matters. If judges ask "why no DB," the honest answer is
in this section, and it's correct hackathon engineering judgment, not a
missing feature.

---

## 6. Deployment checklist (when ready)

1. `requirements.txt` — pin exact versions (`fastapi`, `uvicorn`, `scikit-learn`, `shap`, `pandas`, `numpy`, `requests`).
2. Push `backend/` to GitHub.
3. Render → New Web Service → connect repo → root directory `backend/` →
   start command `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Set CORS origin to the deployed Lovable/Vercel frontend URL once known.
5. Confirm `/docs` (FastAPI's auto Swagger UI) loads — that's your
   fastest sanity check that every endpoint in `ARCHITECTURE.md` is live.

---

## 7. Build order (maps to your video-learning phases)

| Order | File | Unlocked by |
|---|---|---|
| 1 | ✅ `data/data_fetch.py` | done |
| 2 | `data/anomaly_injector.py` | Day 41–43 (Outliers) |
| 3 | `model/features.py` | no video dependency — pure engineering |
| 4 | `model/train.py` | Day 80, 84, 88, 91–93 (Trees → Ensembles → RF) |
| 5 | `model/detect.py` + eval | Day 61, 75, 76 (overfitting, precision/recall/F1) |
| 6 | `model/explain.py` | Day 97 + standalone SHAP video |
| 7 | `simulator.py`, `state.py`, `main.py` | pure engineering, ties everything together |
