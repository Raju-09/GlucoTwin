# GlucoTwin — System Architecture

> Status: Draft · Last updated: 2026-09-23

## Overview

GlucoTwin is structured as a pipeline of loosely coupled modules. Each module has a defined input contract and output contract, making failures easy to isolate and test independently.

```
Data Sources
    │
    ▼
┌─────────────────────────────────────────────┐
│              Ingestion Layer                │
│  discover_files → load_raw → validate       │
└──────────────────────┬──────────────────────┘
                       │  validated DataFrame
                       ▼
┌─────────────────────────────────────────────┐
│           Preprocessing Layer               │
│  normalize_timestamps → resolve_duplicates  │
│  → convert_units → interpolate_if_justified │
└──────────────────────┬──────────────────────┘
                       │  clean, unit-normalized DataFrame
                       ▼
┌─────────────────────────────────────────────┐
│        Supervised Window Construction       │
│  create_windows(lookback_n, horizon=120min) │
│  → split_by_patient_and_time               │
└──────────────────────┬──────────────────────┘
                       │  windows_{train,val,test}.parquet
                       ▼
┌─────────────────────────────────────────────┐
│           Feature Engineering               │
│  extract_tabular_features(window_row)       │
│  (fitted only on train split)               │
└──────────────────────┬──────────────────────┘
                       │  feature matrix X, target y
                       ▼
┌─────────────────────────────────────────────┐
│              Model Layer                    │
│  PersistenceBaseline (always present)       │
│  RandomForest / LightGBM pipeline           │
│  (optional) PatientAdaptation               │
└──────────────────────┬──────────────────────┘
                       │  raw forecast value
                       ▼
┌─────────────────────────────────────────────┐
│           Reliability Layer                 │
│  ReliabilityChecker → status + reason_code  │
│  available / degraded / withheld            │
└──────────────────────┬──────────────────────┘
                       │  ForecastResponse
                       ▼
┌─────────────────────────────────────────────┐
│              FastAPI Backend                │
│  POST /forecast                             │
│  GET /health, /model-info, /patients/...    │
└──────────────────────┬──────────────────────┘
                       │  JSON response
                       ▼
┌─────────────────────────────────────────────┐
│           Streamlit Dashboard               │
│  PatientSelector → GlucoseChart → Forecast  │
│  ReliabilityBadge → ModelEvidence           │
└─────────────────────────────────────────────┘
```

## Data Streams

| Stream | Contents | Source |
|---|---|---|
| Historical / static context | Synthetic demographics, diagnoses, baseline metrics | Synthetic demo or EHR dataset |
| Dynamic stream | Time-stamped CGM glucose readings | CGM dataset (DiaTrend / synthetic) |
| Target | Future glucose value at t+120 min | Observed from same CGM dataset |

These streams are kept **conceptually separate**. The target is never included in the input window.

## Key Design Decisions

### 1. Leakage prevention
All preprocessing (scalers, imputers, feature selection) is fitted **only on the training split**. The test set is used exactly once for final evaluation.

### 2. Chronological split
Earlier time periods → training. Later time periods → test. No patient data from a later period appears in training. See `src/glucotwin/data/splits.py`.

### 3. Reliability-first API
Every `/forecast` response includes a reliability status. When reliability is `withheld`, no prediction value is returned — the API returns the reason code instead.

### 4. Synthetic demo independence
The Streamlit dashboard and API demo mode work entirely from `data/synthetic_demo/`. No access to restricted data is required to run a demo.

## Module Inventory

| Module | Path | Responsibility |
|---|---|---|
| Loader | `src/glucotwin/data/load.py` | File discovery and raw ingestion |
| Validator | `src/glucotwin/data/validate.py` | Schema, timestamps, units, range |
| Preprocessor | `src/glucotwin/data/preprocess.py` | Normalization, deduplication, unit conversion |
| Window builder | `src/glucotwin/features/windows.py` | Supervised example construction |
| Splitter | `src/glucotwin/data/splits.py` | Train/val/test splits |
| Feature builder | `src/glucotwin/features/build_features.py` | Tabular feature extraction |
| Persistence model | `src/glucotwin/models/persistence.py` | Baseline forecaster |
| Trainer | `src/glucotwin/models/train.py` | Pipeline training and serialization |
| Metrics | `src/glucotwin/evaluation/metrics.py` | MAE, RMSE, evaluation reports |
| Reliability | `src/glucotwin/reliability/checks.py` | Input quality and forecast gating |
| API | `src/glucotwin/api/main.py` | FastAPI serving layer |
| Dashboard | `frontend/streamlit_app.py` | Clinician-facing UI |
| Replay engine | `src/glucotwin/demo/replay.py` | Synthetic demo scenarios |

## Deployment Topology

```
Docker Compose
├── api        (FastAPI, port 8000)
└── frontend   (Streamlit, port 8501)
```

See `docker-compose.yml` and `deployment/` for container definitions.
