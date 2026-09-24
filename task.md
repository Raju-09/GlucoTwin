# GlucoTwin — Task Tracker

## Phase 0 — Repository scaffold + research framing docs
- [x] Root files: README, LICENSE, .gitignore, .env.example, requirements.txt, pyproject.toml
- [x] docs/: problem_definition.md, dataset_card.md, feasibility_checklist.md, architecture.md, model_card.md, limitations.md
- [x] data/ skeleton with READMEs and .gitkeep
- [x] src/glucotwin/ package: __init__.py, config.py
- [x] scripts/README.md, tests/conftest.py
- [x] notebooks/ skeleton

## Phase 1 — Dataset audit
- [x] src/glucotwin/data/load.py
- [x] src/glucotwin/data/validate.py
- [x] scripts/audit_data.py
- [x] tests/test_validate.py (24/24 passing)
- [ ] Run audit on real data — fill dataset_card.md (Waiting for dataset in data/raw)

## Phase 2 — Preprocessing & leakage-safe windows
- [x] src/glucotwin/data/preprocess.py
- [x] src/glucotwin/features/windows.py
- [x] src/glucotwin/data/splits.py
- [x] scripts/build_dataset.py
- [x] tests/test_windows.py, tests/test_splits.py (all passing)
- [x] Supervised dataset generated: 24,647 train, 5,273 val, 7,316 test windows (data/processed/)

## Phase 3 — Baseline forecasting & evaluation
- [x] src/glucotwin/models/persistence.py (PersistenceBaseline & LinearTrendBaseline)
- [x] src/glucotwin/evaluation/metrics.py (MAE, RMSE, MAPE, percentiles, per-patient)
- [x] scripts/evaluate.py
- [x] tests/test_metrics.py, tests/test_baselines.py (all passing)
- [x] Baseline evaluated on held-out test split (7,316 examples):
  - Persistence: MAE = 53.25 mg/dL, RMSE = 71.26 mg/dL
  - Trend: MAE = 97.38 mg/dL, RMSE = 145.10 mg/dL (proves linear extrapolation overshoots)

## Phase 4 — First learned model
- [x] src/glucotwin/features/build_features.py (lag features, rolling stats, velocities, circadian)
- [x] src/glucotwin/models/train.py (GlucoTwinForecaster & training pipeline)
- [x] scripts/train_model.py (CLI training)
- [x] artifacts/model_v0.1/ (saved model.joblib, feature_schema.json, model_meta.json, training_metrics.json)
- [x] tests/test_train.py, tests/test_features.py (all passing)
- [x] Evaluated on held-out test split (7,316 examples) vs Persistence:
  - Persistence: MAE = 53.25 mg/dL, RMSE = 71.26 mg/dL, MAPE = 31.78%
  - GlucoTwin v0.1: MAE = 18.69 mg/dL, RMSE = 30.84 mg/dL, MAPE = 11.79%
  - Relative improvement: 64.9% error reduction over persistence floor

## Phase 5 — Personalization feasibility
- [x] scripts/audit_personalization.py (empirical audit on 7,316 held-out test windows)
- [x] docs/personalization_feasibility.md & reports/personalization_audit.json
- [x] Finding documented: Selective benefit (+11.3% error reduction for SYNTH_002, overall 18.69 -> 18.48 mg/dL)

## Phase 6 — Reliability layer
- [x] src/glucotwin/reliability/checks.py (available / degraded / withheld gating)
- [x] tests/test_reliability.py (7/7 tests passing: stale, gaps, bounds, missing, empty)
- [x] Machine-readable reason codes: OK_DATA_INTACT, ERR_STALE_READING, ERR_INSUFFICIENT_HISTORY, ERR_EXCESSIVE_GAP, ERR_PHYSIOLOGICAL_OUT_OF_BOUNDS, WARN_PARTIAL_SEQUENCE

## Phase 7 — FastAPI backend
- [x] src/glucotwin/api/schemas.py (Pydantic models with validation)
- [x] data/synthetic_demo/patients.json (deterministic demo patient metadata)
- [x] src/glucotwin/api/main.py (GET /health, GET /model-info, GET /patients/demo, GET /patients/{id}/timeline, POST /forecast)
- [x] tests/test_api.py (8/8 tests passing: clean, stale, out-of-bounds, degraded, metadata)
- [x] All 56 unit/API tests passing across entire test suite

## Phase 8 — Dashboard
- [x] frontend/streamlit_app.py (interactive clinician dashboard with Altair timeline chart, KPI metrics, reliability badges, and model evidence)

## Phase 9 — Synthetic replay
- [x] src/glucotwin/demo/replay.py (ReplayEngine with scenario injectors: stale sensor, excessive gap, hardware spike, dropouts)
- [x] Tested interactive scenario injection: system reacts in real time by degrading or withholding forecasts with explicit reason codes

## Phase 10 — Deployment
- [x] docker-compose.yml (multi-container FastAPI + Streamlit configuration)
- [x] deployment/Dockerfile.api
- [x] deployment/Dockerfile.frontend
- [x] scripts/healthcheck.ps1 (PowerShell health check for API & model status)

## Phase 11 — Research audit & submission
- [x] Filled README.md with real results (+64.9% error reduction over persistence)
- [x] Filled docs/model_card.md with real metrics & per-patient breakdown
- [x] docs/limitations.md & docs/personalization_feasibility.md complete
- [x] Submission ready: reproducible scripts, deterministic fixtures, 56 unit/API tests passing
