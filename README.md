# GlucoTwin

> **Research prototype — not a medical device.**  
> GlucoTwin forecasts blood glucose 120 minutes ahead from CGM history and verified patient context. It is intended for researcher and clinician review of synthetic demo patients only.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-prototype-orange)

---

## Hackathon Submission — Happiest Health Digital Twin Challenge 2026

| Field | Details |
|---|---|
| **Team Name** | [YOUR_TEAM_NAME] |
| **Team Leader** | [YOUR_NAME] |
| **College / Incubator** | [YOUR_COLLEGE_NAME] |
| **Team Members** | [YOUR_NAME] *(solo / list all members)* |
| **Contact Email** | [YOUR_EMAIL] |
| **GitHub Repository** | `https://github.com/Raju-09/GlucoTwin` |
| **Demo Video** | [YouTube unlisted link — add before submission] |
| **Architecture Diagram** | [`docs/architecture_diagram.pdf`](docs/architecture_diagram.pdf) |
| **Presentation** | [`docs/glucotwin_presentation.pdf`](docs/glucotwin_presentation.pdf) |
| **License** | MIT — see [LICENSE](LICENSE) |

---

## Problem

India has **101 million people living with diabetes** — the largest diabetic population in the world (IDF Diabetes Atlas, 2023). Continuous Glucose Monitors (CGMs) record a reading every 5 minutes, but all current CGM alert systems are **reactive** — they fire *after* glucose crosses a threshold. By the time a clinician sees the number, the window to intervene has already passed.

**GlucoTwin addresses this gap:** it forecasts glucose 120 minutes ahead, fuses static EHR context with dynamic CGM history, and — critically — tells the clinician *when not to trust the prediction* through a built-in reliability layer.

**Research question:** Does a time-series model improve on a simple persistence baseline on a leakage-safe held-out test set, and when should the system withhold a forecast?

## Intended User

A clinician or researcher reviewing a **synthetic demo patient's** glucose trajectory and forecast. GlucoTwin is **not** a diagnostic system, treatment recommender, or clinical decision support tool.

## Prediction Target

| Property | Value |
|---|---|
| Input | Last 24 CGM readings (~2 h history at 5-min cadence) + EHR context |
| Horizon | **120 minutes** |
| Output | Point forecast of blood glucose (mg/dL) |
| Reliability | `available` / `degraded` / `withheld` with machine-readable reason code |

## Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Model | scikit-learn HistGradientBoostingRegressor (41 features) |
| Feature Engineering | CGM lags, rolling stats, rate-of-change, circadian encoding, EHR context (9 fields) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Altair |
| Containerisation | Docker + docker-compose |
| Tests | pytest (69 passing) |
| Data | 60 synthetic T2D patients — mechanistic EHR-physiology coupling, CGM sensor physics layer |

## Data Streams (Dual Fusion)

| Stream | Contents | Source |
|---|---|---|
| **Static / Historical** | Demographics, HbA1c, BMI, fasting glucose, comorbidities, medications | Synthetic EHR (60 patients, mechanistically generated) |
| **Dynamic / Real-time** | CGM readings at 5-min cadence, heart rate, HRV, step count, sleep stages | Synthetic wearable telemetry (235,112 rows) |

## Architecture

See [`docs/architecture_diagram.pdf`](docs/architecture_diagram.pdf) for the visual diagram.

```
Static EHR ──────────────────────────┐
                                     ▼
                          Feature Engineering (50 feats)
                                     │
CGM Stream → Validation → Windows ──►├──► HistGBM Model ──► Reliability Gate ──► FastAPI ──► Streamlit
                                     │         │                    │
Wearable Stream ────────────────────►┘    (120-min forecast)   available /              Dashboard
                                                               degraded /              (Doctor view)
                                                               withheld
```

## Setup

```bash
# 1. Clone
git clone https://github.com/Raju-09/GlucoTwin.git
cd GlucoTwin

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
copy .env.example .env

# 5. Run tests (all 69 should pass)
pytest -q tests/
```

## Running the Demo (Synthetic Data Only)

```bash
# Terminal 1 — Start the API
uvicorn src.glucotwin.api.main:app --reload --port 8000

# Terminal 2 — Start the dashboard
streamlit run frontend/streamlit_app.py

# Or run both with Docker
docker-compose up
```

The dashboard uses synthetic demo patients only. No real patient data is required.

## Evaluation

```bash
# Build supervised windows
python scripts/build_dataset.py --data-dir data/raw --out data/processed/

# Evaluate persistence baseline
python scripts/evaluate.py --model persistence --split test

# Evaluate learned model
python scripts/evaluate.py --model rf --split test

# Run LOPO cross-validation (takes ~25 min on 60 patients)
python experiments/run_exp01_break_claims.py
```

## Results

### Ablation Study — 60 Synthetic Patients, 43,888 Held-Out Test Windows

Evaluated under strict chronological hold-out with a 120-minute boundary safety buffer:

| Configuration | Data Streams | Test MAE (mg/dL) | vs. CGM-only |
|---|---|:---:|:---:|
| **Persistence Baseline** | Naive y(t+120) = y(t) | **53.25** | — |
| **Config A — CGM Only** | 24 lags + rolling stats (41 feats) | **18.31** | Reference |
| **Config B — CGM + EHR** | + Demographics, labs, FBG (50 feats) | **17.61** | −0.70 mg/dL |
| **Config C — CGM + Wearables** | + HR, HRV, steps, sleep (47 feats) | **18.32** | +0.01 mg/dL |
| **Config D — Full Fusion** | All streams (56 feats) | **17.67** | −0.64 mg/dL |

### Leave-One-Patient-Out Cross-Validation (N=60 patients, 224,681 windows)

| Config | Mean LOPO MAE | Delta | 95% CI | p-value |
|---|:---:|:---:|:---:|:---:|
| CGM-only | 17.65 | — | — | — |
| CGM + Real EHR | **17.65** | +0.01 | [−0.34, +0.33] | **0.96 (n.s.)** |
| CGM + Permuted EHR | 20.47 | +2.82 | — | — |

**Honest interpretation:** EHR context improves same-patient forecasting (−0.70 mg/dL temporal split) but shows no benefit on held-out patients (LOPO p=0.96). The permuted-EHR catastrophic degradation (+2.82 mg/dL) confirms the model encodes real patient identity. This is documented honestly — not hidden.

*Full results in `artifacts/experiment_manifest.json` and `experiments/results/exp01_break_your_claim.json`.*

## Model & Data Cards

- [`docs/model_card.md`](docs/model_card.md)
- [`docs/dataset_card.md`](docs/dataset_card.md)
- [`docs/limitations.md`](docs/limitations.md)

## Safety and Limitations

- This is a **research prototype**, not a medical device.
- Do not use predictions to guide insulin dosing, medication changes, or any clinical decision.
- All demo data is **synthetic** — no real patient data was used or stored.
- See [`docs/limitations.md`](docs/limitations.md) for the complete list of known failure cases.

## License

Code: [MIT](LICENSE)  
Data: Synthetic only — see [`docs/dataset_card.md`](docs/dataset_card.md).

## Folder Structure

```
GlucoTwin/
├── src/glucotwin/          # Core package (API, models, features, reliability, twin)
├── frontend/               # Streamlit dashboard
├── tests/                  # 69 pytest tests
├── scripts/                # Evaluation and data generation scripts
├── experiments/            # LOPO and break-your-claim experiments
├── docs/                   # Model card, dataset card, architecture, limitations
├── data/synthetic_demo/    # Committed synthetic patient data (60 patients)
├── artifacts/              # Trained models and evaluation results
└── submission/             # Demo video script, pitch deck outline, judges Q&A
```

