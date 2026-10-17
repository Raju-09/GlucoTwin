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
| **Project Title** | **GlucoTwin: A Reliability-Aware Digital Twin for 120-Minute Glucose Forecasting** |
| **Team Name** | **Team Synapse** |
| **Team Leader** | **Sammeta Siva Prasad Raju** |
| **Team Members** | **Sammeta Siva Prasad Raju, Bedadakota Devi Chandrika** (2 Members) |
| **College / Incubator** | **Mohan Babu University** |
| **GitHub Repository** | `https://github.com/Raju-09/GlucoTwin` (Public) |
| **Demo Video** | [YouTube Unlisted Link — 15–20 Minute Walkthrough] |
| **Architecture Diagram** | [`docs/architecture_diagram.pdf`](docs/architecture_diagram.pdf) (Vector PDF) |
| **Presentation** | [`docs/glucotwin_presentation.pdf`](docs/glucotwin_presentation.pdf) (10-Slide Deck PDF) |
| **License** | [MIT License](LICENSE) (Open Source) |

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
# 1. Evaluate persistence baseline on held-out test split (43,888 windows)
python scripts/evaluate.py --model persistence --split test

# 2. Evaluate learned model (HistGradientBoostingRegressor)
python scripts/evaluate.py --model ml --split test

# 3. Run full multimodal 4-way ablation study
python scripts/run_multimodal_ablation.py

# 4. Evaluate clinical adverse event classification (>180 mg/dL & <70 mg/dL)
python scripts/evaluate_adverse_events.py

# 5. Evaluate reliability gate risk-coverage trade-off
python scripts/evaluate_risk_coverage.py

# 6. Run Leave-One-Patient-Out (LOPO) cross-validation (60 patients, 224,681 windows)
python experiments/run_exp01_break_claims.py
```

## Results & Scientific Evidence

### 1. Multimodal Ablation Study (Held-Out Test Split: 43,888 Windows, 60 Patients)

Evaluated under strict chronological hold-out (Days 12–14) with a 120-minute boundary safety buffer:

| Configuration | Data Streams | Test MAE (mg/dL) | Δ MAE (Model − Ref) | 95% Bootstrap CI |
|---|---|:---:|:---:|:---:|
| **Persistence Baseline** | Naive $y_{t+120} = y_t$ | **53.25** | *+34.94 mg/dL* | — |
| **Config A (CGM Only)** | 24 lags + rolling stats (41 feats) | **18.31** | *Reference (0.00)* | — |
| **Config B (CGM + Static EHR)** | Demographics + Labs + FBG (50 feats) | **17.61** | **−0.70 mg/dL** | **[+0.49, +0.96]** ($p = 5.6 \times 10^{-6}$) |
| **Config C (CGM + Wearables)** | Decoupled HR, HRV, steps, sleep (47 feats) | **18.32** | +0.01 mg/dL | [−0.08, +0.07] ($p = 0.95$, n.s.) |
| **Config D (Full Fusion)** | EHR + CGM + Wearables (56 feats) | **17.67** | **−0.64 mg/dL** | [+0.44, +0.88] |

*Takeaway:* Static EHR context provides a modest, statistically significant error reduction (−0.70 mg/dL) on subsequent time periods for the same patients. Wearable telemetry under this simulator provided negligible incremental predictive value.

### 2. Leave-One-Patient-Out Cross-Validation (N=60 Patients, 224,681 Windows)

| Configuration | Mean LOPO MAE | Delta vs CGM-only | 95% Bootstrap CI | p-value |
|---|:---:|:---:|:---:|:---:|
| **CGM Only** | 17.65 mg/dL | — | — | — |
| **CGM + Real EHR** | **17.65 mg/dL** | +0.01 mg/dL | [−0.34, +0.33] | **0.956 (n.s.)** |
| **CGM + Permuted EHR (Control)** | **20.47 mg/dL** | **+2.82 mg/dL** | — | — |

*Scientific Interpretation:* When evaluated on completely unseen patients (LOPO), EHR context provides no transfer gain ($p=0.956$). However, randomly shuffling EHR between patients causes a **catastrophic +2.82 mg/dL degradation**, proving the model encodes true patient identity rather than population-wide physiology. We report this boundary openly rather than claiming unvalidated generalization.

### 3. Clinical Adverse Event Classification (t+120 min Horizon)

| Adverse Event | Model | Events (N) | Sensitivity (Recall) | Specificity | Precision | F1-Score | False Alarms / Patient-Day |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hyperglycemia (>180 mg/dL)** | Persistence Baseline | 12,246 | 26.6% | 70.4% | 25.8% | 0.262 | 61.4 alarms/day |
| **Hyperglycemia (>180 mg/dL)** | **GlucoTwin Two-Stream** | 12,246 | **76.2%** | **90.6%** | **75.8%** | **0.760** | **19.6 alarms/day (−68%)** |
| **Hypoglycemia (<70 mg/dL)** | Persistence Baseline | 71 | 12.7% | 99.9% | 12.7% | 0.127 | 0.4 alarms/day |
| **Hypoglycemia (<70 mg/dL)** | **GlucoTwin Two-Stream** | 71 | **0.0%** | **100.0%** | **0.0%** | **0.000** | **0.0 alarms/day** |

*Takeaway:* GlucoTwin catches nearly 3× more postprandial hyperglycemic spikes 2 hours in advance while reducing false alarms by **68%**, combating alarm fatigue. In our adult T2D cohort, hypoglycemia is a rare tail event (<0.2% prevalence); standard MSE regression regresses towards the mean (~150 mg/dL) and misses rare nadirs below 70 mg/dL. This is an explicit limitation requiring asymmetric loss functions in future work.

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

- **Software Code:** Licensed under the [MIT License](LICENSE).
- **Synthetic Data & Documentation:** Available for research and educational purposes. See [`docs/dataset_card.md`](docs/dataset_card.md) for data-specific terms. The software license does not automatically grant rights to third-party medical datasets or dependencies.

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

