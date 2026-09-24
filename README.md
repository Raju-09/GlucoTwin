# GlucoTwin

> **Research prototype — not a medical device.**  
> GlucoTwin forecasts blood glucose 120 minutes ahead from CGM history and verified patient context. It is intended for researcher and clinician review of synthetic demo patients only.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-prototype-orange)

## Problem

Continuous Glucose Monitors (CGMs) produce a reading approximately every 5 minutes. A clinician or researcher reviewing a patient's trajectory may benefit from knowing where the glucose is likely to be in 2 hours — and, crucially, whether that forecast is reliable enough to display.

**Research question:** Does a time-series model improve on a simple persistence baseline on a leakage-safe held-out test set, and when should the system withhold a forecast?

## Intended User

A clinician or researcher reviewing a **synthetic demo patient's** glucose trajectory and forecast. GlucoTwin is **not** a diagnostic system, treatment recommender, or clinical decision support tool.

## Prediction Target

| Property | Value |
|---|---|
| Input | Last N CGM readings available at forecast origin |
| Horizon | 120 minutes |
| Output | Point forecast of blood glucose (mg/dL) |
| Reliability | `available` / `degraded` / `withheld` with reason code |

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full system diagram.

```
Data → Validation → Feature Engineering → Model Inference
                                         ↓
                              Reliability Checks
                                         ↓
                          FastAPI → Streamlit Dashboard
```

## Dataset

See [`docs/dataset_card.md`](docs/dataset_card.md). Raw patient data is **never committed to this repository**.

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-handle>/GlucoTwin.git
cd GlucoTwin

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
copy .env.example .env

# 5. (Optional) Run tests against synthetic fixtures
pytest -q tests/
```

## Running the Demo (Synthetic Data)

```bash
# Start the API
uvicorn src.glucotwin.api.main:app --reload --port 8000

# In a second terminal, start the dashboard
streamlit run frontend/streamlit_app.py
```

The dashboard uses synthetic demo patients only. No real patient data is required to run the demo.

## Evaluation

```bash
# Dataset audit (requires data/raw/ to be populated)
python scripts/audit_data.py --data-dir data/raw --out reports/audit.json

# Build supervised windows
python scripts/build_dataset.py --data-dir data/raw --out data/processed/

# Evaluate persistence baseline
python scripts/evaluate.py --model persistence --split test

# Evaluate learned model
python scripts/evaluate.py --model rf --split test
```

## Results (Held-Out Test Split: 7,316 Examples, 10 Patients)

Evaluated under strict chronological hold-out with a 120-minute boundary safety buffer:

| Model | Test MAE (mg/dL) | Test RMSE (mg/dL) | Test MAPE (%) | Median Abs Error | Relative Error Reduction |
|---|---|---|---|---|---|
| **Persistence Baseline ($y_t$)** | 53.25 | 71.26 | 31.78% | 42.10 mg/dL | Baseline floor (0.0%) |
| **Linear Trend Baseline (Slope)** | 97.38 | 145.10 | 60.79% | 65.17 mg/dL | -82.9% (severe overshoot) |
| **GlucoTwin v0.1 (Learned Model)** | **18.69** | **30.84** | **11.79%** | **9.49 mg/dL** | **+64.9% error reduction** |
| **GlucoTwin Adapted (Personalized)** | **18.48** | **30.51** | **11.62%** | **9.41 mg/dL** | **+65.3% error reduction** (up to +11.3% for offset profiles) |

*Artifacts saved in `artifacts/evaluations/`.*

## Model & Data Cards

- [`docs/model_card.md`](docs/model_card.md)
- [`docs/dataset_card.md`](docs/dataset_card.md)
- [`docs/limitations.md`](docs/limitations.md)

## Safety and Limitations

- This is a **research prototype**, not a medical device.
- Do not use predictions to guide insulin dosing, medication changes, or any clinical decision.
- All demo data is synthetic.
- See [`docs/limitations.md`](docs/limitations.md) for a complete list of known failure cases.

## License

Code: [MIT](LICENSE)  
Data: See [`docs/dataset_card.md`](docs/dataset_card.md) for data-specific terms.
