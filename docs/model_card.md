# GlucoTwin — Model Card

> Status: Evaluated & Verified · Date: 2026-09-24  
> Format follows Mitchell et al. (2019) Model Cards for Model Reporting

## 1. Model Details

| Field | Value |
|---|---|
| **Model name** | GlucoTwin v0.1 |
| **Model type** | HistGradientBoostingRegressor with StandardScaler pipeline |
| **Framework** | scikit-learn (Pipeline: StandardScaler + HistGradientBoostingRegressor) |
| **Prediction task** | Point forecast of blood glucose at t+120 minutes (mg/dL) |
| **Input features** | 41 engineered features (24 raw lags, rolling 30/60/120m stats, velocity, acceleration, circadian sin/cos) |
| **Output** | Bounded predicted glucose [20, 600] mg/dL + reliability status |
| **Training date** | 2026-09-24 |
| **Model version** | v0.1 |
| **Artifact path** | `artifacts/model_v0.1/model.joblib` |

## 2. Intended Use

**Intended uses:**
- Research and educational prototype demonstration for the Digital Twin Challenge 2026.
- Comparison against persistence baseline and trend baseline under strict chronological separation.
- Demonstration of input-integrity gating and forecast withholding.

**Out-of-scope uses:**
- Clinical diagnosis or decision support.
- Insulin dosing guidance or medication adjustments.
- Real patient clinical monitoring without medical validation.

## 3. Evaluation Design

| Property | Value |
|---|---|
| **Split design** | Chronological hold-out with 120-minute safety buffer |
| **Training period** | Days 1–9 (24,647 windows) |
| **Validation period** | Days 10–11 (5,273 windows) |
| **Test set period** | Days 12–14 (7,316 windows) |
| **N test examples** | 7,316 |
| **N test patients** | 10 |

## 4. Quantitative Results

Evaluated on the held-out test split (7,316 examples across 10 patients):

| Model | Test MAE (mg/dL) | Test RMSE (mg/dL) | Test MAPE (%) | Median Abs Error | Relative Error Reduction |
|---|---|---|---|---|---|
| **Persistence Baseline ($y_t$)** | 53.25 | 71.26 | 31.78% | 42.10 mg/dL | Baseline floor (0.0%) |
| **Linear Trend Baseline (Slope)** | 97.38 | 145.10 | 60.79% | 65.17 mg/dL | -82.9% (severe overshoot) |
| **GlucoTwin v0.1 (Population Model)** | **18.69** | **30.84** | **11.79%** | **9.49 mg/dL** | **+64.9% error reduction** |
| **GlucoTwin Adapted (Personalized)** | **18.48** | **30.51** | **11.62%** | **9.41 mg/dL** | **+65.3% error reduction** (up to +11.3% for offset profiles) |

### Per-Patient Breakdown (GlucoTwin v0.1)

| Patient ID | Test N | MAE (mg/dL) | RMSE (mg/dL) | Median Err (mg/dL) | 90th %ile Err (mg/dL) |
|---|---|---|---|---|---|
| `SYNTH_001` | 741 | 21.13 | 33.03 | 11.91 | 55.46 |
| `SYNTH_002` | 726 | 20.52 (18.20 adapted) | 29.94 | 13.72 | 50.32 |
| `SYNTH_003` | 733 | 16.84 | 29.53 | 7.73 | 45.47 |
| `SYNTH_004` | 727 | 18.51 | 29.15 | 9.78 | 46.71 |
| `SYNTH_005` | 720 | 15.27 | 22.77 | 9.79 | 34.99 |
| `SYNTH_006` | 734 | 14.02 | 24.73 | 5.92 | 36.38 |
| `SYNTH_007` | 733 | 17.64 | 29.03 | 8.63 | 45.83 |
| `SYNTH_008` | 729 | 19.31 | 32.13 | 9.14 | 46.98 |
| `SYNTH_009` | 736 | 19.30 | 33.09 | 8.86 | 49.62 |
| `SYNTH_010` | 737 | 24.29 | 41.04 | 9.08 | 76.26 |

## 5. Training Data & Preprocessing

Preprocessing scalers were fitted **strictly on the training split only**. Features are derived strictly from readings at or before the forecast origin $t$. Zero test observations were used for feature selection, scaling, threshold tuning, or hyperparameter optimization.

## 6. Caveats & Ethical Considerations

- Tested on a transparent synthetic physiological benchmark. Results establish algorithmic and software validity, not clinical efficacy.
- Forecast withholding is enforced when sensors disconnect, inputs are stale ($>20$ min), or gaps exceed 30 minutes.
- The model must never be used for insulin dosing or medical decision making.
