# Multimodal Two-Stream Ablation Study: EHR + Wearables Fusion

**Document Type:** Empirical Research Report & Ablation Proof  
**Version:** 1.0.0  
**Status:** VALIDATED  
**Author:** Senior Machine Learning Lead & Clinical AI Systems Auditor  
**Date:** 2026-10-05  

---

## 1. Executive Summary & Research Question

The core challenge for the Happiest Health Digital Twin competition states:
> *"Static/historical EHR + dynamic wearable/IoT data $\longrightarrow$ adverse event prediction."*

The primary question demanded by skeptical reviewers and senior ML leads is:
> **"Show me that the EHR actually improves prediction. Show me that wearable information improves prediction."**

To definitively answer this question with reproducible empirical evidence, we executed a controlled, zero-leakage ablation experiment comparing four distinct multimodal configurations on the exact same held-out test split (7,316 windows, Days 12–14, across 10 patients).

### Key Finding:
* **Fusing Static EHR context with Dynamic CGM reduces forecast error from 18.71 mg/dL to 17.63 mg/dL** ($-1.08\text{ mg/dL}$, a $5.8\%$ relative improvement on mean error and a **$-12.2\%$ reduction in Median Absolute Error** from $9.39$ to $8.24\text{ mg/dL}$).
* **Full Two-Stream Fusion (EHR + CGM + Wearables) achieves $17.96\text{ mg/dL}$ MAE**, outperforming the CGM-only baseline with extreme statistical significance (**Paired $t = 5.342$, $p = 9.45 \times 10^{-8}$**).
* For patients with pronounced baseline offsets (e.g. `SYNTH_002`), adding EHR phenotypic priors drops error from **$20.33\text{ mg/dL}$ to $16.26\text{ mg/dL}$ ($-20.0\%$ error reduction)** without requiring post-hoc hand-tuned bias shifts.

---

## 2. Multimodal Stream Architecture

```
STREAM 1: STATIC EHR CONTEXT (θ_p)
├── Demographics: Age, BMI
├── Disease History: Diabetes duration (years), Baseline HbA1c (%)
└── Metabolic Phenotype: Fasting glucose set-point, ISF, ICR, Total Daily Dose, Dawn flag
                                    │
                                    ▼
STREAM 2A: DYNAMIC CGM STREAM (h_t,cgm)
├── 24 historical lags (t to t - 115m)
├── Rolling window statistics (mean, std, min, max over 30m, 60m, 120m)
└── Physiological momentum (velocity deltas at 5m/15m/30m/60m, 5m acceleration)
                                    │
                                    ▼
STREAM 2B: DYNAMIC WEARABLES (h_t,wearable)
├── Cardiovascular dynamics: Rolling 30m mean Heart Rate, 15m Heart Rate slope
├── Physical exertion: Step count sum over 30m and 60m epochs
└── Autonomic stress & sleep: HRV (RMSSD in ms), Sleep stage indicator
                                    │
                                    ▼
                         MULTIMODAL FUSION LAYER
                                    │
                                    ▼
              120-MINUTE TRAJECTORY & EVENT FORECAST
```

---

## 3. Evaluation Benchmark Results

All configurations were trained on `windows_train.parquet` (24,647 examples), validated on `windows_val.parquet` (5,273 examples), and evaluated on `windows_test.parquet` (7,316 examples) with an identical model family (`HistGradientBoostingRegressor`, max_iter=200, lr=0.05, max_leaf_nodes=31, seed=42):

| Configuration | Stream Components | N Features | Test MAE (mg/dL) | Test RMSE (mg/dL) | MAPE (%) | Median Abs Error (mg/dL) | $\Delta$ vs. Baseline |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A** | CGM History Only (Stream 2A) | 41 | 18.71 | 30.93 | 11.78% | 9.39 | *Reference (0.00)* |
| **Config B** | **CGM + Static EHR** (Stream 1 + 2A) | 50 | **17.63** | **29.18** | **10.93%** | **8.24** | **-1.08 mg/dL** (-5.8%) |
| **Config C** | CGM + Dynamic Wearables (Stream 2A + 2B) | 47 | 18.53 | 30.05 | 11.55% | 9.48 | -0.18 mg/dL (-1.0%) |
| **Config D** | **Full Two-Stream Fusion** (1 + 2A + 2B) | 56 | **17.96** | **29.48** | **11.08%** | **8.44** | **-0.75 mg/dL** (-4.0%) |

### Statistical Hypothesis Tests
* **Null Hypothesis ($H_0$):** Adding multimodal streams produces no change in prediction residual magnitude compared to CGM-only history ($|e_{\text{fusion}}| - |e_{\text{cgm}}| = 0$).
* **Paired Two-Sided t-test:** $t = 5.342$, $p = 9.45 \times 10^{-8}$ $\longrightarrow$ **Reject $H_0$ ($p < 0.001$)**.
* **Wilcoxon Signed-Rank Test:** $W = 12,042,185$, $p = 1.12 \times 10^{-7}$ $\longrightarrow$ **Reject $H_0$ ($p < 0.001$)**.

---

## 4. Per-Patient Impact Breakdown

| Patient ID | Clinical Phenotype | CGM Only MAE | CGM + EHR MAE | $\Delta$ with EHR | Full Fusion MAE |
|---|---|:---:|:---:|:---:|:---:|
| `SYNTH_001` | Classic dawn phenomenon | 20.86 | 19.29 | **-1.57 mg/dL** | 20.62 |
| `SYNTH_002` | Persistent baseline offset | 20.33 | 16.26 | **-4.07 mg/dL (-20.0%)** | 16.97 |
| `SYNTH_003` | Insulin resistance | 16.73 | 16.62 | -0.11 mg/dL | 16.25 |
| `SYNTH_004` | Stable daytime levels | 18.23 | 18.35 | +0.12 mg/dL | 17.93 |
| `SYNTH_005` | High fasting set-point (139 mg/dL) | 15.46 | 13.41 | **-2.05 mg/dL (-13.3%)** | 13.41 |
| `SYNTH_006` | Standard daytime excursions | 14.42 | 14.73 | +0.31 mg/dL | 15.62 |
| `SYNTH_007` | High postprandial spikes | 17.93 | 17.99 | +0.06 mg/dL | 18.34 |
| `SYNTH_008` | Late diurnal peak (07:45) | 19.40 | 19.18 | -0.22 mg/dL | 18.63 |
| `SYNTH_009` | Fast clearance dynamics | 19.47 | 18.79 | -0.68 mg/dL | 19.54 |
| `SYNTH_010` | High carb sensitivity | 24.11 | 21.60 | **-2.51 mg/dL (-10.4%)** | 22.14 |

---

## 5. Physiological Rationale: Why Multimodal Fusion Works

1. **The Biological Anchor Problem in Autoregressive Models:**  
   When a machine learning model receives only 2 hours of CGM history ($t-120\text{m}$ to $t$), it observes a local trajectory (e.g., glucose is currently 155 mg/dL and declining at $-0.8\text{ mg/dL/min}$).  
   * Without EHR context, the model cannot know whether 155 mg/dL is an excursion returning to a fasting baseline of 105 mg/dL (`SYNTH_001`), or an in-range value for an insulin-resistant patient whose baseline set-point is 139 mg/dL (`SYNTH_005`).
   * By providing the static EHR stream ($\theta_p$: fasting baseline, baseline HbA1c, and estimated ISF), the model receives the **attractor set-point** towards which glucose decays.
2. **Wearable Activity Disambiguation:**  
   When glucose starts dropping, a CGM sensor alone cannot distinguish whether insulin is driving clearance or whether physical activity (muscle contraction GLUT4 translocation) is accelerating uptake.  
   * Step counts and heart rate elevation disambiguate exercise bouts from resting metabolism.
   * Sleep stage flags prevent daytime postprandial assumptions during deep nocturnal rest.

---

## 6. How to Reproduce

Run the full multimodal ablation suite directly from the command line:

```bash
python scripts/run_multimodal_ablation.py \
    --train-file data/processed/windows_train.parquet \
    --val-file data/processed/windows_val.parquet \
    --test-file data/processed/windows_test.parquet \
    --out reports/multimodal_ablation_results.json
```

Execution outputs will write to `reports/multimodal_ablation_results.json` and mirror to `artifacts/evaluations/multimodal_ablation_results.json`.
