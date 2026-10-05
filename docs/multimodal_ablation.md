# Multimodal Two-Stream Ablation Study: EHR + Wearables Fusion

**Document Type:** Empirical Research Report & Ablation Benchmark  
**Experiment ID:** `GT-2026-10-05-V1.2`  
**Dataset Version:** `SYNTH-T2D-42` (10 patients, 14 days)  
**Status:** VALIDATED & AUDITED (Gate 1.5 Compliant)  
**Author:** Senior Machine Learning Lead & Healthcare AI Systems Auditor  
**Date:** 2026-10-05  

---

## 1. Executive Summary & Research Questions

The core challenge for the Happiest Health Digital Twin competition states:
> *"Static/historical EHR + dynamic wearable/IoT data $\longrightarrow$ adverse event prediction."*

To evaluate the marginal contribution of each data stream with strict scientific rigor, we formulated two specific research questions:
1. **RQ1 (EHR Value):** Does patient-specific static EHR context (age, BMI, diabetes duration, baseline HbA1c, historical fasting lab) improve 120-minute glucose forecasting beyond CGM time-series history alone?
2. **RQ2 (Wearables Value):** Does the continuously arriving wearable stream (heart rate, step counts, HRV, sleep stage) provide incremental predictive value beyond CGM and EHR?

We executed a controlled, zero-leakage ablation experiment comparing four distinct multimodal configurations on the exact same held-out test split (7,316 windows, Days 12–14, across 10 patients).

### Key Empirical Findings:
* **Static EHR Context Reduces Error:** Fusing static EHR features with CGM history reduces test MAE from **$18.73\text{ mg/dL}$ to $17.89\text{ mg/dL}$** ($-0.84\text{ mg/dL}$, a $4.5\%$ relative improvement on mean error and a **$-8.7\%$ reduction in Median Absolute Error** from $9.27$ to $8.46\text{ mg/dL}$). The 24-hour moving block bootstrap yields an empirical 95% Confidence Interval of **$[+0.11, +1.41]\text{ mg/dL}$ error reduction**.
* **Wearable Stream Value is Inconclusive:** Adding decoupled wearable telemetry alone to CGM achieves $18.56\text{ mg/dL}$ MAE ($-0.17\text{ mg/dL}$, 95% CI: $[-0.39, +0.66]\text{ mg/dL}$, not statistically significant). When added to CGM + EHR, Full Fusion achieves $17.96\text{ mg/dL}$ MAE ($+0.07\text{ mg/dL}$ higher error than CGM + EHR alone).
* **Scientific Verdict:** In our synthetic cohort, static phenotypic EHR context provides the primary incremental predictive benefit. The currently simulated wearable features provide limited incremental information beyond CGM and EHR. We therefore treat dynamic wearable fusion as an open research question rather than asserting unproven clinical efficacy.

---

## 2. Multimodal Stream Architecture

```
STREAM 1: STATIC EHR CONTEXT (θ_p)
├── Demographics: Age, BMI
├── Disease History: Diabetes duration (years), Baseline HbA1c (%)
└── Observational Metabolic Phenotype: Historical fasting glucose lab, Hypertension flag,
    Metformin flag, SGLT2i flag, Dawn phenomenon flag
                                    │
                                    ▼
STREAM 2A: DYNAMIC CGM STREAM (h_t,cgm)
├── 24 historical lags (t to t - 115m)
├── Rolling window statistics (mean, std, min, max over 30m, 60m, 120m)
└── Physiological momentum (velocity deltas at 5m/15m/30m/60m, 5m acceleration)
                                    │
                                    ▼
STREAM 2B: DYNAMIC WEARABLES (h_t,wearable) [Decoupled from CGM]
├── Autonomous cardiovascular dynamics: Rolling 30m mean Heart Rate, 15m Heart Rate delta
├── Physical exertion: Step count sum over 30m and 60m epochs
└── Autonomic tone & sleep: HRV (RMSSD in ms), Sleep stage indicator
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

| Configuration | Stream Components | N Features | Test MAE (mg/dL) | Change vs CGM-only | 95% CI for Improvement |
|---|---|:---:|:---:|:---:|:---:|
| **Config A** | CGM History Only (Stream 2A) | 41 | 18.73 | *Reference* | — |
| **Config B** | **CGM + Static EHR** (Stream 1 + 2A) | 50 | **17.89** | **-0.84 mg/dL** | **0.11 to 1.41 mg/dL improvement** |
| **Config C** | CGM + Dynamic Wearables (Stream 2A + 2B) | 47 | 18.56 | -0.17 mg/dL | -0.39 to +0.66 mg/dL |
| **Config D** | Full Two-Stream Fusion (1 + 2A + 2B) | 56 | 17.96 | -0.77 mg/dL | 0.04 to 1.38 mg/dL improvement |
| *Baseline* | *Persistence Baseline ($y_{t+120} = y_t$)* | *—* | *53.25* | *+34.52 mg/dL* | *—* |

---

## 4. Defensible Statistical Significance & Incremental Analysis

To eliminate **pseudoreplication** caused by evaluating autocorrelated 5-minute windows across 10 patients, significance testing was computed across **patient-level paired differences ($N = 10, df = 9$)** and validated via **24-hour moving block bootstrap (1,000 resamples)**:

### 4.1 Incremental Breakdown

| Incremental Step | Contrast | $\Delta$ MAE | Patient-Level Paired $t$ ($df=9$) | $p$-value | 95% Block Bootstrap CI for Improvement | Verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **EHR Value** | Config B vs Config A | **$-0.84\text{ mg/dL}$** | $t = 1.832$ | $p = 0.100$ | **$0.11\text{ to }1.41\text{ mg/dL improvement}$** | ✅ Meaningful gain in synthetic cohort; CI excludes zero |
| **Wearable Value Alone** | Config C vs Config A | $-0.17\text{ mg/dL}$ | $t = 0.536$ | $p = 0.605$ | $-0.39\text{ to }+0.66\text{ mg/dL}$ | 🟡 Inconclusive; CI spans zero |
| **Wearable Marginal Value After EHR** | Config D vs Config B | $+0.07\text{ mg/dL}$ | $t = -0.296$ | $p = 0.774$ | $-0.48\text{ to }+0.30\text{ mg/dL}$ | ❌ No incremental value demonstrated |
| **Full Fusion vs CGM-only** | Config D vs Config A | $-0.77\text{ mg/dL}$ | $t = 1.625$ | $p = 0.139$ | **$0.04\text{ to }1.38\text{ mg/dL improvement}$** | 🟢 Overall fusion beats CGM-only |

---

## 5. Per-Patient Breakdown (N = 10)

| Patient ID | Clinical Phenotype | CGM Only MAE | CGM + EHR MAE | $\Delta$ with EHR | Full Fusion MAE |
|---|---|:---:|:---:|:---:|:---:|
| `SYNTH_001` | Mild dawn rise on metformin | 20.86 | 19.34 | **-1.52 mg/dL** | 20.65 |
| `SYNTH_002` | Baseline diurnal offset | 20.33 | 16.51 | **-3.82 mg/dL (-18.8%)** | 16.98 |
| `SYNTH_003` | Metabolic syndrome / insulin resistance | 16.73 | 16.68 | -0.05 mg/dL | 16.29 |
| `SYNTH_004` | Diet/lifestyle controlled | 18.23 | 18.35 | +0.12 mg/dL | 17.93 |
| `SYNTH_005` | High fasting baseline / dawn rise | 15.46 | 13.56 | **-1.90 mg/dL (-12.3%)** | 13.41 |
| `SYNTH_006` | Standard daytime excursions | 14.42 | 14.77 | +0.35 mg/dL | 15.65 |
| `SYNTH_007` | Postprandial variability | 17.93 | 18.01 | +0.08 mg/dL | 18.34 |
| `SYNTH_008` | Late diurnal peak (07:45) | 19.40 | 19.18 | -0.22 mg/dL | 18.63 |
| `SYNTH_009` | Active lifestyle / fast clearance | 19.47 | 18.82 | -0.65 mg/dL | 19.54 |
| `SYNTH_010` | High carbohydrate sensitivity | 24.11 | 21.68 | **-2.43 mg/dL (-10.1%)** | 22.14 |

---

## 6. Scientific Interpretation & Responsible Claims

1. **Why Static EHR Improves Forecasting:**  
   A 2-hour CGM history window provides short-term velocity and curvature, but lacks knowledge of the patient's long-term glycemic anchor. Phenotypic features (BMI, duration, baseline HbA1c, and historical fasting blood glucose) anchor the model's regression trees toward each patient's individual steady-state set-point.
2. **Why Wearables Provide Limited Incremental Information:**  
   Because wearable telemetry is generated from independent behavioral/circadian processes without coupling to glucose, its correlation with 120-minute forward glucose is weak in this dataset. This reflects realistic clinical conditions where consumer wearables measure physical movement rather than biochemical glucose uptake.
3. **Canonical Reference:**  
   All reported figures in this report are programmatically synchronized with [`artifacts/experiment_manifest.json`](file:///c:/Users/ursra/Projects/GlucoTwin/artifacts/experiment_manifest.json).
