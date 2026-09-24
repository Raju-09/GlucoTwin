# GlucoTwin — Feature Ablation Study Report

**Auditor:** Senior Time-Series ML Researcher  
**Split:** Chronological hold-out (120-min safety buffer)  
**Test Cohort:** 7,316 examples across 10 synthetic patients  
**Model Architecture:** `StandardScaler` + `HistGradientBoostingRegressor` (identical hyperparameters across all configurations)  

---

## 1. Experimental Results Summary

| Configuration | Feat Count | Test MAE (mg/dL) | Test RMSE (mg/dL) | Test MAPE (%) | Median Error | Rel. Gain vs Persistence |
|---|---|---|---|---|---|---|
| **Persistence Baseline ($y_t$)** | — | 53.25 | 71.26 | 31.78% | 42.10 | 0.0% (Baseline Floor) |
| **Config A (Latest Glucose Only)** | 1 | **34.05** | 45.61 | 21.52% | 25.41 | **+36.1%** |
| **Config B (Raw Lags History)** | 24 | **28.88** | 41.40 | 18.08% | 18.39 | **+45.8%** |
| **Config C (Lags + Rolling Stats)** | 34 | **28.93** | 41.98 | 18.07% | 18.17 | **+45.7%** |
| **Config D (Lags + Rolling + Velocity)** | 39 | **28.91** | 42.25 | 18.09% | 17.97 | **+45.7%** |
| **Config E (Full Feature Set)** | 41 | **18.69** | 30.84 | 11.79% | 9.49 | **+64.9%** |

---

## 2. Deep-Dive Answers to Core Research Questions

### Comparative Progress Breakdown
```text
Baseline (Persistence yt)    [53.25 mg/dL] ████████████████████████████████████ (Floor)
Config A (Latest yt only)    [34.05 mg/dL] ███████████████████████ (-19.20 mg/dL, +36.1%)
Config B (24 raw lags)       [28.88 mg/dL] ███████████████████     (-5.17 mg/dL, +45.8%)
Config C (+ Rolling stats)   [28.93 mg/dL] ███████████████████     (+0.05 mg/dL, redundant)
Config D (+ Velocity/accel)  [28.91 mg/dL] ███████████████████     (-0.02 mg/dL, redundant)
Config E (+ Circadian sin/cos)[18.69 mg/dL] ████████████           (-10.22 mg/dL, +64.9%)
```

### Q1: Which feature groups improve performance?
1. **Latest Glucose Non-Linear Calibration (Persistence $\to$ Config A):**
   - Fitting a gradient boosted tree on just $y_t$ reduces MAE from 53.25 to 34.05 mg/dL. This $-19.20\text{ mg/dL}$ drop reflects the tree learning non-linear reversion toward the population mean across 120 minutes.
2. **Historical Lags / Temporal State (Config A $\to$ Config B):**
   - Adding 24 historical lags ($t-5\text{m}$ to $t-115\text{m}$) further drops MAE from 34.05 to 28.88 mg/dL (a **$-5.17\text{ mg/dL}$ or 15.2% incremental error reduction**). This confirms that recent trajectory history carries essential predictive momentum beyond the instantaneous point measurement.
3. **Circadian Diurnal Sin/Cos (Config D $\to$ Config E):**
   - Adding `sin_hour` and `cos_hour` causes the largest single jump: MAE drops from 28.91 to 18.69 mg/dL (a **$-10.22\text{ mg/dL}$ or 35.3% incremental error reduction**). At a 2-hour horizon, time-of-day information strongly informs the model whether the patient is in nocturnal fasting, morning dawn phenomenon, or standard post-meal periods.

### Q2: Which feature groups provide little or no benefit?
- **Rolling Statistics (Config B $\to$ Config C):**
  - MAE shifts from 28.88 to 28.93 mg/dL ($+0.05\text{ mg/dL}$ change). 10 rolling summary moments (mean, std, min, max across 30m, 60m, full lookback) provide zero additive predictive power.
- **Rate-of-Change Velocity & Acceleration (Config C $\to$ Config D):**
  - MAE shifts from 28.93 to 28.91 mg/dL ($-0.02\text{ mg/dL}$ change). 5 delta and acceleration features are essentially redundant.
- **Why?**
  - Decision tree ensembles inherently split on individual axis-aligned lag values. Multiple sequential splits on `lag_0` and `lag_1` already approximate $(x_0 - x_1)$ and rolling thresholds. Hand-crafted aggregations do not add new mutual information when all 24 raw lags are already present.

### Q3: Does the result support the claim that GlucoTwin maintains a time-series state rather than merely copying the latest reading?
- **Yes, conclusively.**
  - If the model merely relied on the latest reading, Config B (24 lags) would not improve upon Config A (latest only) by 5.17 mg/dL.
  - Furthermore, persistence ($y_{t+120} = y_t$) has an MAE of 53.25 mg/dL, while GlucoTwin achieves 18.69 mg/dL (a 64.9% error reduction). The model actively incorporates temporal trajectory and diurnal phase.

---

## 3. Methodological Caveats & Honesty
1. **Circadian Regularity in In Silico Data:**
   - In this procedural synthetic dataset, diurnal rhythm follows a stationary cosine wave across all 14 days. The dramatic $-10.22\text{ mg/dL}$ gain from `sin_hour`/`cos_hour` reflects the high regularity of this simulation. On real human patient CGM data with irregular sleep schedules, variable meal times, and shift work, circadian features will likely carry higher variance and lower certainty.
2. **No Causal Claim:**
   - Feature predictive importance does not imply biochemical causality. Features like `sin_hour` are temporal correlates of habitual meal and sleep behaviors, not metabolic drivers in themselves.
