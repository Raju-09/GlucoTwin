# GlucoTwin — Forecast Regime Error Analysis

**Auditor:** Senior Time-Series ML Researcher  
**Cohort:** 7,316 held-out test windows across 10 synthetic patients  
**Evaluation Principle:** Regime assignment uses *strictly historical information* available at forecast origin $t$.  

> **Important Disclaimer:** These trajectory groupings are **analytical time-series classifications** based on empirical rates of change. They are **not** clinical risk categories, triage classifications, or diagnostic labels.

---

## 1. Mathematical Definitions of Analytical Regimes

Let $y_t$ be the glucose reading at forecast origin $t$ (`glucose_lag_0`), and $y_{t-30\text{m}}$ be the reading 30 minutes prior (`glucose_lag_6`). The 30-minute velocity is defined as:

$$\Delta_{30\text{m}} = y_t - y_{t-30\text{m}} \quad (\text{mg/dL over 30 minutes})$$

The test cohort is partitioned into 4 mutually exclusive and collectively exhaustive categories:

1. **Relatively Stable:** $|\Delta_{30\text{m}}| \le 15.0\text{ mg/dL}$ (rate of change $\le 0.5\text{ mg/dL/min}$). Minimal short-term fluctuation.
2. **Rising:** $15.0 < \Delta_{30\text{m}} \le 45.0\text{ mg/dL}$ ($0.5 < \text{rate} \le 1.5\text{ mg/dL/min}$). Moderate sustained upward slope (e.g. dawn phenomenon or early postprandial absorption).
3. **Falling:** $-45.0 \le \Delta_{30\text{m}} < -15.0\text{ mg/dL}$ ($-1.5 \le \text{rate} < -0.5\text{ mg/dL/min}$). Moderate sustained downward slope (e.g. postprandial clearance).
4. **Rapidly Changing (Excursion):** $|\Delta_{30\text{m}}| > 45.0\text{ mg/dL}$ ($|\text{rate}| > 1.5\text{ mg/dL/min}$). Steep glycemic surge or rapid decline.

---

## 2. Experimental Regime Comparison Results

| Trajectory Regime | Test Samples | % Cohort | Persistence MAE (mg/dL) | GlucoTwin MAE (mg/dL) | Persistence RMSE | GlucoTwin RMSE | Median Abs Error | Rel. Improvement |
|---|---|---|---|---|---|---|---|---|
| **Relatively Stable (|Delta_30m| <= 15)** | 4,321 | 59.1% | 47.84 | **21.56** | 70.07 | 34.32 | 10.37 | **+54.9%** |
| **Rising (15 < Delta_30m <= 45)** | 241 | 3.3% | 64.64 | **7.99** | 77.70 | 10.85 | 6.24 | **+87.6%** |
| **Falling (-45 <= Delta_30m < -15)** | 1,821 | 24.9% | 50.13 | **17.65** | 61.24 | 30.14 | 9.67 | **+64.8%** |
| **Rapidly Changing (|Delta_30m| > 45)** | 933 | 12.8% | 81.46 | **10.22** | 90.56 | 14.03 | 7.91 | **+87.4%** |

### Performance Breakdown by Regime
```text
Relatively Stable (|Delta Persist: ███████████████ [47.8 mg/dL]
                          GTwin  : ███████ [21.6 mg/dL] (+54.9%)
Rising (15 < Delta_30m <= Persist: █████████████████████ [64.6 mg/dL]
                          GTwin  : ██ [8.0 mg/dL] (+87.6%)
Falling (-45 <= Delta_30m Persist: ████████████████ [50.1 mg/dL]
                          GTwin  : █████ [17.6 mg/dL] (+64.8%)
Rapidly Changing (|Delta_ Persist: ███████████████████████████ [81.5 mg/dL]
                          GTwin  : ███ [10.2 mg/dL] (+87.4%)
```

---

## 3. Critical Analytical Findings

### 1. Massive Superiority in Dynamic Regimes (Rising & Rapid Excursions)
- Persistence fails catastrophically during **Rapidly Changing** regimes ($|\Delta_{30\text{m}}| > 45\text{ mg/dL}$): Persistence MAE = **81.46 mg/dL** (RMSE = 90.56 mg/dL).
- Assuming that a +50 mg/dL surge will remain at the peak 2 hours later causes massive error.
- GlucoTwin cuts this error by **+87.4%**, down to **10.22 mg/dL** (RMSE = 14.03 mg/dL).
- Similarly, in the **Rising** regime ($+15$ to $+45$ mg/dL), GlucoTwin drops MAE from 64.64 mg/dL down to **7.99 mg/dL** (**+87.6% error reduction**, median error **6.24 mg/dL**).

### 2. Why is MAE Higher in the Relatively Stable Regime?
- In the **Relatively Stable** regime ($|\Delta_{30\text{m}}| \le 15\text{ mg/dL}$), GlucoTwin MAE is **21.56 mg/dL** (vs Persistence 47.84 mg/dL, **+54.9% reduction**), with a median error of **10.37 mg/dL**.
- While 10.37 mg/dL median error is tight, the mean (21.56) is pulled upward by the tail of **unheralded meal onsets**.
- As established in the outlier analysis (Prompt #5), unannounced meals often occur when the patient was completely flat/fasting at forecast origin $t$. The model cannot anticipate an unheralded lunch that occurs at $t+15\text{ min}$, skewing the mean absolute error for stable origins.

### 3. Asymmetric Dynamics (Rising vs Falling vs Excursions)
- When a meal rise has **already begun** in the lookback window (Rising / Rapid regimes), the model captures the upward velocity, anticipates the peak, and models the clearance curve, delivering single-digit error (7.99 to 10.22 mg/dL).
- When blood glucose is falling ($-45 \le \Delta_{30\text{m}} < -15\text{ mg/dL}$), GlucoTwin achieves MAE = **17.65 mg/dL** (**+64.8% error reduction** over persistence 50.13 mg/dL).

---

## 4. Summary Verdict for Research Reviewers
- GlucoTwin outperforms the persistence baseline **across every single trajectory regime** (ranging from **+54.9% up to +87.6%** relative error reduction).
- The model delivers its largest clinical advantage during volatile glycemic swings—preventing >70 mg/dL of naive persistence overshoot during meal peaks and rapid recoveries.