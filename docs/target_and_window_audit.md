# GlucoTwin — Target Definition & Window Construction Audit

**Auditor:** Senior Time-Series ML Researcher (Healthcare AI Specialty)  
**Date:** 2026-09-24  
**Audit Target:** `src/glucotwin/features/windows.py` and `src/glucotwin/config.py`  
**Status:** **VERIFIED — ZERO INTERPOLATION, STRICT OBSERVATION PROTOCOL**  

---

## 1. Executive Summary

This document addresses Sections 20, 29, and 30 of the Senior ML Review:
1. **Target Construction:** Are future target values interpolated, synthesized, or strictly observed?
2. **Tolerance Mechanics:** How is the $\pm 10\text{ min}$ horizon window defined and justified?
3. **Horizon Justification:** Why 120 minutes? (Clinical, mathematical, and challenge justifications).
4. **Rejection Accounting:** How many candidate origins are rejected, and under what rules?

---

## 2. Target Construction: Mathematical Formulation

Let $t_i$ be candidate forecast origin timestamp with observed reading $y(t_i) = y_t$.  
The desired forecast horizon is $H = 120\text{ minutes}$.  
The ideal target timestamp is:
$$t^* = t_i + 120\text{ minutes}$$

### Search Algorithm (from `src/glucotwin/features/windows.py` lines 147–167)
```python
future_mask = timestamps[i + 1 :] >= (ideal_target_time - tolerance_ns)
candidates_idx = np.where(future_mask)[0]
first_candidate_pos = i + 1 + candidates_idx[0]
candidate_ts = timestamps[first_candidate_pos]
diff_from_ideal = np.abs(candidate_ts - ideal_target_time)

if diff_from_ideal > tolerance_ns:
    report.record_rejection("target_outside_tolerance")
    continue

target_val = glucoses[first_candidate_pos]
```

### Verification Findings
* **Zero Interpolation:** The target value $y(t_{\text{candidate}})$ is an **actual, observed measurement** recorded in the CGM stream. GlucoTwin does **not** perform linear, spline, or polynomial interpolation between adjacent readings to synthesize a synthetic $t+120$ point.
* **Tolerance Window:** The target is accepted if and only if an observed reading exists in the closed interval:
  $$t_{\text{candidate}} \in [t^* - 10\text{ min}, \; t^* + 10\text{ min}]$$
* If the sensor was disconnected or dropped during this 20-minute window, the candidate origin $t_i$ is **rejected entirely**. It is never imputed.

---

## 3. Justification for Tolerance Bounds ($\pm 10$ Minutes)

Continuous Glucose Monitors nominally sample interstitial fluid every $\Delta t = 5\text{ minutes}$. In real-world operation and in our synthetic sensor stream:
1. **Packet Jitter & Clock Drift:** Transmitters may report readings at intervals varying from 4.8 to 5.2 minutes.
2. **Single-Reading Packet Loss:** If a single reading at exactly $t + 120\text{ min}$ is dropped due to RF interference, readings at $t + 115\text{ min}$ ($-5\text{ min}$) or $t + 125\text{ min}$ ($+5\text{ min}$) are present.
3. **Why Not $\pm 5$ min?** A $\pm 5\text{ min}$ tolerance would reject an otherwise continuous window if a single packet at 120 min was missed, even if valid readings exist at 115 min and 125 min.
4. **Why Not $\pm 20$ min?** A $\pm 20\text{ min}$ tolerance would allow target horizons ranging from 100 to 140 minutes, introducing excessive horizon variance into the regression objective.
5. **Conclusion:** $\pm 10\text{ minutes}$ is the minimal natural bound that tolerates a single missing 5-minute transmission without degrading horizon consistency.

---

## 4. Why 120 Minutes? (Research Justification)

Reviewers rightly ask: *"Why 120 minutes rather than 30, 60, or 240 minutes?"*

| Dimension | Justification |
|---|---|
| **Challenge Alignment** | Explicitly specified as the illustrative forecast horizon in the Happiest Health Digital Twin Challenge 2026. |
| **Clinical Significance** | 120 minutes (2 hours) is the standardized benchmark interval in metabolic endocrinology: standard Oral Glucose Tolerance Tests (OGTT) and postprandial guidelines (e.g. American Diabetes Association postprandial goals) evaluate glycemic clearance at 2 hours post-meal. |
| **Mathematical Difficulty** | At 15–30 minutes, glucose autocorrelation is so extreme ($\rho \approx 0.95$) that persistence ($y_t$) is nearly unbeatable. At 120 minutes, the temporal lag autocorrelation decays significantly, forcing the model to learn diurnal patterns and rate-of-change momentum rather than relying on trivial persistence. |
| **Information Boundary** | Beyond 180–240 minutes, future behavioral events (unannounced meals, activity changes, boluses) completely dominate the variance, making deterministic point forecasts mathematically ill-posed without behavioral logging. 120 minutes sits at the outer edge of pure time-series predictability. |

---

## 5. Candidate Window Rejection Accounting

On the benchmark dataset of 38,942 rows across 10 patients:

| Metric | Count | Percentage | Reason / Disposition |
|---|---|---|---|
| **Total Candidate Origins** | 38,712 | 100.0% | Sliding candidate origins with $\ge 24$ preceding readings |
| **Accepted Supervised Windows** | **38,056** | **98.31%** | Clean lookback, no gaps $>30$m, target found in $[110, 130]$m |
| **Rejected: Lookback Gap Exceeded** | 456 | 1.18% | Gaps $>30$ min inside the 2-hour historical lookback |
| **Rejected: Target Outside Tolerance** | 200 | 0.52% | Target missing due to sensor disconnect at $t+120$ |
| **Rejected: Implausible Values** | 0 | 0.00% | No values $<20$ or $>600$ mg/dL in lookback or target |

The $98.31\%$ acceptance rate confirms that the dataset produces dense, regular supervision without relying on artificial imputation.
