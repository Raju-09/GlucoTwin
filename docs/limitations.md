# GlucoTwin — Known Limitations and Failure Cases

> This document is part of responsible disclosure. A system that cannot describe its failure modes should not be trusted.

## 1. What This Prototype Does NOT Claim

- It is **not** a medical device.
- It is **not** clinically validated.
- It does **not** provide diagnosis, treatment recommendations, or insulin dosing advice.
- It does **not** generalize to patient populations outside the training cohort.
- Forecast accuracy on real patients in clinical settings is **unknown**.
- The system has **not** been tested with real-time wearable hardware.

## 2. Data Failures

| Failure | System behavior | Limitation |
|---|---|---|
| Missing CGM readings | Reliability checker flags `degraded` or `withheld` | Threshold is a rule, not clinically validated |
| Duplicate timestamps | Resolved by documented policy (keep_first) | May not match clinical ground truth |
| Stale latest reading | Withheld if older than freshness threshold | Threshold is configurable, not evidence-based |
| mg/dL vs mmol/L mismatch | Flagged and rejected | Relies on value-range heuristic |
| Physiologically impossible values | Flagged and excluded | Definition of "impossible" uses documented bounds |
| Clock / time-zone errors | Parsing validated; edge cases documented | Daylight-saving transitions may still cause errors |
| Activity / meal stream absent | Feature set uses missing-channel indicator | Model behavior under missing context is uncharacterized |
| Patient ID mismatch | Enforced at ingestion | Cannot detect mislabeled records within a single patient ID |

## 3. Modeling Failures

| Failure | Notes |
|---|---|
| New patient with no history | Falls back to population model or withholds if personalization required |
| Distribution shift | Not detected; predictions may silently degrade |
| Rare glucose excursions | Average MAE may hide clinically important high-error events |
| Model predicts smooth curve during sharp change | Persistence baseline may outperform at transitions |
| Personalization overfitting | Guarded by chronological split; still possible with small per-patient datasets |
| Treatment change | Historical behavior may not transfer; model does not detect this |

## 4. Evaluation Limitations

| Limitation | Notes |
|---|---|
| Single dataset | Results may not generalize to other cohorts, devices, or conditions |
| Controlled study data | Real-world data may have higher noise, more irregular cadence |
| Horizon sensitivity | MAE at exactly 120 min may differ from shorter horizons; not separately reported unless explicitly evaluated |
| Event-specific error | Aggregate MAE does not capture false-negative rate on high-glucose events |
| Calibration | Uncertainty intervals (if shown) are validated on held-out data only; coverage may not hold under distribution shift |

## 5. Product and Human Failures

| Failure | Required behavior |
|---|---|
| Clinician misreads forecast as diagnosis | Clear labels, limitations visible on every screen |
| Model service unavailable | API returns service-unavailable status; dashboard shows last known state |
| Wrong demo patient selected | Strong patient context label; no real-patient data in demo |
| New model version deployed | Version metadata shown in UI; regression tests required |
| Missing input treated as zero | Explicit missing-indicator feature; not silent zero-imputation |

## 6. Claims Explicitly NOT Made

- We do not claim that GlucoTwin improves clinical outcomes.
- We do not claim that synthetic demo performance predicts real-world performance.
- We do not claim that uncertainty intervals are calibrated for patients outside the evaluation cohort.
- We do not claim that personalization always helps; it is an empirical question answered by our evaluation.
- We do not claim that the 120-minute forecast horizon is clinically actionable without further validation.
