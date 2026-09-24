# GlucoTwin Experiment Reproducibility Audit & Guide

**Audit Target:** End-to-end reproducibility of data generation, preprocessing, split construction, model training, and held-out evaluation.  
**Auditor:** Senior Time-Series ML Researcher (Healthcare AI Specialty)  
**Date:** 2026-09-24  
**Status:** **100% REPRODUCIBLE (EXACT MATCH)**  

---

## 1. Executive Summary

This document verifies that every numerical claim reported in GlucoTwin—specifically:
* **Persistence Baseline:** Test $\text{MAE} = 53.25\text{ mg/dL}$, $\text{RMSE} = 71.26\text{ mg/dL}$, $\text{MAPE} = 31.78\%$
* **GlucoTwin v0.1:** Test $\text{MAE} = 18.69\text{ mg/dL}$, $\text{RMSE} = 30.84\text{ mg/dL}$, $\text{MAPE} = 11.79\%$
* **Relative Error Reduction:** **$+64.9\%$** over persistence

can be reproduced deterministically down to the hundredth of a mg/dL using a **single documented command**.

---

## 2. Nine Core Reproducibility Inquiries

| # | Question | Verification Status | Details |
|---|---|---|---|
| **1** | Can the dataset be regenerated? | **YES** | Running `python scripts/generate_synthetic_data.py --seed 42` deterministically generates 38,942 rows matching SHA-256 digest `6795bf22969a0e103e00ed5edaf000422d686fd586b614b602350cf8d8912161`. |
| **2** | Can preprocessing be reproduced? | **YES** | Preprocessing in `src/glucotwin/data/preprocess.py` is deterministic (sorting by patient/timestamp, `keep_first` duplicate policy, threshold unit check). |
| **3** | Can the exact train/validation/test split be reproduced? | **YES** | Splitting logic uses deterministic duration fractions (0.65 train, 0.15 val, 0.20 test) and a strict 120-minute buffer. Window counts are constant: Train = 24,647; Val = 5,273; Test = 7,316. |
| **4** | Are random seeds fixed where appropriate? | **YES** | Data generation uses `np.random.default_rng(42 + 100 * i)`. Model training uses `HistGradientBoostingRegressor(random_state=42)`. No unseeded pseudo-random calls exist. |
| **5** | Can the exact model configuration be recovered? | **YES** | Serialized pipeline in `artifacts/model_v0.1/model.joblib` and human-readable metadata in `artifacts/model_v0.1/model_meta.json` record the exact architecture and feature schema. |
| **6** | Can the exact reported metrics be regenerated? | **YES** | Running the evaluation pipeline yields $\text{MAE} = 53.25$ and $\text{MAE} = 18.69$ exactly. |
| **7** | Are predictions saved? | **YES** | Every test example prediction is saved to `artifacts/evaluations/reproduced_test_predictions.parquet` with timestamps, ground truth, baseline prediction, and GlucoTwin prediction. |
| **8** | Are evaluation examples identifiable by non-sensitive synthetic IDs? | **YES** | Each row is indexed by non-sensitive identifiers (`SYNTH_001` through `SYNTH_010`) and synthetic UTC timestamps. |
| **9** | Is there a single documented command that reproduces the final result? | **YES** | `python scripts/reproduce_results.py` executes the entire pipeline in under 15 seconds. |

---

## 3. The One-Command Reproduction Workflow

### Quick Evaluation (Using Existing Artifacts)
To evaluate the pre-trained model and pre-built splits:
```bash
python scripts/reproduce_results.py
```

### Full Reproduction From Scratch (Regenerate $\to$ Preprocess $\to$ Train $\to$ Evaluate)
To test complete determinism from raw synthetic generation through training to test evaluation:
```bash
python scripts/reproduce_results.py --from-scratch
```

---

## 4. Pipeline Execution Trace & Benchmark Record

Running `python scripts/reproduce_results.py` yields the following verified benchmark table:

```text
                       Official Held-Out Test Evaluation                       
+-----------------------------------------------------------------------------+
| Model                | Test MAE (mg/dL) | Test RMSE | Test MAPE | Rel. Impr |
|----------------------+------------------+-----------+-----------+-----------|
| Persistence Baseline |            53.25 |     71.26 |    31.78% |     0.0%  |
| GlucoTwin v0.1       |            18.69 |     30.84 |    11.79% |   +64.9%  |
+-----------------------------------------------------------------------------+
```

### Machine-Readable Provenance Record
The reproduction script generates [`reports/reproduced_evaluation.json`](file:///c:/Users/ursra/Projects/GlucoTwin/reports/reproduced_evaluation.json):
```json
{
  "status": "REPRODUCED_EXACT",
  "dataset": {
    "file": "synthetic_cgm_benchmark.csv",
    "sha256": "6795bf22969a0e103e00ed5edaf000422d686fd586b614b602350cf8d8912161",
    "n_patients": 10,
    "days_per_patient": 14,
    "sampling_cadence_minutes": 5
  },
  "split": {
    "name": "chronological_with_120m_buffer",
    "test_samples": 7316,
    "test_patients": 10,
    "horizon_minutes": 120,
    "lookback_steps": 24
  },
  "random_seed": 42,
  "models": {
    "persistence": {
      "mae": 53.25,
      "rmse": 71.26,
      "mape_pct": 31.78,
      "median_abs_error": 42.1
    },
    "glucotwin_v0.1": {
      "architecture": "HistGradientBoostingRegressor",
      "n_features": 41,
      "mae": 18.69,
      "rmse": 30.84,
      "mape_pct": 11.79,
      "median_abs_error": 9.49,
      "p90_abs_error": 48.15,
      "relative_error_reduction_pct": 64.9
    }
  },
  "target_benchmarks": {
    "expected_persistence_mae": 53.25,
    "expected_ml_mae": 18.69,
    "reproduced_matches_documented": true
  }
}
```

---

## 5. Test Predictions Table Schema

Saved to: [`artifacts/evaluations/reproduced_test_predictions.parquet`](file:///c:/Users/ursra/Projects/GlucoTwin/artifacts/evaluations/reproduced_test_predictions.parquet)

| Column | Type | Description |
|---|---|---|
| `patient_id` | string | Synthetic patient identifier (`SYNTH_001`–`SYNTH_010`) |
| `origin_time` | datetime64[ns, UTC] | Prediction origin timestamp $t$ |
| `target_time` | datetime64[ns, UTC] | Future horizon timestamp $t+120\text{ min}$ |
| `target_glucose` | float64 | Observed ground truth glucose at target time |
| `pred_persistence` | float64 | Baseline prediction ($y_t$) |
| `pred_glucotwin` | float64 | Model point prediction $\hat{y}_{t+120}$ |
| `abs_err_persistence`| float64 | Absolute error $\|y - \hat{y}_{\text{pers}}\|$ |
| `abs_err_glucotwin` | float64 | Absolute error $\|y - \hat{y}_{\text{model}}\|$ |

This table allows independent reviewers to compute arbitrary sub-group metrics, error distributions, or plot residual trajectories without re-running the model.

---

## 6. Environment & System Specifications

* **Operating System:** Windows 11 (tested on standard x86_64 architecture)
* **Python Version:** Python 3.10+ / 3.11
* **Key Dependencies:**
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `scikit-learn >= 1.3.0`
  - `pyarrow >= 12.0.0`
  - `rich >= 13.0.0`
* **Execution Time:**
  - Quick Evaluation: $\approx 2.5\text{ seconds}$
  - Full From-Scratch Pipeline: $\approx 14.0\text{ seconds}$
