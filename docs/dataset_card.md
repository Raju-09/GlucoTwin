# GlucoTwin — Dataset Card: Synthetic CGM Benchmark

> Status: Verified via audit · Date: 2026-09-23  
> Audit report: `reports/audit.json` (SHA-256: `6795bf22969a0e103e00ed5edaf000422d686fd586b614b602350cf8d8912161`)

## 1. Dataset Identity

| Field | Value |
|---|---|
| **Name** | GlucoTwin Synthetic CGM Benchmark |
| **Version / release** | v1.0 |
| **Source URL** | Generated via `scripts/generate_synthetic_data.py` |
| **Access platform** | Local repository (`data/synthetic_demo/` and `data/raw/`) |
| **Access status** | Generated & Audited locally |
| **Date accessed** | 2026-09-23 |

## 2. License and Terms

| Field | Value |
|---|---|
| **License** | MIT License (open source) |
| **Redistribution permitted?** | Yes |
| **Publication of derived results permitted?** | Yes |
| **Raw data committed to repo?** | Synthetic demo copy in `data/synthetic_demo/` is committed. Real restricted data (e.g. DiaTrend/OhioT1DM) remains gitignored. |
| **Permitted use in this project** | Algorithmic validation, reproducible benchmarking, software integrity testing, and live dashboard demo. |

## 3. Cohort

| Field | Value |
|---|---|
| **Population** | Synthetic digital cohort (Type 1 diabetes physiological profile simulation) |
| **N patients** | 10 (`SYNTH_001` through `SYNTH_010`) |
| **Age range** | Simulated adult dynamics |
| **Condition** | Simulated insulin-dependent diabetes dynamics with dawn phenomenon |
| **Device(s)** | Simulated Continuous Glucose Monitor (CGM) with sensor noise ($AR(1)$) |
| **Study period** | 2024-03-01 00:00:00+00:00 to 2024-03-14 23:55:00+00:00 (14 days) |
| **Geography** | Synthetic / In-silico |

## 4. Data Structure

### Files

| Filename | Format | Rows | Description |
|---|---|---|---|
| `synthetic_cgm_benchmark.csv` | CSV | 38,942 | Time-stamped glucose and meal events for 10 patients |

### Confirmed Columns

| Column name | Type | Description | Units | Verified Range / Notes |
|---|---|---|---|---|
| `patient_id` | string | Patient identifier | — | 10 distinct IDs (`SYNTH_001` to `SYNTH_010`) |
| `timestamp` | datetime | Reading timestamp | UTC | Monotonically ascending, 0 duplicates |
| `glucose` | float | CGM sensor reading | mg/dL | Min 75.50, Median 138.40, Max 380.00 mg/dL |
| `meal_carbs_g` | float | Logged meal carbohydrates | grams | Sparse event column (98.8% NaN as expected outside meals) |
| `is_synthetic` | bool | Synthetic provenance flag | boolean | True for all records |

### Columns NOT present (verified absent)

- Basal insulin rate
- Bolus insulin dosage
- Physical activity / steps / heart rate
- Sleep staging

## 5. Data Characteristics

| Property | Value |
|---|---|
| **CGM sampling cadence** | 5.0 minutes (median: 5.0 min, 96.9% within $\pm 1.0$ min) |
| **Median sequence length per patient** | 336.0 hours (14 days) |
| **Total readings** | 38,942 across 10 patients (~3,894 readings/patient) |
| **Duplicate timestamps found?** | No (0 duplicates) |
| **Glucose unit** | mg/dL (confirmed by median 138.40 mg/dL) |
| **Physiologically impossible values?** | None (all within $[20, 600]$ mg/dL) |
| **Long gaps ($>30$ min)** | 15 sensor dropouts/gaps intentionally injected to test reliability logic |

## 6. Feasibility for 120-Minute Forecasting

| Check | Status | Evidence |
|---|---|---|
| Sufficient sequence length for input windows? | PASSED | Median sequence length is 14 days per patient ($>3,800$ readings) |
| Target ($t+120$) observable without interpolation? | PASSED | 5-min regular cadence allows targeting 24 steps ahead ($\pm 10$ min tolerance) |
| Enough patients for train/val/test split? | PASSED | 10 patients allow chronological within-patient and cross-patient evaluations |
| Patient-time split feasible without leakage? | PASSED | Strict chronological cut (e.g. Days 1-9 train, 10-11 val, 12-14 test) |
| Persistence baseline constructible? | PASSED | Last valid observation in window provides $y_t$ |

## 7. Synthetic Disclaimer

> **IMPORTANT**: This benchmark was created for end-to-end algorithmic and software validation. Performance metrics obtained on this benchmark demonstrate software correctness and relative algorithmic differences, but **do not establish clinical safety or efficacy** on human patients.
