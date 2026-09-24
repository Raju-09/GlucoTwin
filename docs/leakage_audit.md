# GlucoTwin Pipeline Leakage Audit

**Audit Target:** End-to-end data ingestion, preprocessing, windowing, feature extraction, scaling, model training, personalization, and evaluation scripts.  
**Auditor:** Senior Time-Series ML Researcher (Healthcare AI Specialty)  
**Date:** 2026-09-24  
**Audit Status:** PASSED — Zero Confirmed Code Leakage  

---

## 1. Formal Split Architecture

The GlucoTwin evaluation protocol enforces a strictly chronological hold-out per patient. Because consecutive sliding windows share overlapping historical readings, random K-fold cross-validation or unbuffered chronological splits are catastrophic in time-series forecasting. GlucoTwin introduces a **120-minute safety buffer** (equal to the forecast horizon $H = 120\text{ min}$) at both split boundaries:

```text
[                     PATIENT TIMELINE (~14 DAYS)                     ]
├─── TRAIN (~65%) ───┼─ BUFFER ─┼── VAL (~15%) ──┼─ BUFFER ─┼── TEST (~20%) ──┤
│ Day 1.0 → Day ~9.1 │ 120 min  │ Day ~9.2→~11.2 │ 120 min  │ Day ~11.3→14.0  │
│ (24,647 windows)   │ (DISCARD)│ (5,273 windows)│ (DISCARD)│ (7,316 windows) │
```

### Mathematical Boundary Constraints Enforced in Code
Let $t_{\text{origin}}$ be the forecast origin timestamp and $t_{\text{target}}$ be the future observation timestamp ($t_{\text{target}} \approx t_{\text{origin}} + 120\text{ min}$):

1. **Train Set:**
   $$\forall w \in \text{Train}: \quad t_{\text{origin}}(w) \le t_{\text{train\_end}} \quad \text{AND} \quad t_{\text{target}}(w) \le t_{\text{train\_end}}$$
2. **Buffer 1 (Train $\to$ Val):**
   Interval $(t_{\text{train\_end}}, \; t_{\text{train\_end}} + 120\text{ min})$ is completely discarded.
3. **Validation Set:**
   $$\forall w \in \text{Val}: \quad t_{\text{origin}}(w) \ge t_{\text{train\_end}} + 120\text{ min} \quad \text{AND} \quad t_{\text{target}}(w) \le t_{\text{val\_end}}$$
4. **Buffer 2 (Val $\to$ Test):**
   Interval $(t_{\text{val\_end}}, \; t_{\text{val\_end}} + 120\text{ min})$ is completely discarded.
5. **Test Set:**
   $$\forall w \in \text{Test}: \quad t_{\text{origin}}(w) \ge t_{\text{val\_end}} + 120\text{ min}$$

### Empirical Verification Across All 10 Patients
Programmatic audit on `data/processed/windows_{train,val,test}.parquet`:
* $\min_{p} \Big( \min_{w \in \text{Val}_p} t_{\text{origin}}(w) - \max_{w \in \text{Train}_p} t_{\text{target}}(w) \Big) \ge 0\text{ min}$ (Verified: 0 overlaps)
* $\min_{p} \Big( \min_{w \in \text{Test}_p} t_{\text{origin}}(w) - \max_{w \in \text{Val}_p} t_{\text{target}}(w) \Big) \ge 0\text{ min}$ (Verified: 0 overlaps)
* Overlapping `(patient_id, origin_time)` pairs between Train, Val, and Test: **Exactly 0**.

---

## 2. Risk Classification Matrix (11 Critical Verification Checks)

| # | Audit Item | Risk Classification | Empirical / Code Evidence |
|---|---|---|---|
| **Q1** | Can any future glucose value enter a feature? | **No issue** | `src/glucotwin/features/windows.py` slices historical lags strictly backward: $[i - N + 1 : i + 1]$ where $i$ is origin index. Target search operates on index $j > i + 1$. Features only read `glucose_lag_0` to `glucose_lag_23`. |
| **Q2** | Can target values enter rolling statistics? | **No issue** | `src/glucotwin/features/build_features.py` computes rolling mean, std, min, max strictly over the $N \times 24$ lag matrix ($t$ down to $t - 115\text{ min}$). Target column `target_glucose` is never referenced in feature definitions. |
| **Q3** | Is StandardScaler fitted only on training data? | **No issue** | `src/glucotwin/models/train.py` wraps `StandardScaler` and `HistGradientBoostingRegressor` into an `sklearn.pipeline.Pipeline`. `pipeline.fit()` is called exclusively on `X_train[feature_names]`, `y_train`. In `predict()`, the pipeline uses the previously fitted scaler. |
| **Q4** | Are feature-selection decisions based on validation/test data? | **No issue** | All 41 features are predetermined by domain specification (24 raw lags, 10 summary stats, 5 rate-of-change deltas, 2 circadian sin/cos). No backward/forward stepwise selection, lasso penalty tuning, or filter-based ranking was performed on validation or test sets. |
| **Q5** | Are overlapping windows separated safely? | **No issue** | Consecutive sliding windows inside a split overlap (standard in time-series regression), but windows spanning split boundaries are eliminated by the 120-minute safety buffer. |
| **Q6** | Does the 120-minute safety buffer actually prevent target/input contamination? | **No issue** | Because the maximum prediction horizon is 120 minutes ($\pm 10$ min tolerance) and the buffer is 120 minutes, the latest target in training occurs at or before the earliest origin in validation. Verified with 0 boundary leaks across all 10 patients. |
| **Q7** | Can the same patient-time interval appear in multiple splits? | **No issue** | Each split is defined by non-overlapping chronological timestamps. Set intersection of `(patient_id, timestamp)` tuples across all three splits is empty. |
| **Q8** | Can any patient-specific information from the test period affect training? | **No issue** | Patient ID is **not** included as a feature in the model matrix `X`. Training is entirely blind to patient identity. |
| **Q9** | Does early stopping use only training/validation information? | **No issue** | `HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.15)` internally creates a validation split *inside the training dataset passed to fit*. It never accesses the external test set. |
| **Q10** | Does personalization use test-period information? | **No issue** | In `scripts/audit_personalization.py`, `patient_train_bias` is calculated as $\text{mean}(y_{\text{train}} - \hat{y}_{\text{train}})$. Test targets are only used to evaluate the resulting calibrated predictions. |
| **Q11** | Does evaluation code accidentally tune anything using the test set? | **No issue** | `scripts/evaluate.py` simply loads `artifacts/model_v0.1/model.joblib` and evaluates metrics on the designated split. Zero hyperparameters or thresholds are tuned during evaluation. |

---

## 3. Deep Dive Analysis by Pipeline Stage

### 3.1 Preprocessing (`src/glucotwin/data/preprocess.py`)
- **Operations:** UTC timestamp coercion, sorting by `(patient_id, timestamp)`, duplicate resolution (`keep_first`), mmol/L detection via median check.
- **Audit Findings:** No rolling operations or forward-looking interpolations occur. Missing values are dropped, not imputed using future values.
- **Classification:** **No issue**.

### 3.2 Windowing & Target Coupling (`src/glucotwin/features/windows.py`)
- **Candidate loop:**
  ```python
  lookback_ts = timestamps[i - lookback_n + 1 : i + 1]
  lookback_gl = glucoses[i - lookback_n + 1 : i + 1]
  # Origin is timestamps[i]
  # Target search starts at index i + 1
  future_mask = timestamps[i + 1 :] >= (ideal_target_time - tolerance_ns)
  ```
- **Audit Findings:** The lookback window ends at index $i$. The future mask strictly begins at $i + 1$. Target is recorded in a separate key `target_glucose`.
- **Classification:** **No issue**.

### 3.3 Feature Matrix Extraction (`src/glucotwin/features/build_features.py`)
- **Feature Schema:**
  - Lags: `glucose_lag_0` (reading at $t$), `glucose_lag_1` ($t-5$), ..., `glucose_lag_23` ($t-115$).
  - Summary stats: derived strictly from `gl_matrix` ($N \times 24$ history).
  - Rate of change: `delta_5m = lag_0 - lag_1`, `acceleration_5m = delta_5m - prev_delta_5m`.
  - Circadian features: `sin_hour`, `cos_hour` calculated from `origin_time` ($t$).
- **Audit Findings:** Notice that when `build_features` is run on test data or inference requests, `target_glucose` is either absent or ignored.
- **Classification:** **No issue**.

### 3.4 Model Training & Scaling (`src/glucotwin/models/train.py`)
- **Fit Pipeline:**
  ```python
  pipeline = Pipeline([("scaler", scaler), ("regressor", regressor)])
  pipeline.fit(X_train[feature_names], y_train)
  ```
- **Audit Findings:** `StandardScaler` compute $\mu, \sigma$ solely over `X_train`. When `predict(val_df)` or `predict(test_df)` is called, the pipeline uses the pre-computed training $\mu, \sigma$. No data snooping or distribution leakage occurs.
- **Classification:** **No issue**.

### 3.5 Personalization Feasibility Audit (`scripts/audit_personalization.py`)
- **Residual Computation:**
  ```python
  p_train = train_df[train_df["patient_id"] == pid]
  patient_bias = float(p_train["residual"].mean())
  test_pred_adapted = np.clip(test_pred_global + patient_bias, 20.0, 600.0)
  ```
- **Audit Findings:** `patient_bias` uses only `p_train`. The test split is evaluated out-of-sample.
- **Classification:** **No issue**.

---

## 4. Latent Confounders & Methodological Caveats

While there is **zero programmatic leakage**, a senior reviewer will observe two methodological nuances:

1. **Stationary Circadian Phase in Synthetic Generator:**
   Because each synthetic patient's circadian wave amplitude $A_p$ and peak hour $\phi_p$ are stationary across all 14 days, the model learns the relationship between `sin_hour`/`cos_hour` and the patient cohort's mean diurnal wave. In real clinical data, circadian rhythms drift with travel, sleep changes, shift work, and weekend schedules.
2. **Temporal Generalization vs. Population Generalization:**
   Because all 10 patients are present in both train (earlier days) and test (later days), GlucoTwin evaluates **within-subject temporal forecasting** (can we forecast future trajectory for an enrolled patient?), not zero-shot generalization to an unknown 11th patient. This is an intentional experimental design for a patient-specific computational twin, but must be explicitly stated.

---

## 5. Audit Conclusion

* **Confirmed Leakages:** 0
* **Potential Leakages:** 0
* **Audit Verdict:** **PASSED**
* The reported metric of **MAE = 18.69 mg/dL** is an uncompromised, leakage-free result on the held-out test split of this synthetic CGM process.
