# Synthetic Data Provenance & Generative Process Audit

**Project:** GlucoTwin (Research Prototype)  
**Role:** Senior Machine Learning Researcher & Skeptical Healthcare-AI Reviewer  
**Audit Target:** Procedural CGM Generator (`scripts/generate_synthetic_data.py`), Supervised Windowing (`src/glucotwin/features/windows.py`), and Chronological Splitting (`src/glucotwin/data/splits.py`)  
**Date:** 2026-09-24  

---

## 1. Executive Summary & Purpose

The benchmark dataset powering the current GlucoTwin prototype is **100% in silico generated**. Because no real patient data (e.g., from DiaTrend or OhioT1DM) was downloaded, the observed performance metrics (MAE = 18.69 mg/dL vs. persistence 53.25 mg/dL) reflect learning ability **on this specific mathematical simulation**, rather than generalizable clinical accuracy on human physiology.

This document formally records the exact generative equations, patient parameter distributions, meal excursion mechanics, noise models, seed controls, and architectural boundaries governing the synthetic data.

---

## 2. Generative Process Specifications

### 2.1 File Location & Generation Target
* **Generator Source Code:** [`scripts/generate_synthetic_data.py`](file:///c:/Users/ursra/Projects/GlucoTwin/scripts/generate_synthetic_data.py)
* **Output Artifacts:**
  - `data/raw/synthetic_cgm_benchmark.csv` (SHA-256: `6795bf22969a0e103e00ed5edaf000422d686fd586b614b602350cf8d8912161`)
  - `data/synthetic_demo/synthetic_cgm_benchmark.csv` (Identical copy for demo & UI)
* **Row Count:** 38,942 rows (post-dropout filtering)
* **Sampling Cadence:** 5.0 minutes nominal ($\Delta t = 5\text{ min}$)
* **Cohort:** 10 simulated patients (`SYNTH_001` to `SYNTH_010`), 14 calendar days each (2024-03-01 00:00:00 to 2024-03-14 23:55:00 UTC).

---

### 2.2 Mathematical Model of Glucose Trajectories

For patient $p$ at step $i$ corresponding to time $t$ (in hours, $t \in [0, 24)$):

$$G_p(t_i) = \text{clip}\Big( B_p(t_i) + M_p(t_i) + \eta_p(t_i), \; 45.0, \; 380.0 \Big)$$

#### Component 1: Circadian Basal Oscillation $B_p(t_i)$
Models fasting glucose and the dawn phenomenon as a diurnal cosine wave:
$$B_p(t_i) = \mu_{B, p} + A_p \cdot \cos\left( \frac{2\pi (t_i - \phi_p)}{24.0} \right)$$
Where patient-specific parameters are drawn uniformly at instantiation:
* $\mu_{B, p} \sim \mathcal{U}(105.0, 140.0)\text{ mg/dL}$ (individual fasting baseline)
* $A_p \sim \mathcal{U}(8.0, 20.0)\text{ mg/dL}$ (circadian amplitude)
* $\phi_p \sim \mathcal{U}(5.0, 8.0)\text{ hours}$ (morning peak hour, dawn phenomenon timing)

#### Component 2: Postprandial Excursions $M_p(t_i)$
Meals are scheduled 3 times daily, plus an optional afternoon snack ($p=0.6$ probability):
* Breakfast: $t_{\text{meal}} \sim \mathcal{U}(7.5, 9.0)\text{ h}$, Carbs $C \sim \mathcal{U}(35, 65)\text{ g}$
* Lunch: $t_{\text{meal}} \sim \mathcal{U}(12.5, 14.0)\text{ h}$, Carbs $C \sim \mathcal{U}(50, 85)\text{ g}$
* Dinner: $t_{\text{meal}} \sim \mathcal{U}(18.5, 20.5)\text{ h}$, Carbs $C \sim \mathcal{U}(55, 95)\text{ g}$
* Snack (optional): $t_{\text{meal}} \sim \mathcal{U}(15.5, 16.5)\text{ h}$, Carbs $C \sim \mathcal{U}(15, 30)\text{ g}$

Each meal event triggers a dual-exponential absorption/clearance curve over a 240-minute window:
$$E(\tau) = \max\left(0, \; 2.5 \cdot \left(e^{-\tau / 60.0} - e^{-\tau / 20.0}\right)\right), \quad \tau \in [0, 240\text{ min}]$$
The excursion curve is normalized to peak at 1.0 and scaled by carb sensitivity $S_p$:
$$\Delta G_{\text{meal}}(\tau) = C \cdot S_p \cdot \frac{E(\tau)}{\max_\tau E(\tau)}$$
* $S_p \sim \mathcal{U}(1.8, 3.2)\text{ mg/dL per gram carb}$ (individual insulin/carb sensitivity factor)
Multiple overlapping meal curves add linearly: $M_p(t_i) = \sum_m \Delta G_{\text{meal}, m}(t_i - t_{\text{meal}, m})$.

#### Component 3: Autoregressive Noise $\eta_p(t_i)$
To simulate physiological momentum and sensor measurement drift, noise follows a stationary Gaussian AR(1) process:
$$\eta_p(t_i) = 0.85 \cdot \eta_p(t_{i-1}) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 2.5^2)$$
With $\rho = 0.85$, consecutive 5-minute readings exhibit strong autocorrelation ($\text{Corr}(\eta_t, \eta_{t+1}) = 0.85$), mirroring continuous interstitial fluid sensor dynamics.

#### Component 4: Missingness and Sensor Dropouts
1. **Random Dropouts:** Independent Bernoulli dropout mask per reading with probability $p = 0.03$.
2. **Sensor Reconnection/Warmup Gaps:** $K \sim \text{DiscreteUniform}(1, 2)$ longer continuous dropouts per patient, duration 40 to 70 minutes (8 to 14 consecutive readings dropped).
3. The resulting DataFrame drops unobserved timestamps rather than imputing them, leaving real timestamp discontinuities for the data audit script to detect.

---

## 3. Detailed Answers to 14 Critical Research Questions

### Q1: Where exactly is the synthetic dataset generated?
**Answer:** In `scripts/generate_synthetic_data.py`. When run as CLI or imported via `generate_benchmark_dataset()`, it writes identical CSV files to `data/raw/synthetic_cgm_benchmark.csv` and `data/synthetic_demo/synthetic_cgm_benchmark.csv`.

### Q2: What mathematical/process assumptions generate glucose?
**Answer:** Glucose is assumed to be an additive sum of:
1. Deterministic smooth diurnal baseline (cosine wave).
2. Deterministic dual-exponential impulse response from ingested carbohydrates.
3. Linear Gaussian AR(1) stochastic noise.
4. Hard boundary physiological saturation clipping at $[45.0, 380.0]\text{ mg/dL}$.

### Q3: How are patient differences generated?
**Answer:** Through 4 stationary scalars sampled independently per patient at dataset generation time:
- Mean basal glucose ($\mu_{B, p} \in [105, 140]\text{ mg/dL}$)
- Circadian wave amplitude ($A_p \in [8, 20]\text{ mg/dL}$)
- Circadian phase peak ($\phi_p \in [5, 8]\text{ hours}$)
- Carbohydrate sensitivity ($S_p \in [1.8, 3.2]\text{ mg/dL/g}$)  
Patients do *not* share random generator instances; each receives an isolated seed.

### Q4: How are meal-related excursions generated?
**Answer:** Meals are discrete impulse events with randomized timing and carb amounts. Each meal produces an excursion shaped by a 2-compartment impulse response: $\tau_{\text{rise}} = 20\text{ min}$, $\tau_{\text{decay}} = 60\text{ min}$. The maximum blood glucose excursion occurs at approximately 45–60 minutes post-ingestion.

### Q5: How is noise generated?
**Answer:** As an AR(1) process with autoregression coefficient $\phi = 0.85$ and white noise innovation standard deviation $\sigma = 2.5\text{ mg/dL}$. The theoretical asymptotic variance of this noise is $\sigma_\eta^2 = \frac{2.5^2}{1 - 0.85^2} \approx 22.5\text{ (mg/dL)}^2$, meaning $\text{Std}(\eta) \approx 4.74\text{ mg/dL}$.

### Q6: How are missing readings/gaps generated?
**Answer:** Random point-missingness is generated via independent Bernoulli trials ($p=0.03$). Structural missingness is generated via 1–2 contiguous block erasures of 8–14 steps (40–70 minutes) randomly placed between 20% and 80% of the patient timeline. Missing rows are removed from the DataFrame, resulting in non-uniform $\Delta t$.

### Q7: Are the same random seeds used?
**Answer:** Yes. The global seed in `src/glucotwin/config.py` is `SEED = 42`. The generator initializes patient $i \in \{1, \dots, 10\}$ with `seed = 42 + i * 100` via NumPy's modern `np.random.default_rng(seed)`.

### Q8: Can the dataset be regenerated exactly?
**Answer:** Yes. Running `python scripts/generate_synthetic_data.py --seed 42` is 100% deterministic on any standard platform. It yields the exact same 38,942 rows and identical SHA-256 checksum (`6795bf...`).

### Q9: Is there any possibility that the target-generation mechanism leaks directly into the model features?
**Answer:** No direct software leakage exists in code:
- Windowing extracts historical lags ending at origin $t$ (`glucose_lag_0` to `glucose_lag_23`).
- Target $y_{t+120}$ is queried from timestamps near $t + 120\text{ min}$.
- Feature engineering operates strictly on the lag columns and origin timestamp.
*However, there is an intrinsic generative correlation:* Because the underlying process contains deterministic cosine and exponential functions without unmodeled confounders (e.g., exercise, illness, stress, insulin dosing errors), a model with sufficient capacity can learn the underlying analytical differential equations of the synthetic generator.

### Q10: Are train/validation/test sequences generated independently or derived from the same continuous patient trajectories?
**Answer:** They are derived from the **same continuous 14-day trajectory** for each of the 10 patients via chronological splitting:
- Days 1 to ~9.1: Training
- Days ~9.2 to ~11.2: Validation
- Days ~11.3 to 14.0: Test  
The test set evaluates the model on **unseen future time periods** of known patients (temporal generalization), not on completely unseen patients (cohort cross-validation).

### Q11: Does the chronological split contain future information through normalization, rolling statistics, imputation, or feature construction?
**Answer:** No.
- Preprocessing (`src/glucotwin/data/preprocess.py`) only parses timestamps, checks units, and resolves same-timestamp duplicates.
- StandardScaler is fitted **exclusively on the training split** inside `src/glucotwin/models/train.py`:
  ```python
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.transform(X_val)
  X_test_scaled = scaler.transform(X_test)
  ```
- Rolling statistics are computed inside individual sliding windows, referencing only lags $t, t-5, \dots, t-115$.
- No global imputation or backward-looking rolling smoothers spanning split boundaries are used.

### Q12: Are there duplicated or overlapping windows across splits?
**Answer:** No. The chronological split enforces a strict `buffer_minutes = 120` (equal to the forecast horizon):
- Train windows must have both origin and target $\le t_{\text{train\_end}}$.
- Validation windows must have origin $\ge t_{\text{train\_end}} + 120\text{ min}$ and target $\le t_{\text{val\_end}}$.
- Test windows must have origin $\ge t_{\text{val\_end}} + 120\text{ min}$.
This buffer permanently discards candidate windows at the boundaries, guaranteeing that no test window contains inputs that overlap with validation targets, and no validation window contains inputs that overlap with training targets.

### Q13: Is the 120-minute target actually observed/generated independently at the target timestamp?
**Answer:** It is a forward time-step reading from the same simulated continuous trajectory, located at $t + 120\text{ min} \pm 10\text{ min}$. It is not generated post-hoc to match a model expectation; it is whatever value the continuous simulation generated for that patient at that clock time.

### Q14: Which aspects of the dataset are unrealistic or simplified compared with real CGM data?
**Answer:**
1. **No Exogenous Insulin:** The generator lacks insulin delivery curves (basal insulin rates or bolus injections). In real T1D/T2D patients, insulin-on-board (IOB) strongly drives glucose downward over 3–5 hours.
2. **Fixed Meal Dynamics:** Meal shapes in the generator use identical absorption curves ($\tau_{\text{decay}} = 60\text{ min}$) for all meals; real absorption varies drastically by fat, protein, and glycemic index.
3. **No Circadian Drift or Weekday/Weekend Variance:** The synthetic diurnal baseline repeats with identical phase $\phi_p$ and amplitude $A_p$ for all 14 days without behavioral shift.
4. **Sensor Noise is Uncorrelated with Hypo/Hyperglycemia:** Real CGM sensors (e.g., Dexcom, Freestyle Libre) suffer increased MARD (Mean Absolute Relative Difference) and compression artifacts (e.g., nighttime pressure-induced sensor cutoffs) during rapid rate of change or deep hypoglycemia; the synthetic AR(1) noise is homoscedastic.
5. **No Sensor Calibration Steps:** Real sensors experience step-discontinuities when calibrated against fingerstick glucometers; the synthetic signal is continuous.

---

## 4. Machine-Readable Provenance File

A companion file [`reports/synthetic_provenance.json`](file:///c:/Users/ursra/Projects/GlucoTwin/reports/synthetic_provenance.json) stores these parameters and checksums for automated verification.

---

## 5. Audit Conclusions & Summary Classifications

### A. Verified Facts
- Dataset exists, is 100% synthetic, and is generated by `scripts/generate_synthetic_data.py`.
- 10 patients, 14 days, 38,942 rows, 5-minute sampling cadence.
- Deterministic with global seed 42 (patient seeds $42 + 100 \cdot i$).
- Scaler fitting is train-only (`fit_transform` on train, `transform` on test).
- Chronological split enforces a 120-minute exclusion buffer between partitions.

### B. Potential Leakage Risks Evaluated
- **Feature/Target Leakage:** None found in code. Targets are excluded from features.
- **Temporal Boundary Contamination:** None. The 120-minute buffer discards boundary windows.
- **Process Over-Fitting Risk (High):** Because the synthetic process is smooth and governed by low-dimensional ODEs, the HistGradientBoosting model learns the generator's underlying cosine and exponential functions. This explains the low MAE (18.69 mg/dL) relative to noisy real-world CGM data.

### C. Synthetic-Data Limitations
- No insulin modeling (basal or bolus).
- Homoscedastic Gaussian noise (real CGM noise is state-dependent and asymmetric).
- Deterministic meal clearance curves.
- No illness, physical exercise, hormonal cycles, or psychological stress.
- Performance cannot be claimed as representative of real clinical forecasting.

### D. Recommended Experiments (Research Hardening)
1. **Feature Ablation:** Dissect whether circadian sin/cos features alone capture the bulk of the predictive gain over persistence.
2. **Trajectory Regime Breakdown:** Evaluate error separately on rapid meal rises vs. steady nocturnal baselines.
3. **Worst-Case Error Analysis:** Inspect whether peak errors occur at unheralded meal onset (where the model has no prior meal carbs or early delta signal).
