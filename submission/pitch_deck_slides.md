# GlucoTwin — Official Pitch Deck Slides
**Challenge:** Happiest Health · Digital Twin Challenge 2026  
**File Artifact:** `docs/glucotwin_presentation.pdf` (10 Slides Landscape)  
**Team:** Team Synapse · Mohan Babu University  
**Members:** Sammeta Siva Prasad Raju ([@Raju-09](https://github.com/Raju-09), Team Leader) · Bedadakota Devi Chandrika ([@Devi-chandrika24](https://github.com/Devi-chandrika24), Contributor)  
**Code:** `https://github.com/Raju-09/GlucoTwin` · **Video:** `https://youtu.be/Vsmb3QibpKI` (Unlisted) · **License:** MIT  

---

### Slide 1: Title Slide
* **Title:** **GlucoTwin**
* **Subtitle:** A reliability-aware digital twin for 120-minute glucose forecasting
* **Affiliation:** Team Synapse · Mohan Babu University
* **Team:** Sammeta Siva Prasad Raju ([@Raju-09](https://github.com/Raju-09), Team Leader) · Bedadakota Devi Chandrika ([@Devi-chandrika24](https://github.com/Devi-chandrika24), Contributor)
* **Metadata:**
  * Code: `github.com/Raju-09/GlucoTwin`
  * Video: `youtu.be/Vsmb3QibpKI` (unlisted)
  * License: MIT license
* **Disclaimer:** *Research prototype on synthetic data. Not a medical device.*

---

### Slide 2: The Problem
* **Headline:** A reading every 5 minutes, but alerts that look only a short way ahead
* **Key Metrics:**
  * **101M:** people with diabetes in India, per the ICMR-INDIAB study (*The Lancet Diabetes & Endocrinology*, 2023).
  * **~20 min:** typical look-ahead of commercial predictive alerts, such as Dexcom Urgent Low Soon. Many alerts are plain thresholds.
  * **120 min:** the horizon GlucoTwin explores: enough time for a meal change or a walk, if the forecast can be trusted.
* **Research Question:** Does a time-series model beat a persistence baseline on a leakage-safe held-out test, and when should the system withhold a forecast?

---

### Slide 3: Approach
* **Headline:** The virtual patient is three things at every time step
* **Mathematical State Equation:**
  $$S(t) = (\theta_p, h_t, \delta_t)$$
* **Trio Architecture:**
  * **Static phenotype ($\theta_p$) — EHR context · 9 features:** Age, sex, BMI, baseline HbA1c, diabetes duration, historical fasting glucose, metformin, SGLT2 inhibitor and dawn-phenomenon flags.
  * **Dynamic history ($h_t$) — Causal 120-minute window:** 24 CGM lags at 5-minute cadence, rolling mean and spread, velocity and acceleration, circadian terms, plus simulated wearable signals.
  * **Twin integrity ($\delta_t$) — Stream quality state:** Available, degraded or withheld, with a reason code, data freshness and a quality score. This decides whether a forecast is shown.

---

### Slide 4: Architecture
* **Headline:** From raw streams to a gated forecast
* **Pipeline Flow:**
  1. **Ingest:** CGM every 5 minutes, wearables, EHR (9 fields) $\rightarrow$
  2. **Validate:** Freshness, range, gap and jump checks $\rightarrow$
  3. **Features:** Lags, rolling stats, circadian: 41 / 50 / 56 by configuration $\rightarrow$
  4. **Forecast:** HistGradientBoosting, +120 min $\rightarrow$
  5. **Gate and serve:** Available, degraded or withheld; FastAPI; Streamlit
* **Pillars:**
  * **Simulated sensor layer:** Interstitial lag ($\tau = 10\text{ min}$), AR(1) noise and a rate clamp of $3.8\text{ mg/dL/min}$ mimic real CGM behavior.
  * **Leakage control:** Chronological split with a 120-minute buffer between training and test windows.
  * **Cohort:** 60 synthetic type 2 diabetes patients, 224,681 windows, fixed generator seed.

---

### Slide 5: Reliability Gate
* **Headline:** Before any forecast is shown, the input stream must pass a check
* **Finite-State Machine (FSM):**
  | State | Conditions | Behavior |
  |---|---|---|
  | **Available** | Data fresh (under 15 min), full 24-step history, values within 40–400 mg/dL | Forecast shown |
  | **Degraded** | Latency of 15–20 min or minor packet loss; history intact | Forecast shown with a warning and reason code |
  | **Withheld** | Stale stream (over 20 min), gap over 30 min, sensor dropout, or a jump above 4.0 mg/dL/min | No number returned, only an error code |
* **Safety Principle:** **Fail closed.** The gate is rule-based and checks the data, not the model. The risk–coverage trade-off, meaning how much error is avoided for how many forecasts withheld, is evaluated in `scripts/evaluate_risk_coverage.py`.

---

### Slide 6: Result 1 · Multimodal Ablation Study
* **Headline:** EHR context helps within patients; wearables add nothing here
* **Benchmark Results Table:**
  | Configuration | MAE (mg/dL) | $\Delta$ vs CGM only | 95% CI of $\Delta$ |
  |---|:---:|:---:|:---:|
  | **Persistence baseline** | 46.88 | +28.57 | n/a |
  | **Linear extrapolation (30-min slope)** | 86.83 | +68.52 | n/a |
  | **A · CGM only (41 features)** | 18.31 | reference | n/a |
  | **B · CGM + EHR (50)** | **17.61** | **−0.70** | **[−0.96, −0.49]** |
  | **C · CGM + wearables (47)** | 18.32 | +0.01 | [−0.08, +0.07] |
  | **D · All streams (56)** | **17.67** | **−0.64** | **[−0.88, −0.44]** |
* **Protocol:** MAE in mg/dL at +120 min. Chronological hold-out, days 12–14, 43,888 windows, 60 patients. Training and test share the same patients, so this is a within-patient test. Wearable features gave no measurable gain under this simulator; we report that as a finding.

---

### Slide 7: Result 2 · Testing Our Own Claim (LOPO)
* **Headline:** On unseen patients, EHR context gave no benefit
* **Three-Fold Comparison:**
  * **CGM only:** **17.65 mg/dL**, mean LOPO MAE
  * **CGM + real EHR:** **17.65 mg/dL**, $\Delta +0.01$, $p = 0.956$
  * **CGM + shuffled EHR:** **20.47 mg/dL**, $\Delta +2.82\text{ mg/dL}$, $p < 0.001$
* **Scientific Finding:** Leave-one-patient-out, 60 patients, 224,681 windows. The within-patient gain did not transfer to new patients. Shuffling EHR across patients made predictions worse, which is consistent with the model using EHR as a patient-specific signal. We have not yet tested why; relating the error penalty to the distance between true and swapped EHR is the planned next check.

---

### Slide 8: Result 3 · Threshold Events at +120 Min
* **Headline:** High glucose is detected far better than the baseline; lows are missed
* **Adverse Event Performance:**
  | Event | Model | Positive windows | Sensitivity | Precision | False alarms / day |
  |---|---|:---:|:---:|:---:|:---:|
  | **Above 180 mg/dL** | Persistence | 12,246 | 26.6% | 25.8% | 61.4 |
  | **Above 180 mg/dL** | **GlucoTwin** | 12,246 | **76.2%** | **75.8%** | **19.6** |
  | **Below 70 mg/dL** | Persistence | 71 | 12.7% | 12.7% | 0.4 |
  | **Below 70 mg/dL** | **GlucoTwin** | 71 | **0.0%** | **0.0%** | **0.0** |
* **Clinical Insight:**
  * **High glucose:** Sensitivity rose from 26.6% to 76.2% and false-alarm windows per day fell from 61.4 to 19.6 (−68%). About 28% of test windows are above 180, so this is a common state, not a rare spike.
  * **Low glucose:** The model detected none of the 71 windows below 70. A mean-seeking regression model rarely predicts extremes, so this needs a different loss function.
  * Counts are 5-minute windows, not distinct episodes; the test set spans 152.4 simulated patient-days. Synthetic held-out test; not evidence of clinical effectiveness.

---

### Slide 9: Clinician Dashboard
* **Headline:** A doctor can inspect the virtual patient and break the data stream
* **Feature Pillars:**
  * **Patient state view:** 60 synthetic patients, a 14-day timeline slider, the three-part state $S(t)$, and badges that show where each value came from.
  * **Failure injection:** Inject a stale reading, a sensor gap, a 600 mg/dL spike or packet dropouts and watch the gate move to degraded or withheld.
  * **Serving and tests:** FastAPI latency over 100 calls: median 7.8 ms, p95 11.1 ms, p99 13.3 ms. 69 unit tests pass. Docker Compose setup included.
* **Live Demo Command:** `streamlit run frontend/streamlit_app.py`

---

### Slide 10: Limits and Next Steps
* **Headline:** What this does not show yet, and how we would find out
* **Known Limits:**
  * All data is synthetic; no real-patient performance is claimed.
  * No lows detected: 0 of 71 windows below 70 mg/dL.
  * The EHR benefit did not transfer to unseen patients.
  * Wearables added no measurable value in this simulator.
  * Forecasts are point estimates without uncertainty intervals.
* **Next Steps:**
  * Validate on open CGM datasets under their access terms.
  * Add a time-of-day baseline and a new-onset event analysis.
  * Report alarms as episodes with a refractory period.
  * Add calibrated prediction intervals tied to the gate.
  * Train with an asymmetric loss for low glucose.
* **Closing Motto:**  
  *Forecast earlier. Intervene smarter. Withhold when the data cannot be trusted.*  
  Team Synapse · Mohan Babu University · `github.com/Raju-09/GlucoTwin` · MIT license · *Not a medical device*
