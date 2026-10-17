# GlucoTwin — 20-Minute Master Video Recording Script
**Competition:** Happiest Health Digital Twin Challenge 2026  
**Target Runtime:** 20 to 22 minutes (Complies with minimum 20-minute requirement)  
**Recording Tool:** OBS Studio (1920×1080, 30/60 fps, microphone tested)

---

## Technical Setup Before Hitting Record
1. Terminal 1: `uvicorn src.glucotwin.api.main:app --reload --port 8000`
2. Terminal 2: `streamlit run frontend/streamlit_app.py`
3. Browser Tab 1: `http://localhost:8501` (GlucoTwin Dashboard, patient `SYNTH_002` selected)
4. Browser Tab 2: `http://localhost:8000/docs` (FastAPI Swagger Interactive Documentation)
5. VS Code: Project workspace open, `docs/architecture.md`, `tests/`, and `experiments/` accessible.

---

## Timed Scene-by-Scene Script

### Section 1: Introduction, Clinical Problem & India Context [0:00 – 3:00]
* **Screen:** Title slide or GlucoTwin README.
* **Narration:**
  * "Good day, members of the technical jury and evaluation panel at Happiest Health. We are presenting GlucoTwin, an engineering and clinical research prototype of a personalized Digital Twin for Continuous Glucose Forecasting and proactive adverse event prediction."
  * "Why does this matter, especially in India? India is the diabetes capital of the world, with over 101 million individuals diagnosed with diabetes. Current clinical management relying on Continuous Glucose Monitors (CGMs) is fundamentally *reactive*. Alarms only fire *after* blood glucose breaches dangerous thresholds—such as 70 mg/dL for hypoglycemia or 180 mg/dL for postprandial hyperglycemia. By the time a patient or caregiver hears the alarm, physiological distress has already begun."
  * "Our challenge in this hackathon was to build a foundational Digital Twin that ingests two separate data streams—static EHR historical context and dynamic real-time CGM/wearable telemetry—and predicts adverse glycemic excursions 120 minutes in advance, while knowing *when to stay silent* when input streams degrade."

### Section 2: Digital Twin Architecture & Dual Data Stream Fusion [3:00 – 6:30]
* **Screen:** Switch to Visual Architecture Diagram / VS Code `docs/architecture.md`.
* **Narration:**
  * "A Digital Twin is not just a forecasting model wrapped around a dashboard. Formally, GlucoTwin maintains an explicit, causal patient computational state: $S(t) = (\theta_p, h_t, \delta_t)$."
  * "$\theta_p$ is the Patient Static Phenotype: Demographics, baseline HbA1c, disease duration, historical fasting blood glucose, and medication flags."
  * "$h_t$ is the Dynamic Observation History: Continuous 5-minute CGM readings, heart rate, step count, and sleep telemetry."
  * "$\delta_t$ is the Twin Integrity State: Real-time stream freshness, sensor packet integrity, and the Reliability FSM."
  * "Notice our strict leakage-prevention pipeline: raw data ingestion validates timestamps, physiological boundaries, and sensor physics. Supervised training windows are created using strict chronological splits with a 120-minute safety buffer to guarantee zero future data leaks into past features."

### Section 3: Live Prototype Walkthrough — Doctor's Dashboard [6:30 – 11:00]
* **Screen:** Switch to Browser Tab 1 (`http://localhost:8501`).
* **Actions & Narration:**
  * **Patient Selection [6:30]**: Select `SYNTH_002`. "Here is our clinician dashboard. We can switch between 60 in-silico patients whose physiology is mechanistically coupled to their EHR phenotypes."
  * **State Inspector [7:15]**: Click expander `Inspect Patient Digital State S(t)`. "Notice our transparent provenance labels: `[Observed EHR]` for clinical intake, `[Sensor]` for continuous telemetry, `[Kinematic]` for discrete derivatives, and `[Model]` for predictions. We show the raw unclamped discrete velocity ($dG/dt$) so no anomalous physiological spike is hidden."
  * **Interactive Timeline Scrubbing [8:30]**: Move playback slider across 14 days of telemetry. "As we scrub through time, the blue curve traces observed continuous history. The pink dashed curve illustrates the trajectory smoothly interpolating current velocity to the single 120-minute forecast at the target horizon."
  * **Adverse Event Prediction Cards [9:45]**: Highlight the KPI cards. "Our two-stream model flags threshold crossings: High Crossing (>180 mg/dL) and Low Crossing (<70 mg/dL). In postprandial periods, the forecast alerts the clinician 2 hours ahead that a glycemic excursion is developing."

### Section 4: The Core Differentiator — Reliability Gating Failure Injection [11:00 – 14:00]
* **Screen:** Streamlit Dashboard sidebar → "Telemetry Scenario & Stress Tests".
* **Actions & Narration:**
  * "Now we demonstrate our most critical technical contribution: the Reliability Gate. In clinical AI, an overconfident wrong prediction during sensor malfunction is lethal. GlucoTwin implements a fail-closed integrity layer."
  * **Scenario 1: Stale Telemetry [11:30]**: Select *"Stale Reading (20m delay)"*.
    * "Observe the badge: it immediately switches to `🟡 DEGRADED` with reason code `STALE_DATA_WARNING`. The clinician receives an advisory that the observation is stale."
  * **Scenario 2: Sensor Dropout [12:30]**: Select *"Sensor Dropout (no signal)"*.
    * "Observe: status flips to `🔴 WITHHELD` with reason code `ERR_SENSOR_DROPOUT`. The forecast value is suppressed—no fabricated numbers. The system refuses to guess under uncertainty."
  * **Scenario 3: Non-Physiological Spike [13:15]**: Select *"Physiological Spike (400 mg/dL)"*.
    * "An instantaneous jump exceeding physiological limits triggers `DEGRADED_UNGROUNDED`."
  * **Return to Normal [13:45]**: Reset to *"Normal Telemetry"*. Status recovers to `🟢 AVAILABLE`.

### Section 5: Rigorous Scientific Evidence & The Honest LOPO Verdict [14:00 – 17:30]
* **Screen:** Dashboard Tab 1 ("Model Evidence & Benchmarks") and `artifacts/evaluations/event_prediction_metrics.json`.
* **Narration:**
  * "Here are our benchmark results evaluated on 43,888 held-out test windows across 60 patients under strict chronological hold-out:"
  * "1. **Persistence Baseline**: Naive $y_{t+120} = y_t$ yields an MAE of 53.25 mg/dL."
  * "2. **Config A (CGM Only)**: 18.31 mg/dL."
  * "3. **Config B (CGM + Static EHR)**: 17.61 mg/dL (a statistically significant reduction of 0.70 mg/dL, $p = 5.6 \times 10^{-6}$)."
  * "4. **Config C (CGM + Wearables)**: 18.32 mg/dL. Our ablation honestly shows physical activity telemetry provided no incremental predictive gain over CGM+EHR under this simulator."
  * "5. **Leave-One-Patient-Out (LOPO) Falsification Test**: When evaluated on *unseen held-out patients*, CGM-only achieves 17.65 mg/dL, and CGM+EHR achieves 17.65 mg/dL ($p = 0.96$). EHR provides no cross-patient generalization benefit. However, when we permuted EHR records between patients, error spiked catastrophically to 20.47 mg/dL (+2.82 mg/dL). This proves the model encodes true patient identity rather than transferable population physiology. We report this negative generalization result openly."
  * "6. **Clinical Adverse Event Classification**: On detecting postprandial hyperglycemia (>180 mg/dL), GlucoTwin achieves **76.2% sensitivity** and **90.6% specificity** (F1 = 0.760), while reducing false alarms from 61.4 down to 19.6 per patient-day—a 68% reduction in alarm fatigue."

### Section 6: Codebase Maturity, API & Docker Reproducibility [17:30 – 19:00]
* **Screen:** VS Code / Terminal.
* **Actions & Narration:**
  * "Let us review code quality and engineering rigor."
  * Show `pytest tests/`: run tests live in terminal $\rightarrow$ `69 passed in 6.46s`.
  * Show `src/glucotwin/api/main.py` and Browser Tab 2 (`http://localhost:8000/docs`).
  * "Our FastAPI backend provides fully typed Pydantic endpoints (`/forecast`, `/twin/{patient_id}/state`, `/health`)."
  * Show `docker-compose.yml`: "The entire stack is containerized and reproducible in under 60 seconds with `docker-compose up`."

### Section 7: Conclusion, Safety Disclaimers & Next Steps [19:00 – 20:30]
* **Screen:** Pitch slide or README conclusion.
* **Narration:**
  * "To conclude: GlucoTwin demonstrates that a proactive, reliability-aware Digital Twin is technically feasible. We do not claim clinical efficacy—this is an in-silico research prototype that must undergo prospective trials on clinical datasets like DiaTrend or OhioT1DM."
  * "Our genuine contribution is a system that pairs predictive foresight with operational humility: forecasting earlier, intervening smarter, and failing closed when uncertain."
  * "Thank you. We look forward to your questions."

---
