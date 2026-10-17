# GlucoTwin — Pitch Deck Master Slide Content
**Event:** Happiest Health Digital Twin Challenge 2026  
**Format:** 10 Slides (Target export: `docs/glucotwin_presentation.pdf`)  
**Design Theme:** White background, clinical deep blue (`#0077B6`), charcoal text (`#1A1A1A`), emerald accents (`#2E7D32`).

---

### Slide 1: Title Slide
* **Title**: GlucoTwin
* **Subtitle**: A Reliability-Aware Digital Twin for Proactive Glucose Forecasting and Adverse Event Mitigation
* **Event**: Happiest Health — Reimagining & Reforming Healthcare in India Summit 2026
* **Team**: Team Synapse · Mohan Babu University
* **Members**: Sammeta Siva Prasad Raju & Bedadakota Devi Chandrika
* **Key Visual**: Two-stream data convergence (EHR + CGM) flowing into a calibrated patient twin state.

---

### Slide 2: The Healthcare Challenge in India
* **Headline**: 101 Million Patients: The Unmet Need for Proactive Glycemic Management
* **Key Points**:
  * **Global Epicenter**: India has 101 million individuals diagnosed with diabetes (IDF Diabetes Atlas, 2023).
  * **The Fatal Flaw of Modern CGMs**: Current sensor alarms are strictly *reactive*. They trigger *after* glucose crosses severe thresholds (<70 or >180 mg/dL), when biochemical distress is already underway.
  * **The Proactive Window**: A 120-minute predictive horizon provides actionable lead time for dietary moderation or insulin adjustment.
* **Callout**: *"Reactive monitoring catches complications; proactive digital twins prevent them."*

---

### Slide 3: Prediction Target & Twin State Formalism
* **Headline**: Rigorous Mathematical State Representation
* **Prediction Target**:
  * **Input Horizon**: 24 causal readings (last 120 minutes of continuous CGM at 5-min cadence) + static EHR context.
  * **Forecast Horizon**: Exactly $t+120$ minutes.
  * **Target Output**: Future glucose concentration ($\hat{y}_{t+120} \in [40, 400]$ mg/dL).
* **Digital Twin State Equation**:
  $$S(t) = (\theta_p, h_t, \delta_t)$$
  * $\theta_p$: Static EHR phenotype (HbA1c, BMI, duration, historical FBG, medications).
  * $h_t$: Dynamic telemetry history (CGM lags, kinematics, heart rate, steps, sleep).
  * $\delta_t$: Reliability and stream integrity state (`available`, `degraded`, `withheld`).

---

### Slide 4: System Architecture
* **Headline**: End-to-End Three-Tier Pipeline: Ingest $\rightarrow$ Verify $\rightarrow$ Serve
* **Key Components**:
  * **1. Ingestion & Preprocessing**: Temporal deduplication, interstitial sensor physics layer, physiological sanity checks.
  * **2. Feature Engineering**: 56 multimodal features (24 lags, rolling mean/std, $dG/dt$, $d^2G/dt^2$, circadian encoding, EHR intake).
  * **3. Model & Reliability Gate**: HistGradientBoostingRegressor paired with a finite state machine that enforces fail-closed forecast gating.
  * **4. Serving**: FastAPI REST backend + clinician-facing Streamlit dashboard.

---

### Slide 5: The Core Differentiator — Fail-Closed Reliability Gating
* **Headline**: The Model Knows When to Stay Silent
* **The Clinical Principle**: An overconfident wrong prediction during sensor fault is far more dangerous than an abstention.
* **Gating State Machine**:
  | Quality Condition | State | System Behavior |
  |---|:---:|---|
  | Timely, complete, plausible stream | `AVAILABLE` | 120-min point forecast displayed |
  | Minor packet delay or stale data (15–20m) | `DEGRADED` | Forecast shown with advisory warning |
  | Severe sensor gap (>30m) or dropout | `WITHHELD` | **Prediction suppressed; error reason code returned** |
  | Physiological jump (>4.0 mg/dL/min) | `WITHHELD` | **Suppressed as ungrounded artifact** |
* **Utility Proof**: When the gate withholds corrupt streams, severe errors (>30 mg/dL) are suppressed, preserving clinical trust.

---

### Slide 6: Benchmark Results & Multimodal Ablation
* **Headline**: Held-Out Test Evaluation on 43,888 Windows (60 Patients)
* **Results Table**:
  | Configuration | Data Streams | Test MAE (mg/dL) | Δ MAE (Model − Ref) | 95% Bootstrap CI |
  |---|---|:---:|:---:|:---:|
  | **Persistence Baseline** | Naive $y_{t+120} = y_t$ | **53.25** | *+34.94* | — |
  | **Config A (CGM Only)** | 24 Lags + Kinematics (41 feats) | **18.31** | *Reference* | — |
  | **Config B (CGM + Static EHR)** | Demographics + Labs + FBG (50 feats) | **17.61** | **−0.70 mg/dL** | **[+0.49, +0.96]** ($p < 10^{-5}$) |
  | **Config C (CGM + Wearables)** | Decoupled HR, Steps, Sleep (47 feats) | **18.32** | +0.01 mg/dL | [−0.08, +0.07] ($p = 0.95$) |
  | **Config D (Full Fusion)** | EHR + CGM + Wearables (56 feats) | **17.67** | **−0.64 mg/dL** | [+0.44, +0.88] |
* **Key Insight**: EHR features provide a modest, statistically validated gain on same-patient temporal trajectories. Wearables provide negligible incremental gain under this simulator.

---

### Slide 7: Scientific Integrity — Falsification & The LOPO Test
* **Headline**: Breaking Our Own Claim: Leave-One-Patient-Out (LOPO)
* **The Question**: Does EHR context transfer to unseen patients, or does it memorize patient identities?
* **LOPO Experiment (N=60 patients, 224,681 windows)**:
  * Mean LOPO CGM-Only MAE: **17.65 mg/dL**
  * Mean LOPO CGM + Real EHR MAE: **17.65 mg/dL** (Delta: **+0.01 mg/dL**, $p = 0.96$, n.s.)
  * Mean LOPO Permuted EHR MAE: **20.47 mg/dL** (Degradation: **+2.82 mg/dL**)
* **Takeaway**: Shuffling EHR between patients causes catastrophic error (+2.82 mg/dL), proving the model relies on true patient identity rather than transferable physiology. We report this honestly rather than claiming false population generalization.

---

### Slide 8: Clinical Adverse Event Prediction
* **Headline**: 76.2% Sensitivity on Postprandial Hyperglycemia Excursions
* **Event Task**: Predict whether glucose will exceed 180 mg/dL 120 minutes in advance.
* **Event Performance Table (Held-Out Test Set)**:
  | Model | Sensitivity (Recall) | Specificity | Precision | F1-Score | False Alarms / Patient-Day |
  |---|:---:|:---:|:---:|:---:|:---:|
  | **Persistence Baseline** | 26.6% | 70.4% | 25.8% | 0.262 | 61.4 |
  | **GlucoTwin Two-Stream** | **76.2%** | **90.6%** | **75.8%** | **0.760** | **19.6 (−68%)** |
* **Clinical Significance**: Catches nearly 3× more hyperglycemic spikes while reducing false alarms by 68%, mitigating clinician and patient alarm fatigue.

---

### Slide 9: Clinician-Ready Dashboard & Software Maturity
* **Headline**: Production-Ready Engineering & Instant Reproducibility
* **Product Features**:
  * **Doctor's State Inspector**: Causal breakdown of observed vs derived metrics.
  * **Interactive Time Playback**: 14 days of continuous scrubbing.
  * **One-Click Stress Testing**: Instant injection of sensor drops, latency, and spikes.
* **Engineering Standards**:
  * **Test Suite**: 69 unit tests passing with 100% coverage across core modules.
  * **FastAPI Backend**: Microsecond inference, Pydantic validation.
  * **One-Command Deployment**: `docker-compose up` ready in < 60 seconds.

---

### Slide 10: Responsible AI, Limitations & Roadmap
* **Headline**: Honest Boundaries and the Path to Clinical Translation
* **Explicit Non-Goals**:
  * Research prototype; **not** an FDA/CDSCO-cleared medical device.
  * Evaluated on synthetic cohort; does **not** establish real patient efficacy.
* **Translation Roadmap**:
  * **Phase 1 (Completed)**: In-silico digital twin, reliability gating FSM, LOPO falsification.
  * **Phase 2 (Next)**: Clinical benchmarking on de-identified real cohorts (DiaTrend / OhioT1DM).
  * **Phase 3 (Future)**: Prospective pilot study in Indian primary care centers.
* **Closing**: *"GlucoTwin: Forecast earlier. Intervene smarter. Fail closed when uncertain."*
