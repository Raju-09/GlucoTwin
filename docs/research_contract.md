# GlucoTwin — Scientific Contract & Research Boundaries

**Document Type:** Formal Scientific Contract & Pre-Implementation Research Gate  
**Version:** 1.0.0 (Frozen)  
**Status:** ACTIVE  
**Evaluation Role:** Senior Machine Learning Lead & Clinical AI Systems Auditor  
**Date:** 2026-10-05  

---

## 1. Executive Summary & Research Gate Purpose

GlucoTwin has demonstrated a functional, leakage-safe software prototype:
* 56/56 unit and API tests passing.
* Deterministic data ingestion and strict chronological train/val/test splitting with a 120-minute safety buffer.
* Machine-readable reliability gating (`available`, `degraded`, `withheld`) with 6 distinct reason codes.
* Strong computational performance on the benchmark synthetic generator (**MAE = 18.69 mg/dL vs. Persistence = 53.25 mg/dL**).

However, **before writing additional production code or claiming superiority over existing literature**, this Scientific Contract formally establishes:
1. **The Exact Definition of the Digital Twin:** Moving away from buzzwords toward a rigorous, falsifiable definition of *Patient Digital State* and *Twin Drift*.
2. **The Truth About the Data:** Explicit recognition that current results are evaluated on an *in silico* mathematical simulation, not human patients.
3. **The Multimodal Reality:** Acknowledgment that the current machine learning model is an autoregressive CGM-only regressor, not yet a genuine two-stream EHR + Wearable fusion engine.
4. **Forbidden Claims:** Quarantining any clinical efficacy, treatment recommendations, causal assertions, or unvalidated counterfactual claims.
5. **Research Gates Roadmap:** Laying out the sequence of controlled experiments (Prompts 1 through 11) required to make every submission claim defensible.

---

## 2. Precise Definition of the Digital Twin

In many hackathon entries, "digital twin" is used as a vague synonym for "machine learning model" or "time series dashboard." In GlucoTwin, the digital twin is strictly defined by three formal concepts:

```
                      REAL WORLD
                  [Physical Patient]
                     │            │
         Static EHR  │            │ Dynamic Sensor
         Parameters  │            │ Stream (CGM/IoT)
                     ▼            ▼
             ┌────────────────────────────┐
             │    PATIENT DIGITAL STATE   │
             │         S_t = (θ_p, h_t)   │
             └─────────────┬──────────────┘
                           │
             ┌─────────────┴──────────────┐
             │                            │
             ▼                            ▼
   [Trajectory / Event]         [Twin Fidelity Engine]
   Forecast at t+120m           Calculates Twin Drift D_t
             │                            │
             │                            ▼
             │                  Is D_t > Threshold?
             │                   /               \
             │             YES  /                 \  NO
             │                 ▼                   ▼
             │         [TWIN DRIFT WARNING]   [FIDELITY HIGH]
             │         Degrade / Withhold     Confident Output
             │                 │                   │
             └─────────────────┴───────────────────┘
                               │
                               ▼
                        [RELIABLE OUTPUT]
```

### 2.1 Patient Digital State $S_t$
The patient digital state at forecast origin $t$ is a joint representation $S_t = (\theta_p, h_t)$:
* **Static / Baseline Phenotypic Parameters $\theta_p$ (Stream 1 - Historical EHR):** Patient-specific biological priors that evolve slowly or remain invariant over weeks (e.g., fasting baseline set-point, diurnal oscillation phase/dawn timing, insulin-to-carbohydrate sensitivity prior, baseline glycemic variability).
* **Dynamic Physiological Trajectory $h_t$ (Stream 2 - Wearables/CGM):** The recent observation history capturing physiological momentum, rate of change, acceleration, and short-term volatility.

### 2.2 Twin Fidelity and Twin Drift $D_t$
A digital twin is useful only if its internal state reflects the patient's current physiology. When the physical patient undergoes an unmodeled shift (illness, acute stress, intense unscheduled exercise, sensor degradation, or altered insulin resistance), the digital twin diverges from the real patient. We define **Twin Drift** $D_t$:

$$D_t = \text{Distance}\Big( P_{\text{twin}}(y_{t} \mid S_{t-\Delta}), \; y_{\text{observed}, t} \Big)$$

**The Core Differentiator of GlucoTwin:**
> *"GlucoTwin is not merely a model that predicts glucose. It is a Digital Twin that tracks its own fidelity, detects when its computational representation of the patient is diverging from reality (Twin Drift), and actively refuses to make an overconfident prediction until the twin is synchronized and trustworthy again."*

---

## 3. Comprehensive Audit of Challenge Requirements (Items A–E)

The challenge explicitly requires:
> **Static / Historical EHR + Dynamic Wearable/IoT Data $\longrightarrow$ Adverse Event Prediction.**

Below is the objective audit of the current GlucoTwin implementation against each requirement:

| Requirement Area | Challenge Expectation | Current Implementation | Status | Gap & Required Remediation |
|---|---|---|:---:|---|
| **A. Static EHR Stream** | Ingest static/historical patient clinical records (diagnoses, baseline labs, treatment). | JSON metadata (`data/synthetic_demo/patients.json`) displayed in Streamlit sidebar. | 🔴 **FAIL** | Zero EHR features enter the feature matrix $X \in \mathbb{R}^{43}$. $X$ is 100% CGM-derived. Patient ID is dropped. *Remediation: Integrate structured phenotypic EHR parameters into feature extraction.* |
| **B. Dynamic Wearable Stream** | Ingest multi-sensor wearable/IoT streams (HR, HRV, steps, sleep, CGM). | Only 5-minute CGM glucose readings are ingested. `meal_carbs_g` is mostly null and ignored. | 🔴 **FAIL** | Non-CGM wearable telemetry does not exist in the pipeline. *Remediation: Explicitly define whether dynamic stream is CGM-centric wearable biosensing or incorporate synthetic wearable telemetry with ablation.* |
| **C. Multimodal Two-Stream Fusion** | Fused architecture uniting static baseline context and dynamic continuous streams. | Single-stream autoregressive tabular gradient booster (`HistGradientBoostingRegressor`). | 🔴 **FAIL** | There is no multi-stream fusion layer. The project currently has no empirical evidence that adding EHR improves forecasting. *Remediation: Run a controlled ablation experiment (Prompt 1) proving the marginal value of fusion.* |
| **D. Localized Adverse Event Target** | Predict localized acute adverse medical events before they manifest. | Continuous point regression predicting glucose value at $t+120$ minutes ($y \in \mathbb{R}$, mg/dL). | 🟡 **WARNING** | Continuous regression alone does not report clinical classification metrics (AUROC, AUPRC, sensitivity/specificity for hypoglycemia $<70$ mg/dL or hyperglycemia $>180$ mg/dL). *Remediation: Formulate dual-target prediction: continuous trajectory + acute event risk classification.* |
| **E. Twin Drift & Reliability Gating** | Model self-awareness: detect when predictions are unreliable and withhold output. | `ReliabilityChecker` with 6 reason codes and 3 statuses (`AVAILABLE`, `DEGRADED`, `WITHHELD`). | 🟡 **WARNING** | Current gating checks input sensor faults (stale data, gaps, range), not physiological twin drift. *Remediation: Formalize the Twin Drift metric and integrate it into the gating decision engine.* |
| **F. Leakage Safety** | Zero future information or target leakage in feature engineering or splits. | Chronological 65/15/20 split across 14 days with a 120-minute blackout buffer. | 🟢 **PASS** | Audited in `docs/leakage_audit.md`. Feature scaling and statistics computed strictly on train split. |
| **G. Experimental Reproducibility** | All reported figures reproducible via fixed-seed executable scripts. | `scripts/reproduce_results.py`, `scripts/evaluate_baseline_suite.py`, `scripts/run_ablation_study.py`. | 🟢 **PASS** | Evaluated numbers (18.69 vs 53.25 mg/dL) reproduce bit-for-bit with `SEED=42`. |
| **H. External Validity & Generalizability** | Validation across real clinical cohorts or distinct data distributions. | Evaluated only on 10 synthetic patients from procedural Generator A; tested on Generator B. | 🔴 **FAIL** | Zero real-world human data (DiaTrend, OhioT1DM) has been processed. Performance on human diabetes patients is unknown. *Remediation: Clear labeling of in silico boundary.* |
| **I. Counterfactual Sandbox** | Simulated interventions (exercise, meals, medications). | Conceptually mentioned in earlier discussions, but unvalidated mathematically. | 🟡 **WARNING** | No validated PK/PD backing. *Remediation: Quarantined strictly as an exploratory hypothesis simulation sandbox; never claim proven clinical outcome.* |

---

## 4. Component Tier Classification

Every module, feature, and claim in the GlucoTwin repository is assigned to one of four strict tiers:

```
TIER 1: EXPERIMENTALLY VALIDATED
├── 120-minute autoregressive CGM forecasting on synthetic benchmark
├── Chronological train/validation/test split with 120-minute exclusion buffer
├── Baseline comparison suite (Persistence, Trend, Recent Mean, Diurnal Climatology)
├── Feature ablation (lag depth, summary stats, rate-of-change, circadian harmonics)
├── Regime-stratified error analysis (fast rising, fast falling, in-range, nocturnal)
├── Per-patient error distribution across 10 synthetic subjects
├── Input integrity reliability gating (stale sensor, excessive gaps, out-of-bounds)
└── Cross-generator holdout test (transfer from Generator A to Generator B)

TIER 2: IMPLEMENTED BUT UNVALIDATED IN CLINICAL/MULTIMODAL SENSE
├── Replay scenario injector (ReplayEngine) for sensor dropouts and spikes
├── FastAPI serving layer (/forecast, /health, /patients endpoints)
└── Local fallback execution engine in Streamlit

TIER 3: UI-ONLY / PRESENTATION ARTIFACTS
├── Static patient demographics and clinical notes displayed in UI sidebar
├── Time in Range reference band (70–180 mg/dL) on charts
├── Traffic light reliability badges
└── Hardcoded patient-specific bias adjustment for SYNTH_002 (-7.17 mg/dL)

TIER 4: UNVALIDATED HYPOTHESIS SIMULATION SANDBOX
├── Simulated exercise effect on glucose clearance
├── Simulated SGLT2i / pharmacodynamic medication response
├── Whole-body digital twin metabolic claims
└── Causal claims derived from SHAP or tree feature importance
```

---

## 5. Strictly Forbidden Claims (Quarantine Matrix)

To preserve scientific integrity during judging and clinical scrutiny, the following claims are **STRICTLY FORBIDDEN** across all documentation, code docstrings, presentation decks, and video scripts:

1. **FORBIDDEN:** *"GlucoTwin is a clinically validated digital twin or medical device."*  
   **PERMITTED:** *"GlucoTwin is an engineering and research prototype evaluating reliability-aware forecasting and twin drift detection on synthetic benchmark cohorts."*
2. **FORBIDDEN:** *"GlucoTwin reduces forecast error on human diabetes patients by 64.9%."*  
   **PERMITTED:** *"On our synthetic benchmark dataset (10 simulated patients over 14 days), GlucoTwin achieved an MAE of 18.69 mg/dL compared to a persistence baseline of 53.25 mg/dL. External validity on human subjects has not been established."*
3. **FORBIDDEN:** *"GlucoTwin can be used to recommend insulin doses or medication changes."*  
   **PERMITTED:** *"GlucoTwin outputs are for researcher and clinician review only and must never be used for treatment recommendations or insulin titration."*
4. **FORBIDDEN:** *"Feature importance / SHAP values prove the physiological cause of the patient's glucose changes."*  
   **PERMITTED:** *"SHAP and feature importances reflect statistical correlations within the learned model, not physiological or causal mechanisms."*
5. **FORBIDDEN:** *"GlucoTwin accurately simulates the therapeutic effect of SGLT2 inhibitors or exercise on a patient."*  
   **PERMITTED:** *"Counterfactual scenarios represent an exploratory mathematical sandbox for scenario exploration, not clinically validated pharmacokinetics."*

---

## 6. Open Scientific Questions & Gate Roadmap

The remaining development will proceed through discrete, verifiable research gates. No step will be skipped:

```
[PROMPT 0: RESEARCH CONTRACT] ──► Freeze boundaries, classify tiers, define Twin Drift (THIS STEP)
            │
            ▼
[PROMPT 1: TWO-STREAM FUSION] ──► Build & evaluate Multimodal Ablation:
                                  CGM-only vs. CGM+EHR vs. CGM+Wearables vs. Full Fusion
            │
            ▼
[PROMPT 2: TWIN STATE ENGINE] ──► Formalize Patient Digital State S_t = (θ_p, h_t)
            │
            ▼
[PROMPT 3: TWIN DRIFT SUBSYSTEM] ──► Implement online Twin Drift metric D_t & divergence triggers
            │
            ▼
[PROMPT 4: SELECTIVE PREDICTION] ──► Quantify error vs. coverage curves under fault injection
            │
            ▼
[PROMPT 5: ADVERSE EVENT TARGET] ──► Add acute excursion classification (AUROC/AUPRC/Recall)
            │
            ▼
[PROMPT 6: SANDBOX ISOLATION] ──► Quarantine counterfactuals into an explicit exploration sandbox
            │
            ▼
[PROMPT 7: UNSEEN TWIN EVALUATION] ──► Test generalization on unseen patient twins
            │
            ▼
[PROMPT 8: STRESS TEST MATRIX] ──► Systematic stress testing of twin synchronization
            │
            ▼
[PROMPT 9: JURY-FACING UI] ──► Upgrade dashboard to showcase Twin Drift & Multimodal Evidence
            │
            ▼
[PROMPT 10: RED TEAM AUDIT] ──► Hostile review to eliminate weak claims before final freeze
            │
            ▼
[PROMPT 11: REPRODUCIBLE BUNDLE] ──► Complete packaging with automated verification
```

---

## 7. Sign-off Criteria for Gate 0

Gate 0 is satisfied when:
1. `docs/research_contract.md` is fully detailed, setting unambiguous boundaries and non-goals.
2. `reports/research_contract_audit.json` contains the structured machine-readable audit matrix.
3. No code modifications or model changes are made until this contract is locked.
4. The user and reviewer confirm alignment on the Twin Drift pivot and two-stream fusion plan.
