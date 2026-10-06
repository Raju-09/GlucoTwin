# Gate 1.5 — Hostile Scientific Audit: Data Generating Process, Target Leakage, and Multimodal Evidence

**Document Type:** Formal Scientific Audit & Peer-Review Red Team  
**Gate:** Gate 1.5 (Prerequisite to Gate 2 Digital Twin Engine)  
**Status:** CONDITIONAL FAIL / REMEDIATION REQUIRED  
**Auditor:** Senior ML Research Lead & Healthcare AI Compliance Auditor  
**Date:** 2026-10-05  

---

## 1. Executive Summary

This audit serves as a hostile pre-submission critique of the GlucoTwin multimodal data pipeline and recent experimental claims. While the engineering execution, zero-leakage windowing (`pd.merge_asof(direction="backward")`), and testing rigor remain top-tier, the scientific justification underpinning recent multimodal claims contains critical vulnerabilities that would be immediately penalized by a peer-review panel or competition jury.

### Key Audit Findings:
1. **Target Proxy Leakage in Static EHR (P0):** The synthetic EHR record for each patient contains `fasting_glucose_baseline`, which is directly copied from the synthetic generator's latent parameter `basal_mean`. The GBDT model exploits this as a direct target-generating parameter proxy.
2. **Wearable Generator Circularity (P1):** Dynamic wearable signals (Heart Rate, HRV) are generated directly from the simulated CGM glucose values (`glucose - 125`), rather than from an independent or shared physiological state. Wearables do not contain independent physiological information.
3. **Wearable Stream Negative Incremental Gain (P1):** Adding wearables to CGM + EHR increases test MAE from $17.63\text{ mg/dL}$ to $17.96\text{ mg/dL}$ ($+0.33\text{ mg/dL}$). The claim that "wearables provide statistically significant improvements" is empirically false and attributes EHR-driven gains to wearables.
4. **Pseudoreplication in Statistical Testing (P1):** The reported $p = 9.45 \times 10^{-8}$ is derived from a naïve paired $t$-test across $7,316$ highly autocorrelated 5-minute rolling windows spanning only 10 patients. Effective degrees of freedom are severely overstated.
5. **Clinical Condition Mismatch (P2):** The prototype is labeled "Type 1 Diabetes", yet the generator lacks insulin pharmacokinetic curves, and the competition specification explicitly targets chronic lifestyle diseases in India (Type 2 Diabetes).
6. **Documentation Drift and Stale Benchmarks (P2):** The codebase maintains contradictory metrics ($18.69$, $18.71$, $18.72$, $17.63$, $17.96$) hardcoded across UI, API, and documentation due to the absence of a canonical `experiment_manifest.json`.

---

## 2. Audit 1: Data Generating Process (DGP) & Dependency Graph

### 2.1 Variable Taxonomy & Lineage

| Variable | Source / Generator Expression | Classification | Risk Analysis |
|---|---|---|---|
| `basal_mean` | `rng.uniform(105.0, 140.0)` | Latent Parameter | Unobserved ground truth parameter determining patient basal attractor |
| `circadian_amp` | `rng.uniform(8.0, 20.0)` | Latent Parameter | Unobserved amplitude of diurnal cosine wave |
| `circadian_peak_hour` | `rng.uniform(5.0, 8.0)` | Latent Parameter | Unobserved phase shift of diurnal cosine wave |
| `carb_sensitivity` | `rng.uniform(1.8, 3.2)` | Latent Parameter | Excursion multiplier per gram of carbohydrate |
| `cgm_glucose(t)` | $\text{basal}(t) + \text{meal}(t) + \text{noise}_{\text{AR1}}(t)$ | Observed Dynamic | Primary continuous input signal |
| `ehr_fasting_glucose` | Copied directly from `basal_mean` | **Direct Leakage Proxy** | **Model shortcut: exposes the exact mean of the unobserved basal wave** |
| `ehr_estimated_isf` | Inverse proxy for `carb_sensitivity` | Questionable Proxy | Indirectly discloses the meal response scale |
| `wearable_steps` | Stochastic bouts (sleep vs active) | Semi-Independent | Independently generated from time/sleep schedule |
| `wearable_heart_rate` | $\text{resting} + \text{steps} + 0.08 \times \max(0, \text{glucose}-125)$ | **Derived from CGM** | **Circularity: Heart rate is generated as a function of glucose** |
| `wearable_hrv` | $\text{baseline} - \text{steps}/30 - 0.4 \times (\text{HR} - \text{resting})$ | Derived from HR | Downstream echo of HR and glucose |

### 2.2 Dependency Graph

```
[Latent Patient Physiology]
       │
       ├──► basal_mean ────────────────────────────┐ (DIRECT EXPOSURE LEAKAGE)
       ├──► carb_sensitivity                       │
       └──► circadian_params                       │
               │                                   │
               ▼                                   ▼
        [Synthetic CGM Generator]          [EHR Records]
               │                                   │
        cgm_glucose(t)                     ehr_fasting_glucose
               │                                   │
       ┌───────┴───────┐                           │
       │               ▼                           │
       │     [Wearable Generator]                  │
       │     (HR = f(glucose - 125))               │
       │               │                           │
       ▼               ▼                           ▼
[CGM Features]  [Wearable Features]         [EHR Features]
 (Lags, Roll)    (Mean HR, Steps)            (FastGluc, ISF)
       │               │                           │
       └───────────────┼───────────────────────────┘
                       ▼
              [Multimodal Model]
                       │
                       ▼
          [Future Glucose Forecast]
```

**Scientific Verdict:**  
The generator topology violates the principle of independent multimodal observation. Instead of latent physiology simultaneously causing glucose variations and cardiovascular responses, the simulation feeds observed glucose into the wearable generator. Wearables are merely a noisy transformation of past CGM values.

---

## 3. Audit 2: Static EHR Target Proxy Leakage

| EHR Variable | Assessment | Severity | Findings & Justification |
|---|---|:---:|---|
| `ehr_fasting_glucose` | **CONFIRMED LEAKAGE** | **P0** | In `generate_synthetic_data.py`, `basal_mean` is the constant center around which diurnal glucose oscillates. In `ehr_records.json`, `fasting_glucose_baseline` is set to this exact value ($\pm 0.5\text{ mg/dL}$). The tree splits on this feature to shift its baseline prediction, bypassing temporal estimation. |
| `ehr_estimated_isf` | **QUESTIONABLE** | **P1** | In synthetic adult dynamics, insulin sensitivity factor is tightly coupled to carb sensitivity. Exposing ISF enables the tree to scale meal-decay predictions without estimating clearance dynamics. |
| `ehr_estimated_icr` | **QUESTIONABLE** | **P1** | Parallels ISF; represents a mechanistic controller setting rather than an observational electronic health record finding. |
| `ehr_total_daily_dose` | **SAFE** | **P3** | Total daily insulin dose is a common EHR summary variable; realistic clinical correlate. |
| `ehr_dawn_flag` | **SAFE** | **P3** | Binary diagnostic code representing dawn phenomenon; realistic clinical annotation. |
| `ehr_age`, `ehr_bmi` | **SAFE** | **P3** | Genuine demographic covariates with physiological realism. |
| `ehr_diabetes_duration` | **SAFE** | **P3** | Chronological disease duration; realistic clinical covariate. |
| `ehr_baseline_hba1c` | **SAFE** | **P3** | 90-day glycemic lab; realistic long-term macro anchor. |

**Remediation Required:**
Replace `fasting_glucose_baseline`, `estimated_isf`, and `estimated_icr` with realistic observational clinical variables (e.g., historical fasting blood glucose lab with clinical variance $\sigma = 15\text{ mg/dL}$, hypertension comorbidity flag, anti-hyperglycemic prescription category). The generator's latent parameters must remain strictly hidden.

---

## 4. Audit 3: Wearable Incremental Information

### 4.1 Empirical Incremental Breakdown

From the 7,316 held-out test windows:

$$\Delta \text{MAE}_{\text{EHR}} = \text{MAE}(\text{CGM}) - \text{MAE}(\text{CGM}+\text{EHR}) = 18.71 - 17.63 = \mathbf{-1.08\text{ mg/dL}} \quad (\text{Meaningful gain})$$

$$\Delta \text{MAE}_{\text{Wearable}} = \text{MAE}(\text{CGM}) - \text{MAE}(\text{CGM}+\text{Wear}) = 18.71 - 18.53 = \mathbf{-0.18\text{ mg/dL}} \quad (\text{Negligible gain})$$

$$\Delta \text{MAE}_{\text{Wearable|EHR}} = \text{MAE}(\text{CGM}+\text{EHR}) - \text{MAE}(\text{Full Fusion}) = 17.63 - 17.96 = \mathbf{+0.33\text{ mg/dL}} \quad (\text{Wearables degrade performance})$$

$$\Delta \text{MAE}_{\text{EHR|Wearable}} = \text{MAE}(\text{CGM}+\text{Wear}) - \text{MAE}(\text{Full Fusion}) = 18.53 - 17.96 = \mathbf{-0.57\text{ mg/dL}} \quad (\text{EHR retains benefit})$$

### 4.2 Scientific Assessment
The current wearable stream fails to provide independent predictive value. The $0.33\text{ mg/dL}$ error increase when adding wearables to EHR indicates that the 6 wearable summary features introduce feature space dimensionality and noise without novel mutual information with the target.

---

## 5. Audit 4: Statistical Testing & Autocorrelation Validity

### 5.1 Critique of Current Approach
The current ablation report states:
> *Paired Two-Sided t-test: $t = 5.342$, $p = 9.45 \times 10^{-8}$ $\longrightarrow$ Reject $H_0$.*

This calculation treats $N = 7,316$ evaluation windows as independent observations. In reality:
1. Windows are generated every 5 minutes along continuous time series. Consecutive windows $w_t$ and $w_{t+1}$ share $95.8\%$ identical input data ($23$ out of $24$ CGM readings).
2. The residuals $e_t = |y_t - \hat{y}_t|$ exhibit strong autocorrelation ($\rho_1 > 0.90$).
3. All windows are sampled from only 10 unique patients ($731$ windows per patient).

Applying standard parametric $t$-tests to autocorrelated clustered time series commits **pseudoreplication**, causing drastic underestimation of standard errors and spuriously astronomical $p$-values.

### 5.2 Defensible Alternative Methodologies
1. **Patient-Level Paired Analysis ($N = 10$):**
   Compute the mean test MAE per patient for each model. Perform a Wilcoxon signed-rank test or paired $t$-test on the 10 paired patient means ($df = 9$).
2. **Moving Block Bootstrap:**
   Resample 24-hour continuous temporal blocks with replacement to preserve intra-day autocorrelation, calculating empirical 95% Confidence Intervals for $\Delta \text{MAE}$.
3. **Cluster-Robust Standard Errors:**
   Cluster by patient ID to account for patient-level variance heterogeneity.

---

## 6. Audit 5: Review of Claims

| Claim in Documentation / UI | Audited Verdict | Rationale & Remediation |
|---|:---:|---|
| *"Dynamic wearables disambiguate resting digestion from physical activity bouts, providing statistically significant improvements."* | **REJECTED** | Wearables alone improve MAE by only $0.18\text{ mg/dL}$, and increase error by $0.33\text{ mg/dL}$ when combined with EHR. The statistical significance cited ($p = 9.45 \times 10^{-8}$) was for Full Fusion vs CGM-only, which was driven entirely by EHR. |
| *"Full two-stream fusion improves prediction."* | **MISLEADING** | Full Fusion ($17.96$) is superior to CGM-only ($18.71$), but inferior to CGM+EHR ($17.63$). Framing this as a victory for "two-stream multimodal fusion" obscures that the wearable stream degraded the model. |
| *"GlucoTwin MAE = 18.69 vs Persistence = 53.25 $\rightarrow$ 64.9% error reduction."* | **QUALIFIED** | Valid computational result on initial single-stream test set, but obsolete. Stale relative to current multimodal benchmarks ($18.71$ / $17.63$ / $17.96$). |

---

## 7. Audit 6: Clinical Condition Alignment (Type 1 vs Type 2 Diabetes)

### 7.1 Challenge Requirements vs Implementation
* **Challenge Specification:** Emphasizes chronic lifestyle conditions prevalent in India, explicitly naming **Type 2 Diabetes (T2D)** as the reference condition.
* **Current Prototype:** Labels all patients as *Type 1 Diabetes (Simulated adult dynamics)* and records T1D-specific parameters (`estimated_isf`, `estimated_icr`, `total_daily_dose`).
* **Simulator Reality:** The synthetic generator does not model subcutaneous insulin infusion or rapid-acting insulin pharmacokinetics. It simulates a steady circadian baseline plus carbohydrate-driven excursions with exponential decay. This is physiologically closer to impaired glucose tolerance / Type 2 diabetes with lifestyle meals than brittle Type 1 diabetes.

### 7.2 Strategic Recommendation
Realign the project to **Type 2 Diabetes & Lifestyle Metabolic Care** or use the neutral formulation **Synthetic Glycemic Cohort (T2D Phenotype)**. Replace T1D therapy parameters with T2D clinical covariates (BMI, HbA1c, fasting glucose lab, exercise profile, oral agent regimen).

---

## 8. Audit 7: Stale Benchmark Numbers & Provenance Drift

The codebase contains conflicting benchmark figures across components:

| Location | Reported Metric | Origin / Context | Status |
|---|---|---|:---:|
| `frontend/streamlit_app.py` (L377, L388) | MAE = **18.69** mg/dL | Baseline single-stream GBDT | **STALE** |
| `src/glucotwin/api/main.py` (L150) | MAE = **18.69** mg/dL | Hardcoded fallback | **STALE** |
| `README.md` (L103) | MAE = **18.69** mg/dL | Phase 4 single-stream | **STALE** |
| `reports/multimodal_ablation_results.json` | CGM = **18.71**, EHR = **17.63**, Full = **17.96** | Gate 1 Multimodal run | **CANONICAL** |
| `docs/reproducibility.md` | MAE = **18.69** mg/dL | Early reproducibility audit | **STALE** |

**Remediation:**  
Establish a single machine-readable manifest (`artifacts/experiment_manifest.json`). API, Streamlit UI, and documentation must dynamically ingest metrics from this manifest rather than hardcoding static floats.

---

## 9. Comprehensive Issue Severity & Action Matrix

| Issue ID | Severity | Category | Description | Exact Recommended Correction | Acceptance Criterion |
|---|:---:|---|---|---|---|
| **ISSUE-01** | **P0** | Leakage | `ehr_fasting_glucose` matches generator `basal_mean` | Perturb EHR fasting baseline with realistic laboratory measurement noise ($\sigma = 15\text{ mg/dL}$); hide latent parameters | Model cannot access exact `basal_mean`; evaluation rerun confirms robust gain |
| **ISSUE-02** | **P1** | Causality | Wearable HR generated from `glucose - 125` | Decouple wearable generator from glucose; drive HR/steps from circadian/activity schedule | Wearable generator receives only timestamp, patient phenotype, and activity bouts |
| **ISSUE-03** | **P1** | Scientific Claim | Claiming wearable significance based on Full Fusion vs CGM | Update docs/reports to state EHR provides primary gain; wearables are currently inconclusive | Docs explicitly state: "Wearables provided limited incremental gain beyond EHR" |
| **ISSUE-04** | **P1** | Statistics | Naïve paired $t$-test on 7,316 autocorrelated windows | Replaced with patient-level paired analysis ($N = 10$) and block bootstrap 95% CI | P-values and CIs computed over patient-level differences ($df = 9$) |
| **ISSUE-05** | **P2** | Condition | Condition labeled as T1D without insulin dynamics | Re-label cohort as "Synthetic Cohort (Type 2 Diabetes Phenotype)" | All UI labels, metadata, and EHR records reflect T2D lifestyle context |
| **ISSUE-06** | **P2** | Provenance | Benchmark numbers ($18.69$ vs $17.63$) hardcoded across UI | Create `artifacts/experiment_manifest.json`; dynamically load in API and UI | No hardcoded benchmark metrics in `streamlit_app.py` or `main.py` |
| **ISSUE-07** | **P2** | Safety UI | Green chart region labeled as "Safe Zone" | Rename to "Illustrative Reference Range (70–180 mg/dL)" | UI includes disclaimer that reference range is not a clinical threshold |

---

## 10. Audit Sign-off

**Gate 1.5 Verdict:** **CONDITIONAL PASS PENDING REMEDIATION**  
Before proceeding to Gate 2 (Patient Digital State Engine), the team must resolve P0 (EHR Target Proxy Leakage) and P1 (Scientific Claim & Statistical Testing Corrections). This guarantees that the digital twin state is constructed upon defensible evidence rather than simulator artifacts.
