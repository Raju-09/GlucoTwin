# GlucoTwin — Pitch Deck Outline
**Competition:** Happiest Health Digital Twin Challenge 2026
**Deck format:** 10 slides, ~5 min presentation + 5 min Q&A

> **Design note:** Use a clean, clinical aesthetic. White backgrounds. Accent color: `#0077B6` (deep ocean blue). No stock photos of stethoscopes. Use your own charts and screenshots.

---

## Slide 1 — Title

**Headline:** GlucoTwin
**Subline:** Patient-Specific Glucose Forecasting with Data-Quality Awareness
**Footer:** Happiest Health Digital Twin Challenge 2026 | Research Prototype

*Visual:* Single glucose timeline with a forecast point glowing at t+120 min.

---

## Slide 2 — The Problem (Why this matters)

**Headline:** 120 minutes is the difference between intervention and emergency

**3 bullet points:**
- Hypoglycemic episodes develop over 1–3 hours — invisible to reactive monitoring
- Clinicians see glucose *after* the swing, not *before*
- Existing CGM alerts are reactive thresholds, not predictive forecasts

**1 data point:** (cite: ADA 2023 standards or similar — do not invent a number; use a real one or omit)

*Visual:* A glucose curve with a "danger zone" reached before any alert fires vs. GlucoTwin's forecast catching it 120 min early.

---

## Slide 3 — The Prediction Target (Precise scope)

**Headline:** One well-defined prediction

**Content:**
> **Given:** The last 24 CGM readings (≈120 min of history, 5-min intervals)
> **Predict:** Glucose value 120 minutes in the future (±10 min tolerance)
> **For whom:** Synthetic patients — feasibility prototype, not clinical deployment

**Why 120 min?**
- Long enough to enable meaningful clinical lead time
- Short enough to be learnable from CGM lags without external inputs

*Visual:* Simple input → model → output diagram. Label the 24-reading window and the 120-min horizon.

---

## Slide 4 — System Architecture

**Headline:** A three-layer architecture: Ingest → Forecast → Verify

**3 layers:**
1. **Data Layer** — Synthetic CGM (38,942 rows, 10 patients, 14 days). Validates schema, units, gaps, physiological range.
2. **Model Layer** — HistGradientBoosting, 41 features (lags, rolling stats, rate-of-change, circadian encoding).
3. **Reliability Layer** — 6 gating reason codes. Status: `available / degraded / withheld`. Prediction suppressed when data quality is insufficient.

**Stack:** Python · scikit-learn · FastAPI · Streamlit · Docker

*Visual:* The architecture diagram from `docs/architecture.md` — simplified to 3 columns.

---

## Slide 5 — The Reliability Layer (Key differentiator)

**Headline:** The model knows when to stay silent

**Table:**

| Scenario | Status | Prediction returned? |
|---|---|---|
| Clean 24-reading window | AVAILABLE | Yes |
| Stale reading (>20 min) | DEGRADED | Yes + warning |
| Sensor dropout | WITHHELD | No |
| Physiologically impossible value | WITHHELD | No |
| Insufficient history (<12 readings) | WITHHELD | No |

**Key message:** An overconfident wrong prediction is more dangerous than no prediction. GlucoTwin refuses to guess under uncertainty.

*Visual:* The three status badges (green/yellow/red) from the dashboard screenshot.

---

## Slide 6 — Results (Evidence slide)

**Headline:** 64.9% error reduction over the persistence baseline

**Results table:**

| Model | MAE (mg/dL) | RMSE (mg/dL) | MAPE |
|---|---|---|---|
| Persistence Baseline | 53.25 | 71.26 | 31.8% |
| Linear Trend Baseline | 97.38 | 145.10 | — |
| **GlucoTwin v0.1** | **18.69** | **30.84** | **11.8%** |

**Personalization result:**
- Overall: 18.69 → 18.48 mg/dL (−1.1%)
- Best patient (SYNTH_002): −11.3%
- 6/10 patients benefit from bias correction

**Test methodology:** Held-out test split, strict chronological ordering, 120-min safety buffer between train/test boundary.

*Visual:* Bar chart comparing MAE across three models.

---

## Slide 7 — The Dashboard (Live product)

**Headline:** A clinician-ready interface, not just a model

**4 features to highlight:**
1. **Patient selector** — switch between synthetic patients
2. **Timeline playback** — scrub through 14 days of history
3. **Scenario injector** — inject stale/gap/spike/dropout conditions in one click
4. **Model Evidence tab** — validation metrics, per-patient breakdown, feature importance

*Visual:* Full-width screenshot of Streamlit dashboard (SYNTH_002 selected, green badge).

---

## Slide 8 — What We Learned (Technical depth)

**Headline:** Three non-obvious findings

1. **The trend baseline is worse than persistence at 120 min** — Linear extrapolation amplifies noise at long horizons. Persistence wins as a naive baseline. Our model beats both.

2. **Meal context isn't needed for competitive forecasting** — `meal_carbs_g` is 98.8% sparse in real CGM data. GlucoTwin achieves MAE=18.69 using only glucose lags + circadian encoding. Meal data would help but isn't required.

3. **Reliability gating matters more than accuracy** — A model that predicts wrong during a sensor dropout but shows a plausible number is actively harmful. The withheld state is not a failure — it's a feature.

---

## Slide 9 — Limitations & Honest Scope

**Headline:** What GlucoTwin is not

**Non-goals (verbatim from `docs/limitations.md`):**
- Not a medical device
- Not a diagnostic system
- Not a treatment recommender
- Not validated on real patient data
- Not a substitute for clinical judgment

**Path to real deployment:**
- Real patient data (DiaTrend, OhioT1DM, or partner data)
- Clinical validation study
- Regulatory pathway (FDA Software as Medical Device)
- Privacy/security review

*Visual:* Simple roadmap: Research Prototype → Pilot Study → Clinical Validation → Regulatory Review

---

## Slide 10 — Close & Ask

**Headline:** What's next

**3-point close:**
1. GlucoTwin demonstrates that a 120-min glucose forecast with reliability gating is feasible on synthetic CGM data
2. The full pipeline (data → model → API → dashboard) is reproducible in < 60 seconds with Docker
3. The framework is ready for real patient data — the limiting factor is data access, not engineering

**Ask:** (tailor to the competition's judging criteria)
- Access to de-identified CGM dataset for clinical validation
- Mentorship from endocrinology / clinical informatics collaborators

**Footer:** `github.com/[your-handle]/glucotwin` | Apache/MIT License | Contact: [your-email]

---

## Appendix slides (have ready for Q&A)

- **A1:** Feature importance chart (top 10 of 41 features)
- **A2:** Per-patient MAE breakdown table (all 10 patients)
- **A3:** Personalization audit — before/after per patient
- **A4:** `docker-compose up` → running in 60 seconds (terminal screenshot)
- **A5:** API response JSON for `/forecast` endpoint
