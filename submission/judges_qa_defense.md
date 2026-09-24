# GlucoTwin — Judges' Q&A Defense
**Competition:** Happiest Health Digital Twin Challenge 2026

Prepare short, confident answers to the 12 toughest questions a panel of clinicians, ML engineers, and product judges might ask. Each answer is grounded in what was *actually built and measured* — no invented claims.

---

## Technical Questions

### Q1: Why HistGradientBoosting and not a Transformer or LSTM?

**Answer:**
> "Two reasons: interpretability and data size. Our synthetic dataset has 38,942 rows — not enough to reliably train a Transformer without significant regularization engineering. HistGradientBoosting with hand-crafted features gives us competitive performance (MAE=18.69) while every feature is auditable. We deliberately avoided deep learning so a clinical reviewer can understand *why* the model made a prediction. That said, the architecture is modular — the feature pipeline is already in place to feed an LSTM or TFT as a next step if we get access to a larger real-world dataset."

---

### Q2: Your dataset is synthetic. How do we know the model would work on real patients?

**Answer:**
> "It wouldn't — not without retraining and validation on real data. That's an explicit non-goal and is documented in the model card. What we've demonstrated is that *the architecture works*: data ingestion, leakage-free windowing, evaluation, API, reliability gating, and dashboard — all functioning end-to-end. The synthetic data reproduces documented CGM characteristics: 5-min cadence, AR(1) noise, circadian basal patterns, postprandial excursions, and sensor dropout gaps. The next milestone is plug-and-replace with DiaTrend or OhioT1DM."

---

### Q3: What's your test/train split methodology? Could there be data leakage?

**Answer:**
> "We use strict chronological splitting per patient. No patient's future data can appear in training. There's a 120-minute safety buffer between the training boundary and the validation boundary — equal to our forecast horizon — to prevent any target leakage. All windowing is timestamp-ordered. The windowing code has 4 dedicated leakage-prevention tests. You can inspect `src/glucotwin/features/windows.py` and `tests/test_windows.py` directly."

---

### Q4: MAE of 18.69 mg/dL — is that clinically meaningful?

**Answer:**
> "It's a legitimate question. The clinical standard used in CGM evaluation is the Clarke Error Grid. At 120-minute prediction horizon, MAE=18.69 mg/dL means the average absolute error is about one CGM reading width. To be clinically useful, you'd want Zone A+B coverage on the Clarke grid — we haven't measured that, and we've documented it as a next step in the model card. What we *can* say is that 18.69 is a 64.9% improvement over the persistence baseline (53.25), which is the standard naive comparator for CGM forecasting research."

---

### Q5: What does 'personalization' mean here? Did you train a separate model per patient?

**Answer:**
> "No. We train a single population model, then estimate per-patient bias using training-split data only (strictly no test-set information). At inference time, we add the patient's bias estimate to the model's output. This is a bias-correction approach, not a separate per-patient model. It's conservative by design — we only apply it when there's enough patient-specific history to estimate a stable bias. The audit showed 6/10 patients benefit; SYNTH_002 sees an 11.3% MAE reduction. We've documented the methodology and LaTeX formulas in `docs/personalization_feasibility.md`."

---

### Q6: What are the 41 features? Could you describe the most important ones?

**Answer:**
> "The features fall into four groups:
> 1. **24 lag readings** — glucose at t, t-5min, …, t-115min. The most recent readings dominate.
> 2. **Rolling statistics** — mean and std over 30/60/120-minute windows.
> 3. **Rate-of-change deltas** — Δglucose over 5/15/30/60 minutes, and a 5-min acceleration (second derivative).
> 4. **Circadian encoding** — sin(hour) and cos(hour) to capture time-of-day patterns without ordinal assumptions.
>
> The most predictive are almost certainly lag_0 (current reading) and the short-window deltas — which is expected: at 120 minutes, current level and recent trend dominate. We can show feature importance from the trained model."

---

## Clinical / Safety Questions

### Q7: This is a healthcare application. What are your safety guardrails?

**Answer:**
> "Three layers. First, the system explicitly states it is a research prototype and not a medical device, diagnostic system, or treatment recommender — this is in the README, the model card, the API responses, and the dashboard UI. Second, the reliability layer suppresses predictions when input quality falls below threshold — stale readings, sensor dropouts, physiologically impossible values, insufficient history. Third, all advisory messages in the dashboard are informational only — there are no dosing recommendations, no alerts to act on, no clinical actions triggered. The system is designed to inform, not decide."

---

### Q8: Why do you withhold predictions instead of just showing a low-confidence interval?

**Answer:**
> "Because a displayed number — even with a wide confidence interval — anchors clinical judgment. If a clinician sees '160 mg/dL (±40)' during a sensor dropout, they may still act on it. A withheld state ('No prediction — insufficient data') is unambiguous. This is a deliberate design decision documented in `docs/limitations.md`. In a future version, we could display prediction intervals for the 'degraded' state, but never for the 'withheld' state."

---

### Q9: What happens if someone uses this as a medical device anyway?

**Answer:**
> "The same thing that happens if someone uses a calculator to dose insulin — it's misuse of a tool that explicitly states it's not for clinical use. We've done everything reasonably possible: disclaimer in README, disclaimer in the API response payload, disclaimer in the dashboard UI, and a `docs/limitations.md` file. If this progresses toward clinical deployment, it would need FDA Software as Medical Device clearance, a clinical validation study, IRB approval, and a privacy/security review. We're nowhere near that and we say so clearly."

---

## Product / Business Questions

### Q10: Who is your user?

**Answer:**
> "For this prototype: a clinical researcher or endocrinologist reviewing a synthetic patient's glucose trajectory in a research or educational context. The dashboard is designed for someone who understands CGM data and wants to explore what a reliability-aware forecast system could look like. It's a proof of concept for a clinical decision support tool — not a patient-facing app."

---

### Q11: How long did this take to build? Is it reproducible?

**Answer:**
> "The full pipeline — from data generation through model training, API, dashboard, and Docker deployment — was built in a single focused sprint. Reproducibility is a first-class requirement: `git clone` + `pip install -e .` + `python scripts/generate_synthetic_data.py` + `python scripts/build_dataset.py` + `python scripts/train_model.py` gives you a trained model in minutes. `docker-compose up` gives you the full stack. 56 automated tests validate the pipeline end-to-end."

---

### Q12: What would you do with 3 more months and access to real data?

**Answer:**
> "In order of priority:
> 1. **Real data integration** — plug in DiaTrend or OhioT1DM; validate the pipeline handles real-world data quality issues.
> 2. **Clarke Error Grid evaluation** — replace MAE-only reporting with clinical accuracy grids.
> 3. **Prediction intervals** — use conformal prediction or quantile regression to report 90% confidence intervals.
> 4. **Per-patient model adaptation** — move from bias correction to proper online learning as patient history accumulates.
> 5. **User study** — recruit 2–3 endocrinologists to evaluate the dashboard for clinical utility and usability."

---

## Backup answers to have ready

- **"Why not use OpenAI / a large language model?"** → LLMs don't natively handle time-series forecasting with data-quality gating. A structured ML pipeline is more appropriate, auditable, and deployable without API costs or rate limits.
- **"What's your data source?"** → Fully synthetic, procedurally generated to match documented CGM characteristics. No real patient data used. No HIPAA/GDPR concerns.
- **"Did you validate on multiple datasets?"** → Not yet. Single synthetic dataset. Multi-dataset validation is a documented next step.
- **"What is the carbon footprint of your model?"** → The model is tiny (~KB). HistGradientBoosting inference is CPU-bound and takes milliseconds. No GPU required.
