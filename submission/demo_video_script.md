# GlucoTwin — Demo Video Script
**Competition:** Happiest Health Digital Twin Challenge 2026
**Deadline:** October 20, 2026, 7 PM IST
**Target runtime:** 3 min 30 sec (keep under 4 min)

---

## Pre-recording checklist
- [ ] Streamlit running on `http://localhost:8501`
- [ ] FastAPI running on `http://localhost:8000`
- [ ] Screen resolution: 1920×1080
- [ ] Mic tested. No background noise.
- [ ] Browser zoom: 100%. Dark mode off (Streamlit default).
- [ ] Patient SYNTH_002 pre-selected (best personalization uplift: +11.3%)
- [ ] Scenario selector: "Normal" pre-set

---

## Script

### [0:00 – 0:20] Hook — Open on the problem (20 sec)

> "Every 4 seconds, someone with diabetes experiences an unexpected glucose swing. The challenge? By the time a clinician sees the number, the window to intervene has already passed. GlucoTwin changes that by forecasting glucose 120 minutes before it happens."

**Screen:** Title slide or static README banner.

---

### [0:20 – 0:50] What it is — One sentence, no jargon (30 sec)

> "GlucoTwin is a patient-specific glucose forecasting prototype. It ingests the last 2 hours of continuous glucose readings and outputs a 120-minute-ahead forecast — with a reliability layer that tells you *when not to trust the prediction*."

**Screen:** Architecture diagram from `docs/architecture.md`.

Highlight the three pillars: **Input → Model → Reliability Gate**

---

### [0:50 – 1:30] Live demo — Dashboard walkthrough (40 sec)

> "Here's the live dashboard. I'm looking at patient SYNTH_002 — a synthetic patient with realistic circadian glucose patterns."

**Action:** Click patient selector → SYNTH_002

> "This timeline shows 14 days of glucose history. The green band is the target range — 70 to 180 mg/dL. I can scrub through time with this slider."

**Action:** Move playback slider slowly from left to right.

> "The blue line is actual glucose. The orange dot — that's the 120-minute forecast. Watch how it tracks the patient's trajectory."

**Action:** Pause at a postprandial spike.

> "At this point the model predicted 168 mg/dL. Actual was 172. That's a 4 mg/dL error — well within clinical relevance."

---

### [1:30 – 2:10] Reliability scenarios — The differentiator (40 sec)

> "Here's what makes GlucoTwin different from a plain prediction model. It has a built-in reliability layer. Watch what happens when data quality degrades."

**Action:** Select scenario → "Stale Reading (20 min old)"

> "The badge just changed to DEGRADED. The model is still running, but the dashboard is warning the clinician that the input data is stale."

**Action:** Select scenario → "Sensor Dropout (no data)"

> "Now it's WITHHELD. No prediction. Because an overconfident wrong prediction is more dangerous than no prediction. The system knows when to stay silent."

**Action:** Select scenario → "Glucose Spike (400 mg/dL)"

> "With an out-of-range reading, the system flags it as physiologically implausible and degrades gracefully."

**Action:** Return to → "Normal"

> "Back to clean data — prediction restored, status green."

---

### [2:10 – 2:40] Evidence — Numbers, not claims (30 sec)

> "This isn't just a dashboard. Here are the actual validation numbers from our held-out test split."

**Action:** Click "Model Evidence" tab in the dashboard.

> "The persistence baseline — just predicting the current reading won't change — gives a MAE of 53 mg/dL at 120 minutes. GlucoTwin v0.1 achieves 18.69 mg/dL. That's a 64.9% error reduction."

> "We also ran a personalization audit. For SYNTH_002, adding a patient-level bias correction cut error by a further 11.3%. The framework is there — we just need real patient data to unlock it."

---

### [2:40 – 3:05] Architecture in 15 seconds (25 sec)

> "Under the hood: scikit-learn HistGradientBoosting trained on 41 handcrafted CGM features — lags, rolling stats, rate-of-change, and circadian time encoding. No deep learning, no black box. Every feature is interpretable."

> "The API is FastAPI. The frontend is Streamlit. The whole stack runs locally in Docker in under 60 seconds."

**Screen:** Show `docker-compose.yml` briefly or the `/health` API response in browser.

---

### [3:05 – 3:20] Limitations — Honesty is a feature (15 sec)

> "What GlucoTwin is *not*: it's not a medical device, not a diagnostic system, and not a treatment recommender. It's a research prototype that demonstrates the feasibility of data-quality-aware glucose forecasting. Real deployment would require clinical validation, regulatory review, and real patient data."

---

### [3:20 – 3:30] Close (10 sec)

> "GlucoTwin. Forecast earlier. Intervene smarter. Stay silent when uncertain."

**Screen:** GitHub repo URL + QR code (if available). Fade out.

---

## Post-recording notes
- Export at 1080p, 30fps
- Max file size for submission: check challenge portal
- Add captions if portal supports it (accessibility)
- Filename: `glucotwin_demo_v1.mp4`
