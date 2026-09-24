# GlucoTwin — Problem Definition

> Status: Draft · Last updated: 2026-09-23

## 1. Prediction Target

| Property | Value |
|---|---|
| **Input signal** | Blood glucose readings from a Continuous Glucose Monitor (CGM), recorded approximately every 5 minutes |
| **Forecast horizon** | 120 minutes |
| **Input lookback** | Last N readings available at forecast origin (N determined from dataset cadence audit) |
| **Output** | Point forecast of blood glucose at t+120 min (mg/dL) |
| **Reliability output** | `available`, `degraded`, or `withheld`, with a machine-readable reason code |
| **Optional context** | Any verified contextual features (e.g., time of day, patient baseline) confirmed to exist in the selected dataset |

## 2. Intended User

**Primary:** A clinician or researcher reviewing a synthetic demo patient's glucose trajectory and forecast.

**Secondary:** A researcher evaluating whether a time-series model improves on a persistence baseline and whether patient-specific context helps.

## 3. Intended Use

- Review a synthetic or permitted patient's recent glucose history.
- Inspect a 120-minute forecast and assess its reliability.
- Understand when and why the system withholds a forecast.
- Compare model performance against simple baselines on held-out evaluation data.

## 4. Non-Goals (Explicit)

- **Not** a diagnostic system.
- **Not** a treatment recommender.
- **Not** an insulin dosing advisor.
- **Not** a clinical decision support tool.
- **Not** validated for use with real patients in a clinical setting.
- **Not** a whole-body digital twin or general health platform.
- **Not** a real-time wearable integration.
- **Not** a second disease or multi-condition system.

## 5. Central Research Question

> Does a time-series model improve on a simple persistence baseline on a leakage-safe held-out test set — and when should the system withhold a forecast?

### Secondary research questions

- Does patient-specific historical context improve forecasting beyond history-only features?
- Does patient-specific adaptation (personalization) improve performance on held-out periods for the same patient?
- How does forecast error change under realistic data failures (missing, stale, irregular readings)?

## 6. Safety Limitations

- This is a **research/educational prototype only**.
- Forecasts must never be used to guide insulin dosing, medication changes, or any clinical decision.
- All demo patient data is synthetic.
- The system must explicitly label synthetic and simulated streams.
- The system must show when a forecast is unavailable due to missing, stale, invalid, or out-of-distribution inputs.
- Uncertainty intervals, if displayed, must be empirically calibrated — not decorative.
- The system must not claim clinical validity or clinical utility.

## 7. Success Criteria (Research Prototype)

1. Persistence baseline is implemented and evaluated on a leakage-safe held-out set.
2. A learned model is compared against persistence on the same test examples.
3. The reliability layer correctly withholds forecasts under defined failure conditions.
4. All evaluation results are traceable to real output files.
5. The demo runs end-to-end with synthetic data without requiring access to restricted patient files.
