# Personalization Feasibility Audit

> Evaluated on held-out test split (7,316 windows across 10 patients)  
> Audit report: `reports/personalization_audit.json`

## Research Question

> Does adjusting the global population model with a patient's own historical calibration data improve forecasting on an unseen, held-out future period for that same patient?

## Methodology & Leakage Safeguards

1. **Strict Temporal Separation**:
   - The global model was trained on the chronological training split (Days 1–9).
   - Patient-specific bias was estimated **strictly from training split residuals**:
     $$\text{bias}_i = \frac{1}{N_{train,i}} \sum_{j=1}^{N_{train,i}} (y_{i,j} - \hat{y}_{\text{global},i,j})$$
   - Calibration adjustments were evaluated on the future, untouched test split (Days 12–14).
   - Zero test data was used to estimate the adaptation parameters.

2. **Adapted Prediction**:
   $$\hat{y}_{\text{adapted}, i, t+120} = \text{clip}\left(\hat{y}_{\text{global}, i, t+120} + \text{bias}_i, 20, 600\right)$$

## Observed Results

| Patient ID | Train Samples | Train Bias (mg/dL) | Global MAE | Adapted MAE | MAE Change | % Improvement |
|---|---|---|---|---|---|---|
| `SYNTH_001` | 2,501 | +1.12 | 21.13 | 21.56 | -0.43 | -2.05% |
| `SYNTH_002` | 2,451 | -7.17 | 20.52 | 18.20 | **+2.32** | **+11.30%** |
| `SYNTH_003` | 2,466 | -0.06 | 16.84 | 16.84 | 0.00 | +0.02% |
| `SYNTH_004` | 2,460 | -2.98 | 18.51 | 18.37 | +0.15 | +0.80% |
| `SYNTH_005` | 2,478 | +2.06 | 15.27 | 14.79 | +0.48 | +3.16% |
| `SYNTH_006` | 2,513 | +0.10 | 14.02 | 14.05 | -0.02 | -0.17% |
| `SYNTH_007` | 2,473 | +4.71 | 17.64 | 17.40 | +0.24 | +1.38% |
| `SYNTH_008` | 2,436 | +2.02 | 19.31 | 19.27 | +0.04 | +0.20% |
| `SYNTH_009` | 2,445 | -5.32 | 19.30 | 19.30 | -0.01 | -0.03% |
| `SYNTH_010` | 2,424 | +5.18 | 24.29 | 24.91 | -0.63 | -2.59% |
| **Overall** | **24,647** | — | **18.69** | **18.48** | **+0.21** | **+1.12%** |

## Findings & Recommendation

1. **Heterogeneous Benefit**: Personalization is not a silver bullet. It yields a strong improvement for patients with systematic physiological baseline offsets (e.g. `SYNTH_002` with **+11.3%** improvement), but has negligible or slightly negative effect on patients whose baseline varies non-stationarily (`SYNTH_001`, `SYNTH_010`).
2. **Recommendation: GO with selective calibration**: Provide patient calibration as an inspectable option in the dashboard while reporting both global and adapted metrics transparently.
