# Submission Pack

Files in this folder prepare the Happiest Health Digital Twin Challenge 2026 submission.

| File | Purpose |
|---|---|
| [`demo_video_script.md`](demo_video_script.md) | Timed 3.5-min recording script with on-screen actions |
| [`pitch_deck_outline.md`](pitch_deck_outline.md) | 10-slide deck structure with content for each slide |
| [`judges_qa_defense.md`](judges_qa_defense.md) | 12 toughest judge questions + defensible answers |

## Recording order

1. Set up environment (API + Streamlit running, SYNTH_002 selected)
2. Record using `demo_video_script.md` — exact cues are timestamped
3. Build slides from `pitch_deck_outline.md` — use real screenshots from recording
4. Rehearse Q&A from `judges_qa_defense.md` — every answer is grounded in real metrics

## Key numbers (do not change without re-running evaluation)

| Metric | Value | Source |
|---|---|---|
| Persistence MAE | 53.25 mg/dL | `artifacts/evaluations/eval_persistence_test.json` |
| GlucoTwin v0.1 MAE | 18.69 mg/dL | `artifacts/evaluations/eval_ml_test.json` |
| Error reduction | 64.9% | Computed |
| SYNTH_002 personalization uplift | 11.3% | `reports/personalization_audit.json` |
| Test suite | 56/56 passing | `pytest tests/` |
