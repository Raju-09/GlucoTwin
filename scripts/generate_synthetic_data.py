"""Generate a transparent, physiologically plausible synthetic CGM dataset.

This script creates synthetic continuous glucose monitoring (CGM) time series
for multiple patients with realistic properties:
- Circadian basal oscillation (circadian rhythm)
- Postprandial glucose excursions (meals with absorption & clearance curves)
- Realistic autocorrelated sensor noise
- Occasional missing readings and sensor dropouts (to test reliability gating)
- Clearly labeled with `is_synthetic=True` on every record

Intended use:
Allows testing and reproducible evaluation of the entire GlucoTwin pipeline
(audit, windowing, baselines, learned models, reliability, API, and UI)
without requiring access to restricted real patient data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from glucotwin.config import SEED


def simulate_patient_cgm(
    patient_id: str,
    days: int = 14,
    start_time: str = "2024-03-01 00:00:00",
    cadence_minutes: int = 5,
    seed: int = 42,
    dropout_rate: float = 0.03,
) -> pd.DataFrame:
    """Simulate realistic CGM time series for a single synthetic patient."""
    rng = np.random.default_rng(seed)

    # 1. Timeline
    dt = pd.Timedelta(minutes=cadence_minutes)
    start_ts = pd.Timestamp(start_time, tz="UTC")
    total_steps = int((days * 24 * 60) / cadence_minutes)
    timestamps = [start_ts + i * dt for i in range(total_steps)]

    # 2. Patient baseline parameters
    basal_mean = rng.uniform(105.0, 140.0)  # average fasting glucose
    circadian_amp = rng.uniform(8.0, 20.0)  # dawn phenomenon / circadian swing
    circadian_peak_hour = rng.uniform(5.0, 8.0)  # morning peak
    carb_sensitivity = rng.uniform(1.8, 3.2)  # mg/dL rise per gram carb

    time_hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])

    # 3. Circadian basal wave
    basal = basal_mean + circadian_amp * np.cos(
        2 * np.pi * (time_hours - circadian_peak_hour) / 24.0
    )

    # 4. Meals simulation
    # Schedule ~3 meals per day: Breakfast (7-9), Lunch (12-14), Dinner (18-20)
    # plus occasional snack
    meal_signal = np.zeros(total_steps)
    meal_carbs_col = np.full(total_steps, np.nan)

    steps_per_day = int((24 * 60) / cadence_minutes)
    for day in range(days):
        day_offset = day * steps_per_day

        # Daily meal times & carbs
        daily_meals = [
            (rng.uniform(7.5, 9.0), rng.uniform(35.0, 65.0)),   # Breakfast
            (rng.uniform(12.5, 14.0), rng.uniform(50.0, 85.0)), # Lunch
            (rng.uniform(18.5, 20.5), rng.uniform(55.0, 95.0)), # Dinner
        ]
        if rng.random() > 0.4:
            # Afternoon snack
            daily_meals.append((rng.uniform(15.5, 16.5), rng.uniform(15.0, 30.0)))

        for meal_hour, carbs in daily_meals:
            meal_step = day_offset + int((meal_hour * 60) / cadence_minutes)
            if meal_step < total_steps:
                meal_carbs_col[meal_step] = round(carbs, 1)

            # Glucose excursion curve: peak at ~45-60 min, decay over ~150-180 min
            # Modeled via double exponential: (e^(-t/tau_decay) - e^(-t/tau_rise))
            peak_rise = carbs * carb_sensitivity
            duration_steps = int(240 / cadence_minutes)  # 4 hours
            t_min = np.arange(duration_steps) * cadence_minutes
            curve = (np.exp(-t_min / 60.0) - np.exp(-t_min / 20.0)) * 2.5
            curve = np.maximum(0, curve)
            if np.max(curve) > 0:
                curve = (curve / np.max(curve)) * peak_rise

            end_step = min(total_steps, meal_step + len(curve))
            actual_len = end_step - meal_step
            if actual_len > 0:
                meal_signal[meal_step:end_step] += curve[:actual_len]

    # 5. AR(1) physiological momentum + sensor noise
    noise = np.zeros(total_steps)
    current_noise = 0.0
    for i in range(total_steps):
        current_noise = 0.85 * current_noise + rng.normal(0, 2.5)
        noise[i] = current_noise

    # Combine signals
    glucose = basal + meal_signal + noise
    glucose = np.clip(glucose, 45.0, 380.0)  # plausible physiological range

    # 6. Inject realistic gaps and dropouts
    # Random dropouts
    mask_valid = rng.random(total_steps) >= dropout_rate

    # Inject 1-2 longer sensor calibration/disconnect gaps (40-60 mins)
    for _ in range(rng.integers(1, 3)):
        gap_start = rng.integers(int(0.2 * total_steps), int(0.8 * total_steps))
        gap_len = rng.integers(8, 14)  # 40-70 mins
        mask_valid[gap_start : gap_start + gap_len] = False

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "timestamp": timestamps,
            "glucose": np.round(glucose, 1),
            "meal_carbs_g": meal_carbs_col,
            "is_synthetic": True,
        }
    )

    # Filter out dropouts (missing readings in CGM stream)
    df = df[mask_valid].reset_index(drop=True)
    return df


def generate_benchmark_dataset(
    n_patients: int = 10,
    days: int = 14,
    out_paths: list[Path] | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate multi-patient synthetic benchmark."""
    all_patients: list[pd.DataFrame] = []
    for i in range(1, n_patients + 1):
        pid = f"SYNTH_{i:03d}"
        p_df = simulate_patient_cgm(
            patient_id=pid,
            days=days,
            seed=seed + i * 100,
        )
        all_patients.append(p_df)

    combined = pd.concat(all_patients, ignore_index=True)

    if out_paths:
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(p, index=False)
            print(f"Saved synthetic benchmark to: {p} ({len(combined):,} rows)")

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CGM benchmark.")
    parser.add_argument("--patients", type=int, default=10, help="Number of patients (default: 10)")
    parser.add_argument("--days", type=int, default=14, help="Days per patient (default: 14)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw" / "synthetic_cgm_benchmark.csv"
    demo_path = project_root / "data" / "synthetic_demo" / "synthetic_cgm_benchmark.csv"

    generate_benchmark_dataset(
        n_patients=args.patients,
        days=args.days,
        out_paths=[raw_path, demo_path],
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
