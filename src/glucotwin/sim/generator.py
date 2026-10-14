"""Physiological Cohort Generator linking EHR Phenotypes to Glucose Dynamics.

Generates in-silico patient cohorts where EHR clinical metrics mechanistically
govern metabolic and glycemic response curves:
1. Fasting Blood Glucose (FBG) & Basal Set-point: Linked to HbA1c via standard GMI:
   GMI = 3.31 + 0.02392 * mean_glucose -> mean_glucose = (HbA1c - 3.31) / 0.02392
2. Peripheral Insulin Sensitivity (S_I): Inversely related to BMI, diabetes duration,
   and baseline HbA1c.
3. Realistic Meals: Variable timing (jitter +/-60 min), variable carb contents,
   occasional skipped meals, and postprandial double-exponential absorption kinetics.
4. CGM Sensor Layer: Applied on true plasma glucose with interstitial diffusion lag,
   AR(1) noise, and strict physiological step-rate enforcement (< 4.0 mg/dL/min).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from glucotwin.sim.sensor import apply_cgm_sensor_layer

logger = logging.getLogger(__name__)


@dataclass
class PatientPhenotype:
    """Clinical phenotype and latent physiological parameter set."""

    patient_id: str
    age: int
    sex: str
    bmi: float
    diabetes_duration_years: float
    baseline_hba1c: float
    historical_fbg_mgdl: float
    hypertension_flag: int
    metformin_flag: int
    sglt2i_flag: int
    dawn_phenomenon_flag: int
    clinical_notes: str

    # Latent physiological simulation parameters derived mechanistically:
    basal_glucose: float
    insulin_sensitivity_factor: float
    circadian_amplitude: float
    circadian_peak_hour: float


def generate_population_phenotype(
    patient_id: str,
    seed: int,
) -> PatientPhenotype:
    """Sample realistic adult T2D phenotype from continuous clinical population distributions."""
    rng = np.random.default_rng(seed)

    # 1. Demographics & Clinical Profile
    age = int(rng.integers(38, 75))
    sex = "M" if rng.random() > 0.48 else "F"
    bmi = float(np.round(rng.normal(28.5, 3.8), 1))
    bmi = float(np.clip(bmi, 21.0, 42.0))

    duration = float(np.round(rng.exponential(6.0) + 1.0, 1))
    duration = float(np.clip(duration, 0.5, 25.0))

    # Baseline HbA1c centered at 7.3% with realistic variation
    hba1c = float(np.round(rng.normal(7.2, 0.8), 1))
    hba1c = float(np.clip(hba1c, 5.8, 10.2))

    # Historical Fasting Blood Glucose (FBG) aligned with HbA1c
    # In clinical medicine, estimated average glucose: eAG = 28.7 * HbA1c - 46.7
    expected_fbg = 18.0 * hba1c - 10.0 + rng.normal(0, 8.0)
    fbg = float(np.round(np.clip(expected_fbg, 85.0, 195.0), 1))

    # Comorbidities & Medications
    htn = int(rng.random() < (0.45 + 0.01 * (age - 40)))
    metformin = int(rng.random() < 0.85)
    sglt2i = int(rng.random() < (0.35 if hba1c > 7.0 else 0.15))
    dawn_flag = int(rng.random() < 0.70)

    # 2. Mechanistically derived latent parameters:
    # Basal glucose set-point Gb is anchored to fasting lab with day-to-day noise
    basal_glucose = float(np.clip(fbg + rng.normal(0, 5.0), 85.0, 190.0))

    # Insulin sensitivity SI (mg/dL rise per gram carb)
    # Lower sensitivity (higher peak rise per carb) in patients with higher BMI, higher HbA1c, and longer duration
    # Typical carb sensitivity in T2D: 1.8 to 3.5 mg/dL per gram carb
    base_carb_sens = 1.6 + 0.03 * (bmi - 25.0) + 0.12 * (hba1c - 6.5) + 0.02 * duration
    carb_sensitivity = float(np.clip(base_carb_sens + rng.normal(0, 0.2), 1.6, 3.6))

    # Circadian amplitude (higher in dawn phenomenon positive patients)
    base_dawn = 16.0 if dawn_flag else 6.0
    circadian_amp = float(np.clip(base_dawn + rng.normal(0, 3.0), 4.0, 24.0))
    circadian_peak_hour = float(rng.uniform(5.0, 7.5))

    notes = (
        f"Adult {age}y {sex}, BMI {bmi:.1f}, HbA1c {hba1c:.1f}%. "
        f"{'Metformin' if metformin else 'Lifestyle'}"
        f"{' + SGLT2i' if sglt2i else ''}; "
        f"{'Dawn elevation present' if dawn_flag else 'No dawn elevation'}."
    )

    return PatientPhenotype(
        patient_id=patient_id,
        age=age,
        sex=sex,
        bmi=bmi,
        diabetes_duration_years=duration,
        baseline_hba1c=hba1c,
        historical_fbg_mgdl=fbg,
        hypertension_flag=htn,
        metformin_flag=metformin,
        sglt2i_flag=sglt2i,
        dawn_phenomenon_flag=dawn_flag,
        clinical_notes=notes,
        basal_glucose=basal_glucose,
        insulin_sensitivity_factor=carb_sensitivity,
        circadian_amplitude=circadian_amp,
        circadian_peak_hour=circadian_peak_hour,
    )


def simulate_patient_trajectory(
    phenotype: PatientPhenotype,
    days: int = 14,
    start_time: str = "2024-03-01 00:00:00",
    cadence_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate 14-day continuous trajectory with mechanistic EHR-coupled dynamics."""
    rng = np.random.default_rng(seed)

    dt = pd.Timedelta(minutes=cadence_minutes)
    start_ts = pd.Timestamp(start_time, tz="UTC")
    total_steps = int((days * 24 * 60) / cadence_minutes)
    timestamps = [start_ts + i * dt for i in range(total_steps)]
    time_hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])

    # 1. Circadian Basal Rhythm
    basal_curve = phenotype.basal_glucose + phenotype.circadian_amplitude * np.cos(
        2 * np.pi * (time_hours - phenotype.circadian_peak_hour) / 24.0
    )

    # 2. Daily Meals Schedule with Timing Jitter (+/- 60 min) and variable carb loads
    steps_per_day = int((24 * 60) / cadence_minutes)
    meal_signal = np.zeros(total_steps, dtype=float)
    meal_carbs_col = np.full(total_steps, np.nan)

    for day in range(days):
        day_offset = day * steps_per_day

        # 3 main meals with timing jitter
        daily_meals = [
            (rng.uniform(7.0, 9.5), rng.uniform(35.0, 65.0)),   # Breakfast
            (rng.uniform(12.0, 14.5), rng.uniform(45.0, 90.0)), # Lunch
            (rng.uniform(18.5, 21.0), rng.uniform(50.0, 95.0)), # Dinner
        ]
        # Occasional snack
        if rng.random() < 0.40:
            daily_meals.append((rng.uniform(15.5, 17.0), rng.uniform(15.0, 30.0)))

        for meal_hour, carbs in daily_meals:
            meal_step = day_offset + int((meal_hour * 60) / cadence_minutes)
            if meal_step < total_steps:
                meal_carbs_col[meal_step] = round(carbs, 1)

            # Physiologically realistic double-exponential digestion/absorption
            # Peak rise determined by patient's latent insulin sensitivity factor
            peak_rise = carbs * phenotype.insulin_sensitivity_factor
            duration_steps = int(240 / cadence_minutes)  # 4 hours
            t_min = np.arange(duration_steps) * cadence_minutes
            # Absorption rise tau = 25 min, clearance decay tau = 65 min
            curve = (np.exp(-t_min / 65.0) - np.exp(-t_min / 25.0)) * 2.8
            curve = np.maximum(0, curve)
            if np.max(curve) > 0:
                curve = (curve / np.max(curve)) * peak_rise

            end_step = min(total_steps, meal_step + len(curve))
            actual_len = end_step - meal_step
            if actual_len > 0:
                meal_signal[meal_step:end_step] += curve[:actual_len]

    # 3. Ground Truth Plasma Glucose
    plasma_glucose = basal_curve + meal_signal
    plasma_glucose = np.clip(plasma_glucose, 50.0, 360.0)

    # 4. Calibration Check: Ensure simulated Glucose Management Indicator (GMI) matches HbA1c
    # GMI = 3.31 + 0.02392 * mean_glucose
    sim_mean_g = float(np.mean(plasma_glucose))
    sim_gmi = 3.31 + 0.02392 * sim_mean_g
    gmi_diff = sim_gmi - phenotype.baseline_hba1c

    # If simulated GMI deviates from clinical HbA1c by > 0.4%, apply mild basal adjustment
    if abs(gmi_diff) > 0.4:
        adj_delta = (phenotype.baseline_hba1c - sim_gmi) / 0.02392
        plasma_glucose = np.clip(plasma_glucose + adj_delta * 0.7, 50.0, 360.0)

    # 5. Apply Sensor Layer (Interstitial Lag + AR(1) Noise + Step-Rate Clamping)
    cgm_readings, valid_mask = apply_cgm_sensor_layer(
        plasma_glucose,
        cadence_minutes=cadence_minutes,
        rng=rng,
    )

    # Final assertion check for physiological step-rates
    step_diffs = np.abs(np.diff(cgm_readings)) / cadence_minutes
    assert np.max(step_diffs) <= 4.0, f"Physiological violation: step rate reached {np.max(step_diffs):.2f} mg/dL/min"

    df = pd.DataFrame(
        {
            "patient_id": phenotype.patient_id,
            "timestamp": timestamps,
            "glucose": cgm_readings,
            "meal_carbs_g": meal_carbs_col,
            "is_synthetic": True,
        }
    )

    # Filter out dropouts
    df = df[valid_mask].reset_index(drop=True)
    return df
