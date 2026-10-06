"""Wearable and IoT data generation, loading, and preprocessing.

This module provides synchronized multi-sensor physiological signals
(Heart Rate, Steps, Heart Rate Variability, Sleep state) that pair
with the patient timeline.

Strict causality and independence guarantee:
Telemetry is generated from autonomous behavioral and circadian processes
(sleep schedule, activity bouts, diurnal autonomic tone) without feeding in
contemporaneous CGM glucose values, preventing simulator circularity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from glucotwin.config import SEED

logger = logging.getLogger(__name__)


def generate_wearable_telemetry(
    cgm_df: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate synchronized wearable telemetry for patient timelines.

    Parameters
    ----------
    cgm_df:
        DataFrame containing `patient_id` and `timestamp`.
    seed:
        Random seed for reproducible telemetry simulation.

    Returns
    -------
    pd.DataFrame
        Columns: [patient_id, timestamp, heart_rate, steps, hrv_rmssd, sleep_state]
    """
    df = cgm_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    records: list[dict] = []

    # Process per patient for physiological continuity
    for pid, group in df.groupby("patient_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue

        p_idx = int(pid.split("_")[-1]) if "_" in pid and pid.split("_")[-1].isdigit() else 1
        p_rng = np.random.default_rng(seed + p_idx * 37)

        # Baseline resting HR for this patient (e.g. 62 to 74 bpm)
        resting_hr = p_rng.uniform(62.0, 74.0)
        # Baseline HRV for this patient (e.g. 35 to 55 ms)
        baseline_hrv = p_rng.uniform(35.0, 55.0)

        ts = group["timestamp"]
        hours = ts.dt.hour + ts.dt.minute / 60.0

        # 1. Sleep state: 23:00 to 07:00 (with individual jitter)
        sleep_start = p_rng.uniform(22.5, 23.5)
        sleep_end = p_rng.uniform(6.5, 7.5)
        is_asleep = ((hours >= sleep_start) | (hours < sleep_end)).astype(int)

        # 2. Activity / Step count (bouts of daytime physical activity)
        steps = np.zeros(n, dtype=int)
        for i in range(n):
            if is_asleep[i]:
                steps[i] = 0
            else:
                # Base ambient movements: 10 - 70 steps / 5 min
                base_steps = p_rng.integers(10, 75)
                # Occasional brisk walking / exercise bout (p ~ 0.06)
                if p_rng.random() < 0.06:
                    base_steps += p_rng.integers(250, 600)
                steps[i] = base_steps

        # 3. Heart Rate (bpm)
        # Autonomous model: resting base - nocturnal dip + activity spike + diurnal sympathetic tone
        hr = np.full(n, resting_hr)
        # Nocturnal dipping during sleep (-12 bpm)
        hr -= is_asleep * p_rng.uniform(10.0, 14.0)

        # Diurnal sympathetic tone (peaking late morning 10:00 - 12:00)
        diurnal_hr = 4.0 * np.cos(2 * np.pi * (hours - 11.0) / 24.0)
        hr += diurnal_hr

        # Physical activity effect: +1 bpm per 25 steps up to max +55 bpm
        hr += np.clip(steps / 25.0, 0.0, 55.0)

        # Autoregressive AR(1) physiological momentum + sensor noise
        hr_noise = np.zeros(n)
        curr_noise = 0.0
        for i in range(n):
            curr_noise = 0.75 * curr_noise + p_rng.normal(0, 1.8)
            hr_noise[i] = curr_noise
        hr = np.clip(hr + hr_noise, 45.0, 185.0).round(1)

        # 4. HRV (RMSSD in ms): autonomic tone
        # Elevated during restorative sleep, drops during physical activity or stress
        hrv = baseline_hrv + (is_asleep * 16.0) - (steps / 30.0) - ((hr - resting_hr) * 0.35)
        hrv += p_rng.normal(0, 3.0, size=n)
        hrv = np.clip(hrv, 12.0, 95.0).round(1)

        for i in range(n):
            records.append(
                {
                    "patient_id": pid,
                    "timestamp": ts.iloc[i],
                    "heart_rate": float(hr[i]),
                    "steps": int(steps[i]),
                    "hrv_rmssd": float(hrv[i]),
                    "sleep_state": int(is_asleep[i]),
                }
            )

    wearable_df = pd.DataFrame(records)
    logger.info(
        "Generated %d decoupled wearable records across %d patient(s).",
        len(wearable_df),
        wearable_df["patient_id"].nunique(),
    )
    return wearable_df


def save_wearable_telemetry(
    wearable_df: pd.DataFrame,
    out_paths: Sequence[str | Path],
) -> None:
    """Save wearable telemetry to Parquet/CSV files."""
    for p in out_paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".parquet":
            wearable_df.to_parquet(p, index=False)
        else:
            wearable_df.to_csv(p, index=False)
        logger.info("Saved wearable telemetry to %s", p)
