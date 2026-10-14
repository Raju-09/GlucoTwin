"""Explicit CGM Sensor Physics Layer for In-Silico Patients.

Models realistic physiological sensor dynamics:
1. Interstitial Lag: 1st-order low-pass kinetics reflecting plasma -> interstitial glucose transport (tau = 5-10 min).
2. Autocorrelated Sensor Noise: AR(1) process representing electrochemical sensor drift/noise rather than white Gaussian noise.
3. Multiplicative Sensor Calibration Drift & Warm-up Error: Sensor insertion bias decaying over initial 2-6 hours.
4. Physiologically bounded step-rates: Asserts discrete gradient |dG/dt| < 4.0 mg/dL/min to ensure no synthetic step spikes occur.
5. Dropout & Calibration Gaps: Random dropouts and multi-sample calibration disconnects.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_cgm_sensor_layer(
    blood_glucose: np.ndarray,
    cadence_minutes: float = 5.0,
    lag_tau_minutes: float = 7.5,
    ar_phi: float = 0.85,
    noise_sigma: float = 2.0,
    cal_bias_factor: float = 1.02,
    warmup_steps: int = 24,  # First 2 hours
    dropout_rate: float = 0.025,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform ground-truth blood glucose into continuous sensor observations.

    Parameters
    ----------
    blood_glucose : np.ndarray
        True plasma/blood glucose (mg/dL).
    cadence_minutes : float
        Sampling interval in minutes (typically 5.0).
    lag_tau_minutes : float
        Time constant of interstitial diffusion lag (5.0 to 10.0 min).
    ar_phi : float
        Autoregressive coefficient for sensor error (0.80 to 0.90).
    noise_sigma : float
        Innovation standard deviation (mg/dL).
    cal_bias_factor : float
        Multiplicative sensor calibration scale (e.g. 0.96 to 1.04).
    warmup_steps : int
        Number of steps affected by sensor insertion warm-up artifact.
    dropout_rate : float
        Fraction of individual telemetry packets randomly dropped.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (cgm_readings, valid_mask)
        cgm_readings: Filtered, noisy, biased CGM values (mg/dL).
        valid_mask: Boolean array indicating non-dropped readings.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(blood_glucose)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    # 1. 1st-Order Low-Pass Filter for Interstitial Diffusion Lag
    # dG_int/dt = (G_blood - G_int) / tau
    # In discrete form with dt = cadence_minutes:
    # alpha = dt / (tau + dt)
    alpha = cadence_minutes / (lag_tau_minutes + cadence_minutes)
    g_interstitial = np.empty(n, dtype=float)
    g_interstitial[0] = blood_glucose[0]
    for i in range(1, n):
        g_interstitial[i] = (1.0 - alpha) * g_interstitial[i - 1] + alpha * blood_glucose[i]

    # 2. Correlated AR(1) Electrochemical Sensor Noise
    ar_noise = np.empty(n, dtype=float)
    ar_noise[0] = rng.normal(0, noise_sigma)
    for i in range(1, n):
        ar_noise[i] = ar_phi * ar_noise[i - 1] + rng.normal(0, noise_sigma)

    # 3. Warm-up and Multiplicative Calibration Bias
    # Warm-up error decays exponentially over initial hours
    warmup_decay = np.exp(-np.arange(n) / max(1, warmup_steps / 2))
    warmup_error = rng.uniform(-15.0, 15.0) * warmup_decay

    # Combine into raw sensor signal
    cgm_raw = (g_interstitial * cal_bias_factor) + ar_noise + warmup_error
    cgm_raw = np.clip(cgm_raw, 25.0, 500.0)

    # 4. Enforce Physiological Step-Rate Assertion
    # Check max rate of change: |dG/dt| must be < 4.0 mg/dL/min
    step_diffs = np.abs(np.diff(cgm_raw)) / cadence_minutes
    max_rate = float(np.max(step_diffs)) if len(step_diffs) > 0 else 0.0

    if max_rate >= 4.0:
        # Dampen any non-physiological instantaneous jumps
        for i in range(1, n):
            delta = (cgm_raw[i] - cgm_raw[i - 1]) / cadence_minutes
            if abs(delta) >= 3.8:
                cgm_raw[i] = cgm_raw[i - 1] + np.sign(delta) * 3.8 * cadence_minutes

    # 5. Sensor Dropouts & Temporary Disconnects
    valid_mask = rng.random(n) >= dropout_rate

    # Inject 1-2 sensor disconnect gaps (30-60 min)
    n_gaps = int(rng.integers(1, 3))
    for _ in range(n_gaps):
        gap_start = int(rng.integers(int(0.15 * n), int(0.85 * n)))
        gap_len = int(rng.integers(6, 12))  # 30-60 min
        valid_mask[gap_start : min(n, gap_start + gap_len)] = False

    return np.round(cgm_raw, 1), valid_mask
