"""Shared pytest fixtures for GlucoTwin tests.

All fixtures here use synthetic data only. No restricted patient
files are accessed at any point during testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glucotwin.config import SEED


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible test data."""
    return np.random.default_rng(SEED)


@pytest.fixture()
def synthetic_cgm_df() -> pd.DataFrame:
    """A minimal synthetic CGM DataFrame with clean, regular readings.

    Returns
    -------
    pd.DataFrame
        Columns: patient_id (str), timestamp (datetime64[ns, UTC]),
        glucose_mgdl (float).
    """
    rng = np.random.default_rng(SEED)
    n = 60  # 60 readings ≈ 5 hours at 5-min cadence
    base_time = pd.Timestamp("2024-01-15 08:00:00", tz="UTC")
    timestamps = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n)]
    # Simulate a slowly varying glucose signal
    glucose = 100.0 + np.cumsum(rng.normal(0, 1.5, n))
    glucose = np.clip(glucose, 60, 300)
    return pd.DataFrame(
        {
            "patient_id": ["P001"] * n,
            "timestamp": timestamps,
            "glucose_mgdl": glucose.round(1),
        }
    )


@pytest.fixture()
def synthetic_cgm_df_multipatient() -> pd.DataFrame:
    """Synthetic CGM data for 3 patients with different sequence lengths."""
    rng = np.random.default_rng(SEED + 1)
    frames = []
    configs = [
        ("P001", 100, 110.0),  # patient, n_readings, base_glucose
        ("P002", 60, 140.0),
        ("P003", 80, 95.0),
    ]
    base_time = pd.Timestamp("2024-01-15 08:00:00", tz="UTC")
    for pid, n, base in configs:
        timestamps = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n)]
        glucose = base + np.cumsum(rng.normal(0, 1.5, n))
        glucose = np.clip(glucose, 60, 300)
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": [pid] * n,
                    "timestamp": timestamps,
                    "glucose_mgdl": glucose.round(1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def cgm_df_with_duplicates(synthetic_cgm_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic CGM DataFrame with 3 duplicate timestamps inserted."""
    extras = synthetic_cgm_df.iloc[5:8].copy()
    return pd.concat([synthetic_cgm_df, extras], ignore_index=True)


@pytest.fixture()
def cgm_df_with_gap(synthetic_cgm_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic CGM DataFrame with a 45-minute gap in the middle."""
    gap_start = 20
    gap_end = 29  # drop readings 20-28 → ~45-min gap
    return synthetic_cgm_df.drop(
        index=list(range(gap_start, gap_end))
    ).reset_index(drop=True)


@pytest.fixture()
def cgm_df_mmol() -> pd.DataFrame:
    """Synthetic CGM DataFrame with values in mmol/L (should be detected)."""
    rng = np.random.default_rng(SEED + 2)
    n = 40
    base_time = pd.Timestamp("2024-01-15 08:00:00", tz="UTC")
    timestamps = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n)]
    glucose_mmol = 7.0 + np.cumsum(rng.normal(0, 0.1, n))
    glucose_mmol = np.clip(glucose_mmol, 3.0, 20.0)
    return pd.DataFrame(
        {
            "patient_id": ["P_MMOL"] * n,
            "timestamp": timestamps,
            "glucose_mgdl": glucose_mmol.round(2),  # column named mgdl but values are mmol
        }
    )
