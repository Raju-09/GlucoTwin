"""Statistical controls and null-hypothesis falsification tests for GlucoTwin.

Provides:
- Patient-level EHR permutation control (shuffling EHR profiles between patients).
- Patient-level paired bootstrap confidence intervals (resampling over patients, not windows).
- Paired statistical hypothesis tests.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def permute_ehr_patient_level(
    df: pd.DataFrame,
    ehr_cols: list[str],
    patient_col: str = "patient_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Permute EHR features strictly at the patient level.

    Shuffles static phenotype assignments between patients, preserving
    all internal correlations between EHR fields within each patient profile
    while breaking the link between the patient's actual CGM trajectory and EHR.
    """
    rng = np.random.default_rng(random_state)
    # Extract one row per patient
    patient_ehr = df.groupby(patient_col, as_index=False)[ehr_cols].first()
    pids = patient_ehr[patient_col].values
    permuted_pids = rng.permutation(pids)

    # Map original patient_id to shuffled patient_id's EHR profile
    permuted_profiles = patient_ehr.set_index(patient_col).loc[permuted_pids].copy()
    permuted_profiles = permuted_profiles.reset_index(drop=True)
    permuted_profiles[patient_col] = pids

    # Replace EHR columns in df
    df_perm = df.drop(columns=ehr_cols).copy()
    df_perm = df_perm.merge(permuted_profiles, on=patient_col, how="left")
    return df_perm


def bootstrap_patient_deltas(
    deltas: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval resampling over independent patients.

    Parameters
    ----------
    deltas : np.ndarray
        Array of per-patient improvements or differences (length N_patients).
    n_boot : int
        Number of bootstrap iterations.
    ci : float
        Confidence level (e.g. 0.95).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float]
        (mean_delta, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_state)
    n = len(deltas)
    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    mean_val = float(np.mean(deltas))

    return mean_val, lower, upper
