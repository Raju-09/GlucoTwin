"""Leave-One-Patient-Out (LOPO) Cross-Validation and Controls.

Evaluates cross-patient generalization by training on N-1 patients and
evaluating strictly on the unseen held-out patient.
Also provides patient-level permutation control to verify whether
EHR context conveys real physiological signal or patient-identity shortcuts.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def make_standard_gbm(random_state: int = 42) -> Pipeline:
    """Factory for standard HistGradientBoosting pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def run_lopo_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_glucose",
    patient_col: str = "patient_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Leave-One-Patient-Out cross validation.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset with features, target, and patient_id.
    feature_cols : list[str]
        List of feature column names.
    target_col : str
        Target column name.
    patient_col : str
        Grouping column (patient_id).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Per-window predictions with columns:
        [patient_id, y_true, y_pred, abs_error].
    """
    logo = LeaveOneGroupOut()
    groups = df[patient_col].values
    records = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(df, groups=groups)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        test_patient = test_df[patient_col].iloc[0]

        model = make_standard_gbm(random_state=random_state)
        model.fit(train_df[feature_cols], train_df[target_col])
        preds = model.predict(test_df[feature_cols])

        for pid, yt, yp in zip(test_df[patient_col], test_df[target_col], preds):
            records.append(
                {
                    "patient_id": pid,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "abs_error": float(abs(yt - yp)),
                }
            )

    return pd.DataFrame(records)


def compute_patient_level_deltas(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    label_a: str = "cgm_only",
    label_b: str = "cgm_plus_ehr",
) -> pd.DataFrame:
    """Compute per-patient summary and paired improvement deltas."""
    mae_a = results_a.groupby("patient_id")["abs_error"].mean().rename(f"mae_{label_a}")
    mae_b = results_b.groupby("patient_id")["abs_error"].mean().rename(f"mae_{label_b}")

    summary = pd.concat([mae_a, mae_b], axis=1)
    # Improvement = MAE_A - MAE_B (positive means B improves upon A)
    summary["improvement_mgdl"] = summary[f"mae_{label_a}"] - summary[f"mae_{label_b}"]
    # Delta MAE = MAE_B - MAE_A (negative means B has lower error than A)
    summary["delta_mae_mgdl"] = summary[f"mae_{label_b}"] - summary[f"mae_{label_a}"]
    return summary
