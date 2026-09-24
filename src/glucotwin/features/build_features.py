"""Feature engineering for GlucoTwin forecasting models.

Rules enforced here:
1. STRICT ZERO LEAKAGE: Features are computed strictly using information
   available at or before the forecast origin timestamp `t`.
2. No patient ID shortcut is fed to the population model (prevents overfitting
   to specific patient identity).
3. Circadian features capture cyclical 24-hour diurnal rhythm via cyclical sine/cosine.
4. Differences and slopes capture recent physiological acceleration and deceleration.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from glucotwin.config import LOOKBACK_N

logger = logging.getLogger(__name__)


def extract_features_from_windows(
    windows_df: pd.DataFrame,
    lookback_n: int | None = None,
    include_circadian: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Extract tabular feature matrix X and target vector y from supervised windows.

    Parameters
    ----------
    windows_df:
        DataFrame produced by `create_supervised_windows` or an incoming inference window.
    lookback_n:
        Number of historical lag columns. If None, auto-detects all `glucose_lag_*` columns.
    include_circadian:
        Whether to extract cyclical sin/cos hour of day features from `origin_time`.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray | None, list[str]]
        (feature_matrix_df, target_array_or_None, feature_names)
    """
    df = windows_df.copy()
    features = pd.DataFrame(index=df.index)

    # 1. Raw historical lag readings
    if lookback_n is None:
        lag_cols = sorted(
            [c for c in df.columns if c.startswith("glucose_lag_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if not lag_cols:
            raise ValueError("No 'glucose_lag_*' columns found in windows DataFrame.")
        lookback_n = len(lag_cols)
    else:
        lag_cols = [f"glucose_lag_{i}" for i in range(lookback_n)]
        for col in lag_cols:
            if col not in df.columns:
                raise ValueError(f"Required lag column '{col}' missing from windows DataFrame.")

    for col in lag_cols:
        features[col] = df[col].astype(float)

    # 2. Key historical summary statistics
    # Matrix of past readings for fast numpy ops
    gl_matrix = df[lag_cols].to_numpy(dtype=float)  # (N, lookback_n)

    # Short-term (last 30 min = 6 steps: lags 0 to 5)
    gl_last_30 = gl_matrix[:, : min(6, lookback_n)]
    features["mean_last_30m"] = np.mean(gl_last_30, axis=1)
    features["std_last_30m"] = np.std(gl_last_30, axis=1)

    # Medium-term (last 60 min = 12 steps: lags 0 to 11)
    gl_last_60 = gl_matrix[:, : min(12, lookback_n)]
    features["mean_last_60m"] = np.mean(gl_last_60, axis=1)
    features["std_last_60m"] = np.std(gl_last_60, axis=1)
    features["min_last_60m"] = np.min(gl_last_60, axis=1)
    features["max_last_60m"] = np.max(gl_last_60, axis=1)

    # Full lookback (approx 120 min = 24 steps)
    features["mean_full_lookback"] = np.mean(gl_matrix, axis=1)
    features["std_full_lookback"] = np.std(gl_matrix, axis=1)
    features["min_full_lookback"] = np.min(gl_matrix, axis=1)
    features["max_full_lookback"] = np.max(gl_matrix, axis=1)

    # 3. Rate of change (velocity & acceleration)
    # diff_5m: y_t - y_{t-5m}
    features["delta_5m"] = features["glucose_lag_0"] - features["glucose_lag_1"]
    # diff_15m: y_t - y_{t-15m}
    if lookback_n >= 4:
        features["delta_15m"] = features["glucose_lag_0"] - features["glucose_lag_3"]
    # diff_30m: y_t - y_{t-30m}
    if lookback_n >= 7:
        features["delta_30m"] = features["glucose_lag_0"] - features["glucose_lag_6"]
    # diff_60m: y_t - y_{t-60m}
    if lookback_n >= 13:
        features["delta_60m"] = features["glucose_lag_0"] - features["glucose_lag_12"]

    # Recent curvature / acceleration: delta_5m - previous delta_5m
    if lookback_n >= 3:
        prev_delta_5m = features["glucose_lag_1"] - features["glucose_lag_2"]
        features["acceleration_5m"] = features["delta_5m"] - prev_delta_5m

    # 4. Circadian features from origin timestamp
    if include_circadian and "origin_time" in df.columns:
        origin_ts = pd.to_datetime(df["origin_time"], utc=True)
        hour_fraction = origin_ts.dt.hour + origin_ts.dt.minute / 60.0
        features["sin_hour"] = np.sin(2.0 * np.pi * hour_fraction / 24.0)
        features["cos_hour"] = np.cos(2.0 * np.pi * hour_fraction / 24.0)

    feature_names = list(features.columns)
    target = (
        df["target_glucose"].to_numpy(dtype=float)
        if "target_glucose" in df.columns
        else None
    )

    return features, target, feature_names
