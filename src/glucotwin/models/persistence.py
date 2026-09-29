"""Baseline forecasting models for GlucoTwin.

Baselines implemented:
1. Persistence Baseline:
   y_hat_{t+120} = y_t
   The future prediction is simply the latest valid observed glucose value.
   This is the non-negotiable floor every machine learning model must beat.

2. Linear Trend Baseline:
   Extrapolates recent glucose rate-of-change (slope) over the last k readings
   ahead by the forecast horizon, with physiological clipping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from glucotwin.config import (
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    HORIZON_MINUTES,
)


class PersistenceBaseline:
    """Persistence forecaster: predicts the latest observed glucose value."""

    def __init__(self, last_glucose_col: str = "last_glucose") -> None:
        self.last_glucose_col = last_glucose_col

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for each row in df.

        Parameters
        ----------
        df:
            Supervised windows DataFrame with `last_glucose` or `glucose_lag_0`.

        Returns
        -------
        np.ndarray
            Predicted values at horizon.
        """
        if self.last_glucose_col in df.columns:
            return df[self.last_glucose_col].to_numpy(dtype=float)
        if "glucose_lag_0" in df.columns:
            return df["glucose_lag_0"].to_numpy(dtype=float)
        raise ValueError(
            f"Expected '{self.last_glucose_col}' or 'glucose_lag_0' in DataFrame columns."
        )


class LinearTrendBaseline:
    """Linear trend forecaster: extrapolates recent slope over k readings."""

    def __init__(
        self,
        k_readings: int = 6,
        cadence_minutes: float = 5.0,
        horizon_minutes: float = HORIZON_MINUTES,
        min_val: float = GLUCOSE_MIN_MGDL,
        max_val: float = GLUCOSE_MAX_MGDL,
    ) -> None:
        self.k_readings = k_readings
        self.cadence_minutes = cadence_minutes
        self.horizon_minutes = horizon_minutes
        self.min_val = min_val
        self.max_val = max_val

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Estimate recent slope via linear regression over the last k readings and extrapolate."""
        # Lag columns: glucose_lag_0 (t), glucose_lag_1 (t-1), ..., glucose_lag_{k-1}
        lag_cols = [f"glucose_lag_{i}" for i in range(self.k_readings)]
        missing = [c for c in lag_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required lag columns: {missing}")

        # Values from oldest to newest: lag_{k-1}, ..., lag_0
        y_matrix = df[lag_cols[::-1]].to_numpy(dtype=float)  # shape: (N, k)
        N, k = y_matrix.shape

        # Time vector in minutes: 0, 5, 10, ..., (k-1)*5
        x = np.arange(k) * self.cadence_minutes
        x_mean = np.mean(x)
        x_diff = x - x_mean
        denom = np.sum(x_diff**2)

        # Vectorized slope: slope = sum((x - x_mean)*(y - y_mean)) / sum((x - x_mean)^2)
        y_mean = np.mean(y_matrix, axis=1, keepdims=True)
        slopes = np.sum(x_diff * (y_matrix - y_mean), axis=1) / denom  # mg/dL per minute

        # Extrapolate from the latest value (lag_0)
        y_latest = y_matrix[:, -1]
        preds = y_latest + slopes * self.horizon_minutes

        # Clip to physiological bounds to prevent explosive trend extrapolation
        return np.clip(preds, self.min_val, self.max_val)


class RecentMeanBaseline:
    """Predicts the mean of the last k historical CGM readings (e.g. last 30 minutes)."""

    def __init__(self, k_readings: int = 6) -> None:
        self.k_readings = k_readings

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate mean across last k readings."""
        lag_cols = [f"glucose_lag_{i}" for i in range(self.k_readings)]
        missing = [c for c in lag_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required lag columns: {missing}")
        return df[lag_cols].mean(axis=1).to_numpy(dtype=float)


class DiurnalClimatologyBaseline:
    """Predicts historical mean glucose conditioned on target hour of day (circadian climatology)."""

    def __init__(self, target_time_col: str = "target_time") -> None:
        self.target_time_col = target_time_col
        self.hourly_means: dict[int, float] = {}
        self.global_mean: float = 120.0

    def fit(self, train_df: pd.DataFrame, target_col: str = "target_glucose") -> DiurnalClimatologyBaseline:
        """Compute hourly empirical means on training split only."""
        ts = pd.to_datetime(train_df[self.target_time_col], utc=True)
        hours = ts.dt.hour
        grouped = train_df.groupby(hours)[target_col].mean()
        self.hourly_means = {int(h): float(m) for h, m in grouped.items()}
        self.global_mean = float(train_df[target_col].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Map target timestamp hour to empirical historical mean."""
        ts = pd.to_datetime(df[self.target_time_col], utc=True)
        hours = ts.dt.hour
        mapped = hours.map(self.hourly_means).fillna(self.global_mean)
        return mapped.to_numpy(dtype=float)
