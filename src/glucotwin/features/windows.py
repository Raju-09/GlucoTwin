"""Supervised forecasting window construction for GlucoTwin.

Crucial safety rules:
1. STRICT ZERO LEAKAGE: The historical input window ends at origin timestamp `t`.
2. The target value `y_{t+120}` is NEVER present in the input feature sequence.
3. Windows crossing patient boundaries are strictly forbidden.
4. Windows with missing targets or internal gaps exceeding `max_gap_minutes`
   are rejected with auditable rejection codes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from glucotwin.config import (
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    HORIZON_MINUTES,
    LOOKBACK_N,
    MAX_GAP_MINUTES,
    TARGET_TOLERANCE_MINUTES,
)

logger = logging.getLogger(__name__)


@dataclass
class WindowingReport:
    """Audit report for window creation and rejections."""

    total_candidates: int = 0
    accepted_windows: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "accepted_windows": self.accepted_windows,
            "acceptance_rate_pct": (
                round(100.0 * self.accepted_windows / self.total_candidates, 2)
                if self.total_candidates > 0
                else 0.0
            ),
            "rejections": self.rejections,
        }


def create_supervised_windows(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    ts_col: str = "timestamp",
    glucose_col: str = "glucose",
    lookback_n: int = LOOKBACK_N,
    horizon_minutes: int = HORIZON_MINUTES,
    tolerance_minutes: int = TARGET_TOLERANCE_MINUTES,
    max_gap_minutes: int = MAX_GAP_MINUTES,
    step_stride: int = 1,
) -> tuple[pd.DataFrame, WindowingReport]:
    """Construct supervised forecasting windows across all patients.

    Parameters
    ----------
    df:
        Preprocessed DataFrame sorted by (patient, timestamp).
    patient_col:
        Patient identifier column.
    ts_col:
        Timestamp column (must be UTC datetime).
    glucose_col:
        Glucose reading column (mg/dL).
    lookback_n:
        Number of consecutive readings in the historical lookback window.
    horizon_minutes:
        Forecast horizon (e.g., 120 minutes).
    tolerance_minutes:
        Tolerance around target horizon (e.g. ±10 minutes).
    max_gap_minutes:
        Max allowable gap between consecutive readings in the lookback window.
    step_stride:
        Step size between successive candidate forecast origins.

    Returns
    -------
    tuple[pd.DataFrame, WindowingReport]
        DataFrame of supervised examples and audit report.
    """
    report = WindowingReport()
    windows: list[dict[str, Any]] = []

    # Ensure timestamps are datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    grouped = df.groupby(patient_col, sort=False)

    for pid, group in grouped:
        group = group.sort_values(by=ts_col).reset_index(drop=True)
        n_pts = len(group)

        if n_pts < lookback_n + 1:
            # Entire patient has insufficient points
            report.total_candidates += max(1, n_pts)
            report.record_rejection("patient_insufficient_records")
            continue

        timestamps = group[ts_col].values
        glucoses = group[glucose_col].values.astype(float)

        target_delta_ns = np.timedelta64(horizon_minutes, "m")
        tolerance_ns = np.timedelta64(tolerance_minutes, "m")

        # Slide candidate origin t from (lookback_n - 1) to (n_pts - 1)
        for i in range(lookback_n - 1, n_pts, step_stride):
            report.total_candidates += 1

            # 1. Historical lookback window: [i - lookback_n + 1 : i + 1]
            lookback_ts = timestamps[i - lookback_n + 1 : i + 1]
            lookback_gl = glucoses[i - lookback_n + 1 : i + 1]

            # 2. Check gaps inside lookback window
            # Compute consecutive differences in minutes
            dt_min = (
                (lookback_ts[1:] - lookback_ts[:-1]).astype("timedelta64[s]").astype(float)
                / 60.0
            )
            if np.any(dt_min > max_gap_minutes):
                report.record_rejection("lookback_gap_exceeded")
                continue

            # 3. Check physiological plausibility of lookback values
            if np.any(lookback_gl < GLUCOSE_MIN_MGDL) or np.any(lookback_gl > GLUCOSE_MAX_MGDL):
                report.record_rejection("lookback_implausible_value")
                continue

            origin_time = timestamps[i]
            ideal_target_time = origin_time + target_delta_ns

            # 4. Search for future reading near ideal_target_time
            # Narrow search space to future readings for this patient
            future_mask = timestamps[i + 1 :] >= (ideal_target_time - tolerance_ns)
            candidates_idx = np.where(future_mask)[0]

            if len(candidates_idx) == 0:
                report.record_rejection("target_not_found_after_origin")
                continue

            first_candidate_pos = i + 1 + candidates_idx[0]
            candidate_ts = timestamps[first_candidate_pos]
            diff_from_ideal = np.abs(candidate_ts - ideal_target_time)

            if diff_from_ideal > tolerance_ns:
                report.record_rejection("target_outside_tolerance")
                continue

            target_val = glucoses[first_candidate_pos]
            if target_val < GLUCOSE_MIN_MGDL or target_val > GLUCOSE_MAX_MGDL or np.isnan(target_val):
                report.record_rejection("target_implausible_or_nan")
                continue

            # Construct clean supervised example
            # Features: lag_0 is latest observed at t, lag_1 is previous, etc.
            example: dict[str, Any] = {
                "patient_id": pid,
                "origin_time": pd.Timestamp(origin_time),
                "target_time": pd.Timestamp(candidate_ts),
                "target_glucose": round(float(target_val), 2),
                "last_glucose": round(float(lookback_gl[-1]), 2),  # y_t
            }

            # Lag features (lag 0 = t, lag 1 = t-1, ...)
            for lag_idx in range(lookback_n):
                example[f"glucose_lag_{lag_idx}"] = round(
                    float(lookback_gl[lookback_n - 1 - lag_idx]), 2
                )

            windows.append(example)
            report.accepted_windows += 1

    result_df = pd.DataFrame(windows)
    logger.info(
        "Windowing complete: %d accepted / %d candidates (%.1f%%).",
        report.accepted_windows,
        report.total_candidates,
        (100.0 * report.accepted_windows / report.total_candidates)
        if report.total_candidates > 0
        else 0,
    )
    return result_df, report
