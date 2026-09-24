"""Reliability and forecast gating layer for GlucoTwin.

This is a core product feature ensuring that the system NEVER returns
a fabricated or dangerously ungrounded forecast when input data integrity
is compromised.

Status codes:
- `available`: Input data is timely, complete, and within plausible range. Forecast is shown.
- `degraded`: Some inputs are missing or irregular, but sufficient historical context exists.
              Forecast is shown with a clear advisory warning and degraded flag.
- `withheld`: Data is stale, severely missing, out of range, or ungrounded.
              NO FORECAST IS RETURNED. Reason code is explicitly returned instead.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from glucotwin.config import (
    FRESHNESS_THRESHOLD_MINUTES,
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    LOOKBACK_N,
    MAX_GAP_MINUTES,
    MIN_READINGS,
)

logger = logging.getLogger(__name__)

ReliabilityStatusType = Literal["available", "degraded", "withheld"]


@dataclass
class ReliabilityAssessment:
    """Structured assessment of data quality and forecast admissibility."""

    status: ReliabilityStatusType
    should_forecast: bool
    reason_code: str
    message: str
    freshness_minutes: float | None
    missing_count: int
    total_expected: int
    quality_score_pct: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReliabilityChecker:
    """Assesses input time series integrity before invoking any model."""

    def __init__(
        self,
        freshness_threshold_minutes: int = FRESHNESS_THRESHOLD_MINUTES,
        min_readings: int = MIN_READINGS,
        max_gap_minutes: int = MAX_GAP_MINUTES,
        min_glucose: float = GLUCOSE_MIN_MGDL,
        max_glucose: float = GLUCOSE_MAX_MGDL,
        expected_cadence_minutes: float = 5.0,
    ) -> None:
        self.freshness_threshold_minutes = freshness_threshold_minutes
        self.min_readings = min_readings
        self.max_gap_minutes = max_gap_minutes
        self.min_glucose = min_glucose
        self.max_glucose = max_glucose
        self.expected_cadence_minutes = expected_cadence_minutes

    def assess_input_stream(
        self,
        timestamps: list[pd.Timestamp] | Sequence[str] | np.ndarray,
        glucose_values: list[float] | Sequence[float] | np.ndarray,
        current_time: pd.Timestamp | str | None = None,
    ) -> ReliabilityAssessment:
        """Evaluate input sequence against physiological and technical integrity criteria.

        Parameters
        ----------
        timestamps:
            Sequence of timestamps for the recent glucose readings.
        glucose_values:
            Sequence of recent glucose measurements (mg/dL).
        current_time:
            The reference prediction origin timestamp (defaults to latest timestamp if None).

        Returns
        -------
        ReliabilityAssessment
        """
        # 1. Check for empty input
        if len(timestamps) == 0 or len(glucose_values) == 0:
            return ReliabilityAssessment(
                status="withheld",
                should_forecast=False,
                reason_code="ERR_EMPTY_INPUT",
                message="No glucose readings provided in request.",
                freshness_minutes=None,
                missing_count=LOOKBACK_N,
                total_expected=LOOKBACK_N,
                quality_score_pct=0.0,
                details={"error": "empty_input"},
            )

        if len(timestamps) != len(glucose_values):
            return ReliabilityAssessment(
                status="withheld",
                should_forecast=False,
                reason_code="ERR_LENGTH_MISMATCH",
                message="Length mismatch between timestamps and glucose readings.",
                freshness_minutes=None,
                missing_count=abs(len(timestamps) - len(glucose_values)),
                total_expected=LOOKBACK_N,
                quality_score_pct=0.0,
                details={"len_ts": len(timestamps), "len_gl": len(glucose_values)},
            )

        # Parse timestamps and values
        ts_series = pd.to_datetime(pd.Series(timestamps), utc=True).sort_values()
        sorted_indices = ts_series.index
        ts = ts_series.reset_index(drop=True)
        gl = np.asarray(glucose_values, dtype=float)[sorted_indices]

        n_received = len(gl)

        # 2. Check for sufficient readings
        if n_received < self.min_readings:
            return ReliabilityAssessment(
                status="withheld",
                should_forecast=False,
                reason_code="ERR_INSUFFICIENT_HISTORY",
                message=(
                    f"Only {n_received} readings provided; minimum required is "
                    f"{self.min_readings}."
                ),
                freshness_minutes=None,
                missing_count=LOOKBACK_N - n_received,
                total_expected=LOOKBACK_N,
                quality_score_pct=round(100.0 * n_received / LOOKBACK_N, 1),
                details={"received": n_received, "required": self.min_readings},
            )

        # 3. Check Physiological Outliers (implausible sensor values)
        has_implausible = np.any((gl < self.min_glucose) | (gl > self.max_glucose))
        has_nan = np.any(np.isnan(gl))
        if has_implausible or has_nan:
            return ReliabilityAssessment(
                status="withheld",
                should_forecast=False,
                reason_code="ERR_PHYSIOLOGICAL_OUT_OF_BOUNDS",
                message=(
                    f"Sensor readings contain values outside plausible range "
                    f"[{self.min_glucose}, {self.max_glucose}] mg/dL or NaN."
                ),
                freshness_minutes=None,
                missing_count=int(np.sum(np.isnan(gl))),
                total_expected=LOOKBACK_N,
                quality_score_pct=0.0,
                details={
                    "min_val": float(np.nanmin(gl)),
                    "max_val": float(np.nanmax(gl)),
                },
            )

        # 4. Check Data Freshness
        latest_ts = ts.iloc[-1]
        ref_time = (
            pd.to_datetime(current_time, utc=True)
            if current_time is not None
            else latest_ts
        )

        age_seconds = (ref_time - latest_ts).total_seconds()
        age_minutes = max(0.0, age_seconds / 60.0)

        if age_minutes > self.freshness_threshold_minutes:
            return ReliabilityAssessment(
                status="withheld",
                should_forecast=False,
                reason_code="ERR_STALE_READING",
                message=(
                    f"Latest glucose observation is {age_minutes:.1f} minutes old; "
                    f"threshold is {self.freshness_threshold_minutes} minutes."
                ),
                freshness_minutes=round(age_minutes, 1),
                missing_count=LOOKBACK_N - n_received,
                total_expected=LOOKBACK_N,
                quality_score_pct=round(100.0 * n_received / LOOKBACK_N, 1),
                details={"age_minutes": age_minutes, "threshold": self.freshness_threshold_minutes},
            )

        # 5. Check for Large Internal Gaps
        diffs_sec = ts.diff().dropna().dt.total_seconds().values
        diffs_min = diffs_sec / 60.0
        max_internal_gap = float(np.max(diffs_min)) if len(diffs_min) > 0 else 0.0

        if max_internal_gap > self.max_gap_minutes:
            return ReliabilityAssessment(
                status="withheld",
                should_forecast=False,
                reason_code="ERR_EXCESSIVE_GAP",
                message=(
                    f"Sensor gap of {max_internal_gap:.1f} minutes exceeds the allowable "
                    f"threshold of {self.max_gap_minutes} minutes."
                ),
                freshness_minutes=round(age_minutes, 1),
                missing_count=LOOKBACK_N - n_received,
                total_expected=LOOKBACK_N,
                quality_score_pct=round(100.0 * n_received / LOOKBACK_N, 1),
                details={"max_gap_minutes": max_internal_gap, "threshold": self.max_gap_minutes},
            )

        # 6. Check for Degraded (Minor gaps or missing points)
        missing_count = max(0, LOOKBACK_N - n_received)
        quality_score = round(100.0 * min(n_received, LOOKBACK_N) / LOOKBACK_N, 1)

        if missing_count > 0 or max_internal_gap > (self.expected_cadence_minutes * 2):
            return ReliabilityAssessment(
                status="degraded",
                should_forecast=True,
                reason_code="WARN_PARTIAL_SEQUENCE",
                message=(
                    f"Sequence is degraded: {n_received}/{LOOKBACK_N} readings present "
                    f"(max gap: {max_internal_gap:.1f} min). Forecast is available but advisory."
                ),
                freshness_minutes=round(age_minutes, 1),
                missing_count=missing_count,
                total_expected=LOOKBACK_N,
                quality_score_pct=quality_score,
                details={"max_gap_minutes": max_internal_gap, "received": n_received},
            )

        # 7. Clean and Available
        return ReliabilityAssessment(
            status="available",
            should_forecast=True,
            reason_code="OK_DATA_INTACT",
            message="Data integrity verified: sequence is current and complete.",
            freshness_minutes=round(age_minutes, 1),
            missing_count=0,
            total_expected=LOOKBACK_N,
            quality_score_pct=100.0,
            details={"received": n_received, "max_gap_minutes": max_internal_gap},
        )
