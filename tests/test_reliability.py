"""Unit tests for the Reliability & Forecast Gating Layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glucotwin.reliability.checks import ReliabilityChecker


@pytest.fixture
def checker() -> ReliabilityChecker:
    return ReliabilityChecker(
        freshness_threshold_minutes=20,
        min_readings=12,
        max_gap_minutes=30,
        min_glucose=20.0,
        max_glucose=600.0,
    )


def test_clean_input_available(checker: ReliabilityChecker):
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(24)]
    values = [120.0 + i for i in range(24)]

    assessment = checker.assess_input_stream(
        timestamps, values, current_time=timestamps[-1]
    )
    assert assessment.status == "available"
    assert assessment.should_forecast is True
    assert assessment.reason_code == "OK_DATA_INTACT"


def test_stale_reading_withheld(checker: ReliabilityChecker):
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(24)]
    values = [120.0] * 24

    # Current time is 45 minutes after the latest observation (threshold is 20m)
    current_time = timestamps[-1] + pd.Timedelta(minutes=45)

    assessment = checker.assess_input_stream(
        timestamps, values, current_time=current_time
    )
    assert assessment.status == "withheld"
    assert assessment.should_forecast is False
    assert assessment.reason_code == "ERR_STALE_READING"
    assert assessment.freshness_minutes >= 45.0


def test_insufficient_history_withheld(checker: ReliabilityChecker):
    # Only 5 readings provided when 12 is minimum
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(5)]
    values = [110.0] * 5

    assessment = checker.assess_input_stream(
        timestamps, values, current_time=timestamps[-1]
    )
    assert assessment.status == "withheld"
    assert assessment.should_forecast is False
    assert assessment.reason_code == "ERR_INSUFFICIENT_HISTORY"


def test_excessive_gap_withheld(checker: ReliabilityChecker):
    # A 50-minute gap inserted into a 20-reading sequence
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    t1 = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(10)]
    t2 = [t1[-1] + pd.Timedelta(minutes=50 + 5 * i) for i in range(10)]
    timestamps = t1 + t2
    values = [130.0] * 20

    assessment = checker.assess_input_stream(
        timestamps, values, current_time=timestamps[-1]
    )
    assert assessment.status == "withheld"
    assert assessment.should_forecast is False
    assert assessment.reason_code == "ERR_EXCESSIVE_GAP"


def test_physiological_out_of_bounds_withheld(checker: ReliabilityChecker):
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(24)]
    values = [120.0] * 23 + [750.0]  # 750 > 600 max

    assessment = checker.assess_input_stream(
        timestamps, values, current_time=timestamps[-1]
    )
    assert assessment.status == "withheld"
    assert assessment.should_forecast is False
    assert assessment.reason_code == "ERR_PHYSIOLOGICAL_OUT_OF_BOUNDS"


def test_partial_sequence_degraded(checker: ReliabilityChecker):
    # 18 readings (between 12 and 24): degraded status but forecast allowed
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [base_ts + pd.Timedelta(minutes=5 * i) for i in range(18)]
    values = [125.0] * 18

    assessment = checker.assess_input_stream(
        timestamps, values, current_time=timestamps[-1]
    )
    assert assessment.status == "degraded"
    assert assessment.should_forecast is True
    assert assessment.reason_code == "WARN_PARTIAL_SEQUENCE"
    assert assessment.missing_count == 6


def test_empty_input_withheld(checker: ReliabilityChecker):
    assessment = checker.assess_input_stream([], [])
    assert assessment.status == "withheld"
    assert assessment.should_forecast is False
    assert assessment.reason_code == "ERR_EMPTY_INPUT"
