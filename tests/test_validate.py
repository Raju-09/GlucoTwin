"""Unit tests for glucotwin.data.validate.

All tests use synthetic fixtures from conftest.py.
No real patient data is loaded or required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glucotwin.data.validate import (
    check_glucose_units,
    check_missingness,
    check_physiological_range,
    check_sampling_intervals,
    check_schema,
    check_timestamps,
    run_full_validation,
)


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------


class TestCheckSchema:
    def test_all_columns_present(self, synthetic_cgm_df):
        result = check_schema(
            synthetic_cgm_df, ["patient_id", "timestamp", "glucose_mgdl"]
        )
        assert result.passed
        assert not result.errors

    def test_missing_column(self, synthetic_cgm_df):
        result = check_schema(synthetic_cgm_df, ["patient_id", "nonexistent_col"])
        assert not result.passed
        assert len(result.errors) == 1
        assert "nonexistent_col" in result.errors[0].message

    def test_multiple_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = check_schema(df, ["b", "c", "d"])
        assert not result.passed
        assert result.errors[0].count == 3

    def test_empty_expected_cols(self, synthetic_cgm_df):
        result = check_schema(synthetic_cgm_df, [])
        assert result.passed


# ---------------------------------------------------------------------------
# check_timestamps
# ---------------------------------------------------------------------------


class TestCheckTimestamps:
    def test_clean_timestamps(self, synthetic_cgm_df):
        result = check_timestamps(synthetic_cgm_df, "timestamp", "patient_id")
        assert result.passed
        dup_issues = [
            i for i in result.issues if "duplicate" in i.check
        ]
        assert len(dup_issues) == 0

    def test_duplicate_timestamps_detected(self, cgm_df_with_duplicates):
        result = check_timestamps(
            cgm_df_with_duplicates, "timestamp", "patient_id"
        )
        dup_issues = [i for i in result.issues if "duplicate" in i.check]
        assert len(dup_issues) == 1
        assert dup_issues[0].count > 0

    def test_missing_ts_column(self, synthetic_cgm_df):
        result = check_timestamps(synthetic_cgm_df, "nonexistent_ts")
        assert not result.passed

    def test_unsorted_timestamps_flagged(self, synthetic_cgm_df):
        # Shuffle the DataFrame to break ordering
        shuffled = synthetic_cgm_df.sample(frac=1, random_state=0).reset_index(
            drop=True
        )
        result = check_timestamps(shuffled, "timestamp")
        ordering_issues = [i for i in result.issues if "ordering" in i.check]
        assert len(ordering_issues) == 1


# ---------------------------------------------------------------------------
# check_glucose_units
# ---------------------------------------------------------------------------


class TestCheckGlucoseUnits:
    def test_mgdl_values_pass(self, synthetic_cgm_df):
        # synthetic_cgm_df has values ~100 mg/dL, well above threshold
        result = check_glucose_units(synthetic_cgm_df, "glucose_mgdl")
        assert result.passed
        mmol_issues = [i for i in result.issues if "mmol_suspected" in i.check]
        assert len(mmol_issues) == 0

    def test_mmol_values_flagged(self, cgm_df_mmol):
        # Values are ~7.0 (mmol/L) which is below the 30 threshold
        result = check_glucose_units(cgm_df_mmol, "glucose_mgdl")
        mmol_issues = [i for i in result.issues if "mmol_suspected" in i.check]
        assert len(mmol_issues) == 1

    def test_missing_column(self, synthetic_cgm_df):
        result = check_glucose_units(synthetic_cgm_df, "no_such_col")
        assert not result.passed


# ---------------------------------------------------------------------------
# check_missingness
# ---------------------------------------------------------------------------


class TestCheckMissingness:
    def test_no_missing(self, synthetic_cgm_df):
        result = check_missingness(synthetic_cgm_df)
        # Should pass (no errors); may have no issues at all
        assert result.passed

    def test_missing_values_reported(self, synthetic_cgm_df):
        df = synthetic_cgm_df.copy()
        df.loc[0, "glucose_mgdl"] = float("nan")
        result = check_missingness(df)
        missing_issues = [
            i for i in result.issues if "glucose_mgdl" in i.check
        ]
        assert len(missing_issues) == 1
        assert missing_issues[0].count == 1

    def test_empty_dataframe(self):
        result = check_missingness(pd.DataFrame())
        assert any("empty" in i.message.lower() for i in result.issues)


# ---------------------------------------------------------------------------
# check_physiological_range
# ---------------------------------------------------------------------------


class TestCheckPhysiologicalRange:
    def test_normal_range(self, synthetic_cgm_df):
        result = check_physiological_range(synthetic_cgm_df, "glucose_mgdl")
        # synthetic values are clipped to [60, 300], within [20, 600]
        range_issues = [
            i for i in result.issues
            if "below_min" in i.check or "above_max" in i.check
        ]
        assert len(range_issues) == 0

    def test_below_min_flagged(self):
        df = pd.DataFrame(
            {"glucose_mgdl": [100, 10, 5, 120]}  # 10 and 5 are below 20
        )
        result = check_physiological_range(df, "glucose_mgdl")
        below = [i for i in result.issues if "below_min" in i.check]
        assert len(below) == 1
        assert below[0].count == 2

    def test_above_max_flagged(self):
        df = pd.DataFrame({"glucose_mgdl": [100, 700, 800]})
        result = check_physiological_range(df, "glucose_mgdl")
        above = [i for i in result.issues if "above_max" in i.check]
        assert len(above) == 1
        assert above[0].count == 2

    def test_missing_column(self, synthetic_cgm_df):
        result = check_physiological_range(synthetic_cgm_df, "wrong_col")
        assert not result.passed


# ---------------------------------------------------------------------------
# check_sampling_intervals
# ---------------------------------------------------------------------------


class TestCheckSamplingIntervals:
    def test_regular_cadence(self, synthetic_cgm_df):
        result = check_sampling_intervals(
            synthetic_cgm_df, "timestamp", expected_interval_minutes=5.0
        )
        long_gap_issues = [
            i for i in result.issues if "long_gaps" in i.check
        ]
        assert len(long_gap_issues) == 0

    def test_long_gap_detected(self, cgm_df_with_gap):
        result = check_sampling_intervals(
            cgm_df_with_gap, "timestamp", expected_interval_minutes=5.0
        )
        long_gap_issues = [
            i for i in result.issues if "long_gaps" in i.check
        ]
        assert len(long_gap_issues) == 1
        assert long_gap_issues[0].count >= 1

    def test_missing_ts_col(self, synthetic_cgm_df):
        result = check_sampling_intervals(synthetic_cgm_df, "no_such_col")
        assert not result.passed


# ---------------------------------------------------------------------------
# run_full_validation
# ---------------------------------------------------------------------------


class TestRunFullValidation:
    def test_clean_df_passes(self, synthetic_cgm_df):
        result = run_full_validation(
            synthetic_cgm_df,
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            patient_col="patient_id",
        )
        assert result.passed
        assert len(result.errors) == 0

    def test_missing_required_col_fails(self):
        df = pd.DataFrame({"timestamp": ["2024-01-01"], "glucose": [100]})
        result = run_full_validation(
            df,
            ts_col="timestamp",
            glucose_col="glucose",
            patient_col="patient_id",  # not in df
        )
        assert not result.passed

    def test_mmol_and_gap_gives_warnings(self, cgm_df_mmol):
        # mmol values + only one patient = should flag unit suspicion
        result = run_full_validation(
            cgm_df_mmol,
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            patient_col="patient_id",
        )
        assert len(result.warnings) > 0
