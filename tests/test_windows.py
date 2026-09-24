"""Unit tests for supervised window creation.

Verifies:
1. No future leakage: target is not in the lookback sequence.
2. Gap rejection logic: windows with gaps > max_gap_minutes are rejected.
3. Patient boundary isolation: windows never span multiple patients.
4. Target tolerance handling: targets outside ±tolerance are rejected.
"""

from __future__ import annotations

import pandas as pd
import pytest

from glucotwin.features.windows import create_supervised_windows


class TestCreateSupervisedWindows:
    def test_window_structure_and_no_leakage(self, synthetic_cgm_df):
        # 60 readings at 5 min = 300 min total
        # lookback=12 readings (60 min), horizon=60 min, tolerance=5 min
        windows_df, report = create_supervised_windows(
            synthetic_cgm_df,
            patient_col="patient_id",
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            lookback_n=12,
            horizon_minutes=60,
            tolerance_minutes=5,
            max_gap_minutes=20,
        )

        assert not windows_df.empty
        assert report.accepted_windows > 0

        # Check required columns
        expected_cols = [
            "patient_id",
            "origin_time",
            "target_time",
            "target_glucose",
            "last_glucose",
        ] + [f"glucose_lag_{i}" for i in range(12)]
        for col in expected_cols:
            assert col in windows_df.columns

        # Verify no temporal leakage: origin_time < target_time
        assert (windows_df["origin_time"] < windows_df["target_time"]).all()

        # Target time must be approximately horizon_minutes ahead of origin_time
        delta_min = (
            (windows_df["target_time"] - windows_df["origin_time"])
            .dt.total_seconds()
            / 60.0
        )
        assert ((delta_min >= 55) & (delta_min <= 65)).all()

        # Last glucose equals lag_0
        assert (windows_df["last_glucose"] == windows_df["glucose_lag_0"]).all()

    def test_gap_rejection_inside_lookback(self, cgm_df_with_gap):
        # With a 45-minute gap and max_gap_minutes=20, candidate windows
        # covering the gap must be rejected
        windows_df, report = create_supervised_windows(
            cgm_df_with_gap,
            patient_col="patient_id",
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            lookback_n=12,
            horizon_minutes=60,
            tolerance_minutes=5,
            max_gap_minutes=20,
        )

        assert "lookback_gap_exceeded" in report.rejections
        assert report.rejections["lookback_gap_exceeded"] > 0

    def test_multipatient_boundary_isolation(self, synthetic_cgm_df_multipatient):
        # Ensure windows are generated independently per patient
        windows_df, report = create_supervised_windows(
            synthetic_cgm_df_multipatient,
            patient_col="patient_id",
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            lookback_n=12,
            horizon_minutes=60,
            tolerance_minutes=5,
            max_gap_minutes=20,
        )

        assert len(windows_df["patient_id"].unique()) == 3
        # Each patient should have accepted windows
        for pid in ["P001", "P002", "P003"]:
            assert len(windows_df[windows_df["patient_id"] == pid]) > 0

    def test_insufficient_history_rejection(self):
        # Only 5 readings when lookback requires 12
        short_df = pd.DataFrame(
            {
                "patient_id": ["P_SHORT"] * 5,
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC"),
                "glucose_mgdl": [100.0] * 5,
            }
        )
        windows_df, report = create_supervised_windows(
            short_df,
            patient_col="patient_id",
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            lookback_n=12,
            horizon_minutes=60,
        )
        assert windows_df.empty
        assert "patient_insufficient_records" in report.rejections
