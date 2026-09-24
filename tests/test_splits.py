"""Unit tests for chronological splitting and temporal buffer protection.

Verifies:
1. Strict chronological ordering across splits.
2. Safety buffers between train/val and val/test prevent boundary bleeding.
3. No overlap in prediction origin or target times across splits.
"""

from __future__ import annotations

import pandas as pd
import pytest

from glucotwin.data.splits import chronological_split
from glucotwin.features.windows import create_supervised_windows


class TestChronologicalSplit:
    def test_chronological_ordering_and_buffer(self, synthetic_cgm_df_multipatient):
        windows_df, _ = create_supervised_windows(
            synthetic_cgm_df_multipatient,
            patient_col="patient_id",
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            lookback_n=10,
            horizon_minutes=40,
            tolerance_minutes=5,
            max_gap_minutes=20,
        )
        assert not windows_df.empty

        buffer_min = 30
        train_df, val_df, test_df, meta = chronological_split(
            windows_df,
            patient_col="patient_id",
            origin_col="origin_time",
            target_col="target_time",
            train_frac=0.6,
            val_frac=0.2,
            buffer_minutes=buffer_min,
        )

        assert not train_df.empty
        assert not val_df.empty
        assert not test_df.empty

        # Check per-patient chronological integrity
        for pid in train_df["patient_id"].unique():
            p_train = train_df[train_df["patient_id"] == pid]
            p_val = val_df[val_df["patient_id"] == pid]
            p_test = test_df[test_df["patient_id"] == pid]

            # Train target time must be <= Val origin time - buffer
            if not p_train.empty and not p_val.empty:
                train_max_target = p_train["target_time"].max()
                val_min_origin = p_val["origin_time"].min()
                assert val_min_origin - train_max_target >= pd.Timedelta(minutes=buffer_min)

            # Val target time must be <= Test origin time - buffer
            if not p_val.empty and not p_test.empty:
                val_max_target = p_val["target_time"].max()
                test_min_origin = p_test["origin_time"].min()
                assert test_min_origin - val_max_target >= pd.Timedelta(minutes=buffer_min)

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="Cannot split empty"):
            chronological_split(pd.DataFrame())
