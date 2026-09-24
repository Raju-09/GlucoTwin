"""Unit tests for feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from glucotwin.features.build_features import extract_features_from_windows
from glucotwin.features.windows import create_supervised_windows


class TestFeatureExtraction:
    def test_feature_dimensions_and_circadian(self, synthetic_cgm_df):
        windows_df, _ = create_supervised_windows(
            synthetic_cgm_df,
            patient_col="patient_id",
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            lookback_n=12,
            horizon_minutes=60,
            tolerance_minutes=5,
        )
        assert not windows_df.empty

        X, y, feature_names = extract_features_from_windows(
            windows_df, lookback_n=12, include_circadian=True
        )

        assert len(X) == len(windows_df)
        assert len(y) == len(windows_df)
        assert len(feature_names) == len(X.columns)

        # Check key engineered feature columns
        assert "mean_last_30m" in X.columns
        assert "std_last_30m" in X.columns
        assert "delta_5m" in X.columns
        assert "sin_hour" in X.columns
        assert "cos_hour" in X.columns

        # Verify no NaNs in extracted features
        assert not X.isna().any().any()

    def test_missing_lag_col_raises(self):
        df = pd.DataFrame({"glucose_lag_0": [100.0]})  # missing lag_1 .. lag_n
        with pytest.raises(ValueError, match="Required lag column"):
            extract_features_from_windows(df, lookback_n=5)
