"""Unit tests for persistence and trend baseline models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glucotwin.models.persistence import LinearTrendBaseline, PersistenceBaseline


class TestPersistenceBaseline:
    def test_persistence_predicts_last_reading(self):
        df = pd.DataFrame(
            {
                "last_glucose": [120.0, 150.5, 90.0],
                "glucose_lag_0": [120.0, 150.5, 90.0],
            }
        )
        model = PersistenceBaseline()
        preds = model.predict(df)
        np.testing.assert_allclose(preds, [120.0, 150.5, 90.0])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        model = PersistenceBaseline()
        with pytest.raises(ValueError, match="Expected"):
            model.predict(df)


class TestLinearTrendBaseline:
    def test_constant_glucose_predicts_flat(self):
        # All lags are 100 -> slope is 0 -> pred is 100
        df = pd.DataFrame(
            {f"glucose_lag_{i}": [100.0] for i in range(6)}
        )
        model = LinearTrendBaseline(k_readings=6, horizon_minutes=120)
        preds = model.predict(df)
        np.testing.assert_allclose(preds, [100.0])

    def test_positive_slope_extrapolation(self):
        # Rising by 1 mg/dL every 5 minutes (0.2 mg/dL/min)
        # lag_0 (t) = 110, lag_1 (t-5) = 109, lag_2 (t-10) = 108, etc.
        df = pd.DataFrame(
            {f"glucose_lag_{i}": [110.0 - i * 1.0] for i in range(6)}
        )
        model = LinearTrendBaseline(
            k_readings=6, cadence_minutes=5.0, horizon_minutes=60
        )
        # In 60 min at +0.2 mg/dL/min: rise = +12 mg/dL -> 110 + 12 = 122
        preds = model.predict(df)
        np.testing.assert_allclose(preds, [122.0], atol=1e-5)
