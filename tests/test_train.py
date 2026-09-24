"""Unit tests for model training and forecaster serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from glucotwin.features.windows import create_supervised_windows
from glucotwin.models.train import GlucoTwinForecaster, train_gradient_booster


class TestModelTraining:
    def test_train_save_load_predict(self, synthetic_cgm_df_multipatient):
        # 1. Create small windows
        windows_df, _ = create_supervised_windows(
            synthetic_cgm_df_multipatient,
            patient_col="patient_id",
            ts_col="timestamp",
            glucose_col="glucose_mgdl",
            lookback_n=10,
            horizon_minutes=40,
            tolerance_minutes=5,
        )
        assert len(windows_df) > 30

        # Split into small train and val
        split_idx = int(0.7 * len(windows_df))
        train_df = windows_df.iloc[:split_idx].reset_index(drop=True)
        val_df = windows_df.iloc[split_idx:].reset_index(drop=True)

        # 2. Train model
        forecaster, metrics = train_gradient_booster(
            train_windows_df=train_df,
            val_windows_df=val_df,
            max_iter=20,
            seed=42,
        )

        assert metrics["train_mae"] > 0
        assert "val_mae" in metrics

        # 3. Predict on val
        preds = forecaster.predict(val_df)
        assert len(preds) == len(val_df)
        assert ((preds >= 20.0) & (preds <= 600.0)).all()

        # 4. Save and reload
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "test_model"
            forecaster.save(model_dir)

            loaded = GlucoTwinForecaster.load(model_dir)
            loaded_preds = loaded.predict(val_df)

            np.testing.assert_allclose(preds, loaded_preds)
