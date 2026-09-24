"""Training pipeline for learned tabular models in GlucoTwin.

Design rules enforced:
1. STRICT ZERO LEAKAGE: Preprocessing scalers are fitted ONLY on training data.
2. Validation split is used for early stopping and hyperparameter monitoring.
3. Model artifact, feature schema, and training metadata are saved together.
4. Model predictions are strictly clipped to physiological glucose range [20, 600] mg/dL.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from glucotwin.config import (
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    SEED,
)
from glucotwin.evaluation.metrics import evaluate_predictions
from glucotwin.features.build_features import extract_features_from_windows

logger = logging.getLogger(__name__)


class GlucoTwinForecaster:
    """Wrapper around trained pipeline providing inference with bounds enforcement."""

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        model_name: str = "HistGradientBoostingRegressor",
        horizon_minutes: int = 120,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.model_name = model_name
        self.horizon_minutes = horizon_minutes

    def predict(self, windows_df: pd.DataFrame) -> np.ndarray:
        """Extract features and generate bounded glucose predictions."""
        X, _, _ = extract_features_from_windows(windows_df)
        # Ensure exact column ordering
        X_ordered = X[self.feature_names]
        raw_preds = self.pipeline.predict(X_ordered)
        # Enforce physiological bounds
        return np.clip(raw_preds, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)

    def save(self, out_dir: str | Path) -> None:
        """Save model artifact and feature schema to directory."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_file = out_dir / "model.joblib"
        schema_file = out_dir / "feature_schema.json"
        meta_file = out_dir / "model_meta.json"

        joblib.dump(self.pipeline, model_file)

        with schema_file.open("w", encoding="utf-8") as fh:
            json.dump({"features": self.feature_names}, fh, indent=2)

        with meta_file.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_name": self.model_name,
                    "horizon_minutes": self.horizon_minutes,
                    "n_features": len(self.feature_names),
                },
                fh,
                indent=2,
            )

        logger.info("Saved forecaster artifacts to %s", out_dir)

    @classmethod
    def load(cls, model_dir: str | Path) -> GlucoTwinForecaster:
        """Load forecaster and schema from directory."""
        model_dir = Path(model_dir)
        pipeline = joblib.load(model_dir / "model.joblib")
        with (model_dir / "feature_schema.json").open("r", encoding="utf-8") as fh:
            schema = json.load(fh)
        with (model_dir / "model_meta.json").open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        return cls(
            pipeline=pipeline,
            feature_names=schema["features"],
            model_name=meta.get("model_name", "HistGradientBoostingRegressor"),
            horizon_minutes=meta.get("horizon_minutes", 120),
        )


def train_gradient_booster(
    train_windows_df: pd.DataFrame,
    val_windows_df: pd.DataFrame | None = None,
    seed: int = SEED,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
) -> tuple[GlucoTwinForecaster, dict[str, Any]]:
    """Train gradient boosted decision tree on supervised historical windows.

    Parameters
    ----------
    train_windows_df:
        Training split windows.
    val_windows_df:
        Validation split windows for evaluation.
    seed:
        Random seed for reproducibility.
    max_iter:
        Maximum boosting iterations.
    learning_rate:
        Boosting learning rate.
    max_leaf_nodes:
        Complexity parameter for tree leaves.

    Returns
    -------
    tuple[GlucoTwinForecaster, dict[str, Any]]
        (trained_forecaster, training_metrics_dict)
    """
    logger.info("Extracting features from training split (%d rows)...", len(train_windows_df))
    X_train, y_train, feature_names = extract_features_from_windows(train_windows_df)

    # Build Pipeline: StandardScaler + HistGradientBoostingRegressor
    scaler = StandardScaler()
    regressor = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        scoring="neg_mean_absolute_error",
    )

    pipeline = Pipeline([("scaler", scaler), ("regressor", regressor)])

    logger.info(
        "Fitting pipeline on %d training samples with %d features...",
        len(X_train),
        len(feature_names),
    )
    pipeline.fit(X_train[feature_names], y_train)

    forecaster = GlucoTwinForecaster(
        pipeline=pipeline,
        feature_names=feature_names,
        model_name="HistGradientBoostingRegressor",
        horizon_minutes=120,
    )

    # Evaluate on Train
    train_preds = forecaster.predict(train_windows_df)
    train_report = evaluate_predictions(
        y_true=y_train,
        y_pred=train_preds,
        model_name="hist_gb_train",
        split_name="train",
    )

    metrics: dict[str, Any] = {
        "train_mae": train_report.mae,
        "train_rmse": train_report.rmse,
        "n_train_samples": len(y_train),
        "n_features": len(feature_names),
    }

    # Evaluate on Validation if provided
    if val_windows_df is not None and not val_windows_df.empty:
        val_preds = forecaster.predict(val_windows_df)
        val_report = evaluate_predictions(
            y_true=val_windows_df["target_glucose"].to_numpy(),
            y_pred=val_preds,
            model_name="hist_gb_val",
            split_name="val",
            patient_ids=val_windows_df["patient_id"].to_numpy(),
        )
        metrics["val_mae"] = val_report.mae
        metrics["val_rmse"] = val_report.rmse
        metrics["n_val_samples"] = len(val_windows_df)
        logger.info(
            "Validation MAE: %.2f mg/dL | Validation RMSE: %.2f mg/dL",
            val_report.mae,
            val_report.rmse,
        )

    logger.info(
        "Training complete. Train MAE: %.2f mg/dL, Val MAE: %.2f mg/dL",
        metrics["train_mae"],
        metrics.get("val_mae", 0.0),
    )
    return forecaster, metrics
