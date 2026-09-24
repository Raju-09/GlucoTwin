"""Evaluation metrics and reporting for GlucoTwin.

Calculates:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Error distribution percentiles (p50, p90, p95, max)
- Per-patient error breakdown
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EvaluationReport:
    """Structured evaluation report."""

    model_name: str
    split_name: str
    n_examples: int
    n_patients: int
    mae: float
    rmse: float
    mape_pct: float
    median_abs_error: float
    p90_abs_error: float
    max_abs_error: float
    per_patient: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error (expressed as %)."""
    # Guard against division by zero
    denom = np.maximum(y_true, 1e-5)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    split_name: str = "test",
    patient_ids: Sequence[str] | np.ndarray | None = None,
) -> EvaluationReport:
    """Compute comprehensive evaluation metrics globally and per-patient."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions array.")

    abs_errors = np.abs(y_true - y_pred)
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)

    per_patient_stats: list[dict[str, Any]] = []
    if patient_ids is not None:
        p_series = pd.Series(patient_ids)
        for pid in p_series.unique():
            idx = np.where(p_series == pid)[0]
            p_true = y_true[idx]
            p_pred = y_pred[idx]
            p_abs = abs_errors[idx]
            per_patient_stats.append(
                {
                    "patient_id": str(pid),
                    "n_examples": len(idx),
                    "mae": round(compute_mae(p_true, p_pred), 2),
                    "rmse": round(compute_rmse(p_true, p_pred), 2),
                    "median_abs_error": round(float(np.median(p_abs)), 2),
                    "p90_abs_error": round(float(np.percentile(p_abs, 90)), 2),
                }
            )

    report = EvaluationReport(
        model_name=model_name,
        split_name=split_name,
        n_examples=len(y_true),
        n_patients=len(per_patient_stats) if per_patient_stats else 1,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        mape_pct=round(mape, 2),
        median_abs_error=round(float(np.median(abs_errors)), 2),
        p90_abs_error=round(float(np.percentile(abs_errors, 90)), 2),
        max_abs_error=round(float(np.max(abs_errors)), 2),
        per_patient=per_patient_stats,
    )
    return report
