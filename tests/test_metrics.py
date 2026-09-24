"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from glucotwin.evaluation.metrics import (
    compute_mae,
    compute_mape,
    compute_rmse,
    evaluate_predictions,
)


def test_compute_mae():
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([110.0, 140.0, 200.0])
    # abs diffs: 10, 10, 0 -> mean = 6.6666...
    assert compute_mae(y_true, y_pred) == pytest.approx(20.0 / 3.0)


def test_compute_rmse():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([103.0, 204.0])
    # squared diffs: 9, 16 -> mean = 12.5 -> sqrt = 3.5355...
    assert compute_rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_evaluate_predictions_structure():
    y_true = np.array([100.0, 120.0, 140.0, 160.0])
    y_pred = np.array([102.0, 118.0, 145.0, 155.0])
    pids = np.array(["P1", "P1", "P2", "P2"])

    report = evaluate_predictions(y_true, y_pred, patient_ids=pids)
    assert report.n_examples == 4
    assert report.n_patients == 2
    assert report.mae > 0
    assert report.rmse > 0
    assert len(report.per_patient) == 2


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_predictions(np.array([1, 2]), np.array([1]))
