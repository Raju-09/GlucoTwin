"""Compute clinical adverse event prediction metrics on held-out test windows.

Evaluates:
1. Hypoglycemic threshold crossings (target < 70 mg/dL)
2. Hyperglycemic threshold crossings (target > 180 mg/dL)

Metrics computed:
- Sensitivity (Recall / True Positive Rate)
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)
- F1-Score
- False Alarm Rate per simulated patient-day
- Brier Score & Confusion Matrices
- Direct comparison vs Persistence baseline
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Silence joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from glucotwin.config import ARTIFACTS_DIR, DATA_PROCESSED_DIR
from glucotwin.features.multimodal import (
    AblationMode,
    extract_multimodal_features,
    load_ehr_profiles,
    load_wearable_telemetry,
)

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_event_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    is_lower: bool,
    n_patient_days: float,
) -> dict:
    """Compute sensitivity, specificity, precision, F1, and false alarms per day."""
    if is_lower:
        actual_event = (y_true < threshold).astype(int)
        pred_event = (y_pred < threshold).astype(int)
    else:
        actual_event = (y_true > threshold).astype(int)
        pred_event = (y_pred > threshold).astype(int)

    tp = int(np.sum((actual_event == 1) & (pred_event == 1)))
    fp = int(np.sum((actual_event == 0) & (pred_event == 1)))
    fn = int(np.sum((actual_event == 1) & (pred_event == 0)))
    tn = int(np.sum((actual_event == 0) & (pred_event == 0)))

    total_actual = tp + fn
    total_non_event = tn + fp

    sensitivity = tp / total_actual if total_actual > 0 else 0.0
    specificity = tn / total_non_event if total_non_event > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    false_alarms_per_day = fp / n_patient_days if n_patient_days > 0 else 0.0

    return {
        "event_threshold_mgdl": threshold,
        "event_direction": "<" if is_lower else ">",
        "total_events": total_actual,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "sensitivity_recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "false_alarms_per_patient_day": round(false_alarms_per_day, 2),
    }


def main():
    console.rule("[bold blue]GlucoTwin Adverse Event Prediction Audit[/bold blue]")

    test_path = DATA_PROCESSED_DIR / "windows_test.parquet"
    if not test_path.exists():
        console.print(f"[red]Error: {test_path} not found.[/red]")
        return

    test_df = pd.read_parquet(test_path)
    console.print(f"Loaded {len(test_df):,} test windows across {test_df['patient_id'].nunique()} patients.")

    # Calculate total simulated patient-days in test set
    # 43,888 windows at 5-min cadence ≈ 219,440 minutes ≈ 152.4 patient-days
    total_minutes = len(test_df) * 5.0
    n_patient_days = total_minutes / 1440.0

    # Load EHR and Wearables
    ehr_df = load_ehr_profiles()
    wear_df = load_wearable_telemetry()

    # Load trained model
    model_path = ARTIFACTS_DIR / "model_multimodal" / "model.joblib"
    schema_path = ARTIFACTS_DIR / "model_multimodal" / "feature_schema.json"
    model = joblib.load(model_path)
    with open(schema_path, "r") as f:
        schema = json.load(f)["features"]

    console.print("Extracting test features...")
    X_test, _, _ = extract_multimodal_features(
        test_df,
        ehr_df=ehr_df,
        wearable_df=wear_df,
        mode="full_fusion",
    )
    X_test = X_test[schema]

    console.print("Generating 120-min forecasts...")
    y_pred = model.predict(X_test)
    y_true = test_df["target_glucose"].to_numpy()
    y_persistence = test_df["glucose_lag_0"].to_numpy()

    # 1. Hypoglycemia (< 70 mg/dL)
    hypo_model = compute_event_classification_metrics(y_true, y_pred, 70.0, True, n_patient_days)
    hypo_pers = compute_event_classification_metrics(y_true, y_persistence, 70.0, True, n_patient_days)

    # 2. Hyperglycemia (> 180 mg/dL)
    hyper_model = compute_event_classification_metrics(y_true, y_pred, 180.0, False, n_patient_days)
    hyper_pers = compute_event_classification_metrics(y_true, y_persistence, 180.0, False, n_patient_days)

    # Print Rich Table
    table = Table(title="Adverse Event Prediction Evaluation (Held-Out Test Set: 43,888 Windows)")
    table.add_column("Adverse Event", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Events (N)", justify="right")
    table.add_column("Sensitivity", justify="right")
    table.add_column("Specificity", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("False Alarms / Patient-Day", justify="right")

    table.add_row(
        "Hypoglycemia (<70 mg/dL)",
        "Persistence Baseline",
        str(hypo_pers["total_events"]),
        f"{hypo_pers['sensitivity_recall']*100:.1f}%",
        f"{hypo_pers['specificity']*100:.1f}%",
        f"{hypo_pers['precision']*100:.1f}%",
        f"{hypo_pers['f1_score']:.3f}",
        f"{hypo_pers['false_alarms_per_patient_day']:.1f}",
    )
    table.add_row(
        "Hypoglycemia (<70 mg/dL)",
        "GlucoTwin (Two-Stream)",
        str(hypo_model["total_events"]),
        f"[green]{hypo_model['sensitivity_recall']*100:.1f}%[/green]",
        f"[green]{hypo_model['specificity']*100:.1f}%[/green]",
        f"[green]{hypo_model['precision']*100:.1f}%[/green]",
        f"[green]{hypo_model['f1_score']:.3f}[/green]",
        f"[green]{hypo_model['false_alarms_per_patient_day']:.1f}[/green]",
    )

    table.add_row(
        "Hyperglycemia (>180 mg/dL)",
        "Persistence Baseline",
        str(hyper_pers["total_events"]),
        f"{hyper_pers['sensitivity_recall']*100:.1f}%",
        f"{hyper_pers['specificity']*100:.1f}%",
        f"{hyper_pers['precision']*100:.1f}%",
        f"{hyper_pers['f1_score']:.3f}",
        f"{hyper_pers['false_alarms_per_patient_day']:.1f}",
    )
    table.add_row(
        "Hyperglycemia (>180 mg/dL)",
        "GlucoTwin (Two-Stream)",
        str(hyper_model["total_events"]),
        f"[green]{hyper_model['sensitivity_recall']*100:.1f}%[/green]",
        f"[green]{hyper_model['specificity']*100:.1f}%[/green]",
        f"[green]{hyper_model['precision']*100:.1f}%[/green]",
        f"[green]{hyper_model['f1_score']:.3f}[/green]",
        f"[green]{hyper_model['false_alarms_per_patient_day']:.1f}[/green]",
    )

    console.print(table)

    results = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "n_test_windows": len(test_df),
        "n_patients": int(test_df["patient_id"].nunique()),
        "total_simulated_patient_days": round(n_patient_days, 1),
        "hypoglycemia_lt_70": {
            "glucotwin": hypo_model,
            "persistence": hypo_pers,
        },
        "hyperglycemia_gt_180": {
            "glucotwin": hyper_model,
            "persistence": hyper_pers,
        },
    }

    out_file = ARTIFACTS_DIR / "evaluations" / "event_prediction_metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]Saved event metrics to {out_file}[/green]")


if __name__ == "__main__":
    main()
