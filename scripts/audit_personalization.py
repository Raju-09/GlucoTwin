"""Personalization feasibility audit script for GlucoTwin.

This script audits the feasibility of patient-specific adaptation by comparing:
1. Global population model error per patient
2. Patient-specific residual bias (mean error on training split)
3. Performance of patient-adapted calibration: y_hat_adapted = y_hat_global - bias_train
4. Minimum sample size requirements and variance analysis.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from glucotwin.evaluation.metrics import compute_mae, compute_rmse
from glucotwin.models.train import GlucoTwinForecaster

console = Console()


def audit_personalization_feasibility(
    train_file: str | Path = "data/processed/windows_train.parquet",
    test_file: str | Path = "data/processed/windows_test.parquet",
    model_dir: str | Path = "artifacts/model_v0.1",
    out_file: str | Path = "reports/personalization_audit.json",
) -> dict[str, Any]:
    train_path = Path(train_file)
    test_path = Path(test_file)
    model_path = Path(model_dir)

    console.rule("[bold blue]GlucoTwin — Personalization Feasibility Audit")
    console.print(f"Train split : [cyan]{train_path}[/cyan]")
    console.print(f"Test split  : [cyan]{test_path}[/cyan]")

    forecaster = GlucoTwinForecaster.load(model_path)
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Global predictions on train and test
    train_preds = forecaster.predict(train_df)
    test_preds = forecaster.predict(test_df)

    train_df["pred_global"] = train_preds
    train_df["residual"] = train_df["target_glucose"] - train_df["pred_global"]

    test_df["pred_global"] = test_preds

    patient_reports: list[dict[str, Any]] = []

    # Test patient-specific calibration:
    # Estimate patient bias from past train data ONLY
    for pid in sorted(test_df["patient_id"].unique()):
        p_train = train_df[train_df["patient_id"] == pid]
        p_test = test_df[test_df["patient_id"] == pid]

        if len(p_train) == 0:
            continue

        # Patient bias estimated strictly from training data
        patient_bias = float(p_train["residual"].mean())
        train_samples = len(p_train)
        test_samples = len(p_test)

        # Baseline: Global model on test
        y_test = p_test["target_glucose"].to_numpy()
        test_pred_global = p_test["pred_global"].to_numpy()
        global_mae = compute_mae(y_test, test_pred_global)
        global_rmse = compute_rmse(y_test, test_pred_global)

        # Adapted: Global model + patient-specific bias correction
        # y_hat_adapted = clip(y_hat_global + bias, 20, 600)
        test_pred_adapted = np.clip(test_pred_global + patient_bias, 20.0, 600.0)
        adapted_mae = compute_mae(y_test, test_pred_adapted)
        adapted_rmse = compute_rmse(y_test, test_pred_adapted)

        mae_improvement = global_mae - adapted_mae
        pct_improvement = (
            round(100.0 * mae_improvement / global_mae, 2)
            if global_mae > 0
            else 0.0
        )

        patient_reports.append(
            {
                "patient_id": str(pid),
                "train_samples": train_samples,
                "test_samples": test_samples,
                "patient_train_bias": round(patient_bias, 2),
                "global_mae": round(global_mae, 2),
                "adapted_mae": round(adapted_mae, 2),
                "mae_change": round(mae_improvement, 2),
                "pct_improvement": pct_improvement,
                "helps": bool(mae_improvement > 0),
            }
        )

    # Overall adapted metrics
    all_adapted_preds = []
    all_test_y = []
    for pid in test_df["patient_id"].unique():
        p_train = train_df[train_df["patient_id"] == pid]
        p_test = test_df[test_df["patient_id"] == pid]
        bias = float(p_train["residual"].mean()) if len(p_train) else 0.0
        adapted = np.clip(p_test["pred_global"].to_numpy() + bias, 20.0, 600.0)
        all_adapted_preds.extend(adapted.tolist())
        all_test_y.extend(p_test["target_glucose"].to_numpy().tolist())

    overall_global_mae = compute_mae(test_df["target_glucose"].to_numpy(), test_preds)
    overall_adapted_mae = compute_mae(np.array(all_test_y), np.array(all_adapted_preds))

    # Go/No-go recommendation
    helps_count = sum(1 for p in patient_reports if p["helps"])
    go_decision = bool(overall_adapted_mae < overall_global_mae and helps_count >= 5)

    summary = {
        "overall_global_mae": round(overall_global_mae, 2),
        "overall_adapted_mae": round(overall_adapted_mae, 2),
        "overall_mae_reduction": round(overall_global_mae - overall_adapted_mae, 2),
        "patients_helped": f"{helps_count}/{len(patient_reports)}",
        "recommendation": "GO" if go_decision else "NO-GO",
        "rationale": (
            f"Patient-specific calibration reduces overall test MAE from "
            f"{overall_global_mae:.2f} to {overall_adapted_mae:.2f} mg/dL and improves "
            f"forecasting on {helps_count}/{len(patient_reports)} patients."
        )
        if go_decision
        else "Personalization did not consistently improve upon the global model.",
        "per_patient": patient_reports,
    }

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Print summary table
    table = Table(title="Personalization Feasibility: Global vs Adapted", show_lines=True)
    table.add_column("Patient", style="cyan")
    table.add_column("Train N", justify="right")
    table.add_column("Train Bias", justify="right")
    table.add_column("Global MAE", justify="right")
    table.add_column("Adapted MAE", justify="right")
    table.add_column("Improvement", justify="right", style="bold")

    for p in patient_reports:
        color = "green" if p["helps"] else "red"
        sign = "+" if p["mae_change"] > 0 else ""
        table.add_row(
            p["patient_id"],
            f"{p['train_samples']:,}",
            f"{p['patient_train_bias']:+.2f}",
            f"{p['global_mae']:.2f}",
            f"{p['adapted_mae']:.2f}",
            f"[{color}]{sign}{p['mae_change']:.2f} ({sign}{p['pct_improvement']}%) [/{color}]",
        )

    console.print(table)
    console.print(f"\nOverall Global MAE  : [bold]{overall_global_mae:.2f} mg/dL[/bold]")
    console.print(f"Overall Adapted MAE : [bold green]{overall_adapted_mae:.2f} mg/dL[/bold green]")
    console.print(f"Recommendation      : [bold cyan]{summary['recommendation']}[/bold cyan] ({summary['rationale']})")
    console.print(f"Report saved to     : {out_path}")
    return summary


def main() -> None:
    audit_personalization_feasibility()


if __name__ == "__main__":
    main()
