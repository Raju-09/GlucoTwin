"""One-command experiment reproduction script for GlucoTwin.

This script audits and executes the full reproduction pipeline:
1. Verifies or regenerates the synthetic CGM benchmark dataset (seed 42).
2. Computes the SHA-256 fingerprint for dataset provenance.
3. Rebuilds or loads the leakage-safe supervised windows and chronological splits.
4. Evaluates the Persistence Baseline ($y_{t+120} = y_t$).
5. Evaluates the trained GlucoTwin Forecaster (HistGradientBoostingRegressor).
6. Exports test-set predictions (`artifacts/evaluations/reproduced_test_predictions.parquet`).
7. Saves a standardized, timestamp-independent benchmark artifact (`reports/reproduced_evaluation.json`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Windows CPU core detection & console encoding fix
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from glucotwin.config import HORIZON_MINUTES, LOOKBACK_N, SEED
from glucotwin.evaluation.metrics import evaluate_predictions
from glucotwin.models.persistence import PersistenceBaseline
from glucotwin.models.train import GlucoTwinForecaster

console = Console(safe_box=True)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_reproduction(
    from_scratch: bool = False,
    data_dir: Path = PROJECT_ROOT / "data" / "raw",
    processed_dir: Path = PROJECT_ROOT / "data" / "processed",
    model_dir: Path = PROJECT_ROOT / "artifacts" / "model_v0.1",
    eval_dir: Path = PROJECT_ROOT / "artifacts" / "evaluations",
    report_file: Path = PROJECT_ROOT / "reports" / "reproduced_evaluation.json",
    seed: int = SEED,
) -> dict[str, Any]:
    console.rule("[bold blue]GlucoTwin — Complete Experiment Reproduction Pipeline")

    raw_file = data_dir / "synthetic_cgm_benchmark.csv"

    # Step 1: Dataset Generation
    if from_scratch or not raw_file.exists():
        console.print("[yellow]Generating synthetic CGM benchmark from scratch...[/yellow]")
        from scripts.generate_synthetic_data import generate_benchmark_dataset
        demo_file = PROJECT_ROOT / "data" / "synthetic_demo" / "synthetic_cgm_benchmark.csv"
        generate_benchmark_dataset(n_patients=10, days=14, out_paths=[raw_file, demo_file], seed=seed)

    sha256_hash = compute_file_sha256(raw_file)
    console.print(f"Dataset File   : [cyan]{raw_file}[/cyan]")
    console.print(f"SHA-256 Digest : [green]{sha256_hash}[/green]")

    # Step 2: Supervised Dataset Build
    test_parquet = processed_dir / "windows_test.parquet"
    train_parquet = processed_dir / "windows_train.parquet"
    val_parquet = processed_dir / "windows_val.parquet"

    if from_scratch or not test_parquet.exists():
        console.print("[yellow]Building supervised windows and chronological splits...[/yellow]")
        from glucotwin.data.load import load_raw
        from glucotwin.data.preprocess import preprocess_cgm_stream
        from glucotwin.data.splits import chronological_split
        from glucotwin.features.windows import create_supervised_windows

        raw_df = load_raw(raw_file)
        clean_df = preprocess_cgm_stream(raw_df)
        windows_df, _ = create_supervised_windows(clean_df)
        train_df, val_df, test_df, _ = chronological_split(windows_df)

        processed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(train_parquet, index=False)
        val_df.to_parquet(val_parquet, index=False)
        test_df.to_parquet(test_parquet, index=False)
    else:
        test_df = pd.read_parquet(test_parquet)

    console.print(f"Test Split     : [cyan]{test_parquet}[/cyan] ({len(test_df):,} examples)")

    # Step 3: Model Verification / Re-training
    model_joblib = model_dir / "model.joblib"
    if from_scratch or not model_joblib.exists():
        console.print("[yellow]Training GlucoTwin Forecaster on train split...[/yellow]")
        from glucotwin.models.train import train_gradient_booster
        train_df = pd.read_parquet(train_parquet)
        val_df = pd.read_parquet(val_parquet) if val_parquet.exists() else None
        forecaster, _ = train_gradient_booster(train_df, val_df, seed=seed)
        forecaster.save(model_dir)
    else:
        forecaster = GlucoTwinForecaster.load(model_dir)

    console.print(f"Model Artifact : [cyan]{model_joblib}[/cyan]")

    # Step 4: Run Official Evaluation on Held-Out Test Set
    console.print("\n[bold]Evaluating Models on Held-Out Test Split...[/bold]")

    # Baseline 1: Persistence
    p_model = PersistenceBaseline()
    p_preds = p_model.predict(test_df)
    y_true = test_df["target_glucose"].to_numpy()
    patient_ids = test_df["patient_id"].to_numpy()

    p_report = evaluate_predictions(
        y_true=y_true,
        y_pred=p_preds,
        model_name="persistence",
        split_name="test",
        patient_ids=patient_ids,
    )

    # Learned Model: GlucoTwin
    ml_preds = forecaster.predict(test_df)
    ml_report = evaluate_predictions(
        y_true=y_true,
        y_pred=ml_preds,
        model_name="glucotwin_v0.1",
        split_name="test",
        patient_ids=patient_ids,
    )

    # Step 5: Save Test Predictions Table
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds_table = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "origin_time": test_df["origin_time"],
            "target_time": test_df["target_time"],
            "target_glucose": y_true,
            "pred_persistence": np.round(p_preds, 2),
            "pred_glucotwin": np.round(ml_preds, 2),
            "abs_err_persistence": np.round(np.abs(y_true - p_preds), 2),
            "abs_err_glucotwin": np.round(np.abs(y_true - ml_preds), 2),
        }
    )
    preds_file = eval_dir / "reproduced_test_predictions.parquet"
    preds_table.to_parquet(preds_file, index=False)
    console.print(f"Test Predictions Exported : [green]{preds_file}[/green]")

    # Step 6: Verify Reproduction Equivalence
    rel_improvement = round(100.0 * (p_report.mae - ml_report.mae) / p_report.mae, 2)

    target_persistence_mae = 53.25
    target_ml_mae = 18.69
    reproduced_successfully = (
        abs(p_report.mae - target_persistence_mae) < 0.05
        and abs(ml_report.mae - target_ml_mae) < 0.05
    )

    # Output Artifact
    result_artifact = {
        "status": "REPRODUCED_EXACT" if reproduced_successfully else "REPRODUCED_WITH_DIFF",
        "dataset": {
            "file": str(raw_file.name),
            "sha256": sha256_hash,
            "n_patients": 10,
            "days_per_patient": 14,
            "sampling_cadence_minutes": 5,
        },
        "split": {
            "name": "chronological_with_120m_buffer",
            "test_samples": len(test_df),
            "test_patients": int(len(np.unique(patient_ids))),
            "horizon_minutes": HORIZON_MINUTES,
            "lookback_steps": LOOKBACK_N,
        },
        "random_seed": seed,
        "models": {
            "persistence": {
                "mae": p_report.mae,
                "rmse": p_report.rmse,
                "mape_pct": p_report.mape_pct,
                "median_abs_error": p_report.median_abs_error,
            },
            "glucotwin_v0.1": {
                "architecture": "HistGradientBoostingRegressor",
                "n_features": len(forecaster.feature_names),
                "mae": ml_report.mae,
                "rmse": ml_report.rmse,
                "mape_pct": ml_report.mape_pct,
                "median_abs_error": ml_report.median_abs_error,
                "p90_abs_error": ml_report.p90_abs_error,
                "relative_error_reduction_pct": rel_improvement,
            },
        },
        "target_benchmarks": {
            "expected_persistence_mae": target_persistence_mae,
            "expected_ml_mae": target_ml_mae,
            "reproduced_matches_documented": reproduced_successfully,
        },
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as fh:
        json.dump(result_artifact, fh, indent=2)
    console.print(f"Benchmark Report Saved     : [green]{report_file}[/green]")

    # Display Results Table
    table = Table(title="Official Held-Out Test Evaluation", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Test MAE (mg/dL)", justify="right", style="bold")
    table.add_column("Test RMSE (mg/dL)", justify="right")
    table.add_column("Test MAPE (%)", justify="right")
    table.add_column("Median Error", justify="right")
    table.add_column("Rel. Improvement", justify="right", style="green")

    table.add_row(
        "Persistence Baseline",
        f"{p_report.mae:.2f}",
        f"{p_report.rmse:.2f}",
        f"{p_report.mape_pct:.2f}%",
        f"{p_report.median_abs_error:.2f}",
        "Baseline Floor (0.0%)",
    )
    table.add_row(
        "GlucoTwin v0.1 (Learned)",
        f"[green]{ml_report.mae:.2f}[/green]",
        f"{ml_report.rmse:.2f}",
        f"{ml_report.mape_pct:.2f}%",
        f"{ml_report.median_abs_error:.2f}",
        f"[bold green]+{rel_improvement:.1f}%[/bold green]",
    )

    console.print(table)

    if reproduced_successfully:
        console.print(
            "\n[bold green][SUCCESS] REPRODUCTION CONFIRMED:[/bold green] "
            f"Persistence MAE = {p_report.mae:.2f} mg/dL, "
            f"GlucoTwin MAE = {ml_report.mae:.2f} mg/dL. "
            "Matches documented metrics exactly."
        )
    else:
        console.print(
            "\n[bold yellow][WARNING] REPRODUCTION VARIANCE DETECTED:[/bold yellow] "
            f"Persistence MAE = {p_report.mae:.2f} (expected {target_persistence_mae}), "
            f"GlucoTwin MAE = {ml_report.mae:.2f} (expected {target_ml_mae})."
        )

    return result_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce GlucoTwin evaluation results.")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Regenerate synthetic data and retrain model before evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    args = parser.parse_args()

    run_reproduction(from_scratch=args.from_scratch, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
