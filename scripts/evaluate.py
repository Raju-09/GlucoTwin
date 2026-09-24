"""Model evaluation script for GlucoTwin.

Usage
-----
    python scripts/evaluate.py --help
    python scripts/evaluate.py --model persistence --split test
    python scripts/evaluate.py --model trend --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from glucotwin.evaluation.metrics import evaluate_predictions
from glucotwin.models.persistence import LinearTrendBaseline, PersistenceBaseline

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate forecasting models against held-out splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["persistence", "trend", "ml", "gbt"],
        default="persistence",
        help="Model to evaluate (default: persistence).",
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/model_v0.1",
        help="Directory with saved trained model artifacts (default: artifacts/model_v0.1).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "train"],
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory with processed Parquet files (default: data/processed).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to save evaluation reports (default: artifacts).",
    )

    args = parser.parse_args()

    split_file = Path(args.data_dir) / f"windows_{args.split}.parquet"
    if not split_file.exists():
        console.print(f"[red]Split file not found: {split_file}[/red]")
        console.print("Run `python scripts/build_dataset.py` first.")
        return 1

    console.rule(f"[bold blue]GlucoTwin — Evaluation: {args.model.upper()} on {args.split.upper()} split")
    console.print(f"Loading split: [cyan]{split_file}[/cyan]")
    df = pd.read_parquet(split_file)
    console.print(f"Total examples: {len(df):,}")

    # Instantiate model
    if args.model == "persistence":
        model = PersistenceBaseline()
    elif args.model == "trend":
        model = LinearTrendBaseline(k_readings=6)
    elif args.model in ("ml", "gbt"):
        from glucotwin.models.train import GlucoTwinForecaster
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            console.print(f"[red]Model directory not found: {model_dir}[/red]")
            console.print("Run `python scripts/train_model.py` first.")
            return 1
        console.print(f"Loading trained model from: [cyan]{model_dir}[/cyan]")
        model = GlucoTwinForecaster.load(model_dir)
    else:
        console.print(f"[red]Unknown model: {args.model}[/red]")
        return 1

    # Run inference
    with console.status(f"[bold green]Running {args.model} forecasting..."):
        preds = model.predict(df)

    y_true = df["target_glucose"].to_numpy()
    patient_ids = df["patient_id"].to_numpy()

    # Compute metrics
    report = evaluate_predictions(
        y_true=y_true,
        y_pred=preds,
        model_name=args.model,
        split_name=args.split,
        patient_ids=patient_ids,
    )

    # Save artifact
    out_dir = Path(args.artifacts_dir) / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{args.model}_{args.split}.json"
    report.save_json(out_file)

    # Print summary table
    table = Table(title=f"Evaluation Results: {args.model} ({args.split})", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold green")

    table.add_row("Evaluated Examples", f"{report.n_examples:,}")
    table.add_row("Patients", str(report.n_patients))
    table.add_row("MAE (mg/dL)", f"{report.mae:.2f}")
    table.add_row("RMSE (mg/dL)", f"{report.rmse:.2f}")
    table.add_row("MAPE (%)", f"{report.mape_pct:.2f}%")
    table.add_row("Median Abs Error", f"{report.median_abs_error:.2f} mg/dL")
    table.add_row("90th Percentile Abs Error", f"{report.p90_abs_error:.2f} mg/dL")
    table.add_row("Max Abs Error", f"{report.max_abs_error:.2f} mg/dL")

    console.print(table)

    # Print per-patient breakdown
    p_table = Table(title="Per-Patient Breakdown", show_lines=True)
    p_table.add_column("Patient ID", style="cyan")
    p_table.add_column("N", justify="right")
    p_table.add_column("MAE (mg/dL)", justify="right")
    p_table.add_column("RMSE (mg/dL)", justify="right")
    p_table.add_column("Median Err", justify="right")
    p_table.add_column("90th %ile Err", justify="right")

    for p in report.per_patient:
        p_table.add_row(
            p["patient_id"],
            f"{p['n_examples']:,}",
            f"{p['mae']:.2f}",
            f"{p['rmse']:.2f}",
            f"{p['median_abs_error']:.2f}",
            f"{p['p90_abs_error']:.2f}",
        )
    console.print(p_table)

    console.print(f"\n[green]Structured report saved to:[/green] {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
