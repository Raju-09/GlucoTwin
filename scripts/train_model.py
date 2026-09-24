"""Model training CLI script for GlucoTwin.

Usage
-----
    python scripts/train_model.py --help
    python scripts/train_model.py --train-file data/processed/windows_train.parquet \
        --val-file data/processed/windows_val.parquet \
        --out-dir artifacts/model_v0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from glucotwin.config import SEED
from glucotwin.models.train import train_gradient_booster

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train learned gradient boosted tree model on supervised windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-file",
        default="data/processed/windows_train.parquet",
        help="Path to training windows Parquet file (default: data/processed/windows_train.parquet).",
    )
    parser.add_argument(
        "--val-file",
        default="data/processed/windows_val.parquet",
        help="Path to validation windows Parquet file (default: data/processed/windows_val.parquet).",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/model_v0.1",
        help="Directory to save model artifacts (default: artifacts/model_v0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=250,
        help="Maximum boosting iterations (default: 250).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.04,
        help="Learning rate (default: 0.04).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    out_dir = Path(args.out_dir)

    console.rule("[bold blue]GlucoTwin — Train Gradient Boosted Model")
    console.print(f"Training split   : [cyan]{train_path}[/cyan]")
    console.print(f"Validation split : [cyan]{val_path}[/cyan]")
    console.print(f"Output directory : [cyan]{out_dir}[/cyan]")

    if not train_path.exists():
        console.print(f"[red]Training file not found: {train_path}[/red]")
        return 1

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path) if val_path.exists() else None

    console.print(f"Loaded {len(train_df):,} training examples.")
    if val_df is not None:
        console.print(f"Loaded {len(val_df):,} validation examples.")

    with console.status("[bold green]Training gradient boosted forecaster..."):
        forecaster, metrics = train_gradient_booster(
            train_windows_df=train_df,
            val_windows_df=val_df,
            seed=args.seed,
            max_iter=args.max_iter,
            learning_rate=args.lr,
        )

    # Save model artifacts
    forecaster.save(out_dir)

    # Save training config and metrics
    with (out_dir / "training_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    table = Table(title="Model Training Summary", show_lines=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right", style="bold green")

    table.add_row("Model Architecture", "HistGradientBoostingRegressor")
    table.add_row("Trained Samples", f"{metrics['n_train_samples']:,}")
    table.add_row("Extracted Features", str(metrics["n_features"]))
    table.add_row("Train MAE (mg/dL)", f"{metrics['train_mae']:.2f}")
    table.add_row("Train RMSE (mg/dL)", f"{metrics['train_rmse']:.2f}")
    if "val_mae" in metrics:
        table.add_row("Val MAE (mg/dL)", f"{metrics['val_mae']:.2f}")
        table.add_row("Val RMSE (mg/dL)", f"{metrics['val_rmse']:.2f}")

    console.print(table)
    console.print(f"\n[green]Artifacts successfully saved to:[/green] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
