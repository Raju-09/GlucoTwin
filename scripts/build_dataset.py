"""Build supervised dataset script for GlucoTwin.

Usage
-----
    python scripts/build_dataset.py --help
    python scripts/build_dataset.py --data-dir data/raw --out-dir data/processed

Steps:
1. Loads raw dataset files.
2. Preprocesses stream (timestamp parsing, unit conversion, deduplication).
3. Constructs leakage-safe 120-minute supervised forecasting windows.
4. Generates chronological train/val/test splits with safety buffers.
5. Saves Parquet splits and JSON audit reports to `out-dir`.
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

from glucotwin.config import (
    HORIZON_MINUTES,
    LOOKBACK_N,
    MAX_GAP_MINUTES,
    TARGET_TOLERANCE_MINUTES,
)
from glucotwin.data.load import discover_files, load_raw
from glucotwin.data.preprocess import preprocess_cgm_stream
from glucotwin.data.splits import chronological_split
from glucotwin.features.windows import create_supervised_windows

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
        description="Build supervised windows and splits for GlucoTwin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing raw data files (default: data/raw).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Directory to save processed Parquet files (default: data/processed).",
    )
    parser.add_argument(
        "--patient-col",
        default="patient_id",
        help="Patient identifier column (default: patient_id).",
    )
    parser.add_argument(
        "--ts-col",
        default="timestamp",
        help="Timestamp column (default: timestamp).",
    )
    parser.add_argument(
        "--glucose-col",
        default="glucose",
        help="Glucose value column (default: glucose).",
    )
    parser.add_argument(
        "--lookback-n",
        type=int,
        default=LOOKBACK_N,
        help=f"Number of historical readings in lookback window (default: {LOOKBACK_N}).",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=HORIZON_MINUTES,
        help=f"Forecast horizon in minutes (default: {HORIZON_MINUTES}).",
    )
    parser.add_argument(
        "--tolerance-minutes",
        type=int,
        default=TARGET_TOLERANCE_MINUTES,
        help=f"Tolerance around target in minutes (default: {TARGET_TOLERANCE_MINUTES}).",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=MAX_GAP_MINUTES,
        help=f"Max gap inside lookback window before rejection (default: {MAX_GAP_MINUTES}).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.65,
        help="Fraction of timeline for training (default: 0.65).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of timeline for validation (default: 0.15).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]GlucoTwin — Build Supervised Dataset")
    console.print(f"Data directory : {data_dir}")
    console.print(f"Output directory: {out_dir}")
    console.print(f"Lookback window: {args.lookback_n} steps")
    console.print(f"Horizon        : {args.horizon_minutes} minutes (±{args.tolerance_minutes} min)")

    files = discover_files(data_dir)
    if not files:
        console.print("[red]No data files found in data directory.[/red]")
        return 1

    # Load and combine all discovered files
    raw_dfs: list[pd.DataFrame] = []
    for f in files:
        console.print(f"Loading: [cyan]{f.name}[/cyan]")
        raw_dfs.append(load_raw(f))

    raw_combined = pd.concat(raw_dfs, ignore_index=True)
    console.print(f"Total raw rows: {len(raw_combined):,}")

    # Step 1: Preprocess
    with console.status("[bold green]Preprocessing raw stream..."):
        clean_df = preprocess_cgm_stream(
            raw_combined,
            patient_col=args.patient_col,
            ts_col=args.ts_col,
            glucose_col=args.glucose_col,
        )
    console.print(f"Cleaned rows: {len(clean_df):,}")

    # Step 2: Window Construction
    with console.status("[bold green]Constructing supervised forecasting windows..."):
        windows_df, window_report = create_supervised_windows(
            clean_df,
            patient_col=args.patient_col,
            ts_col=args.ts_col,
            glucose_col=args.glucose_col,
            lookback_n=args.lookback_n,
            horizon_minutes=args.horizon_minutes,
            tolerance_minutes=args.tolerance_minutes,
            max_gap_minutes=args.max_gap_minutes,
        )

    console.print(
        f"Windows created: [bold green]{len(windows_df):,}[/bold green] "
        f"from {window_report.total_candidates:,} candidate origins "
        f"({window_report.to_dict()['acceptance_rate_pct']}%)"
    )

    if windows_df.empty:
        console.print("[red]No valid windows could be constructed. Check dataset and cadence.[/red]")
        return 1

    # Step 3: Chronological Split
    with console.status("[bold green]Splitting dataset chronologically with buffer..."):
        train_df, val_df, test_df, split_meta = chronological_split(
            windows_df,
            patient_col=args.patient_col,
            origin_col="origin_time",
            target_col="target_time",
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            buffer_minutes=args.horizon_minutes,
        )

    # Save outputs as Parquet
    train_path = out_dir / "windows_train.parquet"
    val_path = out_dir / "windows_val.parquet"
    test_path = out_dir / "windows_test.parquet"
    all_path = out_dir / "windows_all.parquet"
    meta_path = out_dir / "build_metadata.json"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    windows_df.to_parquet(all_path, index=False)

    metadata = {
        "windowing_report": window_report.to_dict(),
        "split_metadata": split_meta.to_dict(),
        "parameters": {
            "lookback_n": args.lookback_n,
            "horizon_minutes": args.horizon_minutes,
            "tolerance_minutes": args.tolerance_minutes,
            "max_gap_minutes": args.max_gap_minutes,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
        },
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # Summary table
    table = Table(title="Supervised Windows & Splits Summary", show_lines=True)
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("File", style="green")

    table.add_row("Train", f"{len(train_df):,}", f"{split_meta.train_pct:.1f}%", train_path.name)
    table.add_row("Validation", f"{len(val_df):,}", f"{split_meta.val_pct:.1f}%", val_path.name)
    table.add_row("Test", f"{len(test_df):,}", f"{split_meta.test_pct:.1f}%", test_path.name)
    table.add_row("Total Valid", f"{len(train_df)+len(val_df)+len(test_df):,}", "100.0%", all_path.name)

    console.print(table)
    console.print(f"\n[green]All artifacts saved to:[/green] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
