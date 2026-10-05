"""Run controlled Multimodal Two-Stream Ablation Study for GlucoTwin.

This script rigorously evaluates the core hypothesis:
"Does fusing static EHR context and dynamic wearable signals improve 120-minute
glucose forecasting over a CGM-only model on a held-out test cohort?"

Configurations evaluated on identical chronological splits (7,316 test examples):
1. Config A: CGM History Only (Stream 2A) — 41 features
2. Config B: CGM + Static EHR Context (Stream 1 + 2A) — 50 features
3. Config C: CGM + Dynamic Wearables (Stream 2A + 2B) — 47 features
4. Config D: Full Two-Stream Fusion (Stream 1 + 2A + 2B) — 56 features

Outputs:
- reports/multimodal_ablation_results.json
- artifacts/evaluations/multimodal_ablation_results.json
- Statistical significance tests (paired t-test, Wilcoxon signed-rank test)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure reproducible loky core detection on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from glucotwin.config import (
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    PROJECT_ROOT,
    SEED,
)
from glucotwin.evaluation.metrics import evaluate_predictions
from glucotwin.features.multimodal import (
    AblationMode,
    extract_multimodal_features,
    load_ehr_profiles,
    load_wearable_telemetry,
)

console = Console()
logger = logging.getLogger("multimodal_ablation")


def run_ablation_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ehr_df: pd.DataFrame,
    wearable_df: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train and evaluate all four multimodal ablation configurations."""
    configs: list[tuple[str, str, AblationMode]] = [
        ("Config A", "CGM History Only (Stream 2A)", "cgm_only"),
        ("Config B", "CGM + Static EHR (Stream 1 + 2A)", "cgm_plus_ehr"),
        ("Config C", "CGM + Dynamic Wearables (Stream 2A + 2B)", "cgm_plus_wearables"),
        ("Config D", "Full Two-Stream Fusion (1 + 2A + 2B)", "full_fusion"),
    ]

    results: dict[str, Any] = {}
    predictions_by_config: dict[str, np.ndarray] = {}
    y_test_true = test_df["target_glucose"].to_numpy(dtype=float)
    test_patient_ids = test_df["patient_id"].to_numpy()

    console.rule("[bold blue]GlucoTwin — Multimodal Two-Stream Ablation Study")
    console.print(f"Train split : {len(train_df):,} examples")
    console.print(f"Val split   : {len(val_df):,} examples")
    console.print(f"Test split  : {len(test_df):,} examples across {len(np.unique(test_patient_ids))} patients")
    console.print()

    for config_id, config_label, mode in configs:
        console.print(f"Training [cyan]{config_id}: {config_label}[/cyan] (mode='{mode}')...")

        # 1. Feature extraction
        X_train, y_train, feat_names = extract_multimodal_features(
            train_df, ehr_df=ehr_df, wearable_df=wearable_df, mode=mode
        )
        X_val, y_val, _ = extract_multimodal_features(
            val_df, ehr_df=ehr_df, wearable_df=wearable_df, mode=mode
        )
        X_test, y_test, _ = extract_multimodal_features(
            test_df, ehr_df=ehr_df, wearable_df=wearable_df, mode=mode
        )

        # 2. Pipeline setup
        scaler = StandardScaler()
        regressor = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
            scoring="neg_mean_absolute_error",
        )
        pipeline = Pipeline([("scaler", scaler), ("regressor", regressor)])

        # 3. Fit
        pipeline.fit(X_train[feat_names], y_train)

        # 4. Predict on Test
        raw_preds = pipeline.predict(X_test[feat_names])
        bounded_preds = np.clip(raw_preds, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)
        predictions_by_config[mode] = bounded_preds

        # 5. Evaluate
        report = evaluate_predictions(
            y_true=y_test_true,
            y_pred=bounded_preds,
            model_name=config_id,
            split_name="test",
            patient_ids=test_patient_ids,
        )

        results[mode] = {
            "config_id": config_id,
            "label": config_label,
            "mode": mode,
            "n_features": len(feat_names),
            "features": feat_names,
            "metrics": {
                "mae": report.mae,
                "rmse": report.rmse,
                "mape_pct": report.mape_pct,
                "median_abs_error": report.median_abs_error,
                "p90_abs_error": report.p90_abs_error,
            },
            "per_patient": report.per_patient,
        }

        console.print(
            f"  [green]Done.[/green] Test MAE: [bold]{report.mae:.2f} mg/dL[/bold] | "
            f"RMSE: {report.rmse:.2f} mg/dL | Features: {len(feat_names)}"
        )

    # 6. Statistical Significance Comparison
    base_res = np.abs(y_test_true - predictions_by_config["cgm_only"])
    full_res = np.abs(y_test_true - predictions_by_config["full_fusion"])
    ehr_res = np.abs(y_test_true - predictions_by_config["cgm_plus_ehr"])
    wear_res = np.abs(y_test_true - predictions_by_config["cgm_plus_wearables"])

    t_full, p_full = stats.ttest_rel(base_res, full_res)
    w_full, p_w_full = stats.wilcoxon(base_res, full_res)

    results["statistical_tests"] = {
        "full_fusion_vs_cgm_only": {
            "mean_error_reduction_mgdl": float(np.mean(base_res) - np.mean(full_res)),
            "relative_improvement_pct": float(
                (np.mean(base_res) - np.mean(full_res)) / np.mean(base_res) * 100.0
            ),
            "paired_t_statistic": float(t_full),
            "paired_t_pvalue": float(p_full),
            "wilcoxon_statistic": float(w_full),
            "wilcoxon_pvalue": float(p_w_full),
            "statistically_significant": bool(p_full < 0.05),
        },
        "cgm_plus_ehr_vs_cgm_only": {
            "mean_error_reduction_mgdl": float(np.mean(base_res) - np.mean(ehr_res)),
            "relative_improvement_pct": float(
                (np.mean(base_res) - np.mean(ehr_res)) / np.mean(base_res) * 100.0
            ),
        },
        "cgm_plus_wearables_vs_cgm_only": {
            "mean_error_reduction_mgdl": float(np.mean(base_res) - np.mean(wear_res)),
            "relative_improvement_pct": float(
                (np.mean(base_res) - np.mean(wear_res)) / np.mean(base_res) * 100.0
            ),
        },
    }

    return results


def print_summary_table(results: dict[str, Any]) -> None:
    """Print high-level comparison table to console."""
    table = Table(title="GlucoTwin — Multimodal Two-Stream Ablation Benchmark", show_lines=True)
    table.add_column("Configuration", style="cyan")
    table.add_column("Stream Components", style="white")
    table.add_column("N Features", justify="right")
    table.add_column("Test MAE", justify="right", style="bold green")
    table.add_column("Test RMSE", justify="right")
    table.add_column("MAPE (%)", justify="right")
    table.add_column("Median Abs Err", justify="right")
    table.add_column("Delta vs Baseline", justify="right", style="bold yellow")

    cgm_mae = results["cgm_only"]["metrics"]["mae"]

    for mode in ["cgm_only", "cgm_plus_ehr", "cgm_plus_wearables", "full_fusion"]:
        cfg = results[mode]
        m = cfg["metrics"]
        diff = m["mae"] - cgm_mae
        diff_str = f"{diff:+.2f} mg/dL" if mode != "cgm_only" else "Ref (0.00)"
        table.add_row(
            cfg["config_id"],
            cfg["label"],
            str(cfg["n_features"]),
            f"{m['mae']:.2f} mg/dL",
            f"{m['rmse']:.2f} mg/dL",
            f"{m['mape_pct']:.2f}%",
            f"{m['median_abs_error']:.2f} mg/dL",
            diff_str,
        )

    console.print()
    console.print(table)

    stats_info = results["statistical_tests"]["full_fusion_vs_cgm_only"]
    sig_str = "[green]Statistically Significant[/green]" if stats_info["statistically_significant"] else "[red]Not Significant[/red]"
    console.print(
        f"\n[bold]Hypothesis Test (Full Fusion vs CGM-Only):[/bold] {sig_str} "
        f"(Paired t = {stats_info['paired_t_statistic']:.3f}, p = {stats_info['paired_t_pvalue']:.2e})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled multimodal ablation study.")
    parser.add_argument(
        "--train-file",
        default="data/processed/windows_train.parquet",
        help="Path to train windows Parquet.",
    )
    parser.add_argument(
        "--val-file",
        default="data/processed/windows_val.parquet",
        help="Path to val windows Parquet.",
    )
    parser.add_argument(
        "--test-file",
        default="data/processed/windows_test.parquet",
        help="Path to test windows Parquet.",
    )
    parser.add_argument(
        "--out",
        default="reports/multimodal_ablation_results.json",
        help="Path to save output results JSON.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    train_path = project_root / args.train_file
    val_path = project_root / args.val_file
    test_path = project_root / args.test_file

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    ehr_df = load_ehr_profiles()
    wearable_df = load_wearable_telemetry()

    results = run_ablation_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        ehr_df=ehr_df,
        wearable_df=wearable_df,
        seed=args.seed,
    )

    out_file = project_root / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Mirror to artifacts/evaluations/
    artifact_file = project_root / "artifacts" / "evaluations" / "multimodal_ablation_results.json"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    with artifact_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print_summary_table(results)
    console.print(f"\nSaved report to: [green]{out_file}[/green]")
    console.print(f"Mirrored artifact to: [green]{artifact_file}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
