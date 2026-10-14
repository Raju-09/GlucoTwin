"""Experiment 01: Break Your Own Claim.

Runs:
1. Leave-One-Patient-Out (LOPO) cross validation on all 10 synthetic patients
   comparing Config A (CGM only) vs Config B (CGM + static EHR).
2. Computes per-patient paired differences (MAE_B - MAE_A) and patient-level bootstrap CI.
3. Runs the patient-level Permuted-EHR control under both the old temporal split
   and the LOPO split.

Saves results to experiments/results/exp01_break_your_claim.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Fix Windows CPU core count for joblib
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy import stats

from glucotwin.config import SEED
from glucotwin.eval.controls import bootstrap_patient_deltas, permute_ehr_patient_level
from glucotwin.eval.lopo import compute_patient_level_deltas, make_standard_gbm, run_lopo_cv
from glucotwin.features.multimodal import (
    extract_multimodal_features,
    load_ehr_profiles,
    load_wearable_telemetry,
)

console = Console(safe_box=True)


def main() -> int:
    console.rule("[bold red]GlucoTwin Experiment 01: Break Your Own Claim (LOPO & Controls)")

    # 1. Load data
    train_df = pd.read_parquet(PROJECT_ROOT / "data/processed/windows_train.parquet")
    val_df = pd.read_parquet(PROJECT_ROOT / "data/processed/windows_val.parquet")
    test_df = pd.read_parquet(PROJECT_ROOT / "data/processed/windows_test.parquet")
    ehr_df = load_ehr_profiles()
    wearable_df = load_wearable_telemetry()

    # Combine for LOPO evaluation across the entire cohort (37,236 total windows)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    console.print(f"Total windows loaded for LOPO: [cyan]{len(all_df):,}[/cyan] across [cyan]{all_df['patient_id'].nunique()}[/cyan] patients")

    # 2. Extract features
    console.print("Extracting features for Config A (CGM-only) and Config B (CGM + EHR)...")
    X_cgm, y_all, cgm_feats = extract_multimodal_features(all_df, ehr_df, wearable_df, mode="cgm_only")
    X_ehr, _, ehr_feats = extract_multimodal_features(all_df, ehr_df, wearable_df, mode="cgm_plus_ehr")

    # Build evaluation DataFrames
    cgm_eval_df = X_cgm.copy()
    cgm_eval_df["target_glucose"] = y_all
    cgm_eval_df["patient_id"] = all_df["patient_id"].values

    ehr_eval_df = X_ehr.copy()
    ehr_eval_df["target_glucose"] = y_all
    ehr_eval_df["patient_id"] = all_df["patient_id"].values

    ehr_cols = [c for c in ehr_feats if c.startswith("ehr_")]
    console.print(f"EHR feature count: {len(ehr_cols)} features: {ehr_cols}")

    # -----------------------------------------------------------------------
    # Part 1: Permuted EHR Control on Old Temporal Split
    # -----------------------------------------------------------------------
    console.rule("[bold yellow]Part 1: Permuted-EHR Control on Temporal Split")
    # Feature extraction for train/test splits
    X_tr_ehr, y_tr, _ = extract_multimodal_features(train_df, ehr_df, wearable_df, mode="cgm_plus_ehr")
    X_te_ehr, y_te, _ = extract_multimodal_features(test_df, ehr_df, wearable_df, mode="cgm_plus_ehr")

    # Real Config B model on temporal split
    m_real = make_standard_gbm(random_state=SEED)
    m_real.fit(X_tr_ehr, y_tr)
    preds_real = m_real.predict(X_te_ehr)
    mae_real = float(np.mean(np.abs(y_te - preds_real)))

    # Permuted EHR on temporal split (shuffle EHR at patient level)
    ehr_df_perm = permute_ehr_patient_level(
        ehr_df.reset_index(),
        ehr_cols=ehr_cols,
        patient_col="patient_id",
        random_state=SEED,
    ).set_index("patient_id")

    X_tr_perm, _, _ = extract_multimodal_features(train_df, ehr_df_perm, wearable_df, mode="cgm_plus_ehr")
    X_te_perm, _, _ = extract_multimodal_features(test_df, ehr_df_perm, wearable_df, mode="cgm_plus_ehr")

    m_perm = make_standard_gbm(random_state=SEED)
    m_perm.fit(X_tr_perm, y_tr)
    preds_perm = m_perm.predict(X_te_perm)
    mae_perm = float(np.mean(np.abs(y_te - preds_perm)))

    # Real Config A (CGM only) on temporal split
    X_tr_cgm, _, _ = extract_multimodal_features(train_df, ehr_df, wearable_df, mode="cgm_only")
    X_te_cgm, _, _ = extract_multimodal_features(test_df, ehr_df, wearable_df, mode="cgm_only")
    m_cgm = make_standard_gbm(random_state=SEED)
    m_cgm.fit(X_tr_cgm, y_tr)
    preds_cgm = m_cgm.predict(X_te_cgm)
    mae_cgm = float(np.mean(np.abs(y_te - preds_cgm)))

    console.print(f"Temporal Split CGM-only MAE:      [cyan]{mae_cgm:.2f} mg/dL[/cyan]")
    console.print(f"Temporal Split Real EHR MAE:      [green]{mae_real:.2f} mg/dL[/green] (delta = {mae_real - mae_cgm:+.2f} mg/dL)")
    console.print(f"Temporal Split Permuted EHR MAE:  [magenta]{mae_perm:.2f} mg/dL[/magenta] (delta = {mae_perm - mae_cgm:+.2f} mg/dL)")

    # -----------------------------------------------------------------------
    # Part 2: Leave-One-Patient-Out (LOPO) Cross-Validation
    # -----------------------------------------------------------------------
    console.rule("[bold cyan]Part 2: Leave-One-Patient-Out (LOPO) Evaluation")
    console.print("Running LOPO for Config A (CGM-only)...")
    lopo_cgm_res = run_lopo_cv(cgm_eval_df, feature_cols=cgm_feats, random_state=SEED)

    console.print("Running LOPO for Config B (CGM + Real EHR)...")
    lopo_ehr_res = run_lopo_cv(ehr_eval_df, feature_cols=ehr_feats, random_state=SEED)

    console.print("Running LOPO for Permuted EHR Control...")
    perm_eval_df = permute_ehr_patient_level(
        ehr_eval_df,
        ehr_cols=ehr_cols,
        patient_col="patient_id",
        random_state=SEED,
    )
    lopo_perm_res = run_lopo_cv(perm_eval_df, feature_cols=ehr_feats, random_state=SEED)

    # -----------------------------------------------------------------------
    # Part 3: Per-Patient Analysis & Patient-Level Bootstrap
    # -----------------------------------------------------------------------
    console.rule("[bold green]Part 3: Per-Patient Summary & Statistical Inference")
    summary = compute_patient_level_deltas(lopo_cgm_res, lopo_ehr_res, "cgm_only", "cgm_plus_ehr")
    perm_maes = lopo_perm_res.groupby("patient_id")["abs_error"].mean()
    summary["mae_permuted_ehr"] = perm_maes

    n_p = len(summary)
    table = Table(title=f"Leave-One-Patient-Out (LOPO) Evaluation (N={n_p} Unseen Patients)", show_lines=True)
    table.add_column("Patient ID", style="cyan")
    table.add_column("CGM Only MAE", justify="right")
    table.add_column("CGM + EHR MAE", justify="right", style="green")
    table.add_column("Permuted EHR MAE", justify="right", style="magenta")
    table.add_column("EHR Improvement (MAE_CGM - MAE_EHR)", justify="right", style="bold yellow")
    table.add_column("delta MAE (MAE_EHR - MAE_CGM)", justify="right")

    for pid, row in summary.iterrows():
        table.add_row(
            str(pid),
            f"{row['mae_cgm_only']:.2f}",
            f"{row['mae_cgm_plus_ehr']:.2f}",
            f"{row['mae_permuted_ehr']:.2f}",
            f"{row['improvement_mgdl']:+.2f} mg/dL",
            f"{row['delta_mae_mgdl']:+.2f} mg/dL",
        )

    console.print(table)

    # Patient-level paired statistical tests
    cgm_pt = summary["mae_cgm_only"].values
    ehr_pt = summary["mae_cgm_plus_ehr"].values
    perm_pt = summary["mae_permuted_ehr"].values
    improvements = summary["improvement_mgdl"].values  # MAE_CGM - MAE_EHR

    mean_imp, ci_lower, ci_upper = bootstrap_patient_deltas(improvements, n_boot=2000, random_state=SEED)
    t_stat, p_val = stats.ttest_rel(cgm_pt, ehr_pt)

    console.print()
    console.print(f"[bold]Mean LOPO CGM-Only MAE:[/bold]       {np.mean(cgm_pt):.2f} mg/dL")
    console.print(f"[bold]Mean LOPO CGM+EHR MAE:[/bold]        {np.mean(ehr_pt):.2f} mg/dL")
    console.print(f"[bold]Mean LOPO Permuted EHR MAE:[/bold]   {np.mean(perm_pt):.2f} mg/dL")
    console.print(f"[bold]Mean Per-Patient Improvement:[/bold]  {mean_imp:+.2f} mg/dL")
    console.print(f"[bold]Patient-level 95% Bootstrap CI:[/bold] [{ci_lower:+.2f}, {ci_upper:+.2f}] mg/dL")
    console.print(f"[bold]Patient-level Paired t-test:[/bold]    t = {t_stat:.2f}, p = {p_val:.4f} (df = 9)")

    # Save output artifacts
    out_dir = PROJECT_ROOT / "experiments/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp01_break_your_claim.json"

    results_payload = {
        "experiment_id": "EXP01_BREAK_YOUR_CLAIM",
        "n_patients": len(summary),
        "total_windows": len(all_df),
        "temporal_split": {
            "cgm_only_mae": round(mae_cgm, 3),
            "cgm_plus_ehr_mae": round(mae_real, 3),
            "cgm_plus_permuted_ehr_mae": round(mae_perm, 3),
            "ehr_gain_survived_permutation": bool(mae_real < mae_perm),
        },
        "lopo_cross_validation": {
            "mean_cgm_only_mae": round(float(np.mean(cgm_pt)), 3),
            "mean_cgm_plus_ehr_mae": round(float(np.mean(ehr_pt)), 3),
            "mean_permuted_ehr_mae": round(float(np.mean(perm_pt)), 3),
            "mean_improvement_mgdl": round(mean_imp, 3),
            "bootstrap_95_ci_improvement": [round(ci_lower, 3), round(ci_upper, 3)],
            "paired_t_statistic": round(float(t_stat), 3),
            "p_value": round(float(p_val), 4),
        },
        "per_patient_summary": summary.round(3).to_dict(orient="index"),
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    console.print(f"\n[green]Saved Experiment 01 results to {out_file}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
