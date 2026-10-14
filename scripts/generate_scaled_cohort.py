"""Scale and generate realistic 60-patient in-silico cohort with mechanistic EHR links and sensor layer.

Generates:
- data/synthetic_demo/synthetic_cgm_benchmark.csv (and data/raw/synthetic_cgm_benchmark.csv)
- data/synthetic_demo/ehr_records.json
- data/synthetic_demo/patients.json
- data/synthetic_demo/wearable_telemetry.parquet

Features:
- N = 60 synthetic patients with realistic clinical phenotypic distributions.
- Basal glucose and ISF mechanistically linked to BMI, HbA1c, and duration.
- GMI calibration check: GMI = 3.31 + 0.02392 * mean_glucose ~= HbA1c.
- Sensor layer with interstitial lag (tau=7.5m), AR(1) correlated noise, and step-rate assertions (< 4 mg/dL/min).
- Autonomous wearable telemetry (HR, steps, HRV, sleep) decoupled from glucose.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from glucotwin.config import PROJECT_ROOT, SEED
from glucotwin.data.wearables import generate_wearable_telemetry
from glucotwin.sim.generator import generate_population_phenotype, simulate_patient_trajectory

console = Console(safe_box=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate scaled in-silico cohort.")
    parser.add_argument("--patients", type=int, default=60, help="Number of patients (default: 60)")
    parser.add_argument("--days", type=int, default=14, help="Days per patient (default: 14)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    console.rule(f"[bold cyan]GlucoTwin: Generating Scaled In-Silico Cohort (N={args.patients})")

    demo_dir = PROJECT_ROOT / "data" / "synthetic_demo"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    demo_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    phenotypes = []
    cgm_frames = []
    ehr_records = []
    patient_cards = []

    for i in range(1, args.patients + 1):
        pid = f"SYNTH_{i:03d}"
        p_seed = args.seed + i * 137

        # 1. Generate clinical phenotype
        pheno = generate_population_phenotype(pid, seed=p_seed)
        phenotypes.append(pheno)

        # 2. Simulate 14-day CGM trajectory with sensor physics layer
        p_df = simulate_patient_trajectory(
            pheno,
            days=args.days,
            seed=p_seed + 1,
        )
        cgm_frames.append(p_df)

        # 3. Create EHR record
        ehr_dict = {
            "patient_id": pheno.patient_id,
            "age": pheno.age,
            "sex": pheno.sex,
            "bmi": pheno.bmi,
            "diabetes_duration_years": pheno.diabetes_duration_years,
            "baseline_hba1c": pheno.baseline_hba1c,
            "historical_fbg_mgdl": pheno.historical_fbg_mgdl,
            "hypertension_flag": pheno.hypertension_flag,
            "metformin_flag": pheno.metformin_flag,
            "sglt2i_flag": pheno.sglt2i_flag,
            "dawn_phenomenon_flag": pheno.dawn_phenomenon_flag,
            "clinical_notes": pheno.clinical_notes,
        }
        ehr_records.append(ehr_dict)

        # 4. Create patient card
        med_gl = float(np.round(p_df["glucose"].median(), 1))
        patient_cards.append(
            {
                "patient_id": pheno.patient_id,
                "label": f"Synthetic Patient {i:03d}",
                "condition": "Synthetic diabetes cohort (Simulated adult dynamics)",
                "days_recorded": args.days,
                "median_glucose_mgdl": med_gl,
                "adaptation_recommended": bool(i % 5 == 0),
                "notes": pheno.clinical_notes,
            }
        )

    # Combine all CGM records
    all_cgm = pd.concat(cgm_frames, ignore_index=True)
    console.print(f"Total simulated CGM records: [green]{len(all_cgm):,}[/green] across [cyan]{args.patients}[/cyan] patients")

    # Save CGM benchmarks
    cgm_demo_path = demo_dir / "synthetic_cgm_benchmark.csv"
    cgm_raw_path = raw_dir / "synthetic_cgm_benchmark.csv"
    all_cgm.to_csv(cgm_demo_path, index=False)
    all_cgm.to_csv(cgm_raw_path, index=False)
    console.print(f"Saved CGM benchmark to: {cgm_demo_path}")

    # Save EHR records
    ehr_path = demo_dir / "ehr_records.json"
    with ehr_path.open("w", encoding="utf-8") as f:
        json.dump(ehr_records, f, indent=2)
    console.print(f"Saved EHR records to: {ehr_path}")

    # Save patient cards
    patient_path = demo_dir / "patients.json"
    with patient_path.open("w", encoding="utf-8") as f:
        json.dump(patient_cards, f, indent=2)
    console.print(f"Saved Patient cards to: {patient_path}")

    # 5. Generate decoupled wearable telemetry
    console.print("Generating synchronized autonomous wearable telemetry...")
    wearable_df = generate_wearable_telemetry(all_cgm, seed=args.seed + 999)
    wearable_path = demo_dir / "wearable_telemetry.parquet"
    wearable_df.to_parquet(wearable_path, index=False)
    console.print(f"Saved Wearable telemetry to: {wearable_path} ({len(wearable_df):,} rows)")

    console.print("[bold green]Cohort generation complete![/bold green]")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
