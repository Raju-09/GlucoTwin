"""Unit tests for multimodal two-stream feature engineering and wearable data."""

from __future__ import annotations

import pandas as pd
import pytest

from glucotwin.data.wearables import generate_wearable_telemetry
from glucotwin.features.multimodal import (
    extract_multimodal_features,
    load_ehr_profiles,
)


def test_load_ehr_profiles():
    ehr_df = load_ehr_profiles()
    assert len(ehr_df) >= 10
    expected_cols = [
        "ehr_age",
        "ehr_bmi",
        "ehr_diabetes_duration",
        "ehr_baseline_hba1c",
        "ehr_historical_fbg",
        "ehr_hypertension_flag",
        "ehr_metformin_flag",
        "ehr_sglt2i_flag",
        "ehr_dawn_flag",
    ]
    for col in expected_cols:
        assert col in ehr_df.columns, f"Missing EHR column {col}"
    assert "SYNTH_001" in ehr_df.index
    assert "SYNTH_010" in ehr_df.index


def test_generate_wearable_telemetry(synthetic_cgm_df):
    wearable_df = generate_wearable_telemetry(synthetic_cgm_df)
    assert len(wearable_df) == len(synthetic_cgm_df)
    expected_cols = ["patient_id", "timestamp", "heart_rate", "steps", "hrv_rmssd", "sleep_state"]
    for col in expected_cols:
        assert col in wearable_df.columns
    assert wearable_df["heart_rate"].between(40, 200).all()
    assert (wearable_df["steps"] >= 0).all()
    assert wearable_df["hrv_rmssd"].between(5, 120).all()
    assert set(wearable_df["sleep_state"].unique()).issubset({0, 1})


def test_extract_multimodal_modes():
    windows_path = "data/processed/windows_test.parquet"
    test_windows = pd.read_parquet(windows_path).iloc[:20]

    # Mode 1: cgm_only
    X_cgm, y, names_cgm = extract_multimodal_features(test_windows, mode="cgm_only")
    assert len(names_cgm) == 41
    assert all(not c.startswith("ehr_") for c in names_cgm)
    assert all(not c.startswith("wearable_") for c in names_cgm)

    # Mode 2: cgm_plus_ehr
    X_ehr, _, names_ehr = extract_multimodal_features(test_windows, mode="cgm_plus_ehr")
    assert len(names_ehr) == 50
    assert any(c.startswith("ehr_") for c in names_ehr)
    assert all(not c.startswith("wearable_") for c in names_ehr)

    # Mode 3: cgm_plus_wearables
    X_wear, _, names_wear = extract_multimodal_features(test_windows, mode="cgm_plus_wearables")
    assert len(names_wear) == 47
    assert all(not c.startswith("ehr_") for c in names_wear)
    assert any(c.startswith("wearable_") for c in names_wear)

    # Mode 4: full_fusion
    X_full, _, names_full = extract_multimodal_features(test_windows, mode="full_fusion")
    assert len(names_full) == 56
    assert any(c.startswith("ehr_") for c in names_full)
    assert any(c.startswith("wearable_") for c in names_full)
    assert X_full.isna().sum().sum() == 0  # Zero missing values
