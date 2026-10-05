"""Multimodal feature engineering for Two-Stream GlucoTwin forecasting.

Integrates:
- Stream 1: Static EHR Patient Context (demographics, baseline labs, ISF, ICR)
- Stream 2A: Dynamic CGM Time-Series (lags, rolling stats, velocity, acceleration)
- Stream 2B: Dynamic Wearable Telemetry (heart rate, step counts, HRV, sleep stage)

Strict zero-leakage guarantee:
All features are computed strictly using information available at or before
the window origin timestamp `t`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from glucotwin.config import LOOKBACK_N, PROJECT_ROOT
from glucotwin.features.build_features import extract_features_from_windows

logger = logging.getLogger(__name__)

AblationMode = Literal["cgm_only", "cgm_plus_ehr", "cgm_plus_wearables", "full_fusion"]


def load_ehr_profiles(path: str | Path | None = None) -> pd.DataFrame:
    """Load static patient EHR profiles as a DataFrame indexed by patient_id.

    Parameters
    ----------
    path:
        Optional path to ehr_records.json. Defaults to data/synthetic_demo/ehr_records.json.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns prefixed with 'ehr_', indexed by 'patient_id'.
    """
    if path is None:
        path = PROJECT_ROOT / "data" / "synthetic_demo" / "ehr_records.json"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"EHR records not found at: {path}")

    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    df = pd.DataFrame(records)
    df = df.rename(
        columns={
            "age": "ehr_age",
            "bmi": "ehr_bmi",
            "diabetes_duration_years": "ehr_diabetes_duration",
            "baseline_hba1c": "ehr_baseline_hba1c",
            "historical_fbg_mgdl": "ehr_historical_fbg",
            "hypertension_flag": "ehr_hypertension_flag",
            "metformin_flag": "ehr_metformin_flag",
            "sglt2i_flag": "ehr_sglt2i_flag",
            "dawn_phenomenon_flag": "ehr_dawn_flag",
        }
    )

    keep_cols = [
        "patient_id",
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
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.set_index("patient_id")
    return df


def load_wearable_telemetry(path: str | Path | None = None) -> pd.DataFrame:
    """Load pre-generated wearable telemetry.

    Parameters
    ----------
    path:
        Optional path to wearable_telemetry.parquet. Defaults to data/synthetic_demo.

    Returns
    -------
    pd.DataFrame
        Wearable records with [patient_id, timestamp, heart_rate, steps, hrv_rmssd, sleep_state].
    """
    if path is None:
        path = PROJECT_ROOT / "data" / "synthetic_demo" / "wearable_telemetry.parquet"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Wearable telemetry not found at: {path}")

    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def extract_wearable_features_for_windows(
    windows_df: pd.DataFrame,
    wearable_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute summary wearable features available at window origin time `t`.

    For each window origin time `t` and `patient_id`:
    - `wearable_mean_hr_30m`: Rolling mean HR in [t - 30m, t]
    - `wearable_hr_delta_15m`: HR(t) - HR(t - 15m)
    - `wearable_steps_sum_30m`: Total steps in [t - 30m, t]
    - `wearable_steps_sum_60m`: Total steps in [t - 60m, t]
    - `wearable_hrv_last`: HRV at or immediately before `t`
    - `wearable_sleep_indicator`: Sleep state at `t`

    Parameters
    ----------
    windows_df:
        DataFrame containing `patient_id` and `origin_time`.
    wearable_df:
        Synchronized wearable telemetry.

    Returns
    -------
    pd.DataFrame
        Wearable feature columns matching windows_df.index.
    """
    windows = windows_df[["patient_id", "origin_time"]].copy()
    windows["origin_time"] = pd.to_datetime(windows["origin_time"], utc=True)
    windows["_window_idx"] = windows.index

    w_df = wearable_df.sort_values("timestamp").copy()

    # Pre-sort windows by timestamp for merge_asof
    windows_sorted = windows.sort_values("origin_time").reset_index(drop=True)

    # 1. As-of merge to get reading at origin_time t
    exact_at_t = pd.merge_asof(
        windows_sorted,
        w_df,
        left_on="origin_time",
        right_on="timestamp",
        by="patient_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=10),
    )

    # 2. As-of merge for t - 15m (for HR delta)
    windows_sorted["time_minus_15m"] = windows_sorted["origin_time"] - pd.Timedelta(minutes=15)
    past_15m = pd.merge_asof(
        windows_sorted.sort_values("time_minus_15m"),
        w_df[["patient_id", "timestamp", "heart_rate"]],
        left_on="time_minus_15m",
        right_on="timestamp",
        by="patient_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=10),
        suffixes=("", "_15m"),
    )

    # 3. Rolling window aggregations per patient
    # Compute rolling aggregates on wearable time series
    w_df_indexed = w_df.set_index("timestamp")
    rolling_records = []
    for pid, p_w in w_df_indexed.groupby("patient_id"):
        # Rolling 30m
        r30 = p_w[["heart_rate", "steps"]].rolling("30min", closed="both")
        mean_hr_30 = r30["heart_rate"].mean().rename("wearable_mean_hr_30m")
        steps_30 = r30["steps"].sum().rename("wearable_steps_sum_30m")

        # Rolling 60m
        r60 = p_w[["steps"]].rolling("60min", closed="both")
        steps_60 = r60["steps"].sum().rename("wearable_steps_sum_60m")

        p_roll = pd.concat([mean_hr_30, steps_30, steps_60], axis=1).reset_index()
        p_roll["patient_id"] = pid
        rolling_records.append(p_roll)

    all_rolling = pd.concat(rolling_records, ignore_index=True).sort_values("timestamp")

    # Merge rolling aggregates into windows
    merged_rolling = pd.merge_asof(
        windows_sorted,
        all_rolling,
        left_on="origin_time",
        right_on="timestamp",
        by="patient_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=10),
    )

    # Assemble feature DataFrame restored to original index
    wearable_features = pd.DataFrame(index=windows["_window_idx"])
    
    # Map back by _window_idx
    exact_at_t = exact_at_t.set_index("_window_idx")
    past_15m = past_15m.set_index("_window_idx")
    merged_rolling = merged_rolling.set_index("_window_idx")

    wearable_features["wearable_mean_hr_30m"] = merged_rolling["wearable_mean_hr_30m"].fillna(72.0)
    wearable_features["wearable_hr_delta_15m"] = (
        exact_at_t["heart_rate"] - past_15m["heart_rate"]
    ).fillna(0.0)
    wearable_features["wearable_steps_sum_30m"] = merged_rolling["wearable_steps_sum_30m"].fillna(30.0)
    wearable_features["wearable_steps_sum_60m"] = merged_rolling["wearable_steps_sum_60m"].fillna(60.0)
    wearable_features["wearable_hrv_last"] = exact_at_t["hrv_rmssd"].fillna(42.0)
    wearable_features["wearable_sleep_indicator"] = exact_at_t["sleep_state"].fillna(0).astype(float)

    return wearable_features


def extract_multimodal_features(
    windows_df: pd.DataFrame,
    ehr_df: pd.DataFrame | None = None,
    wearable_df: pd.DataFrame | None = None,
    mode: AblationMode = "full_fusion",
) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Extract tabular feature matrix X and target y according to ablation mode.

    Parameters
    ----------
    windows_df:
        DataFrame of supervised windows.
    ehr_df:
        DataFrame of static EHR features indexed by patient_id.
    wearable_df:
        DataFrame of wearable telemetry.
    mode:
        One of 'cgm_only', 'cgm_plus_ehr', 'cgm_plus_wearables', 'full_fusion'.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray | None, list[str]]
        (X, y, feature_names)
    """
    # 1. Base Stream 2A: CGM Features (43 features)
    cgm_features, target, cgm_names = extract_features_from_windows(windows_df)
    feature_dfs = [cgm_features]

    # 2. Stream 1: Static EHR Features
    if mode in ("cgm_plus_ehr", "full_fusion"):
        if ehr_df is None:
            ehr_df = load_ehr_profiles()

        # Join EHR features on patient_id
        ehr_joined = windows_df[["patient_id"]].join(ehr_df, on="patient_id")
        ehr_cols = [c for c in ehr_joined.columns if c.startswith("ehr_")]
        ehr_features = ehr_joined[ehr_cols].astype(float)
        feature_dfs.append(ehr_features)

    # 3. Stream 2B: Dynamic Wearable Features
    if mode in ("cgm_plus_wearables", "full_fusion"):
        if wearable_df is None:
            wearable_df = load_wearable_telemetry()

        wearable_features = extract_wearable_features_for_windows(windows_df, wearable_df)
        feature_dfs.append(wearable_features)

    # Combine all requested streams
    X = pd.concat(feature_dfs, axis=1)
    feature_names = list(X.columns)

    logger.info(
        "Extracted %d features for mode '%s' across %d examples.",
        len(feature_names),
        mode,
        len(X),
    )
    return X, target, feature_names
