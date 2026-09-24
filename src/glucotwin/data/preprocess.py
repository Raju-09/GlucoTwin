"""Data preprocessing pipeline for GlucoTwin.

Rules enforced here:
1. Raw data remains immutable.
2. Timestamps are parsed to UTC and sorted chronologically per patient.
3. Duplicates are resolved explicitly with documented policy.
4. Glucose units are confirmed as mg/dL (converted from mmol/L if detected).
5. No future information is leaked.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from glucotwin.config import (
    MMOL_DETECTION_THRESHOLD,
    MMOL_TO_MGDL,
)

logger = logging.getLogger(__name__)


def preprocess_cgm_stream(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    ts_col: str = "timestamp",
    glucose_col: str = "glucose",
    duplicate_policy: Literal["keep_first", "keep_last", "mean"] = "keep_first",
) -> pd.DataFrame:
    """Preprocess raw CGM time-series stream.

    Parameters
    ----------
    df:
        Raw input DataFrame.
    patient_col:
        Patient identifier column.
    ts_col:
        Timestamp column.
    glucose_col:
        Blood glucose readings column.
    duplicate_policy:
        How to handle duplicate (patient_id, timestamp) readings:
        - "keep_first": Keep the earliest ingested reading.
        - "keep_last": Keep the latest ingested reading.
        - "mean": Average the conflicting glucose readings.

    Returns
    -------
    pd.DataFrame
        Clean, sorted, unit-normalized DataFrame.
    """
    logger.info("Starting preprocessing on %d rows.", len(df))
    clean = df.copy()

    # 1. Parse timestamps to UTC
    clean[ts_col] = pd.to_datetime(clean[ts_col], utc=True)

    # 2. Sort by patient and timestamp
    clean = clean.sort_values(by=[patient_col, ts_col]).reset_index(drop=True)

    # 3. Handle duplicates
    initial_count = len(clean)
    dup_mask = clean.duplicated(subset=[patient_col, ts_col], keep=False)
    n_dups = int(dup_mask.sum())

    if n_dups > 0:
        logger.warning(
            "Resolving %d duplicate records using policy: %s",
            n_dups,
            duplicate_policy,
        )
        if duplicate_policy == "keep_first":
            clean = clean.drop_duplicates(subset=[patient_col, ts_col], keep="first")
        elif duplicate_policy == "keep_last":
            clean = clean.drop_duplicates(subset=[patient_col, ts_col], keep="last")
        elif duplicate_policy == "mean":
            # Group by patient & timestamp, aggregate numeric by mean, others by first
            agg_funcs = {}
            for c in clean.columns:
                if c in (patient_col, ts_col):
                    continue
                if pd.api.types.is_numeric_dtype(clean[c]):
                    agg_funcs[c] = "mean"
                else:
                    agg_funcs[c] = "first"
            clean = (
                clean.groupby([patient_col, ts_col], as_index=False)
                .agg(agg_funcs)
                .reset_index(drop=True)
            )
        else:
            raise ValueError(f"Unknown duplicate policy: {duplicate_policy}")

        logger.info(
            "Duplicate resolution completed: %d rows -> %d rows (-%d rows).",
            initial_count,
            len(clean),
            initial_count - len(clean),
        )

    # 4. Check & convert units
    valid_glucose = pd.to_numeric(clean[glucose_col], errors="coerce").dropna()
    if not valid_glucose.empty:
        median_glucose = float(valid_glucose.median())
        if median_glucose < MMOL_DETECTION_THRESHOLD:
            logger.warning(
                "Median glucose is %.2f (< %.1f). Converting mmol/L -> mg/dL (x %.4f).",
                median_glucose,
                MMOL_DETECTION_THRESHOLD,
                MMOL_TO_MGDL,
            )
            clean[glucose_col] = clean[glucose_col] * MMOL_TO_MGDL

    clean[glucose_col] = pd.to_numeric(clean[glucose_col], errors="coerce")

    # 5. Drop rows where glucose itself is NaN
    nan_glucose = clean[glucose_col].isna().sum()
    if nan_glucose > 0:
        logger.warning("Dropping %d rows with missing glucose reading.", nan_glucose)
        clean = clean.dropna(subset=[glucose_col]).reset_index(drop=True)

    clean = clean.sort_values(by=[patient_col, ts_col]).reset_index(drop=True)
    logger.info("Preprocessing complete. Output rows: %d.", len(clean))
    return clean
