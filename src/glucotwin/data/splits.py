"""Leakage-safe split construction for GlucoTwin.

Rules enforced here:
1. Chronological splitting: Model is trained on the past and tested on the future.
2. Buffer exclusion: A safety buffer equal to the forecast horizon (120 min)
   is enforced between splits so that windows at the boundary never bleed across
   train, validation, and test.
3. Metadata report: Exact timestamps and counts for each split are recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from glucotwin.config import HORIZON_MINUTES

logger = logging.getLogger(__name__)


@dataclass
class SplitMetadata:
    """Audit metadata for data splitting."""

    strategy: str
    train_count: int
    val_count: int
    test_count: int
    train_pct: float
    val_pct: float
    test_pct: float
    buffer_minutes: int
    details_per_patient: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "train_pct": self.train_pct,
            "val_pct": self.val_pct,
            "test_pct": self.test_pct,
            "buffer_minutes": self.buffer_minutes,
            "details_per_patient": self.details_per_patient,
        }


def chronological_split(
    windows_df: pd.DataFrame,
    patient_col: str = "patient_id",
    origin_col: str = "origin_time",
    target_col: str = "target_time",
    train_frac: float = 0.65,
    val_frac: float = 0.15,
    buffer_minutes: int = HORIZON_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]:
    """Split supervised windows chronologically per patient with safety buffers.

    Parameters
    ----------
    windows_df:
        DataFrame of supervised windows.
    patient_col:
        Patient identifier column.
    origin_col:
        Origin timestamp column.
    target_col:
        Target timestamp column.
    train_frac:
        Fraction of time span allocated to train (default: 0.65).
    val_frac:
        Fraction of time span allocated to validation (default: 0.15).
        Remaining fraction (1.0 - train_frac - val_frac) is allocated to test.
    buffer_minutes:
        Buffer period in minutes discarded between split boundaries to prevent
        temporal leakage.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]
        (train_df, val_df, test_df, metadata)
    """
    if windows_df.empty:
        raise ValueError("Cannot split empty windows DataFrame.")

    df = windows_df.copy()
    df[origin_col] = pd.to_datetime(df[origin_col], utc=True)
    df[target_col] = pd.to_datetime(df[target_col], utc=True)

    train_list: list[pd.DataFrame] = []
    val_list: list[pd.DataFrame] = []
    test_list: list[pd.DataFrame] = []
    per_patient_meta: list[dict[str, Any]] = []

    buffer_delta = pd.Timedelta(minutes=buffer_minutes)

    for pid, group in df.groupby(patient_col, sort=False):
        group = group.sort_values(by=origin_col).reset_index(drop=True)

        t_min = group[origin_col].min()
        t_max = group[target_col].max()
        total_duration = t_max - t_min

        # Cutoffs by chronological duration
        t_train_end = t_min + total_duration * train_frac
        t_val_start = t_train_end + buffer_delta
        t_val_end = t_min + total_duration * (train_frac + val_frac)
        t_test_start = t_val_end + buffer_delta

        # Leakage-safe masking:
        # Train windows must have BOTH origin and target before t_train_end
        mask_train = (group[origin_col] <= t_train_end) & (group[target_col] <= t_train_end)

        # Validation windows must start after t_val_start and finish before t_val_end
        mask_val = (group[origin_col] >= t_val_start) & (group[target_col] <= t_val_end)

        # Test windows must start after t_test_start
        mask_test = group[origin_col] >= t_test_start

        p_train = group[mask_train]
        p_val = group[mask_val]
        p_test = group[mask_test]

        train_list.append(p_train)
        val_list.append(p_val)
        test_list.append(p_test)

        per_patient_meta.append(
            {
                "patient_id": str(pid),
                "total_windows": len(group),
                "train_windows": len(p_train),
                "val_windows": len(p_val),
                "test_windows": len(p_test),
                "boundary_discarded_windows": len(group)
                - (len(p_train) + len(p_val) + len(p_test)),
                "time_min": str(t_min),
                "train_end": str(t_train_end),
                "val_start": str(t_val_start),
                "val_end": str(t_val_end),
                "test_start": str(t_test_start),
                "time_max": str(t_max),
            }
        )

    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    total_valid = len(train_df) + len(val_df) + len(test_df)
    metadata = SplitMetadata(
        strategy="chronological_per_patient_with_buffer",
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        train_pct=round(100.0 * len(train_df) / total_valid, 2) if total_valid else 0.0,
        val_pct=round(100.0 * len(val_df) / total_valid, 2) if total_valid else 0.0,
        test_pct=round(100.0 * len(test_df) / total_valid, 2) if total_valid else 0.0,
        buffer_minutes=buffer_minutes,
        details_per_patient=per_patient_meta,
    )

    logger.info(
        "Chronological split complete: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        metadata.train_count,
        metadata.train_pct,
        metadata.val_count,
        metadata.val_pct,
        metadata.test_count,
        metadata.test_pct,
    )
    return train_df, val_df, test_df, metadata
