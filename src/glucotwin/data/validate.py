"""Data validation for GlucoTwin.

Validation checks are non-destructive: they never modify the input
DataFrame. Instead they return structured :class:`ValidationResult`
objects and log warnings for any issue found.

Design rules:
- A failed check produces a warning, not a silent fix.
- The caller decides how to handle failed checks.
- All checks must work on synthetic fixtures (no real patient data in tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from glucotwin.config import (
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    MMOL_DETECTION_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single data-quality issue found during validation."""

    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    count: int = 0  # number of affected rows, if applicable


@dataclass
class ValidationResult:
    """Aggregated result of one or more validation checks."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.passed = False

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_schema(
    df: pd.DataFrame,
    expected_cols: Sequence[str],
) -> ValidationResult:
    """Check that all expected columns are present in *df*.

    Parameters
    ----------
    df:
        DataFrame to check.
    expected_cols:
        Column names that must be present.

    Returns
    -------
    ValidationResult
        Passed if all columns are present; error for each missing column.
    """
    result = ValidationResult(passed=True)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        issue = ValidationIssue(
            check="schema",
            severity="error",
            message=f"Missing required columns: {missing}",
            count=len(missing),
        )
        result.add(issue)
        logger.error(issue.message)
    else:
        logger.info("Schema check passed. All %d required columns present.", len(expected_cols))
    return result


def check_timestamps(
    df: pd.DataFrame,
    ts_col: str,
    patient_col: str | None = None,
) -> ValidationResult:
    """Check timestamps for parseability, ordering, and duplicates.

    Parameters
    ----------
    df:
        DataFrame to check.
    ts_col:
        Name of the timestamp column.
    patient_col:
        If provided, check ordering within each patient group.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult(passed=True)

    if ts_col not in df.columns:
        result.add(
            ValidationIssue(
                check="timestamps",
                severity="error",
                message=f"Timestamp column '{ts_col}' not found in DataFrame.",
            )
        )
        return result

    # Try parsing if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        try:
            parsed = pd.to_datetime(df[ts_col], utc=True)
            n_failed = parsed.isna().sum() - df[ts_col].isna().sum()
            if n_failed > 0:
                result.add(
                    ValidationIssue(
                        check="timestamps.parse",
                        severity="error",
                        message=f"{n_failed} timestamp(s) could not be parsed.",
                        count=int(n_failed),
                    )
                )
        except Exception as exc:
            result.add(
                ValidationIssue(
                    check="timestamps.parse",
                    severity="error",
                    message=f"Timestamp parsing failed: {exc}",
                )
            )
            return result

    # Duplicate check
    if patient_col and patient_col in df.columns:
        dup_mask = df.duplicated(subset=[patient_col, ts_col], keep=False)
    else:
        dup_mask = df.duplicated(subset=[ts_col], keep=False)

    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        result.add(
            ValidationIssue(
                check="timestamps.duplicates",
                severity="warning",
                message=f"{n_dups} duplicate (patient, timestamp) pair(s) found.",
                count=n_dups,
            )
        )
        logger.warning(
            "Found %d duplicate timestamp(s). "
            "These must be resolved before window creation.",
            n_dups,
        )
    else:
        logger.info("No duplicate timestamps found.")

    # Ordering check (within patients if patient_col given)
    ts_series = pd.to_datetime(df[ts_col], utc=True)
    if patient_col and patient_col in df.columns:
        # Check monotonic increasing within each group
        is_sorted = True
        for _, grp in df.groupby(patient_col, sort=False):
            if not ts_series.loc[grp.index].is_monotonic_increasing:
                is_sorted = False
                break
    else:
        is_sorted = ts_series.is_monotonic_increasing

    if not is_sorted:
        result.add(
            ValidationIssue(
                check="timestamps.ordering",
                severity="warning",
                message="Timestamps are not sorted. Sorting will be applied in preprocessing.",
            )
        )
        logger.warning("Timestamps are not sorted.")
    else:
        logger.info("Timestamps are in ascending order.")

    return result


def check_glucose_units(
    df: pd.DataFrame,
    value_col: str,
) -> ValidationResult:
    """Heuristic check for glucose units.

    If the median value is below :data:`~glucotwin.config.MMOL_DETECTION_THRESHOLD`,
    the values are likely in mmol/L rather than mg/dL.

    Parameters
    ----------
    df:
        DataFrame to check.
    value_col:
        Name of the glucose value column.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult(passed=True)

    if value_col not in df.columns:
        result.add(
            ValidationIssue(
                check="glucose_units",
                severity="error",
                message=f"Column '{value_col}' not found.",
            )
        )
        return result

    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if values.empty:
        result.add(
            ValidationIssue(
                check="glucose_units",
                severity="error",
                message=f"Column '{value_col}' has no numeric values.",
            )
        )
        return result

    median_val = float(values.median())
    logger.info(
        "Glucose column '%s': median=%.2f, min=%.2f, max=%.2f",
        value_col,
        median_val,
        float(values.min()),
        float(values.max()),
    )

    if median_val < MMOL_DETECTION_THRESHOLD:
        result.add(
            ValidationIssue(
                check="glucose_units.mmol_suspected",
                severity="warning",
                message=(
                    f"Median glucose value ({median_val:.2f}) is below "
                    f"{MMOL_DETECTION_THRESHOLD}. Values may be in mmol/L rather "
                    "than mg/dL. Confirm units before proceeding."
                ),
            )
        )
        logger.warning(
            "Glucose units may be mmol/L (median=%.2f). "
            "Conversion will be applied if confirmed.",
            median_val,
        )

    return result


def check_missingness(
    df: pd.DataFrame,
    critical_cols: Sequence[str] | None = None,
) -> ValidationResult:
    """Report missing values per column.

    Parameters
    ----------
    df:
        DataFrame to check.
    critical_cols:
        Columns where high missingness (>=20%) is treated as an error.
        For other columns (e.g. sparse event logs like meals), missingness is a warning.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult(passed=True)
    n = len(df)
    if n == 0:
        result.add(
            ValidationIssue(
                check="missingness",
                severity="warning",
                message="DataFrame is empty.",
            )
        )
        return result

    critical_set = set(critical_cols) if critical_cols else set()

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct = 100.0 * n_missing / n
        if n_missing > 0:
            is_critical = col in critical_set
            severity = "error" if (is_critical and pct >= 20.0) else "warning"
            result.add(
                ValidationIssue(
                    check=f"missingness.{col}",
                    severity=severity,
                    message=f"Column '{col}': {n_missing}/{n} missing ({pct:.1f}%).",
                    count=n_missing,
                )
            )
            logger.log(
                logging.ERROR if severity == "error" else logging.WARNING,
                "Column '%s': %d missing (%.1f%%)",
                col,
                n_missing,
                pct,
            )
        else:
            logger.debug("Column '%s': no missing values.", col)

    return result


def check_physiological_range(
    df: pd.DataFrame,
    value_col: str,
    min_val: float = GLUCOSE_MIN_MGDL,
    max_val: float = GLUCOSE_MAX_MGDL,
) -> ValidationResult:
    """Flag glucose values outside physiologically plausible bounds.

    These bounds are used to detect sensor errors or unit mismatches.
    They are NOT clinical decision thresholds.

    Parameters
    ----------
    df:
        DataFrame to check.
    value_col:
        Name of the glucose column.
    min_val, max_val:
        Inclusive bounds for plausible values (mg/dL).

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult(passed=True)

    if value_col not in df.columns:
        result.add(
            ValidationIssue(
                check="physiological_range",
                severity="error",
                message=f"Column '{value_col}' not found.",
            )
        )
        return result

    values = pd.to_numeric(df[value_col], errors="coerce")
    below = int((values < min_val).sum())
    above = int((values > max_val).sum())

    if below > 0:
        result.add(
            ValidationIssue(
                check="physiological_range.below_min",
                severity="warning",
                message=(
                    f"{below} value(s) below physiological minimum ({min_val} mg/dL). "
                    "These will be flagged and excluded from windows."
                ),
                count=below,
            )
        )
        logger.warning("%d value(s) below %.0f mg/dL.", below, min_val)

    if above > 0:
        result.add(
            ValidationIssue(
                check="physiological_range.above_max",
                severity="warning",
                message=(
                    f"{above} value(s) above physiological maximum ({max_val} mg/dL). "
                    "These may indicate sensor errors or unit mismatches."
                ),
                count=above,
            )
        )
        logger.warning("%d value(s) above %.0f mg/dL.", above, max_val)

    if below == 0 and above == 0:
        logger.info(
            "All glucose values within plausible range [%.0f, %.0f] mg/dL.",
            min_val,
            max_val,
        )

    return result


def check_sampling_intervals(
    df: pd.DataFrame,
    ts_col: str,
    patient_col: str | None = None,
    expected_interval_minutes: float = 5.0,
    tolerance_minutes: float = 1.0,
) -> ValidationResult:
    """Compute and report the distribution of inter-reading intervals.

    Parameters
    ----------
    df:
        DataFrame (should be sorted by timestamp within patient).
    ts_col:
        Timestamp column.
    patient_col:
        If provided, intervals are computed within each patient.
    expected_interval_minutes:
        Expected cadence (used for reporting only).
    tolerance_minutes:
        Readings within ``expected_interval_minutes ± tolerance_minutes``
        are considered regular.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult(passed=True)

    if ts_col not in df.columns:
        result.add(
            ValidationIssue(
                check="sampling_intervals",
                severity="error",
                message=f"Timestamp column '{ts_col}' not found.",
            )
        )
        return result

    ts = pd.to_datetime(df[ts_col], utc=True)

    if patient_col and patient_col in df.columns:
        diffs_minutes: list[float] = []
        for _, grp in df.groupby(patient_col, sort=False):
            t = ts.loc[grp.index].sort_values()
            delta = t.diff().dropna().dt.total_seconds() / 60
            diffs_minutes.extend(delta.tolist())
    else:
        diffs_minutes = (ts.sort_values().diff().dropna().dt.total_seconds() / 60).tolist()

    if not diffs_minutes:
        result.add(
            ValidationIssue(
                check="sampling_intervals",
                severity="warning",
                message="No intervals computed (fewer than 2 readings).",
            )
        )
        return result

    diffs = np.array(diffs_minutes)
    median_interval = float(np.median(diffs))
    pct_regular = float(
        np.mean(
            np.abs(diffs - expected_interval_minutes) <= tolerance_minutes
        )
        * 100
    )
    n_long_gaps = int(np.sum(diffs > 30))

    logger.info(
        "Sampling intervals: median=%.1f min, %.1f%% within expected ±%.1f min, "
        "%d long gap(s) > 30 min",
        median_interval,
        pct_regular,
        tolerance_minutes,
        n_long_gaps,
    )

    if n_long_gaps > 0:
        result.add(
            ValidationIssue(
                check="sampling_intervals.long_gaps",
                severity="warning",
                message=(
                    f"{n_long_gaps} gap(s) > 30 min detected. "
                    "Windows spanning these gaps will be rejected."
                ),
                count=n_long_gaps,
            )
        )

    # Store stats for the audit report
    result.issues.append(
        ValidationIssue(
            check="sampling_intervals.stats",
            severity="info",
            message=(
                f"median={median_interval:.1f} min, "
                f"pct_regular={pct_regular:.1f}%, "
                f"n_long_gaps={n_long_gaps}"
            ),
        )
    )

    return result


def run_full_validation(
    df: pd.DataFrame,
    ts_col: str,
    glucose_col: str,
    patient_col: str | None = None,
    expected_cols: Sequence[str] | None = None,
) -> ValidationResult:
    """Run all validation checks and return an aggregated result.

    Parameters
    ----------
    df:
        DataFrame to validate.
    ts_col, glucose_col, patient_col:
        Column names (patient_col is optional).
    expected_cols:
        Columns that must be present (defaults to ts_col + glucose_col).

    Returns
    -------
    ValidationResult
        Aggregated result with all issues.
    """
    if expected_cols is None:
        expected_cols = [ts_col, glucose_col]
        if patient_col:
            expected_cols = [patient_col] + list(expected_cols)

    combined = ValidationResult(passed=True)

    for check_result in [
        check_schema(df, expected_cols),
        check_timestamps(df, ts_col, patient_col),
        check_glucose_units(df, glucose_col),
        check_missingness(df, critical_cols=[ts_col, glucose_col]),
        check_physiological_range(df, glucose_col),
        check_sampling_intervals(df, ts_col, patient_col),
    ]:
        combined.issues.extend(check_result.issues)
        if not check_result.passed:
            combined.passed = False

    return combined
