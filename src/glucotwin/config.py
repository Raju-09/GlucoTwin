"""Centralised configuration for GlucoTwin.

All tuneable parameters live here so that scripts and modules
never hard-code magic numbers. Values default to sensible
prototype settings and can be overridden via environment
variables (loaded from .env).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels above this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = _PROJECT_ROOT
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR: Path = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_SYNTHETIC_DIR: Path = PROJECT_ROOT / "data" / "synthetic_demo"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Forecasting parameters
# ---------------------------------------------------------------------------

#: Random seed for reproducibility.
SEED: int = int(os.getenv("GLUCOTWIN_SEED", "42"))

#: Forecast horizon in minutes.
HORIZON_MINUTES: int = int(os.getenv("GLUCOTWIN_HORIZON_MINUTES", "120"))

#: Number of historical CGM readings to include in each input window.
#: Default 24 ≈ 2 hours at a 5-minute cadence.
LOOKBACK_N: int = int(os.getenv("GLUCOTWIN_LOOKBACK_N", "24"))

#: Accept a target reading within ±N minutes of the desired horizon.
TARGET_TOLERANCE_MINUTES: int = int(
    os.getenv("GLUCOTWIN_TARGET_TOLERANCE_MINUTES", "10")
)

#: Maximum gap (consecutive missing minutes) allowed within an input window.
#: Windows with a gap larger than this are rejected.
MAX_GAP_MINUTES: int = int(os.getenv("GLUCOTWIN_MAX_GAP_MINUTES", "30"))

# ---------------------------------------------------------------------------
# Reliability thresholds
# ---------------------------------------------------------------------------

#: Forecast is withheld if the latest reading is older than this many minutes.
FRESHNESS_THRESHOLD_MINUTES: int = int(
    os.getenv("GLUCOTWIN_FRESHNESS_THRESHOLD_MINUTES", "20")
)

#: Minimum number of valid readings required in the input window.
MIN_READINGS: int = max(1, LOOKBACK_N // 2)

# ---------------------------------------------------------------------------
# Glucose physiology bounds (for outlier detection only — not clinical)
# ---------------------------------------------------------------------------

#: Values below this threshold (mg/dL) are flagged as physiologically
#: implausible for a living person with a functioning CGM.
GLUCOSE_MIN_MGDL: float = 20.0

#: Values above this threshold (mg/dL) are flagged as implausible.
GLUCOSE_MAX_MGDL: float = 600.0

#: Heuristic: if median of a patient's readings is < 30, assume mmol/L.
MMOL_DETECTION_THRESHOLD: float = 30.0

#: Conversion factor from mmol/L to mg/dL.
MMOL_TO_MGDL: float = 18.01559

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

API_HOST: str = os.getenv("GLUCOTWIN_API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("GLUCOTWIN_API_PORT", "8000"))

#: Path to the saved model artifact (relative to PROJECT_ROOT).
MODEL_PATH: Path = PROJECT_ROOT / os.getenv(
    "GLUCOTWIN_MODEL_PATH", "artifacts/model_v0.1/model.joblib"
)
