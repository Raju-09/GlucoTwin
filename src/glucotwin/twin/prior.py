"""EHR-driven Bayesian priors for mechanistic twin parameters.

Maps observable clinical variables (HbA1c, BMI, diabetes duration, FBG)
to prior distributions over the two most critical latent physiological
parameters used in the Minimal Model-style glucose ODE:

    G'(t) = -p2*(G(t) - Gb) - p3*X(t)*G(t) + Ra(t)/Vd
    X'(t) = -p2*X(t) + p3*I(t)

Parameters inferred here:
    Gb  -- basal/fasting glucose setpoint (mg/dL)
    SI  -- insulin sensitivity index (dimensionless, relative units)

Priors are Gaussian: (mean, std) derived from clinical literature regression.

References (notation only — no real patient data used):
    Dalla Man et al. (2007) IEEE TBME — Minimal Model
    GMI equation: Bergenstal et al. (2018) Diabetes Care
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TwinPrior:
    """Gaussian prior distributions over core twin parameters.

    All parameters are in relative / normalised units unless noted.
    """

    # Basal glucose setpoint prior (mg/dL)
    Gb_mean: float
    Gb_std: float

    # Insulin sensitivity prior (relative, higher = more sensitive)
    SI_mean: float
    SI_std: float

    # Circadian amplitude prior (mg/dL peak-to-trough half-range)
    circ_amp_mean: float
    circ_amp_std: float

    # Meal absorption rate prior (fraction of carb absorbed per minute)
    kabs_mean: float
    kabs_std: float

    # Carb intake per meal prior (grams)
    meal_carb_mean: float
    meal_carb_std: float

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal regression helpers
# ---------------------------------------------------------------------------

def _ehr_to_gb(hba1c: float, fbg_mgdl: float) -> Tuple[float, float]:
    """Map HbA1c + historical FBG to a basal glucose prior (mean, std).

    Uses the GMI (Glucose Management Indicator) equation:
        GMI (%) = 3.31 + 0.02392 * mean_glucose
    Inverted: mean_glucose = (HbA1c - 3.31) / 0.02392

    The historical FBG is blended (60/40) with the GMI-derived estimate
    to set the basal setpoint, because FBG reflects overnight physiology
    while GMI reflects 24h average.
    """
    gmi_implied_mean = (hba1c - 3.31) / 0.02392  # mg/dL
    gmi_implied_mean = float(np.clip(gmi_implied_mean, 70.0, 280.0))

    # Weighted blend: 60% from historical FBG, 40% from GMI-implied mean
    blended = 0.60 * fbg_mgdl + 0.40 * gmi_implied_mean
    blended = float(np.clip(blended, 70.0, 280.0))

    # Higher HbA1c → wider uncertainty (more variable control)
    std = 8.0 + 1.5 * max(0.0, hba1c - 6.5)
    std = float(np.clip(std, 6.0, 25.0))

    return blended, std


def _ehr_to_si(bmi: float, hba1c: float, duration_years: float) -> Tuple[float, float]:
    """Map BMI, HbA1c, and disease duration to an SI (insulin sensitivity) prior.

    SI decreases with:
        - Higher BMI (adiposity → peripheral insulin resistance)
        - Higher HbA1c (chronic hyperglycaemia → glucotoxicity)
        - Longer duration (progressive beta-cell dysfunction)

    Returns (SI_mean, SI_std) in relative units where 1.0 = reference
    (lean, well-controlled, newly diagnosed T2D).
    """
    # Reference: BMI 25, HbA1c 6.5, duration 2y → SI = 1.0
    si_mean = (
        1.0
        - 0.025 * max(0.0, bmi - 25.0)       # -2.5% per BMI unit above 25
        - 0.040 * max(0.0, hba1c - 6.5)       # -4.0% per HbA1c % above 6.5
        - 0.012 * max(0.0, duration_years - 2) # -1.2% per year of T2D beyond 2y
    )
    si_mean = float(np.clip(si_mean, 0.20, 1.60))

    # Uncertainty grows with disease severity
    std = 0.08 + 0.015 * max(0.0, hba1c - 6.5)
    std = float(np.clip(std, 0.06, 0.22))

    return si_mean, std


def _ehr_to_circ_amp(age: int, sglt2i: bool) -> Tuple[float, float]:
    """Circadian amplitude prior from age and SGLT2i use.

    Older patients and SGLT2i users tend to have attenuated dawn phenomenon.
    """
    base_amp = 12.0
    age_effect = -0.05 * max(0.0, age - 50)
    sglt2i_effect = -2.5 if sglt2i else 0.0
    amp_mean = float(np.clip(base_amp + age_effect + sglt2i_effect, 3.0, 22.0))
    amp_std = 3.5
    return amp_mean, amp_std


def _ehr_to_kabs(metformin: bool) -> Tuple[float, float]:
    """Meal carb absorption rate prior.

    Metformin slightly slows glucose absorption via gut mechanisms.
    """
    kabs_mean = 0.022 if not metformin else 0.018  # fraction absorbed per minute
    kabs_std = 0.005
    return kabs_mean, kabs_std


def _ehr_to_meal_carb(bmi: float, age: int) -> Tuple[float, float]:
    """Prior on grams of carbohydrate per main meal.

    Higher BMI patients tend to consume larger meals on average.
    """
    meal_mean = 45.0 + 1.2 * max(0.0, bmi - 25.0) - 0.15 * max(0.0, age - 60)
    meal_mean = float(np.clip(meal_mean, 25.0, 90.0))
    meal_std = 12.0
    return meal_mean, meal_std


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prior_from_ehr(
    hba1c: float,
    bmi: float,
    diabetes_duration_years: float,
    fbg_mgdl: float,
    age: int,
    metformin: bool = True,
    sglt2i: bool = False,
) -> TwinPrior:
    """Build a TwinPrior from EHR clinical variables.

    Parameters
    ----------
    hba1c : float
        HbA1c in percent (e.g. 7.2).
    bmi : float
        Body Mass Index (kg/m^2).
    diabetes_duration_years : float
        Years since T2D diagnosis.
    fbg_mgdl : float
        Historical fasting blood glucose (mg/dL).
    age : int
        Patient age in years.
    metformin : bool
        Whether patient is on metformin.
    sglt2i : bool
        Whether patient is on an SGLT2 inhibitor.

    Returns
    -------
    TwinPrior
        Gaussian prior distributions over twin parameters.
    """
    Gb_mean, Gb_std = _ehr_to_gb(hba1c, fbg_mgdl)
    SI_mean, SI_std = _ehr_to_si(bmi, hba1c, diabetes_duration_years)
    circ_amp_mean, circ_amp_std = _ehr_to_circ_amp(age, sglt2i)
    kabs_mean, kabs_std = _ehr_to_kabs(metformin)
    meal_carb_mean, meal_carb_std = _ehr_to_meal_carb(bmi, age)

    prior = TwinPrior(
        Gb_mean=Gb_mean,
        Gb_std=Gb_std,
        SI_mean=SI_mean,
        SI_std=SI_std,
        circ_amp_mean=circ_amp_mean,
        circ_amp_std=circ_amp_std,
        kabs_mean=kabs_mean,
        kabs_std=kabs_std,
        meal_carb_mean=meal_carb_mean,
        meal_carb_std=meal_carb_std,
    )

    logger.debug(
        "Built TwinPrior | Gb=%.1f+/-%.1f | SI=%.3f+/-%.3f | circ_amp=%.1f",
        Gb_mean, Gb_std, SI_mean, SI_std, circ_amp_mean,
    )
    return prior
