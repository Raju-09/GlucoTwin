"""Formal Digital Twin State Representation for GlucoTwin.

Mathematical Definition:
    S_t = (θ_p, h_t, δ_t)
where:
    - θ_p: Patient Static Phenotype (EHR, clinical intake, demographic & baseline parameters)
    - h_t: Patient Dynamic History (Causal window of continuous sensor streams & derived kinematics)
    - δ_t: Twin Integrity State (Data quality, freshness, drift & gating status)

All state representations are immutable-friendly, fully serializable, and purely causal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import math
import numpy as np


class IntegrityStatus(str, Enum):
    """Integrity status levels for the digital twin state machine."""

    AVAILABLE = "AVAILABLE"      # Stream nominal, all quality gates pass
    DEGRADED = "DEGRADED"        # Minor packet loss or mild jitter, model operable with warning
    DRIFTING = "DRIFTING"        # Consecutive unexpected residual error or physiological drift
    WITHHELD = "WITHHELD"        # Hard quality failure (stale, extreme gap, hardware spike)
    RECOVERING = "RECOVERING"    # Returning to nominal after withheld/degraded state


@dataclass
class PatientStaticPhenotype:
    """Static and slowly-varying EHR phenotype parameters (θ_p)."""

    patient_id: str
    age: int
    sex: str
    bmi: float
    baseline_hba1c: float
    diabetes_duration_years: float
    historical_fbg_mgdl: float
    hypertension_flag: int = 0
    metformin_flag: int = 0
    sglt2i_flag: int = 0
    dawn_phenomenon_flag: int = 0
    data_age_description: str = "Updated: 180 days ago (Historical EHR/Intake)"
    phenotype_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatientDynamicHistory:
    """Causal window of dynamic observation history (h_t)."""

    lookback_n: int = 24  # Standard 2-hour window at 5-min cadence
    last_observation_time: Optional[str] = None
    timestamps: List[str] = field(default_factory=list)
    glucose_readings: List[float] = field(default_factory=list)
    heart_rate: List[float] = field(default_factory=list)
    hrv_rmssd: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    sleep_state: str = "awake"
    cgm_freshness_minutes: float = 0.0
    wearable_freshness_minutes: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivedFeatures:
    """Causally computed kinematic and statistical features derived from h_t and θ_p."""

    current_glucose: float = 120.0
    glucose_velocity_mgdl_min: float = 0.0          # 1st discrete derivative dG/dt
    glucose_acceleration_mgdl_min2: float = 0.0     # 2nd discrete derivative d^2G/dt^2
    rolling_mean_1h: float = 120.0
    rolling_std_1h: float = 0.0
    time_in_illustrative_range_pct: float = 100.0   # % readings in [70, 180] mg/dL
    circadian_sin: float = 0.0
    circadian_cos: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwinIntegrityState:
    """Twin operational integrity and data quality state (δ_t)."""

    status: IntegrityStatus = IntegrityStatus.AVAILABLE
    reason_code: str = "NOMINAL_STREAM"
    quality_score_pct: float = 100.0
    advisory_message: str = "Continuous telemetry verified. Forecasting enabled."
    active_flags: List[str] = field(default_factory=list)
    consecutive_recovery_count: int = 0
    drift_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["status"] = self.status.value
        return res


@dataclass
class StateMetadata:
    """Metadata tracking deterministic execution and state provenance."""

    state_version: str = "2.0.0"
    sequence_step: int = 0
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_synthetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationPacket:
    """Typed incoming observation packet representing a new multi-channel telemetry reading."""

    timestamp: str
    glucose_mgdl: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    hrv_rmssd_ms: Optional[float] = None
    steps: Optional[int] = None
    sleep_state: Optional[str] = None
    sensor_status_flag: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatientDigitalState:
    """Complete, self-contained Digital Twin State S(t) = (θ_p, h_t, δ_t)."""

    phenotype: PatientStaticPhenotype           # θ_p
    history: PatientDynamicHistory             # h_t
    integrity: TwinIntegrityState              # δ_t
    derived: DerivedFeatures                   # Computed kinematics
    metadata: StateMetadata                    # Provenance & step counter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phenotype": self.phenotype.to_dict(),
            "history": self.history.to_dict(),
            "integrity": self.integrity.to_dict(),
            "derived": self.derived.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    def get_stream_freshness_summary(self) -> Dict[str, str]:
        """Returns human-readable freshness descriptions for all multimodal streams."""
        return {
            "ehr": self.phenotype.data_age_description,
            "cgm": f"Updated: {self.history.cgm_freshness_minutes:.1f} min ago ({'Live' if self.history.cgm_freshness_minutes < 10 else 'Delayed'})",
            "wearables": f"Updated: {self.history.wearable_freshness_minutes:.1f} min ago (Continuous)",
        }

    def to_multimodal_feature_dict(self) -> Dict[str, float]:
        """Converts the current state into an aligned feature vector dictionary matching the trained ML pipeline."""
        feats: Dict[str, float] = {}

        # 1. CGM Lags (24 lags)
        gl = self.history.glucose_readings
        lookback = self.history.lookback_n
        if len(gl) < lookback:
            padded_gl = [gl[0] if len(gl) > 0 else 120.0] * (lookback - len(gl)) + list(gl)
        else:
            padded_gl = list(gl[-lookback:])

        feats["last_glucose"] = float(padded_gl[-1])
        for i in range(lookback):
            feats[f"glucose_lag_{i}"] = float(padded_gl[lookback - 1 - i])

        # CGM Kinematics & Statistics
        feats["glucose_mean_1h"] = float(np.mean(padded_gl[-12:]))
        feats["glucose_std_1h"] = float(np.std(padded_gl[-12:]))
        feats["glucose_min_1h"] = float(np.min(padded_gl[-12:]))
        feats["glucose_max_1h"] = float(np.max(padded_gl[-12:]))
        feats["glucose_range_1h"] = feats["glucose_max_1h"] - feats["glucose_min_1h"]
        feats["glucose_velocity_5m"] = float(self.derived.glucose_velocity_mgdl_min * 5.0)
        feats["glucose_velocity_15m"] = float(
            (padded_gl[-1] - padded_gl[-4]) / 15.0 if len(padded_gl) >= 4 else 0.0
        )
        feats["glucose_velocity_30m"] = float(
            (padded_gl[-1] - padded_gl[-7]) / 30.0 if len(padded_gl) >= 7 else 0.0
        )
        feats["glucose_acceleration"] = float(self.derived.glucose_acceleration_mgdl_min2)
        feats["time_in_range_70_180"] = float(self.derived.time_in_illustrative_range_pct / 100.0)
        feats["time_below_70"] = float(np.mean(np.array(padded_gl) < 70.0))
        feats["time_above_180"] = float(np.mean(np.array(padded_gl) > 180.0))
        feats["time_above_250"] = float(np.mean(np.array(padded_gl) > 250.0))
        feats["cgm_sample_count"] = float(len(gl))
        feats["cgm_missing_rate"] = float(1.0 - (len(gl) / lookback))

        # Circadian
        feats["circadian_sin"] = float(self.derived.circadian_sin)
        feats["circadian_cos"] = float(self.derived.circadian_cos)

        # 2. Static EHR Features
        feats["ehr_age"] = float(self.phenotype.age)
        feats["ehr_bmi"] = float(self.phenotype.bmi)
        feats["ehr_baseline_hba1c"] = float(self.phenotype.baseline_hba1c)
        feats["ehr_diabetes_duration_years"] = float(self.phenotype.diabetes_duration_years)
        feats["ehr_historical_fbg"] = float(self.phenotype.historical_fbg_mgdl)
        feats["ehr_hypertension_flag"] = float(self.phenotype.hypertension_flag)
        feats["ehr_metformin_flag"] = float(self.phenotype.metformin_flag)
        feats["ehr_sglt2i_flag"] = float(self.phenotype.sglt2i_flag)
        feats["ehr_dawn_flag"] = float(self.phenotype.dawn_phenomenon_flag)

        # 3. Dynamic Wearable Features
        hr = self.history.heart_rate or [72.0]
        hrv = self.history.hrv_rmssd or [45.0]
        steps = self.history.steps or [0]

        feats["wearable_hr_latest"] = float(hr[-1])
        feats["wearable_hr_mean_1h"] = float(np.mean(hr[-12:]))
        feats["wearable_hr_std_1h"] = float(np.std(hr[-12:]))
        feats["wearable_hrv_rmssd_latest"] = float(hrv[-1])
        feats["wearable_hrv_mean_1h"] = float(np.mean(hrv[-12:]))
        feats["wearable_steps_sum_1h"] = float(np.sum(steps[-12:]))
        feats["wearable_sleep_flag"] = 1.0 if self.history.sleep_state != "awake" else 0.0

        return feats
