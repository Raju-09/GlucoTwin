"""Pydantic schemas for the GlucoTwin REST API.

Enforces strict input validation, multimodal stream context,
and explicit clinical/research response contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request payload for glucose trajectory forecasting."""

    patient_id: str = Field(..., description="Patient pseudonym identifier (e.g. SYNTH_001)")
    origin_time: str = Field(..., description="Timestamp of the forecast origin (ISO 8601 UTC)")
    timestamps: List[str] = Field(..., description="Historical timestamps for each reading (ISO 8601 UTC)")
    glucose_readings: List[float] = Field(..., description="Historical glucose readings in mg/dL")
    current_time: Optional[str] = Field(None, description="Optional reference current wall-clock time for freshness evaluation")
    enable_adaptation: bool = Field(False, description="Whether to apply patient-specific calibration if available")
    ehr_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional static EHR context overrides (age, bmi, baseline_hba1c, historical_fbg_mgdl, etc.)",
    )
    wearable_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional wearable telemetry overrides (heart_rate, steps, hrv_rmssd, sleep_state)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "SYNTH_001",
                "origin_time": "2024-03-15T12:00:00Z",
                "timestamps": [
                    "2024-03-15T10:05:00Z", "2024-03-15T10:10:00Z", "2024-03-15T10:15:00Z",
                    "2024-03-15T10:20:00Z", "2024-03-15T10:25:00Z", "2024-03-15T10:30:00Z",
                    "2024-03-15T10:35:00Z", "2024-03-15T10:40:00Z", "2024-03-15T10:45:00Z",
                    "2024-03-15T10:50:00Z", "2024-03-15T10:55:00Z", "2024-03-15T11:00:00Z",
                    "2024-03-15T11:05:00Z", "2024-03-15T11:10:00Z", "2024-03-15T11:15:00Z",
                    "2024-03-15T11:20:00Z", "2024-03-15T11:25:00Z", "2024-03-15T11:30:00Z",
                    "2024-03-15T11:35:00Z", "2024-03-15T11:40:00Z", "2024-03-15T11:45:00Z",
                    "2024-03-15T11:50:00Z", "2024-03-15T11:55:00Z", "2024-03-15T12:00:00Z"
                ],
                "glucose_readings": [
                    118.0, 119.5, 121.0, 122.5, 124.0, 125.0,
                    127.5, 129.0, 131.0, 132.5, 135.0, 137.0,
                    139.5, 142.0, 144.5, 147.0, 149.0, 151.5,
                    154.0, 156.0, 158.5, 160.0, 161.5, 163.0
                ],
                "current_time": "2024-03-15T12:02:00Z",
                "enable_adaptation": False
            }
        }
    }


class TrajectoryPoint(BaseModel):
    """Projected trajectory coordinate at an intermediate horizon step."""

    minutes_ahead: int
    target_time: str
    projected_glucose: float


class ForecastResponse(BaseModel):
    """Structured response contract for glucose forecasting."""

    patient_id: str
    origin_time: str
    target_time: str
    horizon_minutes: int
    predicted_glucose: Optional[float] = Field(
        None,
        description="Predicted future glucose (mg/dL) at horizon. MUST be None if reliability is withheld.",
    )
    intermediate_trajectory: List[TrajectoryPoint] = Field(
        default_factory=list,
        description="Multi-horizon trajectory projection (t+30, t+60, t+90, t+120 min)",
    )
    reliability_status: Literal["available", "degraded", "withheld"]
    reason_code: str
    advisory_message: str
    quality_score_pct: float
    freshness_minutes: Optional[float]
    hypoglycemia_risk: str = Field("Low", description="Risk of glucose dipping < 70 mg/dL (Low, Moderate, Elevated)")
    hyperglycemia_risk: str = Field("Low", description="Risk of glucose exceeding 180 mg/dL (Low, Moderate, Elevated)")
    clinical_risk_advisory: str = Field("", description="Clinician advisory regarding trajectory trend and risk")
    model_version: str = "v1.2"
    adaptation_applied: bool
    is_synthetic: bool = True
    disclaimer: str = (
        "RESEARCH PROTOTYPE ONLY - NOT A MEDICAL DEVICE. "
        "Do not use for diagnosis, treatment, or insulin dosing."
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    model_loaded: bool
    model_name: Optional[str]


class ModelInfoResponse(BaseModel):
    """Model metadata and held-out evidence response."""

    model_name: str
    version: str
    horizon_minutes: int
    test_mae_mgdl: float
    test_rmse_mgdl: float
    persistence_mae_mgdl: float
    relative_improvement_pct: float
    training_samples: int
    n_features: int
    feature_names: List[str]
    is_synthetic_benchmark: bool = True


class DemoPatient(BaseModel):
    """Summary of a synthetic demo patient."""

    patient_id: str
    label: str
    condition: str
    days_recorded: int
    median_glucose_mgdl: float
    adaptation_recommended: bool
    notes: str


class TimelinePoint(BaseModel):
    """Single time-series point."""

    timestamp: str
    glucose: float
    meal_carbs_g: Optional[float] = None


class PatientTimelineResponse(BaseModel):
    """Patient historical timeline for dashboard plotting."""

    patient_id: str
    total_readings: int
    time_start: str
    time_end: str
    readings: List[TimelinePoint]
    is_synthetic: bool = True


class TwinStateResponse(BaseModel):
    """Formal Digital Twin State representation S(t) = (θ_p, h_t, δ_t)."""

    patient_id: str
    state_version: str
    sequence_step: int
    phenotype: Dict[str, Any]
    dynamic_history: Dict[str, Any]
    integrity_state: Dict[str, Any]
    derived_kinematics: Dict[str, Any]
    stream_freshness: Dict[str, str]
    is_synthetic: bool = True


class ObservationPacketRequest(BaseModel):
    """Observation packet for online digital twin state updates."""

    timestamp: str
    glucose_mgdl: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    hrv_rmssd_ms: Optional[float] = None
    steps: Optional[int] = None
    sleep_state: Optional[str] = "awake"
    current_wallclock_time: Optional[str] = None
