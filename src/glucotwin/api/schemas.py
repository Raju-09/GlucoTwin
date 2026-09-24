"""Pydantic schemas for the GlucoTwin REST API.

Enforces strict input validation and explicit response contracts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request payload for glucose trajectory forecasting."""

    patient_id: str = Field(..., description="Patient pseudonym identifier (e.g. SYNTH_001)")
    origin_time: str = Field(..., description="Timestamp of the forecast origin (ISO 8601 UTC)")
    timestamps: list[str] = Field(..., description="Historical timestamps for each reading (ISO 8601 UTC)")
    glucose_readings: list[float] = Field(..., description="Historical glucose readings in mg/dL")
    current_time: str | None = Field(None, description="Optional reference current wall-clock time for freshness evaluation")
    enable_adaptation: bool = Field(False, description="Whether to apply patient-specific calibration if available")


class ForecastResponse(BaseModel):
    """Structured response contract for glucose forecasting."""

    patient_id: str
    origin_time: str
    target_time: str
    horizon_minutes: int
    predicted_glucose: float | None = Field(
        None,
        description="Predicted future glucose (mg/dL). MUST be None if reliability is withheld.",
    )
    reliability_status: Literal["available", "degraded", "withheld"]
    reason_code: str
    advisory_message: str
    quality_score_pct: float
    freshness_minutes: float | None
    model_version: str
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
    model_name: str | None


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
    feature_names: list[str]
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
    meal_carbs_g: float | None = None


class PatientTimelineResponse(BaseModel):
    """Patient historical timeline for dashboard plotting."""

    patient_id: str
    total_readings: int
    time_start: str
    time_end: str
    readings: list[TimelinePoint]
    is_synthetic: bool = True
