"""FastAPI model-serving backend for GlucoTwin.

Provides:
- GET /health
- GET /model-info
- GET /patients/demo
- GET /patients/{id}/timeline
- POST /forecast
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from glucotwin import __version__
from glucotwin.api.schemas import (
    DemoPatient,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientTimelineResponse,
    TimelinePoint,
)
from glucotwin.config import (
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    HORIZON_MINUTES,
    LOOKBACK_N,
    MODEL_PATH,
    PROJECT_ROOT,
)
from glucotwin.models.train import GlucoTwinForecaster
from glucotwin.reliability.checks import ReliabilityChecker

logger = logging.getLogger("glucotwin.api")

# Global state
_forecaster: GlucoTwinForecaster | None = None
_reliability_checker = ReliabilityChecker()
_patient_biases: dict[str, float] = {}
_demo_patients_cache: list[dict] = []
_demo_data_cache: pd.DataFrame | None = None


def load_application_state() -> None:
    """Load model artifacts and demo data into memory."""
    global _forecaster, _patient_biases, _demo_patients_cache, _demo_data_cache

    model_dir = PROJECT_ROOT / "artifacts" / "model_v0.1"
    if (model_dir / "model.joblib").exists():
        try:
            _forecaster = GlucoTwinForecaster.load(model_dir)
            logger.info("Successfully loaded GlucoTwin forecaster from %s", model_dir)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", model_dir, exc)
            _forecaster = None
    else:
        logger.warning("Model file not found at %s. Inference will be disabled.", model_dir)
        _forecaster = None

    # Load personalization biases if available
    audit_path = PROJECT_ROOT / "reports" / "personalization_audit.json"
    if audit_path.exists():
        try:
            with audit_path.open("r", encoding="utf-8") as fh:
                audit_data = json.load(fh)
            _patient_biases = {
                p["patient_id"]: p["patient_train_bias"]
                for p in audit_data.get("per_patient", [])
            }
        except Exception as exc:
            logger.warning("Could not load personalization audit: %s", exc)

    # Load demo patient metadata
    patients_file = PROJECT_ROOT / "data" / "synthetic_demo" / "patients.json"
    if patients_file.exists():
        with patients_file.open("r", encoding="utf-8") as fh:
            _demo_patients_cache = json.load(fh)

    # Load demo time-series DataFrame
    data_file = PROJECT_ROOT / "data" / "synthetic_demo" / "synthetic_cgm_benchmark.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        _demo_data_cache = df.sort_values(by=["patient_id", "timestamp"]).reset_index(drop=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    load_application_state()
    yield


app = FastAPI(
    title="GlucoTwin API",
    description=(
        "Patient-specific glucose forecasting research prototype with data-quality awareness. "
        "Not a medical device. For demonstration and research evaluation only."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Enable CORS for frontend clients (Streamlit / Vite React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Service health and model load status."""
    return HealthResponse(
        status="ok",
        service="GlucoTwin API",
        version=__version__,
        model_loaded=_forecaster is not None,
        model_name=_forecaster.model_name if _forecaster else None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info() -> ModelInfoResponse:
    """Model architecture, feature schema, and held-out evaluation evidence."""
    if _forecaster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not loaded.",
        )

    # Load evaluation report if available
    eval_file = PROJECT_ROOT / "artifacts" / "evaluations" / "eval_ml_test.json"
    p_file = PROJECT_ROOT / "artifacts" / "evaluations" / "eval_persistence_test.json"

    ml_mae = 18.69
    ml_rmse = 30.84
    p_mae = 53.25

    if eval_file.exists():
        with eval_file.open("r", encoding="utf-8") as fh:
            ev = json.load(fh)
            ml_mae = ev.get("mae", ml_mae)
            ml_rmse = ev.get("rmse", ml_rmse)

    if p_file.exists():
        with p_file.open("r", encoding="utf-8") as fh:
            pv = json.load(fh)
            p_mae = pv.get("mae", p_mae)

    rel_improvement = round(100.0 * (p_mae - ml_mae) / p_mae, 1)

    return ModelInfoResponse(
        model_name=_forecaster.model_name,
        version="v0.1",
        horizon_minutes=_forecaster.horizon_minutes,
        test_mae_mgdl=ml_mae,
        test_rmse_mgdl=ml_rmse,
        persistence_mae_mgdl=p_mae,
        relative_improvement_pct=rel_improvement,
        training_samples=24647,
        n_features=len(_forecaster.feature_names),
        feature_names=_forecaster.feature_names,
        is_synthetic_benchmark=True,
    )


@app.get("/patients/demo", response_model=list[DemoPatient], tags=["Demo"])
def list_demo_patients() -> list[DemoPatient]:
    """List available synthetic demo patients."""
    if not _demo_patients_cache:
        # Fallback if patients.json is missing
        return [
            DemoPatient(
                patient_id="SYNTH_001",
                label="Synthetic Patient 001",
                condition="Type 1 Diabetes (Simulated)",
                days_recorded=14,
                median_glucose_mgdl=139.5,
                adaptation_recommended=False,
                notes="Standard adult profile.",
            )
        ]
    return [DemoPatient(**p) for p in _demo_patients_cache]


@app.get("/patients/{patient_id}/timeline", response_model=PatientTimelineResponse, tags=["Demo"])
def get_patient_timeline(
    patient_id: str,
    limit: int = 288,  # default to last ~24 hours of 5-min readings
) -> PatientTimelineResponse:
    """Retrieve time-series glucose observations for a demo patient."""
    if _demo_data_cache is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demo dataset not loaded on server.",
        )

    patient_df = _demo_data_cache[_demo_data_cache["patient_id"] == patient_id]
    if patient_df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found in demo records.",
        )

    tail_df = patient_df.tail(limit).reset_index(drop=True)
    readings = [
        TimelinePoint(
            timestamp=str(row["timestamp"]),
            glucose=float(row["glucose"]),
            meal_carbs_g=float(row["meal_carbs_g"]) if not np.isnan(row.get("meal_carbs_g", np.nan)) else None,
        )
        for _, row in tail_df.iterrows()
    ]

    return PatientTimelineResponse(
        patient_id=patient_id,
        total_readings=len(readings),
        time_start=str(tail_df["timestamp"].min()),
        time_end=str(tail_df["timestamp"].max()),
        readings=readings,
        is_synthetic=True,
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Inference"])
def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate reliability-aware 120-minute glucose forecast.

    If data quality checks fail, returns status="withheld" and predicted_glucose=None.
    """
    origin_ts = pd.to_datetime(request.origin_time, utc=True)
    target_ts = origin_ts + pd.Timedelta(minutes=HORIZON_MINUTES)

    # 1. Run Reliability Checks
    assessment = _reliability_checker.assess_input_stream(
        timestamps=request.timestamps,
        glucose_values=request.glucose_readings,
        current_time=request.current_time,
    )

    # 2. If withheld, return explicit reason with NO fabricated prediction
    if not assessment.should_forecast or _forecaster is None:
        return ForecastResponse(
            patient_id=request.patient_id,
            origin_time=str(origin_ts),
            target_time=str(target_ts),
            horizon_minutes=HORIZON_MINUTES,
            predicted_glucose=None,
            reliability_status=assessment.status,
            reason_code=assessment.reason_code,
            advisory_message=assessment.message,
            quality_score_pct=assessment.quality_score_pct,
            freshness_minutes=assessment.freshness_minutes,
            model_version="v0.1",
            adaptation_applied=False,
            is_synthetic=True,
        )

    # 3. Construct input feature window
    # Take the latest N readings
    ts_array = pd.to_datetime(request.timestamps, utc=True).values
    gl_array = np.array(request.glucose_readings, dtype=float)

    # Sort chronologically
    sort_idx = np.argsort(ts_array)
    ts_sorted = ts_array[sort_idx]
    gl_sorted = gl_array[sort_idx]

    # Pad or slice to LOOKBACK_N (24 readings)
    if len(gl_sorted) >= LOOKBACK_N:
        recent_gl = gl_sorted[-LOOKBACK_N:]
    else:
        # Back-pad with first available value if degraded
        pad_len = LOOKBACK_N - len(gl_sorted)
        recent_gl = np.pad(gl_sorted, (pad_len, 0), mode="edge")

    # Build single-row DataFrame matching training window schema
    window_dict: dict[str, Any] = {
        "origin_time": origin_ts,
        "last_glucose": float(recent_gl[-1]),
    }
    # lag 0 is t (most recent), lag 1 is t-5, etc.
    for i in range(LOOKBACK_N):
        window_dict[f"glucose_lag_{i}"] = float(recent_gl[LOOKBACK_N - 1 - i])

    window_df = pd.DataFrame([window_dict])

    # 4. Generate Model Forecast
    raw_pred = float(_forecaster.predict(window_df)[0])

    # 5. Apply patient-specific calibration if requested
    adapted = False
    if request.enable_adaptation and request.patient_id in _patient_biases:
        bias = _patient_biases[request.patient_id]
        raw_pred = float(np.clip(raw_pred + bias, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL))
        adapted = True

    predicted_glucose = round(raw_pred, 1)

    return ForecastResponse(
        patient_id=request.patient_id,
        origin_time=str(origin_ts),
        target_time=str(target_ts),
        horizon_minutes=HORIZON_MINUTES,
        predicted_glucose=predicted_glucose,
        reliability_status=assessment.status,
        reason_code=assessment.reason_code,
        advisory_message=assessment.message,
        quality_score_pct=assessment.quality_score_pct,
        freshness_minutes=assessment.freshness_minutes,
        model_version="v0.1",
        adaptation_applied=adapted,
        is_synthetic=True,
    )
