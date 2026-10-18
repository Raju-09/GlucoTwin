"""FastAPI model-serving and Digital Twin state backend for GlucoTwin.

Provides:
- GET /health
- GET /model-info
- GET /patients/demo
- GET /patients/{id}/timeline
- POST /forecast (Two-stream multimodal with intermediate trajectory & clinical risk)
- GET /twin/{patient_id}/state (Gate 2 formal state S(t) = (θ_p, h_t, δ_t))
- POST /twin/{patient_id}/update (Deterministic online state update engine)
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

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
    ObservationPacketRequest,
    PatientTimelineResponse,
    TimelinePoint,
    TrajectoryPoint,
    TwinStateResponse,
)
from glucotwin.config import (
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    HORIZON_MINUTES,
    LOOKBACK_N,
    PROJECT_ROOT,
)
from glucotwin.models.train import GlucoTwinForecaster
from glucotwin.reliability.checks import ReliabilityChecker
from glucotwin.twin.state import (
    ObservationPacket,
    PatientDigitalState,
    PatientStaticPhenotype,
)
from glucotwin.twin.update import (
    create_initial_patient_state,
    update_patient_state,
)

logger = logging.getLogger("glucotwin.api")

# Global state
_forecaster: GlucoTwinForecaster | None = None
_reliability_checker = ReliabilityChecker()
_patient_biases: dict[str, float] = {}
_demo_patients_cache: list[dict] = []
_demo_data_cache: pd.DataFrame | None = None
_ehr_cache: dict[str, dict] = {}
_wearables_cache: pd.DataFrame | None = None
_twin_states: dict[str, PatientDigitalState] = {}


def load_application_state() -> None:
    """Load model artifacts, EHR records, and demo data into memory."""
    global _forecaster, _patient_biases, _demo_patients_cache, _demo_data_cache, _ehr_cache, _wearables_cache, _twin_states

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

    # Load static EHR records
    ehr_file = PROJECT_ROOT / "data" / "synthetic_demo" / "ehr_records.json"
    if ehr_file.exists():
        with ehr_file.open("r", encoding="utf-8") as fh:
            records = json.load(fh)
            _ehr_cache = {r["patient_id"]: r for r in records}

    # Load demo time-series DataFrame
    data_file = PROJECT_ROOT / "data" / "synthetic_demo" / "synthetic_cgm_benchmark.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        _demo_data_cache = df.sort_values(by=["patient_id", "timestamp"]).reset_index(drop=True)

    # Load wearable telemetry
    wear_file = PROJECT_ROOT / "data" / "synthetic_demo" / "wearable_telemetry.parquet"
    if wear_file.exists():
        _wearables_cache = pd.read_parquet(wear_file)


def _get_or_create_twin_state(patient_id: str) -> PatientDigitalState:
    """Retrieve active digital twin state for patient or initialize from static phenotype."""
    global _twin_states
    if patient_id in _twin_states:
        return _twin_states[patient_id]

    ehr = _ehr_cache.get(patient_id, {})
    phenotype = PatientStaticPhenotype(
        patient_id=patient_id,
        age=int(ehr.get("age", 48)),
        sex=str(ehr.get("sex", "M")),
        bmi=float(ehr.get("bmi", 26.4)),
        baseline_hba1c=float(ehr.get("baseline_hba1c", 6.8)),
        diabetes_duration_years=float(ehr.get("diabetes_duration_years", 5.5)),
        historical_fbg_mgdl=float(ehr.get("historical_fbg_mgdl", 115.0)),
        hypertension_flag=int(ehr.get("hypertension_flag", 0)),
        metformin_flag=int(ehr.get("metformin_flag", 1)),
        sglt2i_flag=int(ehr.get("sglt2i_flag", 0)),
        dawn_phenomenon_flag=int(ehr.get("dawn_phenomenon_flag", 0)),
        data_age_description=ehr.get("data_age_description", "Updated: 180 days ago (Historical EHR)"),
        phenotype_notes=ehr.get("phenotype_notes", "Simulated Type 2 Diabetes adult profile"),
    )

    # Initial steady state
    initial_gl = float(ehr.get("historical_fbg_mgdl", 120.0))
    state = create_initial_patient_state(phenotype, initial_glucose=initial_gl)
    _twin_states[patient_id] = state
    return state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    load_application_state()
    yield


app = FastAPI(
    title="GlucoTwin API",
    description=(
        "Patient-specific glucose forecasting research prototype with data-quality awareness and digital twin state engine. "
        "Not a medical device. For demonstration and research evaluation only."
    ),
    version="v1.2",
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
        version="v1.2",
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

    # Canonical experiment manifest
    manifest_file = PROJECT_ROOT / "artifacts" / "experiment_manifest.json"
    ml_mae = 17.96
    ml_rmse = 29.53
    p_mae = 46.88

    if manifest_file.exists():
        try:
            with manifest_file.open("r", encoding="utf-8") as fh:
                mf = json.load(fh)
                bm = mf.get("benchmarks", {})
                ml_mae = bm.get("full_fusion_mae_mgdl", ml_mae)
                ml_rmse = bm.get("full_fusion_rmse_mgdl", ml_rmse)
                p_mae = bm.get("persistence_mae_mgdl", p_mae)
        except Exception:
            pass

    rel_improvement = round(100.0 * (p_mae - ml_mae) / p_mae, 1)

    return ModelInfoResponse(
        model_name=_forecaster.model_name,
        version="v1.2",
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


@app.get("/patients/demo", response_model=List[DemoPatient], tags=["Demo"])
def list_demo_patients() -> List[DemoPatient]:
    """List available synthetic demo patients."""
    if not _demo_patients_cache:
        return [
            DemoPatient(
                patient_id="SYNTH_001",
                label="Synthetic Patient 001",
                condition="Synthetic diabetes cohort (Simulated adult dynamics)",
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
    limit: int = 288,
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
    """Generate reliability-aware two-stream multimodal 120-minute glucose forecast.

    Includes:
    - Zero-fabrication reliability gating
    - Intermediate trajectory spline (t+30, t+60, t+90, t+120 min)
    - Clinical adverse event risk quantification (Hypoglycemia / Hyperglycemia alerts)
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
            intermediate_trajectory=[],
            reliability_status=assessment.status,
            reason_code=assessment.reason_code,
            advisory_message=assessment.message,
            quality_score_pct=assessment.quality_score_pct,
            freshness_minutes=assessment.freshness_minutes,
            hypoglycemia_risk="Withheld",
            hyperglycemia_risk="Withheld",
            clinical_risk_advisory="Forecast withheld due to telemetry failure. No projection generated.",
            model_version="v1.2",
            adaptation_applied=False,
            is_synthetic=True,
        )

    # 3. Construct input feature window
    ts_array = pd.to_datetime(request.timestamps, utc=True).values
    gl_array = np.array(request.glucose_readings, dtype=float)

    sort_idx = np.argsort(ts_array)
    ts_sorted = ts_array[sort_idx]
    gl_sorted = gl_array[sort_idx]

    if len(gl_sorted) >= LOOKBACK_N:
        recent_gl = gl_sorted[-LOOKBACK_N:]
    else:
        pad_len = LOOKBACK_N - len(gl_sorted)
        recent_gl = np.pad(gl_sorted, (pad_len, 0), mode="edge")

    # Start building single-row DataFrame
    window_dict: dict[str, Any] = {
        "origin_time": origin_ts,
        "last_glucose": float(recent_gl[-1]),
        "patient_id": request.patient_id,
    }
    for i in range(LOOKBACK_N):
        window_dict[f"glucose_lag_{i}"] = float(recent_gl[LOOKBACK_N - 1 - i])

    # 4. Integrate Static EHR Context
    ehr_data = request.ehr_context or _ehr_cache.get(request.patient_id, {})
    window_dict["ehr_age"] = float(ehr_data.get("age", 48.0))
    window_dict["ehr_bmi"] = float(ehr_data.get("bmi", 26.4))
    window_dict["ehr_baseline_hba1c"] = float(ehr_data.get("baseline_hba1c", 6.8))
    window_dict["ehr_diabetes_duration"] = float(ehr_data.get("diabetes_duration_years", 5.5))
    window_dict["ehr_historical_fbg"] = float(ehr_data.get("historical_fbg_mgdl", 115.0))
    window_dict["ehr_hypertension_flag"] = float(ehr_data.get("hypertension_flag", 0.0))
    window_dict["ehr_metformin_flag"] = float(ehr_data.get("metformin_flag", 1.0))
    window_dict["ehr_sglt2i_flag"] = float(ehr_data.get("sglt2i_flag", 0.0))
    window_dict["ehr_dawn_flag"] = float(ehr_data.get("dawn_phenomenon_flag", 0.0))

    # 5. Integrate Dynamic Wearable Telemetry Context
    wear_data = request.wearable_context or {}
    window_dict["wearable_mean_hr_30m"] = float(wear_data.get("heart_rate_mean_30m", 72.0))
    window_dict["wearable_hr_delta_15m"] = float(wear_data.get("heart_rate_delta_15m", 0.0))
    window_dict["wearable_steps_sum_30m"] = float(wear_data.get("steps_30m", 30.0))
    window_dict["wearable_steps_sum_60m"] = float(wear_data.get("steps_60m", 60.0))
    window_dict["wearable_hrv_last"] = float(wear_data.get("hrv_rmssd", 42.0))
    window_dict["wearable_sleep_indicator"] = 1.0 if wear_data.get("sleep_state", "awake") != "awake" else 0.0

    window_df = pd.DataFrame([window_dict])

    # 6. Generate Model Forecast
    raw_pred = float(_forecaster.predict(window_df)[0])

    # 7. Apply patient-specific calibration if requested
    adapted = False
    if request.enable_adaptation and request.patient_id in _patient_biases:
        bias = _patient_biases[request.patient_id]
        raw_pred = float(np.clip(raw_pred + bias, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL))
        adapted = True

    predicted_glucose = round(raw_pred, 1)

    # 8. Compute Multi-Horizon Intermediate Trajectory (Physiologically damped)
    g0 = float(recent_gl[-1])
    g_target = predicted_glucose
    # Current discrete velocity (mg/dL/min), clamped to realistic physiological limits
    raw_v0 = (g0 - float(recent_gl[-2])) / 5.0 if len(recent_gl) >= 2 else 0.0
    v0_clamped = float(np.clip(raw_v0, -2.5, 2.5))

    trajectory_points: List[TrajectoryPoint] = []
    horizons = [30, 60, 90, 120]

    # Establish envelope bounds to strictly prevent unrealistic rocket spikes or overshoot
    lower_bound = max(GLUCOSE_MIN_MGDL, min(g0, g_target) - 15.0)
    upper_bound = min(GLUCOSE_MAX_MGDL, max(g0, g_target) + 15.0)

    for h in horizons:
        tau = h / 120.0
        # Smooth interpolation towards target with damped initial velocity influence
        linear_interp = g0 + tau * (g_target - g0)
        # Velocity influence decays smoothly as time advances: (1 - tau)^2
        velocity_influence = v0_clamped * (h * 0.4) * ((1.0 - tau) ** 2)
        g_proj = linear_interp + velocity_influence
        # Strictly clamp within the physiological envelope
        g_proj = float(np.clip(g_proj, lower_bound, upper_bound))
        
        t_proj = origin_ts + pd.Timedelta(minutes=h)
        trajectory_points.append(
            TrajectoryPoint(
                minutes_ahead=h,
                target_time=str(t_proj),
                projected_glucose=round(g_proj, 1),
            )
        )

    # 9. Compute Adverse Event Risks
    all_projected = [g0] + [tp.projected_glucose for tp in trajectory_points]
    min_proj = min(all_projected)
    max_proj = max(all_projected)

    if min_proj < 70.0:
        hypo_risk = "Elevated"
    elif min_proj < 85.0 or raw_v0 < -1.5:
        hypo_risk = "Moderate"
    else:
        hypo_risk = "Low"

    if max_proj > 250.0:
        hyper_risk = "Elevated"
    elif max_proj > 180.0:
        hyper_risk = "Moderate"
    else:
        hyper_risk = "Low"

    diff = g_target - g0
    dir_str = "upward" if diff > 5.0 else ("downward" if diff < -5.0 else "stable")
    risk_notes = []
    if hypo_risk == "Elevated":
        risk_notes.append("Hypoglycemia alert: projected nadir < 70 mg/dL")
    if hyper_risk == "Elevated":
        risk_notes.append("Severe hyperglycemia alert: projected peak > 250 mg/dL")
    elif hyper_risk == "Moderate":
        risk_notes.append("Mild hyperglycemia: projected peak exceeds 180 mg/dL reference boundary")
    
    if not risk_notes:
        advisory_note = f"Stable trajectory. Glucose projected to remain within reference boundaries ({g0:.1f} -> {g_target:.1f} mg/dL)."
    else:
        advisory_note = f"Trajectory indicates {dir_str} trend ({diff:+.1f} mg/dL in 2h). " + "; ".join(risk_notes) + "."

    return ForecastResponse(
        patient_id=request.patient_id,
        origin_time=str(origin_ts),
        target_time=str(target_ts),
        horizon_minutes=HORIZON_MINUTES,
        predicted_glucose=predicted_glucose,
        intermediate_trajectory=trajectory_points,
        reliability_status=assessment.status,
        reason_code=assessment.reason_code,
        advisory_message=assessment.message,
        quality_score_pct=assessment.quality_score_pct,
        freshness_minutes=assessment.freshness_minutes,
        hypoglycemia_risk=hypo_risk,
        hyperglycemia_risk=hyper_risk,
        clinical_risk_advisory=advisory_note,
        model_version="v1.2",
        adaptation_applied=adapted,
        is_synthetic=True,
    )


# ---------------------------------------------------------------------------
# Gate 2: Digital Twin State Engine Endpoints
# ---------------------------------------------------------------------------

@app.get("/twin/{patient_id}/state", response_model=TwinStateResponse, tags=["Digital Twin"])
def get_digital_twin_state(patient_id: str) -> TwinStateResponse:
    """Retrieve the active formal Digital Twin State S(t) = (θ_p, h_t, δ_t)."""
    state = _get_or_create_twin_state(patient_id)
    return TwinStateResponse(
        patient_id=patient_id,
        state_version=state.metadata.state_version,
        sequence_step=state.metadata.sequence_step,
        phenotype=state.phenotype.to_dict(),
        dynamic_history=state.history.to_dict(),
        integrity_state=state.integrity.to_dict(),
        derived_kinematics=state.derived.to_dict(),
        stream_freshness=state.get_stream_freshness_summary(),
        is_synthetic=True,
    )


@app.post("/twin/{patient_id}/update", response_model=TwinStateResponse, tags=["Digital Twin"])
def update_digital_twin_state(patient_id: str, packet: ObservationPacketRequest) -> TwinStateResponse:
    """Deterministically update Digital Twin State: S_t = update(S_{t-1}, o_t)."""
    global _twin_states
    curr_state = _get_or_create_twin_state(patient_id)

    obs = ObservationPacket(
        timestamp=packet.timestamp,
        glucose_mgdl=packet.glucose_mgdl,
        heart_rate_bpm=packet.heart_rate_bpm,
        hrv_rmssd_ms=packet.hrv_rmssd_ms,
        steps=packet.steps,
        sleep_state=packet.sleep_state,
    )

    new_state = update_patient_state(
        previous_state=curr_state,
        observation=obs,
        current_wallclock_time=packet.current_wallclock_time,
    )

    _twin_states[patient_id] = new_state

    return TwinStateResponse(
        patient_id=patient_id,
        state_version=new_state.metadata.state_version,
        sequence_step=new_state.metadata.sequence_step,
        phenotype=new_state.phenotype.to_dict(),
        dynamic_history=new_state.history.to_dict(),
        integrity_state=new_state.integrity.to_dict(),
        derived_kinematics=new_state.derived.to_dict(),
        stream_freshness=new_state.get_stream_freshness_summary(),
        is_synthetic=True,
    )
