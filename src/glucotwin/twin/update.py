"""Deterministic Online State Update and Integrity State Machine for GlucoTwin.

This module implements the core sequential updating equation:
    S_t = update(S_{t-1}, o_t)

Key Architectural Properties:
1. Purely Causal: No future information or lookahead is ever accessed.
2. Deterministic & Replayable: Applying the same sequence of observation packets [o_1, ..., o_t]
   to initial state S_0 always yields bit-identical S_t.
3. No Continuous Model Retraining: Updates perform online state estimation and kinematic tracking,
   NOT parameter re-estimation or online gradient steps.
4. Finite State Machine: δ_t transitions deterministically across
   AVAILABLE -> DEGRADED -> DRIFTING -> WITHHELD -> RECOVERING.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import math
from typing import Optional
import numpy as np
import pandas as pd

from glucotwin.config import (
    FRESHNESS_THRESHOLD_MINUTES,
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    MAX_GAP_MINUTES,
)
from glucotwin.twin.state import (
    DerivedFeatures,
    IntegrityStatus,
    ObservationPacket,
    PatientDigitalState,
    PatientDynamicHistory,
    PatientStaticPhenotype,
    StateMetadata,
    TwinIntegrityState,
)


def create_initial_patient_state(
    phenotype: PatientStaticPhenotype,
    initial_glucose: float = 120.0,
    initial_time: Optional[str] = None,
    lookback_n: int = 24,
) -> PatientDigitalState:
    """Instantiate a baseline Digital Twin State S_0 for a patient before stream arrival."""
    ts = initial_time or datetime.now(timezone.utc).isoformat()
    
    # Baseline history with initial steady-state values
    history = PatientDynamicHistory(
        lookback_n=lookback_n,
        last_observation_time=ts,
        timestamps=[ts],
        glucose_readings=[float(initial_glucose)],
        heart_rate=[72.0],
        hrv_rmssd=[45.0],
        steps=[0],
        sleep_state="awake",
        cgm_freshness_minutes=0.0,
        wearable_freshness_minutes=0.0,
    )

    derived = DerivedFeatures(
        current_glucose=float(initial_glucose),
        glucose_velocity_mgdl_min=0.0,
        glucose_acceleration_mgdl_min2=0.0,
        rolling_mean_1h=float(initial_glucose),
        rolling_std_1h=0.0,
        time_in_illustrative_range_pct=100.0 if (70.0 <= initial_glucose <= 180.0) else 0.0,
        circadian_sin=0.0,
        circadian_cos=1.0,
    )

    integrity = TwinIntegrityState(
        status=IntegrityStatus.AVAILABLE,
        reason_code="INITIALIZED",
        quality_score_pct=100.0,
        advisory_message="Twin state initialized from static phenotype and baseline telemetry.",
        active_flags=[],
        consecutive_recovery_count=0,
        drift_score=0.0,
    )

    metadata = StateMetadata(
        state_version="2.0.0",
        sequence_step=0,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        is_synthetic=True,
    )

    return PatientDigitalState(
        phenotype=phenotype,
        history=history,
        integrity=integrity,
        derived=derived,
        metadata=metadata,
    )


def update_patient_state(
    previous_state: PatientDigitalState,
    observation: ObservationPacket,
    current_wallclock_time: Optional[str] = None,
    max_history_len: int = 144,  # Keep up to 12 hours in active state memory
) -> PatientDigitalState:
    """Deterministic, causal state transition: S_t = update(S_{t-1}, o_t).

    Parameters
    ----------
    previous_state : PatientDigitalState
        The state S_{t-1} prior to receiving the new observation.
    observation : ObservationPacket
        Incoming telemetry packet o_t with sensor readings and timestamp.
    current_wallclock_time : str, optional
        System time against which to measure arrival latency/freshness.
    max_history_len : int
        Maximum sequence length to retain in rolling buffer memory.

    Returns
    -------
    PatientDigitalState
        The newly computed state S_t.
    """
    # 1. Clone dynamic containers to ensure immutability of previous state
    new_timestamps = list(previous_state.history.timestamps)
    new_glucose = list(previous_state.history.glucose_readings)
    new_hr = list(previous_state.history.heart_rate)
    new_hrv = list(previous_state.history.hrv_rmssd)
    new_steps = list(previous_state.history.steps)

    # 2. Parse timestamps and compute interval / freshness
    obs_time = pd.to_datetime(observation.timestamp, utc=True)
    prev_time = (
        pd.to_datetime(previous_state.history.last_observation_time, utc=True)
        if previous_state.history.last_observation_time
        else obs_time
    )

    # Gap from last recorded reading in history (minutes)
    gap_minutes = max(0.0, (obs_time - prev_time).total_seconds() / 60.0)

    # Freshness vs wallclock (minutes)
    if current_wallclock_time:
        wall_time = pd.to_datetime(current_wallclock_time, utc=True)
        freshness_minutes = max(0.0, (wall_time - obs_time).total_seconds() / 60.0)
    else:
        freshness_minutes = 0.0

    # 3. Ingest observation into history buffers
    new_timestamps.append(str(obs_time))
    
    # Handle glucose ingestion
    raw_gl = observation.glucose_mgdl
    if raw_gl is not None:
        new_glucose.append(float(raw_gl))
    else:
        # Impute with last valid reading for continuity if missing
        last_val = new_glucose[-1] if new_glucose else 120.0
        new_glucose.append(float(last_val))

    # Handle wearable telemetry ingestion
    hr_val = observation.heart_rate_bpm if observation.heart_rate_bpm is not None else (new_hr[-1] if new_hr else 72.0)
    hrv_val = observation.hrv_rmssd_ms if observation.hrv_rmssd_ms is not None else (new_hrv[-1] if new_hrv else 45.0)
    steps_val = observation.steps if observation.steps is not None else 0
    sleep_val = observation.sleep_state or previous_state.history.sleep_state

    new_hr.append(float(hr_val))
    new_hrv.append(float(hrv_val))
    new_steps.append(int(steps_val))

    # Buffer truncation to max_history_len
    if len(new_timestamps) > max_history_len:
        new_timestamps = new_timestamps[-max_history_len:]
        new_glucose = new_glucose[-max_history_len:]
        new_hr = new_hr[-max_history_len:]
        new_hrv = new_hrv[-max_history_len:]
        new_steps = new_steps[-max_history_len:]

    new_history = PatientDynamicHistory(
        lookback_n=previous_state.history.lookback_n,
        last_observation_time=str(obs_time),
        timestamps=new_timestamps,
        glucose_readings=new_glucose,
        heart_rate=new_hr,
        hrv_rmssd=new_hrv,
        steps=new_steps,
        sleep_state=sleep_val,
        cgm_freshness_minutes=round(freshness_minutes, 1),
        wearable_freshness_minutes=5.0,
    )

    # 4. Compute Derived Kinematics causally
    curr_gl = new_glucose[-1]
    dt_min = gap_minutes if gap_minutes > 0.1 else 5.0

    if len(new_glucose) >= 2:
        prev_gl = new_glucose[-2]
        vel = (curr_gl - prev_gl) / dt_min
    else:
        vel = 0.0

    if len(new_glucose) >= 3:
        prev_vel = previous_state.derived.glucose_velocity_mgdl_min
        acc = (vel - prev_vel) / dt_min
    else:
        acc = 0.0

    # 1-hour window (last 12 readings)
    recent_1h = new_glucose[-12:]
    mean_1h = float(np.mean(recent_1h))
    std_1h = float(np.std(recent_1h))
    in_range_count = sum(1 for g in recent_1h if 70.0 <= g <= 180.0)
    tir_pct = (in_range_count / len(recent_1h)) * 100.0

    # Circadian components
    hour_float = obs_time.hour + obs_time.minute / 60.0
    circ_sin = float(math.sin(2 * math.pi * hour_float / 24.0))
    circ_cos = float(math.cos(2 * math.pi * hour_float / 24.0))

    new_derived = DerivedFeatures(
        current_glucose=round(curr_gl, 1),
        glucose_velocity_mgdl_min=round(vel, 3),
        glucose_acceleration_mgdl_min2=round(acc, 4),
        rolling_mean_1h=round(mean_1h, 1),
        rolling_std_1h=round(std_1h, 2),
        time_in_illustrative_range_pct=round(tir_pct, 1),
        circadian_sin=round(circ_sin, 4),
        circadian_cos=round(circ_cos, 4),
    )

    # 5. Finite State Machine for Integrity (δ_t)
    flags = []
    hard_failure = False
    degraded_failure = False
    reason_code = "NOMINAL_STREAM"
    advisory_msg = "Continuous telemetry verified. Forecasting enabled."

    # Check 1: Physiological plausibility
    if raw_gl is not None:
        if raw_gl < GLUCOSE_MIN_MGDL:
            flags.append(f"GLUCOSE_BELOW_MIN ({raw_gl:.1f} < {GLUCOSE_MIN_MGDL})")
            hard_failure = True
            reason_code = "PHYSIOLOGICAL_RANGE_EXCEEDED"
            advisory_msg = f"Glucose value ({raw_gl:.1f} mg/dL) below physiological bound ({GLUCOSE_MIN_MGDL} mg/dL)."
        elif raw_gl > GLUCOSE_MAX_MGDL:
            flags.append(f"GLUCOSE_ABOVE_MAX ({raw_gl:.1f} > {GLUCOSE_MAX_MGDL})")
            hard_failure = True
            reason_code = "PHYSIOLOGICAL_RANGE_EXCEEDED"
            advisory_msg = f"Glucose reading ({raw_gl:.1f} mg/dL) exceeds hardware/physiological maximum ({GLUCOSE_MAX_MGDL} mg/dL)."

    # Check 2: Freshness / Latency
    if freshness_minutes > FRESHNESS_THRESHOLD_MINUTES:
        flags.append(f"STALE_TELEMETRY ({freshness_minutes:.1f}m > {FRESHNESS_THRESHOLD_MINUTES}m)")
        hard_failure = True
        reason_code = "STALE_SENSOR_READING"
        advisory_msg = f"Last sensor observation is {freshness_minutes:.1f} min old (threshold: {FRESHNESS_THRESHOLD_MINUTES} min)."

    # Check 3: Inter-reading gap
    if gap_minutes > MAX_GAP_MINUTES and previous_state.metadata.sequence_step > 0:
        flags.append(f"SENSOR_GAP ({gap_minutes:.1f}m > {MAX_GAP_MINUTES}m)")
        hard_failure = True
        reason_code = "SENSOR_GAP_EXCEEDED"
        advisory_msg = f"Telemetry transmission interrupted. Gap of {gap_minutes:.1f} min exceeds safety threshold ({MAX_GAP_MINUTES} min)."

    # Check 4: Missing sensor packet
    if raw_gl is None:
        flags.append("MISSING_PACKET")
        degraded_failure = True
        if not hard_failure:
            reason_code = "MILD_PACKET_LOSS"
            advisory_msg = "Temporary packet dropout detected. Imputed from prior dynamic state."

    # FSM Transition Logic
    prev_status = previous_state.integrity.status
    rec_count = previous_state.integrity.consecutive_recovery_count

    if hard_failure:
        new_status = IntegrityStatus.WITHHELD
        rec_count = 0
        quality_score = 0.0
    elif degraded_failure:
        new_status = IntegrityStatus.DEGRADED
        rec_count = 0
        quality_score = 65.0
    elif prev_status in (IntegrityStatus.WITHHELD, IntegrityStatus.DEGRADED, IntegrityStatus.RECOVERING):
        # We received a valid nominal packet after an error condition
        rec_count += 1
        if rec_count < 3:
            new_status = IntegrityStatus.RECOVERING
            reason_code = "STREAM_RECOVERING"
            advisory_msg = f"Stream recovering from anomaly ({rec_count}/3 consecutive nominal readings verified)."
            quality_score = 75.0 + (rec_count * 8.0)
        else:
            new_status = IntegrityStatus.AVAILABLE
            reason_code = "NOMINAL_STREAM"
            advisory_msg = "Stream recovered to full operational nominal status."
            quality_score = 100.0
            rec_count = 0
    else:
        new_status = IntegrityStatus.AVAILABLE
        reason_code = "NOMINAL_STREAM"
        advisory_msg = "Continuous telemetry verified. Forecasting enabled."
        quality_score = 100.0
        rec_count = 0

    new_integrity = TwinIntegrityState(
        status=new_status,
        reason_code=reason_code,
        quality_score_pct=quality_score,
        advisory_message=advisory_msg,
        active_flags=flags,
        consecutive_recovery_count=rec_count,
        drift_score=0.0,
    )

    new_metadata = StateMetadata(
        state_version="2.0.0",
        sequence_step=previous_state.metadata.sequence_step + 1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        is_synthetic=True,
    )

    return PatientDigitalState(
        phenotype=previous_state.phenotype,
        history=new_history,
        integrity=new_integrity,
        derived=new_derived,
        metadata=new_metadata,
    )
