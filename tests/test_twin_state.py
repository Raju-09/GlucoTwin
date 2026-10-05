"""Unit tests for Gate 2: Patient Digital State and Online Update Engine."""

import pytest
import pandas as pd
import numpy as np

from glucotwin.twin.state import (
    IntegrityStatus,
    ObservationPacket,
    PatientDigitalState,
    PatientDynamicHistory,
    PatientStaticPhenotype,
    TwinIntegrityState,
)
from glucotwin.twin.update import (
    create_initial_patient_state,
    update_patient_state,
)


@pytest.fixture
def sample_phenotype() -> PatientStaticPhenotype:
    return PatientStaticPhenotype(
        patient_id="SYNTH_001",
        age=52,
        sex="M",
        bmi=27.4,
        baseline_hba1c=7.2,
        diabetes_duration_years=6.0,
        historical_fbg_mgdl=128.5,
        hypertension_flag=1,
        metformin_flag=1,
        sglt2i_flag=0,
        dawn_phenomenon_flag=1,
        data_age_description="Updated: 180 days ago (Historical EHR)",
        phenotype_notes="T2D on Metformin monotherapy",
    )


def test_create_initial_patient_state(sample_phenotype):
    state = create_initial_patient_state(
        phenotype=sample_phenotype,
        initial_glucose=135.0,
        initial_time="2026-10-05T08:00:00Z",
    )
    assert isinstance(state, PatientDigitalState)
    assert state.phenotype.patient_id == "SYNTH_001"
    assert state.history.glucose_readings == [135.0]
    assert state.derived.current_glucose == 135.0
    assert state.integrity.status == IntegrityStatus.AVAILABLE
    assert state.metadata.sequence_step == 0


def test_deterministic_replayable_updates(sample_phenotype):
    state_0 = create_initial_patient_state(sample_phenotype, initial_glucose=120.0, initial_time="2026-10-05T08:00:00Z")
    
    packets = [
        ObservationPacket(timestamp="2026-10-05T08:05:00Z", glucose_mgdl=122.0, heart_rate_bpm=74.0, hrv_rmssd_ms=44.0, steps=10),
        ObservationPacket(timestamp="2026-10-05T08:10:00Z", glucose_mgdl=126.0, heart_rate_bpm=76.0, hrv_rmssd_ms=42.0, steps=25),
        ObservationPacket(timestamp="2026-10-05T08:15:00Z", glucose_mgdl=131.0, heart_rate_bpm=80.0, hrv_rmssd_ms=40.0, steps=50),
    ]

    # Run pass 1
    st = state_0
    for p in packets:
        st = update_patient_state(st, p)
    
    # Run pass 2 (independent replay from initial state)
    st_replay = state_0
    for p in packets:
        st_replay = update_patient_state(st_replay, p)

    assert st.derived.current_glucose == st_replay.derived.current_glucose == 131.0
    assert st.derived.glucose_velocity_mgdl_min == st_replay.derived.glucose_velocity_mgdl_min == 1.0  # (131-126)/5.0
    assert st.metadata.sequence_step == st_replay.metadata.sequence_step == 3
    assert st.history.glucose_readings == st_replay.history.glucose_readings


def test_feature_vector_alignment(sample_phenotype):
    state = create_initial_patient_state(sample_phenotype, initial_glucose=120.0, initial_time="2026-10-05T08:00:00Z")
    # Add a few updates
    for i in range(1, 5):
        pkt = ObservationPacket(
            timestamp=f"2026-10-05T08:{i*5:02d}:00Z",
            glucose_mgdl=120.0 + i * 2.0,
            heart_rate_bpm=70.0 + i,
            hrv_rmssd_ms=45.0 - i,
            steps=10 * i,
        )
        state = update_patient_state(state, pkt)

    feat_dict = state.to_multimodal_feature_dict()
    assert "last_glucose" in feat_dict
    assert "glucose_lag_0" in feat_dict
    assert "glucose_lag_23" in feat_dict
    assert "ehr_age" in feat_dict
    assert feat_dict["ehr_age"] == 52.0
    assert "ehr_historical_fbg" in feat_dict
    assert feat_dict["ehr_historical_fbg"] == 128.5
    assert "wearable_hr_latest" in feat_dict
    assert feat_dict["wearable_hr_latest"] == 74.0
    assert "circadian_sin" in feat_dict
    assert "circadian_cos" in feat_dict


def test_fsm_withheld_on_sensor_spike(sample_phenotype):
    state = create_initial_patient_state(sample_phenotype, initial_glucose=120.0, initial_time="2026-10-05T08:00:00Z")
    spike_pkt = ObservationPacket(
        timestamp="2026-10-05T08:05:00Z",
        glucose_mgdl=750.0,  # Exceeds 600 mg/dL physiological bound
    )
    state_after = update_patient_state(state, spike_pkt)
    assert state_after.integrity.status == IntegrityStatus.WITHHELD
    assert state_after.integrity.reason_code == "PHYSIOLOGICAL_RANGE_EXCEEDED"
    assert state_after.integrity.quality_score_pct == 0.0


def test_fsm_withheld_on_large_gap(sample_phenotype):
    state = create_initial_patient_state(sample_phenotype, initial_glucose=120.0, initial_time="2026-10-05T08:00:00Z")
    # Step 1: regular update
    state = update_patient_state(
        state,
        ObservationPacket(timestamp="2026-10-05T08:05:00Z", glucose_mgdl=122.0)
    )
    assert state.integrity.status == IntegrityStatus.AVAILABLE

    # Step 2: 45 min later (gap exceeds 30 min limit)
    gap_pkt = ObservationPacket(
        timestamp="2026-10-05T08:50:00Z",
        glucose_mgdl=125.0,
    )
    state = update_patient_state(state, gap_pkt)
    assert state.integrity.status == IntegrityStatus.WITHHELD
    assert state.integrity.reason_code == "SENSOR_GAP_EXCEEDED"


def test_fsm_withheld_on_stale_telemetry(sample_phenotype):
    state = create_initial_patient_state(sample_phenotype, initial_glucose=120.0, initial_time="2026-10-05T08:00:00Z")
    pkt = ObservationPacket(
        timestamp="2026-10-05T08:05:00Z",
        glucose_mgdl=122.0,
    )
    # Wall clock is 25 minutes after packet timestamp
    state = update_patient_state(
        state,
        pkt,
        current_wallclock_time="2026-10-05T08:30:00Z",
    )
    assert state.integrity.status == IntegrityStatus.WITHHELD
    assert state.integrity.reason_code == "STALE_SENSOR_READING"


def test_fsm_recovery_sequence(sample_phenotype):
    state = create_initial_patient_state(sample_phenotype, initial_glucose=120.0, initial_time="2026-10-05T08:00:00Z")
    
    # 1. Normal step
    state = update_patient_state(state, ObservationPacket(timestamp="2026-10-05T08:05:00Z", glucose_mgdl=120.0))
    assert state.integrity.status == IntegrityStatus.AVAILABLE

    # 2. Hard failure (gap)
    state = update_patient_state(state, ObservationPacket(timestamp="2026-10-05T08:50:00Z", glucose_mgdl=122.0))
    assert state.integrity.status == IntegrityStatus.WITHHELD

    # 3. Recovery step 1 (consecutive nominal packet 1/3)
    state = update_patient_state(state, ObservationPacket(timestamp="2026-10-05T08:55:00Z", glucose_mgdl=124.0))
    assert state.integrity.status == IntegrityStatus.RECOVERING
    assert state.integrity.consecutive_recovery_count == 1

    # 4. Recovery step 2 (consecutive nominal packet 2/3)
    state = update_patient_state(state, ObservationPacket(timestamp="2026-10-05T09:00:00Z", glucose_mgdl=125.0))
    assert state.integrity.status == IntegrityStatus.RECOVERING
    assert state.integrity.consecutive_recovery_count == 2

    # 5. Recovery step 3 -> Back to AVAILABLE
    state = update_patient_state(state, ObservationPacket(timestamp="2026-10-05T09:05:00Z", glucose_mgdl=126.0))
    assert state.integrity.status == IntegrityStatus.AVAILABLE
    assert state.integrity.reason_code == "NOMINAL_STREAM"


def test_stream_freshness_summary(sample_phenotype):
    state = create_initial_patient_state(sample_phenotype, initial_glucose=120.0, initial_time="2026-10-05T08:00:00Z")
    summary = state.get_stream_freshness_summary()
    assert "Historical EHR" in summary["ehr"]
    assert "Live" in summary["cgm"]
    assert "Continuous" in summary["wearables"]
