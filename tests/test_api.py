"""Unit tests for the GlucoTwin FastAPI backend."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from glucotwin.api.main import app, load_application_state


@pytest.fixture(scope="module")
def client() -> TestClient:
    load_application_state()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "GlucoTwin API"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client: TestClient):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["horizon_minutes"] == 120
    assert data["test_mae_mgdl"] < 25.0
    assert data["persistence_mae_mgdl"] > 50.0
    assert data["relative_improvement_pct"] > 50.0


def test_list_demo_patients(client: TestClient):
    resp = client.get("/patients/demo")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    pids = [p["patient_id"] for p in data]
    assert "SYNTH_001" in pids


def test_get_patient_timeline(client: TestClient):
    resp = client.get("/patients/SYNTH_001/timeline?limit=50")
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_id"] == "SYNTH_001"
    assert len(data["readings"]) == 50
    assert data["is_synthetic"] is True


def test_forecast_available_clean_input(client: TestClient):
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [(base_ts + pd.Timedelta(minutes=5 * i)).isoformat() for i in range(24)]
    values = [115.0 + i * 0.5 for i in range(24)]

    payload = {
        "patient_id": "SYNTH_001",
        "origin_time": timestamps[-1],
        "timestamps": timestamps,
        "glucose_readings": values,
        "current_time": timestamps[-1],
        "enable_adaptation": False,
    }

    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["reliability_status"] == "available"
    assert data["predicted_glucose"] is not None
    assert 40.0 <= data["predicted_glucose"] <= 400.0
    assert data["reason_code"] == "OK_DATA_INTACT"


def test_forecast_withheld_stale_reading(client: TestClient):
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [(base_ts + pd.Timedelta(minutes=5 * i)).isoformat() for i in range(24)]
    values = [120.0] * 24

    # Current time is 40 minutes after the latest observation (threshold is 20m)
    current_time = (base_ts + pd.Timedelta(minutes=5 * 23 + 40)).isoformat()

    payload = {
        "patient_id": "SYNTH_001",
        "origin_time": timestamps[-1],
        "timestamps": timestamps,
        "glucose_readings": values,
        "current_time": current_time,
    }

    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["reliability_status"] == "withheld"
    assert data["predicted_glucose"] is None  # Never fabricated when withheld!
    assert data["reason_code"] == "ERR_STALE_READING"


def test_forecast_withheld_out_of_bounds(client: TestClient):
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [(base_ts + pd.Timedelta(minutes=5 * i)).isoformat() for i in range(24)]
    values = [120.0] * 23 + [850.0]  # Out of bounds sensor reading

    payload = {
        "patient_id": "SYNTH_001",
        "origin_time": timestamps[-1],
        "timestamps": timestamps,
        "glucose_readings": values,
        "current_time": timestamps[-1],
    }

    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["reliability_status"] == "withheld"
    assert data["predicted_glucose"] is None
    assert data["reason_code"] == "ERR_PHYSIOLOGICAL_OUT_OF_BOUNDS"


def test_forecast_degraded_partial_sequence(client: TestClient):
    # 16 readings (enough to forecast, but flagged degraded)
    base_ts = pd.Timestamp("2024-03-01 12:00:00", tz="UTC")
    timestamps = [(base_ts + pd.Timedelta(minutes=5 * i)).isoformat() for i in range(16)]
    values = [130.0] * 16

    payload = {
        "patient_id": "SYNTH_001",
        "origin_time": timestamps[-1],
        "timestamps": timestamps,
        "glucose_readings": values,
        "current_time": timestamps[-1],
    }

    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["reliability_status"] == "degraded"
    assert data["predicted_glucose"] is not None
    assert data["reason_code"] == "WARN_PARTIAL_SEQUENCE"
