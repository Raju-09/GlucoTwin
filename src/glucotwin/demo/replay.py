"""Synthetic replay and failure scenario injection engine for GlucoTwin.

Allows a clinician or researcher to simulate live stream updates and test
system robustness under controlled sensor failure scenarios:
- Normal streaming
- Stale sensor reading (delayed packet)
- Excessive gap (>30 min)
- Sensor spike / out of bounds (>600 mg/dL)
- Intermittent dropouts
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from glucotwin.config import LOOKBACK_N

ScenarioType = Literal[
    "normal",
    "stale_sensor",
    "excessive_gap",
    "sensor_spike",
    "intermittent_dropouts",
]


class ReplayEngine:
    """Manages synthetic time-series stepping and failure scenario injection."""

    def __init__(self, patient_df: pd.DataFrame) -> None:
        self.patient_df = patient_df.sort_values(by="timestamp").reset_index(drop=True)
        self.patient_id = str(self.patient_df["patient_id"].iloc[0])
        self.total_records = len(self.patient_df)

    def get_slice_at_step(
        self,
        current_step_idx: int,
        lookback_n: int = LOOKBACK_N,
        scenario: ScenarioType = "normal",
    ) -> dict[str, Any]:
        """Extract a window of lookback readings ending at step_idx and inject scenario."""
        # Ensure sufficient history
        idx = max(lookback_n, min(current_step_idx, self.total_records - 1))
        window = self.patient_df.iloc[idx - lookback_n : idx].copy()

        timestamps = [pd.Timestamp(t) for t in window["timestamp"]]
        glucose = [float(g) for g in window["glucose"]]
        origin_time = timestamps[-1]
        current_wall_time = origin_time
        scenario_note = "Normal continuous stream."

        if scenario == "stale_sensor":
            # Simulate wall-clock advancing 45 minutes ahead while sensor packet was delayed
            current_wall_time = origin_time + pd.Timedelta(minutes=45)
            scenario_note = (
                "Simulated wearable disconnection: wall-clock is 45 min ahead of last reading."
            )

        elif scenario == "excessive_gap":
            # Inject a 55-minute gap in the middle of the lookback window
            mid = len(timestamps) // 2
            timestamps = timestamps[:mid] + [
                t + pd.Timedelta(minutes=55) for t in timestamps[mid:]
            ]
            origin_time = timestamps[-1]
            current_wall_time = origin_time
            scenario_note = "Simulated 55-minute data drop between sensor transmissions."

        elif scenario == "sensor_spike":
            # Corrupt the latest reading with an impossible sensor value
            glucose[-1] = 850.0
            scenario_note = "Simulated hardware artifact: impossible reading of 850 mg/dL."

        elif scenario == "intermittent_dropouts":
            # Drop every 3rd reading (leaving 16 readings instead of 24)
            keep_indices = [i for i in range(len(timestamps)) if i % 3 != 0]
            timestamps = [timestamps[i] for i in keep_indices]
            glucose = [glucose[i] for i in keep_indices]
            origin_time = timestamps[-1]
            current_wall_time = origin_time
            scenario_note = "Simulated intermittent packet loss (16/24 readings present)."

        return {
            "patient_id": self.patient_id,
            "origin_time": origin_time.isoformat(),
            "current_time": current_wall_time.isoformat(),
            "timestamps": [t.isoformat() for t in timestamps],
            "glucose_readings": glucose,
            "scenario": scenario,
            "scenario_note": scenario_note,
            "step_index": idx,
        }
