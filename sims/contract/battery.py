"""Simulated battery state shared by the Trossen simulations."""

from __future__ import annotations

import random
import time

BATTERY_STATE = "sensor_msgs/msg/BatteryState"
NOMINAL_VOLTS = 24.0
MIN_DROOP, MAX_DROOP = 0.01, 0.25
PERIOD = 1.0

BATTERY_STATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "required": ["header", "voltage", "present"], "additionalProperties": False,
    "properties": {
        "header": {"type": "object", "additionalProperties": False, "properties": {
            "stamp": {"type": "object", "required": ["sec", "nanosec"], "additionalProperties": False,
                      "properties": {"sec": {"type": "integer", "minimum": 0},
                                     "nanosec": {"type": "integer", "minimum": 0, "maximum": 999999999}}},
            "frame_id": {"type": "string"}}},
        "voltage": {"type": "number"},
        "present": {"type": "boolean"},
    },
}


def battery_track(interface_id) -> str:
    return f"/{interface_id}/battery_state"


def battery_state(now: float | None = None, rng=random) -> dict:
    """One reading: 24 V minus noise between 0.01 V and 0.25 V."""
    at = time.time() if now is None else now
    sec = int(at)
    return {"header": {"stamp": {"sec": sec, "nanosec": int((at - sec) * 1e9)}, "frame_id": ""},
            "voltage": round(NOMINAL_VOLTS - rng.uniform(MIN_DROOP, MAX_DROOP), 3),
            "present": True}
