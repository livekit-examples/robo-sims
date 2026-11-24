"""Telemetry tracks: schemas, rates and values. Each state message carries simulation time, run ID and sequence."""
from __future__ import annotations

import math

from ..contract.schemas import schema_for
from .simulation import Vacuum

SCHEMA = "https://json-schema.org/draft/2020-12/schema"
FLOAT64 = "std_msgs/msg/Float64"
FLOAT64_SCHEMA = {"$schema": SCHEMA, "type": "object", "required": ["data"], "additionalProperties": False,
                  "properties": {"data": {"type": "number"}}}
ODOMETRY = "nav_msgs/msg/Odometry"
# Custom state messages. The version is part of the schema ID: change the number when a field changes.
BATTERY = "robo_sims/msg/VacuumBatteryStateV1"
BIN = "robo_sims/msg/VacuumDustBinStateV1"
COVERAGE = "robo_sims/msg/VacuumCoverageStateV1"
RUN = "robo_sims/msg/VacuumRunStateV1"

SPEED_TRACK = "/speed_mps"
YAW_RATE_TRACK = "/yaw_rate_rad_s"
BATTERY_TRACK = "/battery_state"
BIN_TRACK = "/dust_bin_state"
COVERAGE_TRACK = "/coverage_state"
RUN_TRACK = "/run_state"
ODOM_TRACK = "/odom"

# name: (message type, rate in Hz)
TRACKS = {
    SPEED_TRACK: (FLOAT64, 20), YAW_RATE_TRACK: (FLOAT64, 20), BATTERY_TRACK: (BATTERY, 10),
    BIN_TRACK: (BIN, 10), COVERAGE_TRACK: (COVERAGE, 10), RUN_TRACK: (RUN, 10), ODOM_TRACK: (ODOMETRY, 20),
}


def stamp(t: float) -> dict:
    sec = int(t)
    return {"sec": sec, "nanosec": min(999999999, int((t - sec) * 1e9))}


def odometry(sim: Vacuum) -> dict:
    x, y, yaw = sim.position()
    vx, vy, wz = sim.velocity()
    return {"header": {"stamp": stamp(sim.time), "frame_id": "map"}, "child_frame_id": "base_link",
            "pose": {"pose": {"position": {"x": x, "y": y, "z": 0.0},
                              "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)}},
                     "covariance": [0.0] * 36},
            "twist": {"twist": {"linear": {"x": vx, "y": vy, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": wz}},
                      "covariance": [0.0] * 36}}


def value(sim: Vacuum, name: str) -> dict:
    if name == SPEED_TRACK:
        return {"data": sim.velocity()[0]}
    if name == YAW_RATE_TRACK:
        return {"data": sim.velocity()[2]}
    if name == ODOM_TRACK:
        return odometry(sim)
    if name == BATTERY_TRACK:
        return {**sim.stamp(), **sim.battery.state()}
    if name == BIN_TRACK:
        return {**sim.stamp(), **sim.bin.state()}
    if name == COVERAGE_TRACK:
        return sim.coverage_state()
    if name == RUN_TRACK:
        state = sim.run_state()
        # Requested motion is reported next to the measured speed, never in its place.
        return {**state, "requested_speed_mps": sim.requested.v, "requested_yaw_rate_rad_s": sim.requested.omega,
                "assist_active": sim.assist_active, "drive_enabled": sim.drive_enabled}
    raise KeyError(name)


def definitions(sim: Vacuum) -> dict:
    """track name -> (message type, JSON schema, rate in Hz)."""
    out = {}
    for name, (msg_type, rate) in TRACKS.items():
        schema = FLOAT64_SCHEMA if msg_type == FLOAT64 else {"$schema": SCHEMA, **schema_for(value(sim, name))}
        out[name] = (msg_type, schema, rate)
    return out
