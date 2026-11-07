"""Executable simulation subset of joints.trajectory.v1. See README.md."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import load_json


def vector(value, size: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value
    ):
        raise ValueError("positions must be a finite number for each named joint")
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class Trajectory:
    names: tuple[str, ...]
    times: tuple[float, ...]
    positions: tuple[tuple[float, ...], ...]

    @classmethod
    def parse(cls, payload: bytes, allowed: list[str]) -> Trajectory:
        if len(payload) > 64 * 1024:
            raise ValueError("trajectory exceeds 64 KiB")
        message = load_json(payload)
        if not isinstance(message, dict) or set(message) - {"header", "joint_names", "points"}:
            raise ValueError("expected a JointTrajectory object")
        header = message.get("header", {})
        if not isinstance(header, dict) or set(header) - {"stamp", "frame_id"}:
            raise ValueError("invalid trajectory header")
        stamp = header.get("stamp", {"sec": 0, "nanosec": 0})
        if not isinstance(stamp, dict) or set(stamp) != {"sec", "nanosec"} \
                or any(type(value) is not int or value != 0 for value in stamp.values()) \
                or header.get("frame_id", "") != "":
            raise ValueError("only immediate trajectories with an empty frame are supported")
        names = message.get("joint_names")
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names) \
                or len(set(names)) != len(names) or not set(names) <= set(allowed):
            raise ValueError("unknown, duplicate, or missing joint names")
        points = message.get("points")
        if not isinstance(points, list) or len(points) > 256:
            raise ValueError("points must be an array of at most 256 waypoints")
        times, positions = [], []
        for point in points:
            if not isinstance(point, dict) or set(point) - {
                "positions", "velocities", "accelerations", "effort", "time_from_start"
            }:
                raise ValueError("invalid waypoint")
            if any(point.get(field, []) != [] for field in ("velocities", "accelerations", "effort")):
                raise ValueError("only position waypoints are supported")
            duration = point.get("time_from_start")
            if not isinstance(duration, dict) or set(duration) != {"sec", "nanosec"}:
                raise ValueError("time_from_start requires sec and nanosec")
            sec, nano = duration["sec"], duration["nanosec"]
            if type(sec) is not int or type(nano) is not int or sec < 0 or not 0 <= nano < 1_000_000_000:
                raise ValueError("invalid waypoint duration")
            at = sec + nano / 1e9
            if at <= (times[-1] if times else 0) or at > 30:
                raise ValueError("waypoint times must increase from zero and finish within 30 s")
            times.append(at)
            positions.append(vector(point.get("positions"), len(names)))
        return cls(tuple(names), tuple(times), tuple(positions))

    def validate_motion(self, start, limits, speeds) -> None:
        previous, before = tuple(start), 0.0
        for at, values in zip(self.times, self.positions):
            for value, old, (low, high), speed in zip(values, previous, limits, speeds):
                if not low <= value <= high:
                    raise ValueError("trajectory exceeds a joint position limit")
                if abs(value - old) > speed * (at - before) + 1e-9:
                    raise ValueError("trajectory exceeds a configured velocity limit")
            previous, before = values, at

    def sample(self, elapsed: float, start) -> tuple[float, ...]:
        previous, before = tuple(start), 0.0
        for at, values in zip(self.times, self.positions):
            if elapsed < at:
                alpha = max(0.0, (elapsed - before) / (at - before))
                return tuple(a + (b - a) * alpha for a, b in zip(previous, values))
            previous, before = values, at
        return previous
