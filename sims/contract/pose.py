"""Executable subset of arm.cartesian_pose.v1; standard library only. See README.md."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import load_json

MAX_DISTANCE = 2.0  # meters from the arm base


@dataclass(frozen=True)
class PoseTarget:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # x, y, z, w; unit length


def _fields(value, names, what):
    if not isinstance(value, dict) or set(value) != set(names) or any(
        isinstance(value[n], bool) or not isinstance(value[n], (int, float)) or not math.isfinite(value[n])
        for n in names
    ):
        raise ValueError(f"{what} needs finite numbers {', '.join(names)}")
    return tuple(float(value[n]) for n in names)


def parse_pose(payload: bytes, frame_id: str) -> PoseTarget:
    """Parse a geometry_msgs/msg/PoseStamped target in the arm's base frame."""
    if len(payload) > 4096:
        raise ValueError("pose exceeds 4 KiB")
    message = load_json(payload)
    if not isinstance(message, dict) or set(message) - {"header", "pose"} or "pose" not in message:
        raise ValueError("expected a PoseStamped object")
    header = message.get("header", {})
    if not isinstance(header, dict) or set(header) - {"stamp", "frame_id"} \
            or header.get("frame_id", "") not in ("", frame_id):
        raise ValueError("pose must be in the arm base frame")
    pose = message["pose"]
    if not isinstance(pose, dict) or set(pose) != {"position", "orientation"}:
        raise ValueError("pose needs position and orientation")
    position = _fields(pose["position"], ("x", "y", "z"), "position")
    quaternion = _fields(pose["orientation"], ("x", "y", "z", "w"), "orientation")
    if math.hypot(*position) > MAX_DISTANCE:
        raise ValueError(f"pose is farther than {MAX_DISTANCE} m from the arm base")
    length = math.hypot(*quaternion)
    if abs(length - 1) > 0.01:
        raise ValueError("orientation must be a unit quaternion")
    return PoseTarget(position, tuple(q / length for q in quaternion))
