"""Load public discovery and local model bindings without importing MuJoCo/LiveKit."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

TRAJECTORY = "joints.trajectory.v1"
GRIPPER = "gripper.position.v1"
JOINT_STATE = "sensor_msgs/msg/JointState"
JOINT_TRAJECTORY = "trajectory_msgs/msg/JointTrajectory"
CONFIGS = Path(__file__).with_name("configs")
ControlKey = str | tuple[str, str]


def channel_key(interface_id: str, operation: dict) -> ControlKey:
    channel = operation.get("control_channel", "primary")
    return interface_id if channel == "primary" else (interface_id, channel)


def operation_joints(interface, operation):
    return operation.get("joint_names", interface["joint_names"])


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(value: str | bytes):
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(value, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def positive(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be positive and finite")
    return float(value)


def names(value, field: str, *, empty=False) -> list[str]:
    if not isinstance(value, list) or (not value and not empty) or any(
        not isinstance(x, str) or not x for x in value
    ) or len(set(value)) != len(value):
        raise ValueError(f"{field} must contain unique non-empty names")
    return value


@dataclass
class Configuration:
    discovery: dict
    bindings: dict

    @classmethod
    def load(cls, path: Path, fps: int) -> Configuration:
        bindings = load_json(path.read_text())
        document = load_json((path.parent / bindings["discovery"]).read_text())
        config = cls(document, bindings)
        config.validate()
        document["default_video_rate_hz"] = positive(fps, "fps")
        return config

    def validate(self) -> None:
        doc, bindings = self.discovery, self.bindings
        if type(doc.get("schema_version")) is not int or doc["schema_version"] != 1 \
                or type(bindings.get("schema_version")) is not int or bindings["schema_version"] != 1:
            raise ValueError("unsupported discovery/binding version")
        if any(key in doc for key in ("topics", "videos", "components", "attachments", "services", "rpcs")):
            raise ValueError("legacy discovery collections are not supported")
        if not isinstance(doc.get("label"), str) or not doc["label"].strip():
            raise ValueError("discovery label is required")
        positive(doc.get("default_video_rate_hz"), "default_video_rate_hz")
        if doc.get("control") != {"scope": "robot", "ownership": "portal_active_operator"}:
            raise ValueError("robot-scoped Portal ownership is required")
        for media in ("data", "video", "audio"):
            patterns = doc.get(f"exclude_{media}_tracks", [])
            names(patterns, f"exclude_{media}_tracks", empty=True)
            for pattern in patterns:
                re.compile(pattern)
        interfaces = doc.get("control_interfaces")
        if not isinstance(interfaces, dict) or not isinstance(bindings.get("control_interfaces"), dict) \
                or interfaces.keys() != bindings["control_interfaces"].keys():
            raise ValueError("discovery and bindings must have the same interface IDs")
        endpoints, backend_arms = set(), set()
        for interface_id, interface in interfaces.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", interface_id):
                raise ValueError(f"invalid interface ID: {interface_id}")
            if not isinstance(interface.get("label"), str) \
                    or not interface["label"].strip() or not interface.get("frame_id"):
                raise ValueError(f"invalid arm interface: {interface_id}")
            joints = names(interface.get("joint_names"), "joint_names")
            binding = bindings["control_interfaces"][interface_id]
            if binding["backend_arm"] in backend_arms:
                raise ValueError("two interfaces cannot drive the same backend arm")
            backend_arms.add(binding["backend_arm"])
            if set(joints) != set(binding["joint_map"]):
                raise ValueError(f"joint binding mismatch for {interface_id}")
            for joint in binding["joint_map"].values():
                positive(joint["max_velocity"], "max_velocity")
                if not joint["joint"] or not joint["actuator"]:
                    raise ValueError("joint and actuator names are required")
            operations = interface.get("operations")
            if not isinstance(operations, dict) or not operations:
                raise ValueError("an arm needs an implemented operation")
            channels = set()
            for operation_id, operation in operations.items():
                if not re.fullmatch(r"[a-z][a-z0-9_]*", operation_id):
                    raise ValueError(f"invalid operation ID: {operation_id}")
                if operation["profile"] not in (TRAJECTORY, GRIPPER):
                    raise ValueError(f"profile is not implemented: {operation['profile']}")
                channel = operation.get("control_channel", "primary")
                if not isinstance(channel, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", channel) \
                        or channel in channels:
                    raise ValueError("simulation requires one implemented operation per control channel")
                channels.add(channel)
                selected = names(operation_joints(interface, operation), "operation joint_names")
                gripper = operation["profile"] == GRIPPER
                if gripper:
                    spec = binding.get("gripper", {})
                    if len(selected) != 1 or selected != [spec.get("name")] or channel == "primary":
                        raise ValueError("gripper requires one explicit bound joint and an independent channel")
                    if "joint_names" not in operation or not spec.get("joint") or not spec.get("actuator") \
                            or not spec.get("coupled_joint"):
                        raise ValueError("gripper joint, actuator and coupling bindings are required")
                    positive(spec.get("max_velocity"), "gripper max_velocity")
                elif not set(selected) <= set(joints):
                    raise ValueError("arm operation selects unbound joints")
                command = operation["command"]
                if command["msg_type"] != JOINT_TRAJECTORY or command["format"] != "json" \
                        or not isinstance(command["track_name"], str) or not command["track_name"]:
                    raise ValueError("trajectory commands require a named JointTrajectory JSON track")
                if command["track_name"] in endpoints:
                    raise ValueError("command destinations must be unambiguous")
                endpoints.add(command["track_name"])
                feedback = names(operation.get("feedback_tracks", []), "feedback_tracks")
                if len(feedback) != 1:
                    raise ValueError("trajectory profile needs exactly one required joint-state track")
                expected = ["gripper_controls"] if gripper else ["joint_sliders"]
                if operation["presentations"] != expected:
                    raise ValueError(f"profile supports {expected}")
                if any(re.fullmatch(p, feedback[0]) for p in doc.get("exclude_data_tracks", [])):
                    raise ValueError("required joint feedback cannot be excluded")
        if endpoints & set(self.feedback_tracks):
            raise ValueError("command and feedback tracks cannot overlap")

    @property
    def command_tracks(self) -> dict[str, tuple[ControlKey, dict]]:
        return {op["command"]["track_name"]: (channel_key(key, op), op)
                for key, interface in self.discovery["control_interfaces"].items()
                for op in interface["operations"].values()}

    @property
    def feedback_tracks(self) -> dict[str, list[ControlKey]]:
        tracks: dict[str, list[ControlKey]] = {}
        for key, interface in self.discovery["control_interfaces"].items():
            for operation in interface["operations"].values():
                for track in operation.get("feedback_tracks", []):
                    target = channel_key(key, operation)
                    if target not in tracks.setdefault(track, []):
                        tracks[track].append(target)
        return tracks
