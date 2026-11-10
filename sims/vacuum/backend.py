"""Twist parsing, driving policy, discovery and the rendered or headless simulation."""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np

from ..contract.config import load_json
from ..mujoco_tools import mujoco
from . import telemetry
from .simulation import (DEADLINE_S, MAX_ACCEL, MAX_FORWARD, MAX_REVERSE, MAX_YAW_ACCEL, MAX_YAW_RATE, ONBOARD,
                         OVERHEAD, PRECISION_SPEED, START_CONFIGS, Command, Vacuum)

PROFILE = "base.twist.v1"
COMMAND = "geometry_msgs/msg/Twist"
COMMAND_TRACK = "/cmd_vel"
KIND = "vacuum.twist.v1"
WATCHDOG_S = 0.15
SCHEMA = "https://json-schema.org/draft/2020-12/schema"
_VECTOR = {"type": "object", "required": ["x", "y", "z"], "additionalProperties": False,
           "properties": {axis: {"type": "number"} for axis in "xyz"}}
TWIST_SCHEMA = {"$schema": SCHEMA, "type": "object", "required": ["linear", "angular"], "additionalProperties": False,
                "properties": {"linear": _VECTOR, "angular": _VECTOR}}


class Backend:
    def __init__(self, *, fps=30, width=960, height=540, video=True, clock=time.monotonic,
                 start="normal", assist=True, auto_start=False):
        self.auto_start = auto_start
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("fps must be an integer between 1 and 60")
        if any(type(v) is not int or v < 2 or v % 2 or v > 1920 for v in (width, height)):
            raise ValueError("video dimensions must be positive even integers at most 1920")
        self.clock = clock
        self.timeout = WATCHDOG_S
        self.last_command = None
        self.timed_out = False
        self.sim = Vacuum(width=width, height=height, start=start, assist=assist)
        self.views = None
        self.feeds = {}
        document = {"schema_version": 1, "label": "Vacuum robot · 2 × 2 m room", "default_video_rate_hz": fps,
                    "control": {"scope": "robot", "ownership": "portal_active_operator"},
                    "control_interfaces": {"base": {"label": "Vacuum robot", "frame_id": "base_link", "operations": {
                        "drive": {"profile": PROFILE,
                                  "command": {"track_name": COMMAND_TRACK, "msg_type": COMMAND, "format": "json"},
                                  "presentations": ["gamepad", "joystick", "keyboard"]}}}}}
        self.config = SimpleNamespace(
            discovery=document,
            command_tracks={COMMAND_TRACK: ("base", document["control_interfaces"]["base"]["operations"]["drive"])})
        if video:
            from .render import Views
            self.views = Views(self.sim, width, height)
            self.feeds = self.views.feeds
            document["camera_info"] = self.camera_info(width, height)

    # --------------------------------------------------------------- policy
    def policy(self) -> dict:
        return {"version": 1, "kind": KIND, "interface_id": "base", "operation_id": "drive",
                "command_track": COMMAND_TRACK, "profile": PROFILE, "control_channel": "primary",
                "min_speed": -MAX_REVERSE, "max_speed": MAX_FORWARD, "max_yaw_rate": MAX_YAW_RATE,
                "precision_max_speed": PRECISION_SPEED, "input_deadband": 0.08, "input_exponent": 2,
                "max_acceleration": MAX_ACCEL,
                "max_braking": MAX_ACCEL, "max_yaw_acceleration": MAX_YAW_ACCEL,
                "command_timeout_ms": round(self.timeout * 1000), "input_interval_ms": 20,
                "neutral": "brake", "deadline_s": DEADLINE_S,
                "run_methods": {"start": "vacuum.start_run", "reset": "vacuum.reset_run"},
                "state_track": telemetry.RUN_TRACK}

    # ------------------------------------------------------------- commands
    def parse(self, payload) -> Command:
        if not isinstance(payload, (bytes, str)) or len(payload) > 4096:
            raise ValueError("Twist command must be JSON of at most 4096 bytes")
        value = load_json(payload)
        if not isinstance(value, dict) or value.keys() != {"linear", "angular"}:
            raise ValueError("expected a Twist with linear and angular")
        vectors = {}
        for name in ("linear", "angular"):
            vector = value[name]
            if not isinstance(vector, dict) or vector.keys() != {"x", "y", "z"}:
                raise ValueError(f"{name} must have x, y and z")
            for axis, number in vector.items():
                if type(number) not in (int, float) or not math.isfinite(number):
                    raise ValueError(f"{name}.{axis} must be a finite number")
            vectors[name] = vector
        if vectors["linear"]["y"] != 0 or vectors["linear"]["z"] != 0:
            raise ValueError("sideways and vertical motion are not supported")
        if vectors["angular"]["x"] != 0 or vectors["angular"]["y"] != 0:
            raise ValueError("only angular.z (yaw rate) is supported")
        v, omega = float(vectors["linear"]["x"]), float(vectors["angular"]["z"])
        if not -MAX_REVERSE <= v <= MAX_FORWARD or abs(omega) > MAX_YAW_RATE:
            raise ValueError("drive command exceeds the advertised limits")
        return Command(v, omega)

    def submit(self, payload):
        command = self.parse(payload)
        if self.auto_start and self.sim.state == "idle" and (command.v or command.omega):
            self.sim.start_run()
        self.sim.requested = command
        self.last_command = self.clock()
        self.timed_out = False

    def hold(self, _interface=None):
        """Request braking within the robot limits. Measured motion stops later, not at once."""
        self.sim.requested = Command()
        self.last_command = None

    def watchdog(self) -> bool:
        if self.last_command is not None and self.clock() - self.last_command > self.timeout:
            self.hold()
            self.timed_out = True
            return True
        return False

    def step(self, dt=0.005):
        self.sim.step(dt)

    # ------------------------------------------------------------ run control
    def start_run(self):
        self.sim.start_run()

    def reset_run(self):
        self.sim.reset()
        self.hold()
        self.timed_out = False

    def record_overload(self):
        self.sim.record_overload()
        self.hold()

    # ------------------------------------------------------------ telemetry
    def instrument_tracks(self) -> dict:
        return telemetry.definitions(self.sim)

    def instrument_value(self, name):
        return telemetry.value(self.sim, name)

    def battery_state(self) -> dict:
        t = self.sim.time
        return {"header": {"stamp": telemetry.stamp(t), "frame_id": "base_link"},
                "voltage": self.sim.battery.voltage(), "present": True}

    def camera_info(self, width, height):
        model = self.sim.model
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, ONBOARD)
        f = (height / 2) / math.tan(model.cam_fovy[cam] * math.pi / 360)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.cam_quat[cam])
        optical = rotation.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, optical.ravel())
        if quat[0] < 0:
            quat = -quat
        w, x, y, z = (round(float(v), 9) for v in quat)
        return {"front": {"track_name": ONBOARD, "width": width, "height": height, "views": {"mono": {
            "frame_id": "front_optical", "image_region": {"x": 0, "y": 0, "width": width, "height": height},
            "intrinsics": {"fx": f, "fy": f, "cx": width / 2, "cy": height / 2,
                           "distortion_model": "none", "distortion_coefficients": []},
            "extrinsics": {"parent_frame_id": "base_link",
                           "translation": [round(float(v), 9) for v in model.cam_pos[cam]],
                           "rotation_xyzw": [x, y, z, w]}}}}}

    # -------------------------------------------------------------- rendering
    batch_rendering = False

    def render_size(self, key):
        return self.views.render_size(key)

    def open_renderers(self):
        self.views.open_renderers()

    def prepare(self, key):
        return self.views.prepare(key)

    def close(self):
        if self.views:
            self.views.close()
