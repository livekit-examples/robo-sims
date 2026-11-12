"""Vehicle policy, strict Ackermann parsing and headless/rendered backend."""
import math
from pathlib import Path
import time
from types import SimpleNamespace

from ..assets import SOURCES
from ..contract.config import load_json
from .dynamics import Car, Drive
from .track import Track
from .telemetry import INSTRUMENT_SCHEMAS, instrument_value

PROFILE = "base.ackermann_drive.v1"
COMMAND = "ackermann_msgs/msg/AckermannDrive"
STATE = "robo_sims/msg/VehicleStateV1"
COMMAND_TRACK = "/ackermann_cmd"
STATE_TRACK = "/vehicle_state"
ODOM_TRACK = "/odom"

ACKERMANN_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "required": ["speed", "steering_angle"], "additionalProperties": False,
    "properties": {key: {"type": "number"} for key in (
        "speed", "steering_angle", "steering_angle_velocity", "acceleration", "jerk")}}


class Backend:
    def __init__(self, *, max_speed=8.0, fps=60, width=960, height=540,
                 renderer="classic", video=True, clock=time.monotonic, track_path=None,
                 camera="tcam", graphics="auto"):
        if type(max_speed) not in (int, float) or not math.isfinite(max_speed) or not 0 < max_speed <= 20:
            raise ValueError("max-speed must be finite, positive and at most the community ceiling of 20 m/s")
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("fps must be an integer between 1 and 60")
        if any(type(v) is not int or v < 2 or v % 2 or v > 1920 for v in (width, height)):
            raise ValueError("video dimensions must be positive even integers at most 1920")
        self.clock = clock
        self.max_speed = float(max_speed)
        self.min_speed = -min(5.0, self.max_speed)
        self.timeout = 0.15
        self.last_command = None
        self.timed_out = False
        path = track_path or SOURCES["f1tenth_racetracks"].path / "Spielberg/Spielberg_centerline.csv"
        self.car = Car(Track(Path(path)))
        self.scene = None
        self.feeds = {}
        document = {"schema_version": 1, "label": "1/10 Formula RC · Spielberg",
            "default_video_rate_hz": fps,
            "control": {"scope": "robot", "ownership": "portal_active_operator"},
            "control_interfaces": {"base": {"label": "Race car", "frame_id": "base_link",
                "operations": {"drive": {"profile": PROFILE,
                    "command": {"track_name": COMMAND_TRACK, "msg_type": COMMAND, "format": "json"}}}}}}
        self.config = SimpleNamespace(discovery=document,
            command_tracks={COMMAND_TRACK: ("base", document["control_interfaces"]["base"]["operations"]["drive"])})
        if video:
            from .scene import Scene
            self.scene = Scene(self.car, width, height, renderer, camera=camera, graphics=graphics)
            self.feeds = self.scene.feeds
            document["camera_info"] = self.scene.camera_info()

    def policy(self):
        p = self.car.p
        return {"version": 1, "kind": "rc.ackermann.v1", "interface_id": "base", "operation_id": "drive",
            "command_track": COMMAND_TRACK, "profile": PROFILE, "control_channel": "primary",
            "min_speed": self.min_speed, "max_speed": self.max_speed,
            "max_steering_angle": p["max_steering"], "max_steering_rate": p["max_steering_rate"],
            "steering_input_exponent": 2,
            "max_acceleration": p["max_acceleration"], "max_braking": p["max_acceleration"],
            "command_timeout_ms": round(self.timeout * 1000), "input_interval_ms": 20,
            "neutral": "brake_and_center", "state_track": STATE_TRACK}

    def parse(self, payload):
        if not isinstance(payload, (bytes, str)) or len(payload) > 4096:
            raise ValueError("Ackermann command must be JSON of at most 4096 bytes")
        value = load_json(payload)
        fields = ACKERMANN_SCHEMA["properties"]
        if not isinstance(value, dict) or not {"speed", "steering_angle"} <= value.keys() or value.keys() - fields.keys():
            raise ValueError("expected AckermannDrive speed and steering_angle")
        for name, v in value.items():
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
            if name not in ("speed", "steering_angle") and v != 0:
                raise ValueError(f"nonzero {name} is not supported; internal policy limits apply")
        if not self.min_speed <= value["speed"] <= self.max_speed \
                or abs(value["steering_angle"]) > self.car.p["max_steering"]:
            raise ValueError("drive command exceeds the advertised limits")
        return Drive(float(value["speed"]), float(value["steering_angle"]))

    def submit(self, payload):
        command = self.parse(payload)
        self.car.target = command
        self.last_command = self.clock()
        self.timed_out = False

    def hold(self, _interface=None):
        self.car.target = Drive(0, 0)
        self.last_command = None

    def watchdog(self):
        if self.last_command is not None and self.clock() - self.last_command > self.timeout:
            self.hold()
            self.timed_out = True
            return True
        return False

    def step(self, dt=0.005):
        self.car.step(dt)

    def state(self):
        return {**self.car.state(), "command_timed_out": self.timed_out,
                "command_speed": self.car.target.speed, "speed_limit": self.max_speed}

    def instrument_tracks(self):
        return dict(INSTRUMENT_SCHEMAS)

    def instrument_value(self, name):
        return instrument_value(self.car, name)

    def battery_state(self):
        sec = int(self.car.time)
        return {"header": {"stamp": {"sec": sec, "nanosec": int((self.car.time - sec) * 1e9)},
                           "frame_id": "base_link"},
                "voltage": self.car.battery.voltage(), "present": True}

    def odometry(self):
        car = self.car
        sec = int(car.time)
        return {"header": {"stamp": {"sec": sec, "nanosec": int((car.time - sec) * 1e9)}, "frame_id": "map"},
            "child_frame_id": "base_link", "pose": {"pose": {
                "position": {"x": float(car.x), "y": float(car.y), "z": 0.065},
                "orientation": {"x": 0, "y": 0, "z": math.sin(car.yaw / 2), "w": math.cos(car.yaw / 2)}},
                "covariance": [0] * 36}, "twist": {"twist": {
                    "linear": {"x": car.vx, "y": car.vy, "z": 0},
                    "angular": {"x": 0, "y": 0, "z": car.yaw_rate}}, "covariance": [0] * 36}}

    @property
    def batch_rendering(self):
        return self.scene.batch_rendering if self.scene else False

    def render_size(self, key):
        return self.scene.render_size(key)

    def open_renderers(self):
        self.scene.open_renderers()

    def prepare(self, key):
        return self.scene.prepare(key)

    def prepare_batch(self, keys):
        return self.scene.prepare_batch(keys)

    def close(self):
        if self.scene:
            self.scene.close()
