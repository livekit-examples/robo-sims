"""MuJoCo room, differential-drive robot and the cleaning run. No networking and no wall clock."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from ..mujoco_tools import mujoco
from . import room
from .floor import Battery, Bin, Floor
from .room import ROBOT_HEIGHT, ROBOT_RADIUS

# Initial driving limits and robot constants from docs/vacuum-design.md.
WHEEL_RADIUS = 0.035
WHEEL_SPACING = 0.235
MAX_FORWARD = 0.45
MAX_REVERSE = 0.25
MAX_YAW_RATE = 1.8
MAX_ACCEL = 0.8
MAX_YAW_ACCEL = 5.0
PRECISION_SPEED = 0.15
CHASSIS_MASS = 3.0
PHYSICS_DT = 0.005
DEADLINE_S = 60.0
CREEP_SPEED = 0.03        # assist never slows below this, so the robot can still touch a wall
REACTION_S = 0.1          # reaction time the assist adds to the stopping distance

START_CONFIGS = {
    "normal": {"battery_wh": 30.0, "bin_used_l": 0.0},
    "low_battery": {"battery_wh": 0.05, "bin_used_l": 0.0},
    "nearly_full_bin": {"battery_wh": 30.0, "bin_used_l": 0.49},
}

OVERHEAD = "overhead"
ONBOARD = "onboard"


def numbers(values) -> str:
    return " ".join(f"{v:.7g}" for v in values)


def wheel_speeds(v: float, omega: float):
    """Wheel angular speeds in rad/s for forward speed v and yaw rate omega. Positive yaw turns left."""
    half = omega * WHEEL_SPACING / 2
    return (v - half) / WHEEL_RADIUS, (v + half) / WHEEL_RADIUS


def limit_wheel_targets(left: float, right: float):
    """Scale both wheel targets by one factor when either exceeds its limit, keeping the turn shape."""
    cap = (MAX_FORWARD + MAX_YAW_RATE * WHEEL_SPACING / 2) / WHEEL_RADIUS
    peak = max(abs(left), abs(right))
    if peak > cap:
        return left * cap / peak, right * cap / peak
    return left, right


def assist_limit(clearance: float, accel: float = MAX_ACCEL, reaction: float = REACTION_S) -> float:
    """Highest forward speed that still stops within `clearance` meters at `accel`."""
    return accel * (math.sqrt(reaction * reaction + 2 * max(0.0, clearance) / accel) - reaction)


def _xyaxes(position, target):
    forward = np.subtract(target, position).astype(float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return numbers([*right, *np.cross(right, forward)])


def model_xml(width=960, height=540) -> str:
    root = Element("mujoco", model="vacuum")
    SubElement(root, "compiler", angle="radian")
    SubElement(root, "option", timestep=str(PHYSICS_DT), integrator="implicitfast", gravity="0 0 -9.81")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    SubElement(visual, "map", znear="0.01", zfar="20")
    SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.5 0.5 0.5", specular="0.05 0.05 0.05")
    # MuJoCo uses the larger friction of two touching geoms. The floor is slick so the casters slide,
    # the wheels carry the grip, and walls and furniture do not hold the chassis.
    defaults = SubElement(root, "default")
    SubElement(defaults, "geom", friction="0.2 0.005 0.0001")
    assets = SubElement(root, "asset")
    SubElement(assets, "texture", name="sky", type="skybox", builtin="flat", rgb1="0.93 0.93 0.95", width="8", height="8")
    # The floor image: one pixel per 1 cm grid square, replaced from the grid at the video rate.
    SubElement(assets, "texture", name="floor", type="2d", builtin="flat", rgb1="0.77 0.67 0.54",
               width=str(room.N), height=str(room.N))
    materials = {
        "floor": dict(texture="floor", specular="0", shininess="0"),
        "wall": dict(rgba="0.92 0.92 0.90 1", specular="0.05"),
        "wood": dict(rgba="0.55 0.38 0.22 1", specular="0.15"),
        "seat": dict(rgba="0.55 0.38 0.22 0.35", specular="0.1"),
        "shade": dict(rgba="0.95 0.85 0.55 0.45", emission="0.3"),
        "cabinet": dict(rgba="0.36 0.45 0.52 1", specular="0.1"),
        "metal": dict(rgba="0.72 0.72 0.75 1", specular="0.5", shininess="0.6"),
        "lamp": dict(rgba="0.95 0.85 0.55 1", emission="0.3"),
        "shell": dict(rgba="0.16 0.17 0.20 1", specular="0.4", shininess="0.5"),
        "lid": dict(rgba="0.80 0.82 0.86 1", specular="0.3"),
        "arrow": dict(rgba="0.10 0.55 0.95 1", emission="0.5"),
        "tyre": dict(rgba="0.05 0.05 0.05 1", specular="0.05"),
    }
    for name, attributes in materials.items():
        SubElement(assets, "material", name=name, **attributes)

    world = SubElement(root, "worldbody")
    SubElement(world, "light", name="sun", pos="0 0 4", dir="0.1 0.1 -1", directional="true",
               castshadow="false", diffuse="0.55 0.55 0.55")
    SubElement(world, "camera", name=OVERHEAD, pos="0 0 3.0", xyaxes="1 0 0 0 1 0", fovy="45")
    SubElement(world, "geom", name="floor", type="box", pos="0 0 -0.01", size=f"{room.HALF} {room.HALF} 0.01",
               material="floor", friction="0.02 0.001 0.0001")

    half, t, h = room.HALF, room.WALL_THICKNESS, room.WALL_HEIGHT
    for name, pos, size in (("wall_n", (0, half + t / 2, h / 2), (half + t, t / 2, h / 2)),
                            ("wall_s", (0, -half - t / 2, h / 2), (half + t, t / 2, h / 2)),
                            ("wall_e", (half + t / 2, 0, h / 2), (t / 2, half, h / 2)),
                            ("wall_w", (-half - t / 2, 0, h / 2), (t / 2, half, h / 2))):
        SubElement(world, "geom", name=name, type="box", pos=numbers(pos), size=numbers(size), material="wall")

    cx, cy = room.CHAIR_CENTER
    seat_z = room.CHAIR_SEAT_BOTTOM
    for i, (lx, ly) in enumerate(room.chair_legs()):
        SubElement(world, "geom", name=f"chair_leg_{i}", type="cylinder", pos=numbers((lx, ly, seat_z / 2)),
                   size=f"{room.CHAIR_LEG_RADIUS} {seat_z / 2}", material="wood")
    reach = room.CHAIR_LEG_SPACING / 2
    SubElement(world, "geom", name="chair_seat", type="box", pos=numbers((cx, cy, seat_z + 0.015)),
               size=numbers((reach + 0.02, reach + 0.02, 0.015)), material="seat")
    for name, pos, size in (("chair_bar_w", (cx - reach, cy, room.CHAIR_BAR_Z), (0.008, reach, 0.008)),
                            ("chair_bar_e", (cx + reach, cy, room.CHAIR_BAR_Z), (0.008, reach, 0.008)),
                            ("chair_bar_s", (cx, cy - reach, room.CHAIR_BAR_Z), (reach, 0.008, 0.008))):
        SubElement(world, "geom", name=name, type="box", pos=numbers(pos), size=numbers(size), material="wood")
    for sx in (-1, 1):  # chair back, above the seat; visual only
        SubElement(world, "geom", type="cylinder", pos=numbers((cx + sx * reach, cy + reach, seat_z + 0.22)),
                   size="0.02 0.22", material="wood", contype="0", conaffinity="0")
    SubElement(world, "geom", type="box", pos=numbers((cx, cy + reach, seat_z + 0.36)), size=numbers((reach, 0.012, 0.08)),
               material="seat", contype="0", conaffinity="0")

    ccx, ccy = room.CABINET_CENTER
    SubElement(world, "geom", name="cabinet", type="box", pos=numbers((ccx, ccy, room.CABINET_HEIGHT / 2)),
               size=numbers((*room.CABINET_HALF, room.CABINET_HEIGHT / 2)), material="cabinet")
    SubElement(world, "geom", type="box", pos=numbers((ccx, ccy, room.CABINET_HEIGHT + 0.01)),
               size=numbers((room.CABINET_HALF[0] + 0.01, room.CABINET_HALF[1] + 0.01, 0.01)), material="wood",
               contype="0", conaffinity="0")
    lx, ly = room.LAMP_CENTER
    SubElement(world, "geom", name="lamp_base", type="cylinder", pos=numbers((lx, ly, room.LAMP_HEIGHT / 2)),
               size=f"{room.LAMP_RADIUS} {room.LAMP_HEIGHT / 2}", material="metal")
    SubElement(world, "geom", name="lamp_pole", type="cylinder", pos=numbers((lx, ly, 0.55)), size="0.012 0.5",
               material="metal")
    SubElement(world, "geom", type="cylinder", pos=numbers((lx, ly, 1.1)), size="0.09 0.07", material="shade",
               contype="0", conaffinity="0")

    # Robot: free chassis, two powered wheel hinges, two low-friction casters.
    spawn_x, spawn_y, spawn_yaw = room.SPAWN
    body = SubElement(world, "body", name="base_link", pos=numbers((spawn_x, spawn_y, 0.0)),
                      euler=numbers((0, 0, spawn_yaw)))
    SubElement(body, "freejoint", name="base_free")
    SubElement(body, "geom", name="chassis", type="cylinder", size=f"{ROBOT_RADIUS} {ROBOT_HEIGHT / 2}",
               pos=numbers((0, 0, room.GROUND_CLEARANCE + ROBOT_HEIGHT / 2)), mass="2.5", material="shell")
    for name, x in (("caster_f", 0.12), ("caster_r", -0.12)):
        SubElement(body, "geom", name=name, type="sphere", size="0.015", pos=numbers((x, 0, 0.015)), mass="0.05",
                   friction="0.02 0.001 0.0001")
    # Decorative details never collide.
    deco = dict(contype="0", conaffinity="0", mass="0")
    top = room.ROBOT_TOP
    SubElement(body, "geom", type="cylinder", size="0.115 0.003", pos=numbers((-0.02, 0, top + 0.003)), material="lid", **deco)
    SubElement(body, "geom", type="cylinder", size="0.03 0.004", pos=numbers((0.06, 0, top + 0.007)), material="shell", **deco)
    # Direction arrow: shaft and chevron head, readable from the overhead camera.
    SubElement(body, "geom", type="box", size="0.07 0.009 0.002", pos=numbers((-0.05, 0, top + 0.009)), material="arrow", **deco)
    for sy in (-1, 1):
        SubElement(body, "geom", type="box", size="0.04 0.009 0.002", pos=numbers((0.05, sy * 0.025, top + 0.009)),
                   euler=numbers((0, 0, -sy * 0.8)), material="arrow", **deco)
    camera_pos = (0.168, 0.0, 0.075)
    SubElement(body, "camera", name=ONBOARD, pos=numbers(camera_pos),
               xyaxes=_xyaxes(camera_pos, (1.0, 0.0, 0.03)),
               fovy=str(math.degrees(2 * math.atan(math.tan(math.radians(100) / 2) * height / width))))
    for side, y in (("l", WHEEL_SPACING / 2), ("r", -WHEEL_SPACING / 2)):
        wheel = SubElement(body, "body", name=f"wheel_{side}", pos=numbers((0, y, WHEEL_RADIUS)))
        SubElement(wheel, "joint", name=f"wheel_{side}", type="hinge", axis="0 1 0", damping="0.001")
        SubElement(wheel, "geom", type="cylinder", size=f"{WHEEL_RADIUS} 0.011", euler=numbers((math.pi / 2, 0, 0)),
                   mass="0.2", friction="1.0 0.005 0.0001", material="tyre")
    actuators = SubElement(root, "actuator")
    cap = (MAX_FORWARD + MAX_YAW_RATE * WHEEL_SPACING / 2) / WHEEL_RADIUS
    for side in ("l", "r"):
        SubElement(actuators, "velocity", name=f"wheel_{side}", joint=f"wheel_{side}", kv="0.15",
                   ctrlrange=numbers((-cap, cap)), forcerange="-0.1 0.1")
    return tostring(root, encoding="unicode")


@dataclass
class Command:
    """Requested motion: forward speed in m/s, yaw rate in rad/s. Positive yaw turns left."""
    v: float = 0.0
    omega: float = 0.0


class Vacuum:
    def __init__(self, *, width=960, height=540, start="normal", assist=True, floor=None):
        if start not in START_CONFIGS:
            raise ValueError(f"start must be one of {', '.join(START_CONFIGS)}")
        self.start = start
        self.assist_enabled = assist
        self.width, self.height = width, height
        self.floor = floor or Floor()
        self.model = mujoco.MjModel.from_xml_string(model_xml(width, height))
        self.data = mujoco.MjData(self.model)
        self.texture_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "floor")
        self.joint_wheels = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"wheel_{s}")]
                             for s in "lr"]
        self.run_counter = 0
        self.uptime = 0.0   # simulated seconds since creation; unlike `time` it survives resets
        self.requested = Command()
        self.reset()

    # ----------------------------------------------------------- lifecycle
    def reset(self):
        """Back to the start pose with a clean floor and a new run ID. Driving needs fresh commands."""
        mujoco.mj_resetData(self.model, self.data)
        x, y, yaw = room.SPAWN
        self.data.qpos[0:3] = [x, y, 0.0]
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        config = START_CONFIGS[self.start]
        self.battery = Battery(remaining_wh=config["battery_wh"])
        self.bin = Bin(used_l=config["bin_used_l"])
        self.floor.reset()
        self.requested = Command()
        self.shaped = Command()
        self.assist_active = False
        self.clearance = room.forward_clearance(x, y, yaw)
        self._assist_cooldown = 0
        self.drive_enabled = True
        self.state = "idle"       # idle, running, complete or timeout
        self.result = None
        self.started_at = None
        self.finished_at = None
        self.elapsed = 0.0
        self.overloads = 0
        self.trial_valid = True
        self.run_counter += 1
        self.run_id = f"run-{self.run_counter:04d}"
        self.sequence = 0
        self.last_xy = self.position()[:2]

    def place(self, x, y, yaw=0.0):
        """Put the robot at a pose with zero velocity. For tests and reference tools."""
        self.data.qpos[0:3] = [x, y, 0.0]
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.last_xy = (x, y)
        self.shaped = Command()

    def start_run(self):
        if self.state != "idle":
            raise ValueError("the run already started; reset before another attempt")
        self.state = "running"
        self.started_at = self.data.time

    def record_overload(self):
        self.overloads += 1
        self.trial_valid = False
        self.drive_enabled = False

    def enable_driving(self):
        if not self.battery.empty:
            self.drive_enabled = True

    # --------------------------------------------------------- measurements
    def position(self):
        q = self.data.qpos
        w, z = q[3], q[6]
        yaw = math.atan2(2 * w * z, 1 - 2 * z * z)
        return float(q[0]), float(q[1]), yaw

    def velocity(self):
        """Measured body-frame forward speed, sideways speed and yaw rate."""
        _, _, yaw = self.position()
        vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
        return (math.cos(yaw) * vx + math.sin(yaw) * vy, -math.sin(yaw) * vx + math.cos(yaw) * vy,
                float(self.data.qvel[5]))

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def vacuum_on(self) -> bool:
        return self.state == "running" and not self.battery.empty and not self.bin.full

    # ------------------------------------------------------------- stepping
    def _approach(self, current, target, rate, dt):
        step = rate * dt
        return current + max(-step, min(step, target - current))

    def _assist(self, v_request, x, y, yaw):
        if not self.assist_enabled or v_request <= 0:
            self.assist_active = False
            return v_request
        if self._assist_cooldown <= 0:  # the clearance scan runs at 50 Hz
            self.clearance = room.forward_clearance(x, y, yaw)
            self._assist_cooldown = 4
        self._assist_cooldown -= 1
        allowed = max(assist_limit(self.clearance), CREEP_SPEED)
        self.assist_active = allowed < v_request
        return min(v_request, allowed)

    def step(self, dt=PHYSICS_DT):
        count = max(1, round(dt / PHYSICS_DT))
        if not math.isfinite(dt) or abs(dt - count * PHYSICS_DT) > 1e-9:
            raise ValueError("the physics step must be a multiple of 0.005 seconds")
        for _ in range(count):
            self._step_once()

    def _step_once(self):
        self.uptime += PHYSICS_DT
        x, y, yaw = self.position()
        enabled = self.drive_enabled and not self.battery.empty
        v = max(-MAX_REVERSE, min(MAX_FORWARD, self.requested.v)) if enabled else 0.0
        omega = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, self.requested.omega)) if enabled else 0.0
        v = self._assist(v, x, y, yaw)
        self.shaped.v = self._approach(self.shaped.v, v, MAX_ACCEL, PHYSICS_DT)
        self.shaped.omega = self._approach(self.shaped.omega, omega, MAX_YAW_ACCEL, PHYSICS_DT)
        left, right = limit_wheel_targets(*wheel_speeds(self.shaped.v, self.shaped.omega))
        self.data.ctrl[:] = [left, right]
        previous = self.last_xy
        mujoco.mj_step(self.model, self.data)
        x, y, _ = self.position()
        speed, _, yaw_rate = self.velocity()
        vacuum_on = self.vacuum_on
        self.battery.step(PHYSICS_DT, speed / MAX_FORWARD, yaw_rate / MAX_YAW_RATE, vacuum_on)
        if vacuum_on:
            cleaned = self.floor.sweep(previous, (x, y), self.bin.cells_allowed())
            self.bin.add_cells(cleaned)
        self.last_xy = (x, y)
        self._advance_run()

    def _advance_run(self):
        if self.state != "running":
            return
        self.elapsed = self.time - self.started_at
        if self.floor.complete:
            self._finish("complete")
        elif self.elapsed >= DEADLINE_S:
            self.elapsed = DEADLINE_S
            self._finish("timeout")

    def _finish(self, result):
        self.state, self.result, self.finished_at = result, result, self.time

    # ------------------------------------------------------------- states
    def stamp(self) -> dict:
        self.sequence += 1
        return {"sim_time_s": self.time, "run_id": self.run_id, "sequence": self.sequence}

    def coverage_state(self) -> dict:
        areas = self.floor.areas_m2()
        return {**self.stamp(), **areas, "percent": self.floor.percent, "vacuum_on": self.vacuum_on}

    def run_state(self) -> dict:
        return {**self.stamp(), "state": self.state, "elapsed_s": self.elapsed, "deadline_s": DEADLINE_S,
                "result": self.result or "none", "overload": self.overloads > 0,
                "overload_count": self.overloads, "trial_valid": self.trial_valid}
