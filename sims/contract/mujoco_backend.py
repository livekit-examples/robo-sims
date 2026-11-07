"""Named position-actuator backend; scene construction/rendering stays in the scene adapter."""

from __future__ import annotations

from dataclasses import dataclass

from .. import mujoco_tools  # noqa: F401 - selects GL before importing MuJoCo
import mujoco
import numpy as np

from .config import Configuration
from .trajectory import Trajectory


@dataclass
class Joint:
    name: str
    qpos: int
    dof: int
    actuator: int
    limits: tuple[float, float]
    speed: float


@dataclass
class Motion:
    trajectory: Trajectory
    start_time: float
    start: tuple[float, ...]


class MujocoBackend:
    def __init__(self, scene, config: Configuration):
        self.scene = scene
        self.model, self.data = scene.model, scene.data
        self.feeds = scene.feeds
        self.joints: dict[str, dict[str, Joint]] = {}
        self.motions: dict[str, Motion] = {}
        self.targets = self.data.ctrl.copy()
        # Keep all scene actuators, including the unadvertised grippers, at their initial pose.
        for arm in scene.arms.values():
            self.targets[arm.actuator_ids] = arm.setpoints
        used_joints, used_actuators = set(), set()
        for key, binding in config.bindings["control_interfaces"].items():
            if binding["backend_arm"] not in scene.arms:
                raise ValueError(f"unknown backend arm: {binding['backend_arm']}")
            base = self._named(mujoco.mjtObj.mjOBJ_BODY, binding["base_body"])
            site = self._named(mujoco.mjtObj.mjOBJ_SITE, binding["end_effector_site"])
            expected = scene.arms[binding["backend_arm"]]
            if base != expected.base_id or site != expected.site_id:
                raise ValueError(f"base/site does not belong to {key}")
            joints = self.joints[key] = {}
            for name in config.discovery["control_interfaces"][key]["joint_names"]:
                spec = binding["joint_map"][name]
                joint = self._named(mujoco.mjtObj.mjOBJ_JOINT, spec["joint"])
                actuator = self._named(mujoco.mjtObj.mjOBJ_ACTUATOR, spec["actuator"])
                if joint in used_joints or actuator in used_actuators:
                    raise ValueError("joint/actuator bindings must be exclusive to one interface")
                used_joints.add(joint)
                used_actuators.add(actuator)
                if joint not in expected.joint_ids[:6] or actuator not in expected.actuator_ids[:6]:
                    raise ValueError(f"joint/actuator does not belong to {key}")
                if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE \
                        or not self.model.jnt_limited[joint]:
                    raise ValueError("trajectory backend requires limited revolute joints in radians")
                if self.model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT \
                        or self.model.actuator_trnid[actuator, 0] != joint \
                        or not np.allclose(self.model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]) \
                        or self.model.actuator_gainprm[actuator, 0] <= 0 \
                        or not np.isclose(self.model.actuator_biasprm[actuator, 1],
                                          -self.model.actuator_gainprm[actuator, 0]):
                    raise ValueError("trajectory backend requires direct joint position servos")
                joints[name] = Joint(name, int(self.model.jnt_qposadr[joint]),
                                     int(self.model.jnt_dofadr[joint]), actuator,
                                     tuple(self.model.jnt_range[joint]), spec["max_velocity"])

    def _named(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"model is missing {name}")
        return result

    def submit(self, interface_id: str, trajectory: Trajectory) -> None:
        joints = self.joints[interface_id]
        selected = [joints[name] for name in trajectory.names]
        if not trajectory.times:
            self.hold(interface_id)
            return
        start = tuple(float(self.targets[j.actuator]) for j in selected)
        trajectory.validate_motion(start, [j.limits for j in selected], [j.speed for j in selected])
        # Replacing one interface cancels its prior motion, leaving other arms alone.
        self.motions.pop(interface_id, None)
        self.motions[interface_id] = Motion(trajectory, float(self.data.time), start)

    def hold(self, interface_id: str | None = None) -> None:
        for key in self.joints if interface_id is None else (interface_id,):
            self.motions.pop(key, None)
            for joint in self.joints[key].values():
                self.targets[joint.actuator] = np.clip(self.data.qpos[joint.qpos], *joint.limits)

    def step(self, dt: float = 0.01) -> None:
        for _ in range(max(1, round(dt / self.model.opt.timestep))):
            for key, motion in list(self.motions.items()):
                elapsed = self.data.time - motion.start_time
                values = motion.trajectory.sample(elapsed, motion.start)
                for name, value in zip(motion.trajectory.names, values):
                    self.targets[self.joints[key][name].actuator] = value
                if elapsed >= motion.trajectory.times[-1]:
                    self.motions.pop(key)
            self.data.ctrl[:] = self.targets
            mujoco.mj_step(self.model, self.data)
            after_step = getattr(self.scene, "after_physics_step", None)
            if after_step is not None:
                after_step()

    def feedback(self, interface_ids: list[str]) -> dict:
        joints = [joint for key in interface_ids for joint in self.joints[key].values()]
        if len({j.name for j in joints}) != len(joints):
            raise ValueError("shared feedback requires globally unique joint names")
        at = float(self.data.time)
        sec = int(at)
        return {"header": {"stamp": {"sec": sec, "nanosec": int((at - sec) * 1e9)}, "frame_id": ""},
                "name": [j.name for j in joints],
                "position": [float(self.data.qpos[j.qpos]) for j in joints],
                "velocity": [float(self.data.qvel[j.dof]) for j in joints], "effort": []}

    def render_size(self, key):
        return self.scene.render_size(key)

    def open_renderers(self):
        self.scene.open_renderers()

    def prepare(self, key):
        return self.scene.prepare(key)

    def close(self):
        self.scene.close()
