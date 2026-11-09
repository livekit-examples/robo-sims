"""Named position-actuator backend; scene construction/rendering stays in the scene adapter."""

from __future__ import annotations

from dataclasses import dataclass

from .. import mujoco_tools  # noqa: F401 - selects GL before importing MuJoCo
import mujoco
import numpy as np

from .config import Configuration, ControlKey, GRIPPER, channel_key, operation_joints
from .trajectory import Trajectory
from .camera_info import validate_camera_info


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
        self.config = config
        self.scene = scene
        self.model, self.data = scene.model, scene.data
        self.feeds = scene.feeds
        self.batch_rendering = scene.batch_rendering
        self.joints: dict[ControlKey, dict[str, Joint]] = {}
        self.motions: dict[ControlKey, Motion] = {}
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
            interface = config.discovery["control_interfaces"][key]
            for operation in interface["operations"].values():
                target = channel_key(key, operation)
                gripper = operation["profile"] == GRIPPER
                joints = self.joints[target] = {}
                for name in operation_joints(interface, operation):
                    spec = binding["gripper"] if gripper else binding["joint_map"][name]
                    joint = self._named(mujoco.mjtObj.mjOBJ_JOINT, spec["joint"])
                    actuator = self._named(mujoco.mjtObj.mjOBJ_ACTUATOR, spec["actuator"])
                    if joint in used_joints or actuator in used_actuators:
                        raise ValueError("joint/actuator bindings must be exclusive to one control channel")
                    used_joints.add(joint)
                    used_actuators.add(actuator)
                    allowed_joints = expected.joint_ids[6:] if gripper else expected.joint_ids[:6]
                    allowed_actuators = expected.actuator_ids[6:] if gripper else expected.actuator_ids[:6]
                    if joint not in allowed_joints or actuator not in allowed_actuators:
                        raise ValueError(f"joint/actuator does not belong to {key}")
                    kind = mujoco.mjtJoint.mjJNT_SLIDE if gripper else mujoco.mjtJoint.mjJNT_HINGE
                    if self.model.jnt_type[joint] != kind or not self.model.jnt_limited[joint]:
                        raise ValueError("profile requires a limited slide gripper or revolute arm joint")
                    if self.model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT \
                            or self.model.actuator_trnid[actuator, 0] != joint \
                            or not np.allclose(self.model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]) \
                            or self.model.actuator_gainprm[actuator, 0] <= 0 \
                            or not np.isclose(self.model.actuator_biasprm[actuator, 1],
                                              -self.model.actuator_gainprm[actuator, 0]):
                        raise ValueError("trajectory backend requires direct joint position servos")
                    low, high = self.model.jnt_range[joint]
                    if self.model.actuator_ctrllimited[actuator]:
                        lo, hi = self.model.actuator_ctrlrange[actuator]
                        low, high = max(low, lo), min(high, hi)
                    if gripper:
                        coupled = self._named(mujoco.mjtObj.mjOBJ_JOINT, spec["coupled_joint"])
                        if self.model.jnt_type[coupled] != mujoco.mjtJoint.mjJNT_SLIDE \
                                or self.model.jnt_bodyid[coupled] == self.model.jnt_bodyid[joint]:
                            raise ValueError("gripper coupling must connect distinct slide joints")
                        matches = [i for i in range(self.model.neq)
                                   if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
                                   and {int(self.model.eq_obj1id[i]), int(self.model.eq_obj2id[i])} == {joint, coupled}
                                   and self.model.eq_active0[i]
                                   and np.allclose(self.model.eq_data[i, :5], [0, 1, 0, 0, 0])]
                        if len(matches) != 1:
                            raise ValueError("gripper requires verified equal-displacement finger coupling")
                        if coupled in used_joints or coupled in self.model.actuator_trnid[:, 0][
                                self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT]:
                            raise ValueError("coupled finger cannot be independently actuated")
                        used_joints.add(coupled)
                        if not self.model.jnt_limited[coupled]:
                            raise ValueError("coupled finger requires position limits")
                        lo, hi = self.model.jnt_range[coupled]
                        low, high = max(low, lo), min(high, hi)
                    if low >= high:
                        raise ValueError("joint and actuator limits do not overlap")
                    joints[name] = Joint(name, int(self.model.jnt_qposadr[joint]),
                                         int(self.model.jnt_dofadr[joint]), actuator,
                                         (float(low), float(high)), spec["max_velocity"])
        for keys in config.feedback_tracks.values():
            public_names = [name for key in keys for name in self.joints[key]]
            if len(set(public_names)) != len(public_names):
                raise ValueError("shared feedback requires globally unique joint names")
        camera_info = scene.camera_info()
        validate_camera_info(camera_info)
        config.discovery["camera_info"] = camera_info

    def policy(self):
        """The same versioned capabilities feed manual discovery and managed runs."""
        operations = {}
        for key, interface in self.config.discovery["control_interfaces"].items():
            entries = operations[key] = {}
            for op_id, op in interface["operations"].items():
                joints = self.joints[channel_key(key, op)]
                gripper = op["profile"] == GRIPPER
                entry = entries[op_id] = {
                    "profile": op["profile"], "control_channel": op.get("control_channel", "primary"),
                    "joint_names": list(joints), "units": "m" if gripper else "rad",
                    "joints": {name: {"min": j.limits[0], "max": j.limits[1], "max_velocity": j.speed}
                               for name, j in joints.items()},
                }
                if gripper:
                    joint = next(iter(joints.values()))
                    entry.update(coordinate_label="Finger displacement", closed_position=joint.limits[0],
                                 open_position=joint.limits[1])
        return {"version": 2, "max_feedback_age_ms": 1000, "operations": operations}

    def _named(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"model is missing {name}")
        return result

    def submit(self, interface_id: ControlKey, trajectory: Trajectory) -> None:
        joints = self.joints[interface_id]
        selected = [joints[name] for name in trajectory.names]
        if not trajectory.times:
            self.hold(interface_id)
            return
        start = tuple(float(self.targets[j.actuator]) for j in selected)
        trajectory.validate_motion(start, [j.limits for j in selected], [j.speed for j in selected])
        # Replacing one channel leaves the same arm's other channels running.
        self.motions.pop(interface_id, None)
        self.motions[interface_id] = Motion(trajectory, float(self.data.time), start)

    def hold(self, interface_id: ControlKey | None = None) -> None:
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

    def feedback(self, interface_ids: list[ControlKey]) -> dict:
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

    def prepare_batch(self, keys):
        return self.scene.prepare_batch(keys)

    def close(self):
        self.scene.close()
