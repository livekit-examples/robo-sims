"""The B601-DM hardware as MuJoCo sees it: which motor drives which joint, and how.

Seeed's MuJoCo model (rebotarm_b601_colored.xml) is loaded unchanged from disk
and adjusted in memory to match the real arm:

- Each Damiao motor becomes an actuator whose force law is set by the emulated
  firmware every step. Joints 1-6 are direct drive in motor coordinates (the
  same 1:1 mapping Seeed's real driver uses).
- The gripper is one 4310 motor driving both fingers through a rack. Seeed's
  real driver treats 0 rad as closed and -5 rad as fully open
  (hardware_manager.py, _G_ANGLE_OPEN), and the URDF limits each finger to
  0.0285 m, so -5 rad moves each finger 0.0285 m. The model's 0.05 m finger
  range is narrowed to match, and the right finger mirrors the left one.
- The motors' rotor inertia is added to the gripper, and the integrator is
  implicit so stiff motor gains stay stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .damiao import MODELS, DamiaoMotor

# Gripper transmission. The ratio is inferred from Seeed's driver and URDF and
# should be confirmed on the real arm.
GRIPPER_OPEN_MOTOR_RAD = -5.0
FINGER_TRAVEL_M = 0.0285
GRIPPER_GEAR = GRIPPER_OPEN_MOTOR_RAD / FINGER_TRAVEL_M  # motor rad per finger metre
# Output-side inertia of the gripper motor and gearbox (kg*m^2), reflected onto
# the finger slide as armature. An estimate.
GRIPPER_MOTOR_INERTIA = 3.0e-4


@dataclass(frozen=True)
class MotorSpec:
    name: str
    motor_id: int
    feedback_id: int
    model: str
    joint: str  # MuJoCo joint the motor drives
    gear: float  # motor rad per joint unit


# Motor IDs, feedback IDs and models from Seeed's rebotarm_dm.yaml (arm.yaml, gripper.yaml).
MOTORS = (
    MotorSpec("joint1", 0x01, 0x11, "4340P", "joint1", 1.0),
    MotorSpec("joint2", 0x02, 0x12, "4340P", "joint2", 1.0),
    MotorSpec("joint3", 0x03, 0x13, "4340P", "joint3", 1.0),
    MotorSpec("joint4", 0x04, 0x14, "4310", "joint4", 1.0),
    MotorSpec("joint5", 0x05, 0x15, "4310", "joint5", 1.0),
    MotorSpec("joint6", 0x06, 0x16, "4310", "joint6", 1.0),
    MotorSpec("gripper", 0x07, 0x17, "4310", "finger_left", GRIPPER_GEAR),
)


def _find_joint(spec: mujoco.MjSpec, name: str):
    finder = getattr(spec, "joint", None) or spec.find_joint
    joint = finder(name)
    if joint is None:
        raise ValueError(f"model has no joint named {name!r}")
    return joint


def build_model(model_path: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(model_path))
    # Compiling resolves meshdir against the working directory, not the model file.
    spec.meshdir = str((model_path.parent / spec.meshdir).resolve())
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    left = _find_joint(spec, "finger_left")
    right = _find_joint(spec, "finger_right")
    left.range = [0.0, FINGER_TRAVEL_M]
    right.range = [-FINGER_TRAVEL_M, 0.0]
    left.armature = left.armature + GRIPPER_MOTOR_INERTIA * GRIPPER_GEAR**2

    # finger_right = -finger_left.
    coupling = spec.add_equality()
    coupling.type = mujoco.mjtEq.mjEQ_JOINT
    coupling.objtype = mujoco.mjtObj.mjOBJ_JOINT
    coupling.name = "gripper_rack"
    coupling.name1 = "finger_right"
    coupling.name2 = "finger_left"
    coupling.data[0:5] = [0.0, -1.0, 0.0, 0.0, 0.0]

    for motor in MOTORS:
        actuator = spec.add_actuator()
        actuator.name = motor.name
        actuator.target = motor.joint
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.gear[0] = motor.gear
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.gainprm[0] = 1.0
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        # The firmware law sets the actual limit every step.
        tmax = MODELS[motor.model].tmax
        actuator.forcerange = [-tmax, tmax]
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec.compile()


class SimulatedArm:
    """MuJoCo physics with one emulated Damiao motor per actuator."""

    def __init__(self, model_path: Path) -> None:
        self.model = build_model(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.motors = [DamiaoMotor(spec.motor_id, spec.feedback_id, spec.model) for spec in MOTORS]
        self._actuators = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.name) for spec in MOTORS]
        )
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint) for spec in MOTORS]
        self._qpos = np.array([self.model.jnt_qposadr[j] for j in joints])
        self._qvel = np.array([self.model.jnt_dofadr[j] for j in joints])
        self._gears = np.array([spec.gear for spec in MOTORS])
        mujoco.mj_forward(self.model, self.data)
        self._measure()

    def _measure(self) -> None:
        positions = self._gears * self.data.qpos[self._qpos]
        velocities = self._gears * self.data.qvel[self._qvel]
        torques = self.data.actuator_force[self._actuators]
        for motor, position, velocity, torque in zip(self.motors, positions, velocities, torques):
            motor.set_measured(float(position), float(velocity), float(torque))

    def step(self) -> None:
        for motor, actuator in zip(self.motors, self._actuators):
            law = motor.torque_law(self.dt)
            self.model.actuator_biasprm[actuator, 0:3] = (law.offset, law.stiffness, law.damping)
            self.model.actuator_forcerange[actuator] = (-law.limit, law.limit)
        mujoco.mj_step(self.model, self.data)
        self._measure()

    def joint_positions(self) -> dict[str, float]:
        """Joint positions in joint units (rad, and m per finger for the gripper)."""
        return {spec.name: float(self.data.qpos[adr]) for spec, adr in zip(MOTORS, self._qpos)}
