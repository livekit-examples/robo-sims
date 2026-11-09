"""Motion and camera adapter for selectable Trossen MuJoCo scenes."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..mujoco_tools import look_at  # Sets MUJOCO_GL before mujoco is imported.
import mujoco
import numpy as np

from . import whack
from .protocol import Command
from .whack import WhackAMole


# The motion timing of the robot, from trossen-livekit/robot/src/arms/trossen_arm.cpp. The
# Trossen arm driver reaches each goal in a set time, so each simulated joint
# moves along a straight line from its last command to its goal in that time.
COMMAND_TIMEOUT = 0.25            # s without a frame ends continuous motion
POSE_UPDATE_PERIOD = 0.02         # s between grip pose goals
MAX_POSE_TRANSLATION_STEP = 0.008  # m per grip pose goal
MAX_POSE_ROTATION_STEP = 0.045     # rad per grip pose goal
POSE_GOAL_TIME = 0.15             # s, also to hold the arm where it is
POSITION_GOAL_TIME = 2.0          # s, for a Cartesian position delta
HOME_GOAL_TIME = 2.0              # s
# The robot caps each speed at a fraction of the joint's velocity limit, or
# at these values. The caps apply to the WidowX AI and WidowX Pro joints.
JOINT_SPEED = 0.5                 # rad/s for an arm joint position move
GRIPPER_SPEED = 0.025             # m/s for a gripper position move
JOINT_VELOCITY = 0.5              # rad/s in joint velocity mode
GRIPPER_VELOCITY = 0.01           # m/s at full gripper velocity
JOINT_MIN_GOAL_TIME, JOINT_GOAL_TIME_FACTOR = 0.7, 2.0
GRIPPER_MIN_GOAL_TIME, GRIPPER_GOAL_TIME_FACTOR = 0.3, 1.9
# The robot keeps gripper goals this far from the gripper endpoints, and
# stops joint velocity this far from a limit.
GRIPPER_ENDPOINT_MARGIN, GRIPPER_ENDPOINT_MARGIN_FRACTION = 0.002, 0.05
JOINT_LIMIT_MARGIN, GRIPPER_LIMIT_MARGIN = 0.005, 0.0002
CONTINUOUS_MODES = ("joint_velocity", "cartesian_velocity", "pose")
# The arm driver tracks its commands closely. With the stiffness of the
# Trossen models, the damping of each servo lets a moving wrist trail its
# command by up to 0.15 rad. Ten times the stiffness keeps it within 0.02 rad.
SERVO_STIFFNESS_FACTOR = 10.0
BLOCK_HALF_SIZE = 0.0127  # 1 inch cubes


@dataclass(frozen=True)
class StereoCamera:
    """Optics of a stereo camera. The MuJoCo camera sits at the left imager."""

    model: str
    baseline: float  # meters between the left and right imagers
    fovy: float  # vertical field of view, degrees
    aspect: float | None = None  # eye width / height; None uses the participant width


# Intel RealSense D405: 18 mm baseline, 87 x 58 degree field of view. The robot
# captures 640x480 eyes, so the eyes keep the participant's 4:3 size.
D405 = StereoCamera("Intel RealSense D405", 0.018, 58.0)
# Stereolabs ZED X Nano: 18 mm baseline, 92 x 65 degrees, 16:10 sensors.
ZED_X_NANO = StereoCamera("Stereolabs ZED X Nano", 0.018, 65.0, 16 / 10)
# Stereolabs ZED X Mini with the 2.2 mm lens: 50 mm baseline, 110 x 80 degrees.
ZED_X_MINI = StereoCamera("Stereolabs ZED X Mini (2.2 mm)", 0.05, 80.0, 16 / 10)


@dataclass(frozen=True)
class CameraFeed:
    camera: str
    name: str
    label: str
    role: str
    stereo: StereoCamera | None = None
    arm_id: str | None = None


def add_stereo_camera(body: mujoco.MjsBody, name: str, center: tuple[float, float, float],
                      pitch_down: float, stereo: StereoCamera) -> None:
    """Add a forward (+x) camera at the left imager of a stereo pair centered on `center`."""
    forward = np.array([math.cos(math.radians(pitch_down)), 0.0,
                        -math.sin(math.radians(pitch_down))])
    left_eye = np.asarray(center) + np.array([0.0, stereo.baseline / 2.0, 0.0])
    body.add_camera(name=name, pos=left_eye.tolist(),
                    quat=look_at(tuple(left_eye), tuple(left_eye + forward)), fovy=stereo.fovy)


def add_environment(spec: mujoco.MjSpec, assets_dir: Path, floor_z: float) -> None:
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.add_texture(name="floor_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4],
                     rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    floor = spec.add_material(name="floor_grid", texuniform=True, texrepeat=[5, 5],
                              reflectance=0.2)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            pos=[0, 0, floor_z], size=[0, 0, 0.05], material="floor_grid")
    add_maple(spec, assets_dir)


def add_maple(spec: mujoco.MjSpec, assets_dir: Path) -> None:
    spec.add_texture(name="maple_tabletop", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=str((assets_dir / "maple_tabletop.png").resolve()))
    maple = spec.add_material(name="maple", texuniform=True, texrepeat=[2, 2])
    maple.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "maple_tabletop"


def add_table(spec: mujoco.MjSpec, center_xy: tuple[float, float],
              half_xy: tuple[float, float], top_z: float, floor_z: float) -> None:
    thickness = 0.03
    spec.worldbody.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[*center_xy, top_z - thickness / 2],
                            size=[*half_xy, thickness / 2], material="maple")
    leg_half_height = (top_z - thickness - floor_z) / 2
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, leg_half_height],
                pos=[center_xy[0] + x_sign * (half_xy[0] - 0.04),
                     center_xy[1] + y_sign * (half_xy[1] - 0.04), floor_z + leg_half_height],
                rgba=[0.25, 0.22, 0.2, 1], contype=0, conaffinity=0)


def add_blocks(spec: mujoco.MjSpec, surface_z: float,
               positions: tuple[tuple[float, float], ...]) -> None:
    colors = {"red": (0.9, 0.08, 0.08, 1.0),
              "green": (0.08, 0.75, 0.12, 1.0),
              "blue": (0.08, 0.2, 0.9, 1.0)}
    for (color, rgba), (x, y) in zip(colors.items(), positions):
        position = (x, y, surface_z + BLOCK_HALF_SIZE)
        body = spec.worldbody.add_body(name=f"{color}_block", pos=list(position))
        body.add_freejoint().name = f"{color}_block_free"
        body.add_geom(name=f"{color}_block_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[BLOCK_HALF_SIZE] * 3, mass=0.05, rgba=rgba,
                      friction=[1.0, 0.005, 0.0001])
        for key in spec.keys:
            key.qpos = np.concatenate((key.qpos, (*position, 1.0, 0.0, 0.0, 0.0)))


def compensate_gravity(base: mujoco.MjsBody) -> None:
    """Cancels gravity on every body of an arm through its actuators."""
    for body in base.find_all(mujoco.mjtObj.mjOBJ_BODY):
        body.gravcomp = 1.0
    for joint in base.find_all(mujoco.mjtObj.mjOBJ_JOINT):
        joint.actgravcomp = True


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    # Go through a quaternion. The trace formula loses the axis near 180 degrees.
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(matrix, dtype=np.float64).reshape(9))
    vector = np.zeros(3)
    mujoco.mju_quat2Vel(vector, quaternion, 1.0)
    return vector


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-9:
        return np.eye(3)
    axis = vector / angle
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross


@dataclass
class ArmState:
    arm_id: str
    joint_ids: list[int]
    qpos_ids: list[int]
    dof_ids: list[int]
    actuator_ids: list[int]
    site_id: int
    base_id: int
    limits: list[tuple[float, float]]
    # The joint commands that the actuators follow, and the straight-line
    # move of each joint that produces them.
    setpoints: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    start_time: np.ndarray
    goal_time: np.ndarray
    mode: str = ""
    deadline: float = 0.0
    velocity_index: int = 0
    velocity_direction: float = 0.0
    # Gripper velocity is separate from the arm mode, as on the robot, so the
    # gripper can move during grip pose.
    gripper_velocity: float = 0.0
    gripper_deadline: float = 0.0
    cartesian_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    desired_position: np.ndarray | None = None
    desired_rotation: np.ndarray | None = None
    anchor_position: np.ndarray | None = None
    anchor_rotation: np.ndarray | None = None
    # The first pose frame of an engagement. Later frames move the arm by
    # their change from it, as on the robot.
    controller_anchor: np.ndarray | None = None
    next_pose_update: float = 0.0


class TrossenSimulation:
    """One or two Trossen arms with scene-specific cameras and movable blocks."""

    def __init__(self, assets_dir: Path, scene: str = "workbench",
                 width: int = 320, height: int = 240,
                 scene_width: int = 960, scene_height: int = 720,
                 model_path: Path | None = None, *, renderer: str = "classic"):
        if renderer not in ("classic", "filament"):
            raise ValueError(f"Unknown renderer: {renderer}")
        self.renderer_backend = renderer
        self.batch_rendering = renderer == "filament"
        self.filament = None
        self.platform = scene
        if scene in ("single_wxai", "whack_wxai"):
            source = assets_dir / "wxai" / "wxai_follower.xml"
        elif scene in ("workbench", "workbench_whack"):
            source = assets_dir / "workbench" / "scene.xml"
        else:
            raise ValueError(f"Unknown scene: {scene}")
        if model_path is not None:
            source = model_path
        if not source.is_file():
            raise FileNotFoundError(f"Trossen MuJoCo scene not found: {source}")
        spec = mujoco.MjSpec.from_file(str(source))
        if scene == "single_wxai":
            self._single_scene(spec, assets_dir)
        elif scene == "whack_wxai":
            self._whack_scene(spec, assets_dir)
        elif scene == "workbench_whack":
            self._workbench_whack_scene(spec, assets_dir)
        else:
            self._workbench_scene(spec, assets_dir)
        self.model = spec.compile()
        # Scene games run on every physics step (see after_physics_step). A game
        # sets up its keyframes, so it starts before the reset below.
        layout = whack.LAYOUTS.get(scene)
        self.game = WhackAMole(self.model, layout) if layout else None
        if scene in ("workbench", "workbench_whack"):
            for geom_id in range(self.model.ngeom):
                if self.model.geom_group[geom_id] == 3 and mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) is None:
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0
        # MuJoCo contacts are soft, so a held block creeps out of the fingers
        # with the default pyramidal friction cone. Elliptic cones and a high
        # impratio make the friction stiff enough to hold a lifted block.
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = 10
        self.data = mujoco.MjData(self.model)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        # Inverse kinematics runs on a copy, so that it does not move the arm.
        self.ik_data = mujoco.MjData(self.model)
        self.width = width
        self.height = height
        self.scene_width = scene_width
        self.scene_height = scene_height
        # Offscreen rendering needs no camera hardware, so spend it on quality:
        # 8x multisampling and sharper shadows.
        self.model.vis.quality.offsamples = 8
        self.model.vis.quality.shadowsize = 8192
        self.renderers: dict[tuple[str, int, int], mujoco.Renderer] = {}
        if scene in ("single_wxai", "whack_wxai"):
            self.arm_model = "wxai_v0"
            self.arms = {"arm": self._arm("arm", "", "left_gripper")}
            self.feeds = {
                "arm_wrist": CameraFeed("cam", "sim.arm.stereo", "Wrist", "wrist", D405, "arm"),
                "scene": CameraFeed("scene", "sim.scene", "Simulation scene", "scene"),
            }
        else:
            self.arm_model = "wxpro"
            self.arms = {arm_id: self._arm(arm_id, f"{arm_id}_", f"{arm_id}_gripper")
                         for arm_id in ("left", "right")}
            self.feeds = {
                "left_wrist": CameraFeed("left_wrist_cam", "sim.left.stereo", "Left wrist", "wrist",
                                         ZED_X_NANO, "left"),
                "center": CameraFeed("center", "sim.center.stereo", "Robot scene", "other", ZED_X_MINI),
                "right_wrist": CameraFeed("right_wrist_cam", "sim.right.stereo", "Right wrist", "wrist",
                                          ZED_X_NANO, "right"),
                "scene": CameraFeed("scene", "sim.scene", "Simulation scene", "scene"),
            }
        for arm in self.arms.values():
            for actuator in arm.actuator_ids[:6]:
                self.model.actuator_gainprm[actuator, 0] *= SERVO_STIFFNESS_FACTOR
                self.model.actuator_biasprm[actuator, 1] = -self.model.actuator_gainprm[actuator, 0]
        for feed in self.feeds.values():
            camera = self._named(mujoco.mjtObj.mjOBJ_CAMERA, feed.camera)
            if feed.stereo:
                self.model.cam_fovy[camera] = feed.stereo.fovy
        # The offscreen framebuffer must hold the largest frame of any feed. A
        # stereo frame is two eyes wide, and an eye can take its width from the
        # camera's aspect ratio.
        sizes = [self.render_size(camera_id) for camera_id in self.feeds]
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth,
                                              *(size[0] for size in sizes))
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight,
                                               *(size[1] for size in sizes))

    @staticmethod
    def _single_scene(spec: mujoco.MjSpec, assets_dir: Path) -> None:
        add_environment(spec, assets_dir, floor_z=0.0)
        spec.worldbody.add_light(name="scene_light", pos=[0.25, -0.35, 1.2],
                                 dir=[-0.2, 0.25, -1],
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 diffuse=[0.9, 0.9, 0.9], ambient=[0.35, 0.35, 0.35])
        spec.worldbody.add_geom(name="block_table", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[0.32, 0, 0.05], size=[0.18, 0.18, 0.05], material="maple")
        scene_pos = (0.311, -1.274, 0.880)
        spec.worldbody.add_camera(name="scene", pos=scene_pos,
                                  quat=look_at(scene_pos, (0.272, 0.253, 0.363)), fovy=65)
        add_blocks(spec, 0.1, ((0.23, -0.08), (0.33, 0), (0.43, 0.08)))

    @staticmethod
    def _whack_scene(spec: mujoco.MjSpec, assets_dir: Path) -> None:
        whack.add_tron_environment(spec, floor_z=0.0, table_top=whack.SINGLE_WXAI.table_top)
        whack.build_scene(spec, whack.SINGLE_WXAI)
        # From the side and above, looking down at the table.
        scene_pos = (0.247, -1.477, 0.923)
        spec.worldbody.add_camera(name="scene", pos=scene_pos,
                                  quat=look_at(scene_pos, (0.432, 0.306, 0.429)), fovy=58)

    @staticmethod
    def _workbench_scene(spec: mujoco.MjSpec, assets_dir: Path) -> None:
        # The Workbench stands on a table. Its arms and ZED X Mini face +x, so
        # the work area is the tabletop in front of it.
        table_z, floor_z = 0.0, -0.75
        spec.geom("floor").pos = [0, 0, floor_z]
        add_maple(spec, assets_dir)
        add_table(spec, (0.25, 0.0), (0.45, 0.55), table_z, floor_z)
        TrossenSimulation._workbench_rig(spec)
        scene_pos = (1.05, -0.75, 0.75)
        spec.worldbody.add_camera(name="scene", pos=scene_pos,
                                  quat=look_at(scene_pos, (0.25, 0, 0.1)), fovy=45)
        add_blocks(spec, table_z, ((0.42, -0.1), (0.42, 0.0), (0.42, 0.1)))

    @staticmethod
    def _workbench_whack_scene(spec: mujoco.MjSpec, assets_dir: Path) -> None:
        layout = whack.WORKBENCH
        whack.add_tron_environment(spec, floor_z=layout.floor_z, table_top=layout.table_top)
        TrossenSimulation._workbench_rig(spec)
        whack.build_scene(spec, layout)
        # Behind and above the right arm, looking across the table at the light bar.
        scene_pos = (-0.55, -1.45, 1.25)
        spec.worldbody.add_camera(name="scene", pos=scene_pos,
                                  quat=look_at(scene_pos, (0.4, 0.0, -0.05)), fovy=55)

    @staticmethod
    def _workbench_rig(spec: mujoco.MjSpec) -> None:
        """Gravity compensation, finger contacts and stereo cameras for the Workbench arms."""
        # The upstream mesh collision hulls overlap in the home pose and pin
        # the arms. Keep the visuals and use simple contacts for this fixture.
        for arm_id in ("left", "right"):
            # The arm driver holds each joint against gravity. The WidowX AI
            # model compensates gravity, but these WidowX Pro arms do not, so
            # they sag below their commands and fall when grip pose starts.
            compensate_gravity(spec.body(f"{arm_id}_base_link"))
            for side in ("left", "right"):
                # In each finger frame, the finger runs along +z and +x points
                # to the other finger. The box covers the base of the finger,
                # and the pad covers the flat inner face from z = 0.048 m to
                # the tip at z = 0.09 m, so that the fingertips hold a block.
                finger = spec.body(f"{arm_id}_gripper_{side}")
                finger.add_geom(
                    name=f"{arm_id}_{side}_finger_contact", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0.005, 0, 0.018], size=[0.015, 0.005, 0.018],
                    group=3, rgba=[0, 0, 0, 0], contype=2, conaffinity=1,
                    friction=[1.2, 0.005, 0.0001])
                finger.add_geom(
                    name=f"{arm_id}_{side}_finger_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0.0172, 0, 0.069], size=[0.004, 0.0065, 0.021],
                    group=3, rgba=[0, 0, 0, 0], contype=2, conaffinity=1,
                    friction=[1.2, 0.005, 0.0001])
            # ZED X Nano front face center and 25 degree tilt, from the
            # "Simplified Zed Nano" mesh in the camera body frame.
            add_stereo_camera(spec.body(f"{arm_id}_camera"), f"{arm_id}_wrist_cam",
                              (0.0098, 0.0, 0.065), 25.0, ZED_X_NANO)
        # ZED X Mini front face center and tilt, from the cam_high mesh.
        add_stereo_camera(spec.body("cam_high"), "center", (0.1125, 0.0, 0.3114), 25.0, ZED_X_MINI)

    def _named(self, kind: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"Trossen scene is missing {name}")
        return value

    def _arm(self, arm_id: str, prefix: str, gripper_actuator: str) -> ArmState:
        names = [f"{prefix}joint_{index}" for index in range(6)]
        names.append(f"{prefix}left_carriage_joint")
        joint_ids = [self._named(mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        actuators = [f"{prefix}joint_{index}" for index in range(6)]
        actuators.append(gripper_actuator)
        qpos_ids = [int(self.model.jnt_qposadr[joint]) for joint in joint_ids]
        limits = [tuple(float(x) for x in self.model.jnt_range[joint]) for joint in joint_ids]
        return ArmState(
            arm_id=arm_id,
            joint_ids=joint_ids,
            qpos_ids=qpos_ids,
            dof_ids=[int(self.model.jnt_dofadr[joint]) for joint in joint_ids[:6]],
            actuator_ids=[self._named(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuators],
            site_id=self._named(mujoco.mjtObj.mjOBJ_SITE, f"{prefix}ee_site"),
            base_id=self._named(mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_link"),
            limits=limits,
            setpoints=self.data.qpos[qpos_ids].copy(),
            start=self.data.qpos[qpos_ids].copy(),
            goal=self.data.qpos[qpos_ids].copy(),
            start_time=np.zeros(7),
            goal_time=np.zeros(7),
        )

    def joint_state(self, arm_id: str) -> tuple[list[float], list[tuple[float, float]]]:
        arm = self.arms[arm_id]
        return self.data.qpos[arm.qpos_ids].tolist(), arm.limits

    def stop_all_motion(self) -> None:
        """Ends continuous motion, as the robot does when its controller leaves.

        A joint position move or a home move continues to its goal.
        """
        for arm in self.arms.values():
            if arm.mode in CONTINUOUS_MODES:
                self._hold(arm)
            self._stop_gripper(arm)

    def _move(self, arm: ArmState, indices, goals, goal_time: float) -> None:
        """Moves joints from their current commands to their goals in goal_time."""
        for index, goal in zip(indices, goals):
            arm.start[index] = arm.setpoints[index]
            arm.goal[index] = np.clip(goal, *arm.limits[index])
            arm.start_time[index] = self.data.time
            arm.goal_time[index] = goal_time

    def _hold(self, arm: ArmState) -> None:
        """Holds the arm where it is, as the robot does when motion stops.

        The gripper keeps its goal, so that it keeps squeezing a held block.
        """
        arm.mode = ""
        arm.desired_position = None
        arm.desired_rotation = None
        self._move(arm, range(6), arm.setpoints[:6].copy(), 0.0)

    def _stop_gripper(self, arm: ArmState) -> None:
        """Ends gripper velocity. The gripper stops where it is, as on the robot."""
        if arm.gripper_velocity:
            arm.gripper_velocity = 0.0
            self._move(arm, [6], [arm.setpoints[6]], 0.0)

    def _commanded_pose(self, arm: ArmState) -> tuple[np.ndarray, np.ndarray]:
        """The end effector pose of the commanded joints.

        The robot plans from the measured pose of an arm that follows its
        commands closely. The simulated joints lag their commands, so the
        simulation plans from the commanded joints instead.
        """
        ik = self.ik_data
        ik.qpos[:] = self.data.qpos
        ik.qpos[arm.qpos_ids[:6]] = arm.setpoints[:6]
        mujoco.mj_kinematics(self.model, ik)
        return ik.site_xpos[arm.site_id].copy(), ik.site_xmat[arm.site_id].reshape(3, 3).copy()

    def apply(self, arm_id: str, command: Command) -> None:
        arm = self.arms[arm_id]
        values = np.asarray(command.values, dtype=float)
        now = time.monotonic()
        if command.kind == "joint":
            index = command.joint_index
            if index >= 7:
                return
            if command.joint_kind == 2 and index == 6:
                if values[0]:
                    arm.gripper_velocity = float(values[0])
                    arm.gripper_deadline = now + COMMAND_TIMEOUT
                else:
                    self._stop_gripper(arm)
                return
            if command.joint_kind == 2:
                if values[0]:
                    if arm.mode != "joint_velocity" or arm.velocity_index != index:
                        self._hold(arm)
                    arm.velocity_index = index
                    arm.velocity_direction = values[0]
                    arm.deadline = now + COMMAND_TIMEOUT
                    arm.mode = "joint_velocity"
                elif arm.mode == "joint_velocity":
                    self._hold(arm)
                return
            # A home frame for one joint is a position command to 0.
            target = 0.0 if command.joint_kind == 3 else values[0]
            current = self.data.qpos[arm.qpos_ids[index]]
            if index == 6:
                # A gripper position command replaces gripper velocity.
                arm.gripper_velocity = 0.0
                low, high = arm.limits[6]
                margin = min(GRIPPER_ENDPOINT_MARGIN, (high - low) * GRIPPER_ENDPOINT_MARGIN_FRACTION)
                goal = float(np.clip(target, low + margin, high - margin))
                goal_time = max(GRIPPER_MIN_GOAL_TIME, GRIPPER_GOAL_TIME_FACTOR * abs(goal - current) / GRIPPER_SPEED)
            else:
                # A joint command on the arm ends any continuous arm motion.
                if arm.mode in CONTINUOUS_MODES:
                    self._hold(arm)
                arm.mode = ""
                goal = float(np.clip(target, *arm.limits[index]))
                goal_time = max(JOINT_MIN_GOAL_TIME, JOINT_GOAL_TIME_FACTOR * abs(goal - current) / JOINT_SPEED)
            self._move(arm, [index], [goal], goal_time)
        elif command.kind == "home":
            if arm.mode in CONTINUOUS_MODES:
                self._hold(arm)
            arm.mode = ""
            arm.gripper_velocity = 0.0
            low, high = arm.limits[6]
            margin = min(GRIPPER_ENDPOINT_MARGIN, (high - low) * GRIPPER_ENDPOINT_MARGIN_FRACTION)
            self._move(arm, range(7), [0.0] * 6 + [low + margin], HOME_GOAL_TIME)
        elif command.kind == "clear_faults":
            pass  # There are no hardware faults in this integration participant.
        elif command.kind == "velocity":
            if command.active:
                if arm.mode != "cartesian_velocity":
                    self._hold(arm)
                    arm.desired_position, arm.desired_rotation = self._commanded_pose(arm)
                arm.cartesian_velocity[:] = values
                arm.deadline = now + COMMAND_TIMEOUT
                arm.mode = "cartesian_velocity"
            elif arm.mode == "cartesian_velocity":
                self._hold(arm)
        elif command.kind == "pose":
            if command.active:
                if arm.mode != "pose":
                    if arm.mode in CONTINUOUS_MODES:
                        self._hold(arm)
                    arm.anchor_position, arm.anchor_rotation = self._commanded_pose(arm)
                    arm.controller_anchor = values.copy()
                    arm.next_pose_update = self.data.time
                anchor = arm.controller_anchor
                arm.desired_position = arm.anchor_position + arm.anchor_rotation @ (values[:3] - anchor[:3])
                arm.desired_rotation = (arm.anchor_rotation @ rotation_from_vector(values[3:])
                                        @ rotation_from_vector(anchor[3:]).T)
                arm.deadline = now + COMMAND_TIMEOUT
                arm.mode = "pose"
            elif arm.mode == "pose":
                self._hold(arm)
        elif command.kind == "position":
            if arm.mode in CONTINUOUS_MODES:
                self._hold(arm)
            arm.mode = ""
            position, rotation = self._commanded_pose(arm)
            base_rotation = self.data.xmat[arm.base_id].reshape(3, 3)
            goals = self._inverse_kinematics(arm, position + base_rotation @ values[:3],
                                             rotation_from_vector(base_rotation @ values[3:]) @ rotation, 30)
            self._move(arm, range(6), goals, POSITION_GOAL_TIME)

    def _inverse_kinematics(self, arm: ArmState, position: np.ndarray, rotation: np.ndarray,
                            iterations: int) -> np.ndarray:
        """Joint goals that bring the end effector site to a pose, from the commanded joints."""
        ik = self.ik_data
        ik.qpos[:] = self.data.qpos
        ids = arm.qpos_ids[:6]
        ik.qpos[ids] = arm.setpoints[:6]
        low = [limit[0] for limit in arm.limits[:6]]
        high = [limit[1] for limit in arm.limits[:6]]
        jac_position = np.zeros((3, self.model.nv))
        jac_rotation = np.zeros((3, self.model.nv))
        for _ in range(iterations):
            mujoco.mj_kinematics(self.model, ik)
            mujoco.mj_comPos(self.model, ik)
            current = ik.site_xmat[arm.site_id].reshape(3, 3)
            error = np.concatenate((position - ik.site_xpos[arm.site_id], rotation_vector(rotation @ current.T)))
            if np.linalg.norm(error) < 1e-5:
                break
            mujoco.mj_jacSite(self.model, ik, jac_position, jac_rotation, arm.site_id)
            jacobian = np.vstack((jac_position, jac_rotation))[:, arm.dof_ids]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.03**2 * np.eye(6), error)
            ik.qpos[ids] = np.clip(ik.qpos[ids] + delta, low, high)
        return ik.qpos[ids].copy()

    def _update_pose(self, arm: ArmState) -> None:
        """Moves toward the grip pose goal by a limited step, as the robot does every 20 ms."""
        position, rotation = self._commanded_pose(arm)
        translation = arm.desired_position - position
        distance = float(np.linalg.norm(translation))
        if distance > MAX_POSE_TRANSLATION_STEP:
            translation *= MAX_POSE_TRANSLATION_STEP / distance
        correction = rotation_vector(arm.desired_rotation @ rotation.T)
        angle = float(np.linalg.norm(correction))
        if angle > MAX_POSE_ROTATION_STEP:
            correction *= MAX_POSE_ROTATION_STEP / angle
        goals = self._inverse_kinematics(arm, position + translation,
                                         rotation_from_vector(correction) @ rotation, 3)
        self._move(arm, range(6), goals, POSE_GOAL_TIME)

    def step(self, dt: float = 0.01) -> None:
        now = time.monotonic()
        for arm in self.arms.values():
            if arm.mode in CONTINUOUS_MODES and now > arm.deadline:
                self._hold(arm)
            if arm.gripper_velocity and now > arm.gripper_deadline:
                self._stop_gripper(arm)
            if arm.gripper_velocity:
                low, high = arm.limits[6]
                position = self.data.qpos[arm.qpos_ids[6]]
                # The robot holds the gripper at a limit until the operator reverses.
                if not ((arm.gripper_velocity < 0 and position <= low + GRIPPER_LIMIT_MARGIN) or
                        (arm.gripper_velocity > 0 and position >= high - GRIPPER_LIMIT_MARGIN)):
                    goal = arm.setpoints[6] + arm.gripper_velocity * GRIPPER_VELOCITY * dt
                    self._move(arm, [6], [goal], dt)
            if arm.mode == "joint_velocity":
                index = arm.velocity_index
                speed = JOINT_VELOCITY
                margin = JOINT_LIMIT_MARGIN
                low, high = arm.limits[index]
                position = self.data.qpos[arm.qpos_ids[index]]
                if ((arm.velocity_direction < 0 and position <= low + margin) or
                        (arm.velocity_direction > 0 and position >= high - margin)):
                    self._hold(arm)  # The robot stops joint velocity at a limit.
                else:
                    goal = arm.setpoints[index] + arm.velocity_direction * speed * dt
                    self._move(arm, [index], [goal], dt)
            elif arm.mode == "cartesian_velocity":
                base_rotation = self.data.xmat[arm.base_id].reshape(3, 3)
                arm.desired_position += base_rotation @ arm.cartesian_velocity[:3] * dt
                arm.desired_rotation = rotation_from_vector(
                    base_rotation @ arm.cartesian_velocity[3:] * dt) @ arm.desired_rotation
                goals = self._inverse_kinematics(arm, arm.desired_position, arm.desired_rotation, 3)
                self._move(arm, range(6), goals, dt)
            elif arm.mode == "pose" and self.data.time >= arm.next_pose_update:
                self._update_pose(arm)
                arm.next_pose_update = self.data.time + POSE_UPDATE_PERIOD
        for _ in range(max(1, round(dt / self.model.opt.timestep))):
            for arm in self.arms.values():
                progress = np.clip((self.data.time - arm.start_time) / np.maximum(arm.goal_time, 1e-9), 0.0, 1.0)
                arm.setpoints[:] = arm.start + (arm.goal - arm.start) * progress
                self.data.ctrl[arm.actuator_ids] = arm.setpoints
            mujoco.mj_step(self.model, self.data)
            self.after_physics_step()

    def after_physics_step(self) -> None:
        """Runs scene logic, such as a game, after each mj_step. Controllers call this."""
        if self.game is not None:
            self.game.update(self.data)

    def eye_size(self, camera_id: str) -> tuple[int, int]:
        feed = self.feeds[camera_id]
        if feed.stereo is None:
            return self.scene_width, self.scene_height
        if feed.stereo.aspect is None:
            return self.width, self.height
        return 2 * round(self.height * feed.stereo.aspect / 2), self.height

    def render_size(self, camera_id: str) -> tuple[int, int]:
        """The size of the feed's rendered frame, with both eyes of a stereo feed."""
        width, height = self.eye_size(camera_id)
        return (2 * width if self.feeds[camera_id].stereo else width), height

    def camera_info(self) -> dict:
        """Calibrate stereo eyes in their fixed mount frames, without creating a renderer."""
        result = {}
        for key, feed in self.feeds.items():
            if feed.stereo is None:
                continue
            camera_id = self._named(mujoco.mjtObj.mjOBJ_CAMERA, feed.camera)
            if self.model.cam_mode[camera_id] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
                raise ValueError("camera_info requires a fixed camera-to-mount transform")
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = camera_id
            projection = mujoco.mjv_camera2GLCamera(self.model, self.data, camera)
            if projection.orthographic:
                raise ValueError("camera_info requires a perspective camera")
            width, height = self.eye_size(key)
            span = projection.frustum_top - projection.frustum_bottom
            half_width = projection.frustum_width or span * width / height / 2
            intrinsics = {
                "fx": width * projection.frustum_near / (2 * half_width),
                "fy": height * projection.frustum_near / span,
                "cx": width / 2 - width * projection.frustum_center / (2 * half_width),
                "cy": height * projection.frustum_top / span,
                "distortion_model": "none", "distortion_coefficients": [],
            }
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, self.model.cam_quat[camera_id])
            rotation = rotation.reshape(3, 3)
            # MuJoCo looks along -Z with +Y up; optical frames use +Z forward, +Y down.
            optical = rotation @ np.diag([1.0, -1.0, -1.0])
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, optical.ravel())
            parent_id = int(self.model.cam_bodyid[camera_id])
            parent = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
            if not parent:
                raise ValueError("camera_info requires a named camera mount")
            views = {}
            for index, eye in enumerate(("left", "right")):
                translation = self.model.cam_pos[camera_id] + index * feed.stereo.baseline * rotation[:, 0]
                views[eye] = {
                    "frame_id": f"{feed.camera}_{eye}_optical",
                    "image_region": {"x": index * width, "y": 0, "width": width, "height": height},
                    "intrinsics": dict(intrinsics),
                    "extrinsics": {"parent_frame_id": parent,
                        "translation": translation.tolist(),
                        "rotation_xyzw": quaternion[[1, 2, 3, 0]].tolist()},
                }
            result[key] = {"track_name": feed.name, "width": 2 * width, "height": height, "views": views}
        return result

    def open_renderers(self) -> None:
        """Create every feed's renderer.

        On macOS, an OpenGL context stays locked to the first thread that uses
        it. Call this, Renderer.render(), and close() on one rendering thread.
        """
        if self.batch_rendering:
            from .filament import FilamentFrames

            self.filament = FilamentFrames(self.model, self.feeds,
                {key: self.render_size(key) for key in self.feeds}, fog=self.game is not None)
        else:
            for camera_id in self.feeds:
                self._renderer(camera_id)

    def prepare_batch(self, keys):
        return self.filament.prepare(self.model, self.data, keys)

    def prepare(self, camera_id: str) -> mujoco.Renderer:
        """Copy the current state into the feed's renderer. Call render() on it next.

        This reads the simulation, so it runs on the thread that steps it. It
        does not use OpenGL.
        """
        feed = self.feeds[camera_id]
        renderer = self._renderer(camera_id)
        renderer.update_scene(self.data, camera=feed.camera)
        if feed.stereo:
            # MuJoCo centers its stereo pair on the camera. The camera sits at the
            # left imager, and the right imager is one baseline to its right.
            camera = self._named(mujoco.mjtObj.mjOBJ_CAMERA, feed.camera)
            left_eye = self.data.cam_xpos[camera]
            right_axis = self.data.cam_xmat[camera].reshape(3, 3)[:, 0]
            renderer.scene.camera[0].pos[:] = left_eye
            renderer.scene.camera[1].pos[:] = left_eye + feed.stereo.baseline * right_axis
        return renderer

    def _renderer(self, camera_id: str) -> mujoco.Renderer:
        feed = self.feeds[camera_id]
        width, height = self.render_size(camera_id)
        key = (camera_id, width, height)
        renderer = self.renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            # The floor reflection draws the whole scene a second time.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            if self.game is not None:
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            if feed.stereo:
                renderer.scene.stereo = mujoco.mjtStereo.mjSTEREO_SIDEBYSIDE
            self.renderers[key] = renderer
        return renderer

    def render(self, camera_id: str) -> np.ndarray:
        if self.batch_rendering:
            if self.filament is None:
                self.open_renderers()
            return self.prepare_batch((camera_id,)).render()[camera_id].copy()
        return self.prepare(camera_id).render().copy()

    def close(self) -> None:
        if self.filament is not None:
            self.filament.close()
            self.filament = None
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
