"""Builds the cluttered table MuJoCo scene: the office workbench, props, and two WidowX AI arms.

The scene comes from the Portal MuJoCo workbench. The arm model comes from the Trossen asset
source (`wxai/wxai_follower.xml`), so the arms are not copied here. Names follow the Trossen
workbench: `left_joint_0`, `left_left_carriage_joint`, `left_gripper`, `left_ee_site`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..mujoco_tools import look_at  # Sets MUJOCO_GL before mujoco is imported.
import mujoco

SCENE_DIR = Path(__file__).resolve().parent / "scene"
PROPS_DIR = SCENE_DIR / "robocasa"
# Arm bases, in the table frame: 0.28 m behind the origin and 0.3 m to each side.
ARM_BASES = {"left": (-0.28, 0.3, 0.0), "right": (-0.28, -0.3, 0.0)}
# Joint angles of the ready pose, and the open gripper, in the Trossen joint names.
HOME = {"joint_0": 0.0, "joint_1": 1.1, "joint_2": 1.2, "joint_3": -0.65,
        "joint_4": 0.0, "joint_5": 0.0, "left_carriage_joint": 0.03, "right_carriage_joint": 0.03}
# The model's end effector site has x along the fingers and y across the jaws. The hand
# tracking control maps motion for a tool frame with z along the fingers and x across the
# jaws. This quaternion (w, x, y, z) turns the site into that frame.
TOOL_FRAME_QUAT = [0.5, 0.5, 0.5, 0.5]
SCENE_CAMERA = ((1.59, -1.57, 1.215), (0.02, 0.0, 0.18), 45)


def add_props(scene: mujoco.MjSpec) -> None:
    """Compose the offline RoboCasa props, each at the position in the manifest."""
    manifest = json.loads((PROPS_DIR / "manifest.json").read_text())
    for name, config in manifest["objects"].items():
        prop = mujoco.MjSpec.from_file(str(PROPS_DIR / name / "model.xml"))
        frame = scene.worldbody.add_frame(pos=config["position"])
        scene.attach(prop, prefix=name + "/", frame=frame)


def build_spec(assets_dir: Path) -> mujoco.MjSpec:
    """The table, the props, and the two arms, before compilation."""
    scene = mujoco.MjSpec.from_file(str(SCENE_DIR / "office.xml"))
    add_props(scene)
    for side, position in ARM_BASES.items():
        arm = mujoco.MjSpec.from_file(str(assets_dir / "wxai" / "wxai_follower.xml"))
        # Keyframe arrays describe one arm, not the composed scene.
        for key in list(arm.keys):
            arm.delete(key)
        # The composed scene uses full implicit integration for stable object contacts.
        arm.option.integrator = scene.option.integrator
        arm.camera("cam").name = "wrist_cam"
        arm.actuator("left_gripper").name = "gripper"
        arm.site("ee_site").quat = TOOL_FRAME_QUAT
        frame = scene.worldbody.add_frame(pos=position)
        scene.attach(arm, prefix=f"{side}_", frame=frame)
    position, target, fovy = SCENE_CAMERA
    scene.worldbody.add_camera(name="scene", pos=list(position), quat=look_at(position, target), fovy=fovy)
    return scene


def finish_model(model: mujoco.MjModel) -> None:
    """Clip planes scale with the scene's extent, and the room is large. Keep the near plane
    clear of the fingers in the wrist cameras."""
    model.vis.map.znear = 0.005 / model.stat.extent  # 5 mm
    model.vis.map.zfar = 20.0 / model.stat.extent  # 20 m


def set_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put both arms in the ready pose. The props keep their start positions."""
    for side in ARM_BASES:
        for name, value in HOME.items():
            data.joint(f"{side}_{name}").qpos[0] = value
    mujoco.mj_forward(model, data)
