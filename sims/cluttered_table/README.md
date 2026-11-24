# Cluttered table

Two WidowX AI arms stand at a maple workbench with household objects on it. The scene comes from the
Portal MuJoCo workbench. The arm model comes from the Trossen asset source, so run
`python -m sims fetch-assets cluttered-table` first.

```sh
python -m sims run cluttered-table --identity simulation-cluttered-table
python -m sims run cluttered-table --check      # print the discovery document
```

The simulation uses the generalized contract gateway (`sims/contract/README.md`) and supports no
other protocol. `--protocol legacy` is refused.

## What the robot serves

`get_robot_config` takes an empty payload. Any other payload returns RPC error 2000. The reply is the
public document in `sims/contract/configs/cluttered-table.discovery.json`, plus the
`simulation_control` extension. It has no model paths and no list of objects.

Each arm (`left_arm`, `right_arm`) has three operations:

| Operation | Profile | Presentations |
|---|---|---|
| `joint_position` | `joints.trajectory.v1` | `joint_sliders` |
| `gripper_position` | `gripper.position.v1` | `gripper_controls` |
| `cartesian_pose` | `arm.cartesian_pose.v1` | `hand_tracking`, `vr` |

The web app picks the first operation of the primary channel and its first presentation as the default. The discovery file lists `cartesian_pose` first and `hand_tracking` first, so hand tracking is the default control.
Pose feedback is on `/left_arm/end_effector_pose` and `/right_arm/end_effector_pose`, in the frame of each
arm base. The end effector frame has z along the fingers and x across the jaws, which is the frame
that hand tracking assumes.

Four cameras publish video: `overhead_camera`, `left_wrist_camera`, `right_wrist_camera`, and
`scene_camera`.

## Files

| Path | Contents |
|---|---|
| `scene/office.xml` | The workbench, room, and three loose desk items. |
| `scene/robocasa/` | Twelve household objects. They use CC-BY-4.0. See `PROVENANCE.md` in that folder. |
| `scene/maple_tabletop.png` | The table texture, from Trossen's assets (BSD-3-Clause). |
| `scene.py` | Builds the model: the scene, the props, and two arms. |
| `managed.py` | Builds the simulation for managed `/try` sessions. |
