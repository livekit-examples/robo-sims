# Trossen arms (`trossen-wxai`, `trossen-workbench`, participant level)

The default `--protocol contract` uses the [generalized gateway](../contract/README.md):
JSON joint feedback, timed joint trajectories, Portal ownership, and public discovery
plus local MuJoCo binding JSON for both scenes. Only implemented trajectory controls
are advertised. The original VR/gripper and lease behavior described below requires
`--protocol legacy`.

These simulations run a [Trossen Arm MuJoCo](https://github.com/TrossenRobotics/trossen_arm_mujoco) scene as one LiveKit robot participant. They don't start the robot program in `trossen-livekit/robot/` or need a physical arm: Trossen's `libtrossen_arm` talks to a proprietary arm controller, so the simulation stands in for the robot program instead. It exercises the operator's camera selection, stereo display, joint panel, Quest grip pose, thumbstick gripper velocity, and platform-wide control lease. 
| Simulation | Platform | Arms | Video feeds |
|---|---|---|---|
| `trossen-wxai` | `single_wxai` | one `wxai_v0` | wrist D405 stereo, simulation scene |
| `trossen-workbench` | `workbench` | two `wxpro` | left wrist ZED X Nano stereo, robot scene ZED X Mini stereo, right wrist ZED X Nano stereo, simulation scene |

Each stereo feed renders two eyes, side by side, from the real camera's imager positions. The MuJoCo camera sits at the left imager, and the right eye is one baseline along the camera's right axis. The Intel RealSense D405 has an 18 mm baseline and a 58 degree vertical field of view. The Stereolabs ZED X Nano has an 18 mm baseline and a 65 degree vertical field of view.

The Workbench robot scene camera is a Stereolabs ZED X Mini, with a 50 mm baseline. The fixture assumes the 2.2 mm lens, with an 80 degree vertical field of view. ZED eyes use the 16:10 sensor shape, so their width is 1.6 times `--height`.

The models come from `assets/trossen_arm_mujoco/`, fetched by `python -m sims fetch-assets` from the `workbench_assets` branch of the Trossen repository, because the `workbench` scene exists only on that branch.

The fixture implements the existing `get_robot_config`, `acquire_control`, and `release_control` RPCs and the advertised v1 data tracks. Only the operator that holds the lease can move either arm. Commands stop on lease owner disconnect or after the 250 ms watchdog for held motion. There is no hardware fault simulation. Each command moves the simulated arm with the timing of the robot in `trossen-livekit/robot/src/arms/trossen_arm.cpp`. For example, grip pose steps at most 8 mm and 0.045 rad toward the controller every 20 ms, and a joint position command moves at most 0.5 rad/s. The fixture has no torque limiter.

## Run

See the top-level README for setup and credentials. From `simulation/`:

```sh
.venv/bin/python -m sims run trossen-workbench --protocol legacy
.venv/bin/python -m sims run trossen-wxai --protocol legacy
```

The identity is `simulation-trossen-workbench-<scientist>` or `simulation-trossen-wxai-<scientist>`. The stereo track and the arms come from `get_robot_config`, so no track override is needed.

Options (`--help` lists them all):

- `--assets-dir` or `TROSSEN_MUJOCO_ASSETS` loads the models from another copy.
- `--width` and `--height` set the resolution of each wrist camera eye. The stereo track width is twice the eye width. `--scene-width` and `--scene-height` set the mono scene camera resolution, 960x720 by default. `--fps` sets the frame rate.
- MuJoCo renders all feeds offscreen, so a higher resolution needs no camera hardware. On Linux it renders through EGL by default, so it runs without an X display. On a machine without a GPU, set `MUJOCO_GL=osmesa` and install `libosmesa6` to use software rendering.

## Run the probe manually

Keep the simulator running with `--protocol legacy` in another terminal. Give the probe two *different* operator tokens for the same room. Their identities must start with `controller-`, for example `controller-test` and `controller-viewer`, with the join, publish, subscribe, and publish data grants. For a local `livekit-server --dev`, see the development token commands in `trossen-livekit/docs/credentials.md`:

```sh
export CONTROLLER_TOKEN='<controller-test token>'
export VIEWER_TOKEN='<controller-viewer token>'
.venv/bin/python -m sims.trossen.probe --robot <simulation identity>
```

The probe checks the manifest, received scene and stereo frames (including different images in the two eyes), joint telemetry, independent left and right motion, grip pose, gripper position, held joint velocity, gripper velocity during grip pose, rejection of commands from the unleased viewer, and control transfer after disconnect. It can be run repeatedly against the same simulator process. Pass the identity of the participant with `--robot`, for example `--robot simulation-trossen-workbench-curie`. The probe needs `trossen-workbench`. Pass `--height`, `--scene-width`, or `--scene-height` if those participant settings differ from the defaults.
