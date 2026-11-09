# Trossen arms (`trossen-wxai`, `trossen-wxai-whack`, `trossen-workbench`, `trossen-workbench-whack`, participant level)

The default `--protocol contract` uses the [generalized gateway](../contract/README.md):
JSON joint feedback, timed joint trajectories, Portal ownership, and public discovery
plus local MuJoCo binding JSON for both scenes. Independent arm and gripper position trajectory controls
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

## Whack-a-mole (`trossen-wxai-whack`)

The WidowX AI arm from `trossen-wxai` in a Tron-style world (black floor with a glowing grid in LiveKit cyan, #1FD5F9, a neon skyline fading into fog) on a black table with three holes. Each hole
pops up a violet, white or blue ball at random, and a light bar behind the table lights one
color. A wooden mallet lies on the table: bring the gripper within 6 cm of the end of its
handle and it snaps into the gripper (a weld constraint), so pickup works with the
joint-only contract. Hitting a ball with the mallet head scores +1 when the ball matches
the lit color, and -1 otherwise (never below 0). The lit color changes after each scoring
hit, or every 12 s. The game lives in `whack.py` and runs on each physics step through
`TrossenSimulation.after_physics_step`, which the contract backend also calls.
`game.state()` returns the score, target color and visible balls, for a future score display.
Tuning constants (hole layout, timings, mallet size) are at the top of `whack.py`.

### Workbench whack-a-mole (`trossen-workbench-whack`)

The same game and Tron world for the two WidowX Pro arms of `trossen-workbench`. The
Workbench stands on a table about twice the size of the single-arm one (1.0 x 1.44 m),
with five holes in a W in front of it and a mallet lying beside each arm. Either arm can
pick up either mallet, and hits by either mallet score. Every hole is in reach of both
arms with the gripper pitched down a little (0.1 to 0.3 rad). Layouts for both tables are
`WhackLayout` values (`SINGLE_WXAI`, `WORKBENCH`) at the top of `whack.py`.

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
- MuJoCo renders all feeds offscreen, so no camera hardware is needed. Filament manages its own graphics context. The classic renderer uses EGL by default on Linux, without an X display. For classic software rendering on a machine without a GPU, use `--renderer classic`, set `MUJOCO_GL=osmesa`, and install `libosmesa6`.
- CLI runs use MuJoCo 4's newer Filament renderer by default. Use `--renderer classic` to select the classic renderer, or `--renderer filament` to select Filament explicitly.

### Filament cameras

```sh
python -m sims run trossen-workbench --fps 30 --video-stats
python -m sims run trossen-workbench --fps 30 --benchmark-video 15
python -m sims run trossen-workbench --renderer classic --fps 30
```

Filament shares one persistent scene across all cameras, batches the views into
one offscreen image, and splits that image into the existing video tracks. Stereo
eyes keep their original resolution, field of view, and physical baseline. A
snapshot of the simulation state lets physics continue while the GPU renders.
Lighting and tone mapping can differ from the classic renderer.

On macOS, use ordinary `python`, not `mjpython`. The runner services the macOS
graphics event queue while Filament runs on its worker thread. The installed
MuJoCo 4.0 macOS build reports the Filament **OpenGL** backend; selecting Filament
does not select native Metal. The bundled MuJoCo Studio import registers the
Filament material assets; this build also emits an upstream Objective-C duplicate
class warning when those native modules are loaded together.

Filament is the CLI default for all four Trossen scenes and either protocol. Web-managed
demos explicitly use the classic renderer while their software rendering path is validated. To exercise real GPU rendering,
stereo output, motion, and game color updates in all four scenes:

```sh
ROBO_SIMS_FILAMENT_TEST=1 python -m unittest discover -s tests -p test_filament.py -v
```

### Measure camera performance

`--fps` is a per-camera target (default 12). Classic renders feeds sequentially;
Filament batches their views on one worker thread. At 30 FPS, the entire camera cycle must finish within 33.3 ms;
stereo feeds include both eyes in one side-by-side frame.

```sh
python -m sims run trossen-workbench --fps 30 --video-stats
python -m sims run trossen-workbench --fps 30 --benchmark-video 15
```

`--video-stats` logs the effective source FPS every five seconds, the mean and
95th-percentile cycle work time, and each camera's preparation, rendering/wait,
and frame submission time. Rendering/wait includes the worker and GPU work plus
time waiting for the event loop to resume. Source FPS is measured before encoding
and network delivery; compare it with the operator's displayed FPS to locate a
downstream bottleneck.

`--benchmark-video SECONDS` runs the same camera loop with physics and real video
sources, without credentials or a LiveKit connection. The timer starts after
renderer initialization. It excludes encoding and networking, so its rate is an
upper bound for a live run under the same load. Stop other simulations before
benchmarking to avoid sharing the GPU. These options also work with the WXAI,
Whack, and legacy protocol variants.

## Run the probe manually

Keep the simulator running with `--protocol legacy` in another terminal. Give the probe two *different* operator tokens for the same room. Their identities must start with `controller-`, for example `controller-test` and `controller-viewer`, with the join, publish, subscribe, and publish data grants. For a local `livekit-server --dev`, see the development token commands in `trossen-livekit/docs/credentials.md`:

```sh
export CONTROLLER_TOKEN='<controller-test token>'
export VIEWER_TOKEN='<controller-viewer token>'
.venv/bin/python -m sims.trossen.probe --robot <simulation identity>
```

The probe checks the manifest, received scene and stereo frames (including different images in the two eyes), joint telemetry, independent left and right motion, grip pose, gripper position, held joint velocity, gripper velocity during grip pose, rejection of commands from the unleased viewer, and control transfer after disconnect. It can be run repeatedly against the same simulator process. Pass the identity of the participant with `--robot`, for example `--robot simulation-trossen-workbench-curie`. The probe needs `trossen-workbench`. Pass `--height`, `--scene-width`, or `--scene-height` if those participant settings differ from the defaults.

In contract mode each arm card includes a separate gripper channel. Enable it to
use a finger-displacement slider or Open/Close while an arm trajectory runs. See
[the gateway's gripper semantics](../contract/README.md#executable-trajectory-semantics).
