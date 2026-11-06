# reBot B601-DM (`rebot-arm`, hardware level)

A simulated Seeed reBot Arm B601-DM that the real arm's driver code drives unchanged.

The simulator pretends to be the arm's hardware. It emulates the Damiao USB2CAN serial bridge and the seven Damiao motors behind it, with MuJoCo standing in for the motors' physics. Any program that drives the real arm through motorbridge on `/dev/ttyACM0` drives the simulator through a different serial device instead. Nothing else changes, so the robot code tested here is the code that runs on the arm.

```
real arm:  trossen_livekit_robot ── RebotDmDriver ── motorbridge ── /dev/ttyACM0 ── USB2CAN ── 7 Damiao motors
simulator: trossen_livekit_robot ── RebotDmDriver ── motorbridge ── pty ─────────── emulated bus ── MuJoCo
```

## Run

```sh
.venv/bin/python -m sims run rebot-arm
```

This starts two processes:

1. **The emulated bus** (`sims.rebot.bus`). It serves a pseudo-terminal, linked by default at `$XDG_RUNTIME_DIR/sims-<uid>/rebot-arm-ttyACM0`, and the scene camera stream.
2. **The robot program**, as `trossen_livekit_robot --platform rebot_b601_dm --arm-connection <that device> --camera-device scene=http://127.0.0.1:<port>/scene.mjpg`, with a token made by the launcher (see the top-level README).

If either process exits, the launcher stops the other. On Ctrl-C, the robot program parks the arm at its home pose on the still-running bus, and then the bus stops.

Options:

- `--bus-port PORT` also serves the bus on a localhost TCP port. This lets a driver in a container or on another machine reach it through `tools/serial_bridge.py`.
- `--status-file PATH` writes the joint and motor state as JSON at 10 Hz.
- `--robot-arg ARG` passes one extra argument to the robot program; repeat it for each argument.
- `--no-scene-camera` runs the robot program without a scene camera. `--camera-size WxH` (default 640x480) and `--camera-fps` (default 30) set the rendered stream. To use a real camera instead, pass `--robot-arg=--camera-device --robot-arg=scene=/dev/video0`.

### Scene camera

The real setup has a UVC scene camera that the robot program opens with OpenCV. The simulation stands in for it at the same boundary. The bus renders the scene from a fixed viewpoint beside the arm, like the Trossen single-arm scene camera, and serves it as MJPEG at `http://127.0.0.1:<port>/scene.mjpg`. The robot program's `OpenCvCamera` opens that URL as it would open `/dev/video0`, and publishes it as its `scene_camera` track. No kernel module or root is needed.

- Rendering runs on its own thread from snapshots of the physics state, so it doesn't slow the 1 kHz physics loop. It shows the table objects where the physics has them.
- It renders offscreen with EGL. On a machine without a GPU, set `MUJOCO_GL=osmesa` and install `libosmesa6`. If no OpenGL context can be made, the bus logs it and keeps running without the camera.
- Seeed's model gives the arm's materials no color, so they would render white. The simulation colors them in memory to approximate the real arm. This doesn't change the physics.
- To run the camera with the bus alone, pass `--camera-port PORT` to `sims.rebot.bus`. The `docker/rebot-bus` image doesn't include it.

### The bus alone

To check a driver without LiveKit, run the bus by itself and point the driver at its device:

```sh
.venv/bin/python -m sims.rebot.bus            # prints its serial device
teleop_arm_check --driver rebot_dm --connection "$XDG_RUNTIME_DIR/sims-$(id -u)/rebot-arm-ttyACM0" --move
```

For drivers on another machine, the bus also runs as a container. It serves the bus on `/dev/ttyACM0` inside the container and on TCP port 5555:

```sh
docker compose -f docker/rebot-bus/compose.yaml up --build -d
python3 tools/serial_bridge.py --connect localhost:5555 --link /tmp/rebot-ttyACM0
```

The port binds to localhost. To reach it from another machine, set `REBOT_SIM_BIND=0.0.0.0`. It has no authentication, and it moves the arm.

## What is emulated

### The bus

`dm_serial.py` implements the bridge's framing: 30-byte frames from the host, 16-byte frames back, matching motorbridge's `dm_serial.rs`. As with one real adapter, replies go to the connection that sent the command.

### The motors

`damiao.py` emulates the Damiao motor firmware. It handles these frames:

- enable, disable, set zero, clear error;
- MIT, POS_VEL, VEL and force-position commands;
- register reads and writes, including the control mode (register 10), the model limits, and the position register 80 that motorbridge reads when switching modes;
- feedback requests.

Every command is answered with a feedback frame from the motor's own ID, as on the real motor.

| Mode | Torque |
|---|---|
| Disabled | Zero. The arm has no brakes, so it sags under gravity. |
| MIT | `kp·(p−q) + kd·(v−q̇) + τ`, clamped to the model's torque limit. This is exact. |
| POS_VEL, VEL, force-position | An approximation. A velocity-limited reference drives a PI speed loop with anti-windup, and the gains are tuned in the simulator. The gains Seeed writes to registers 25–28 are stored but not used, because their firmware units are undocumented. |

The motors integrate physics implicitly (MuJoCo `implicitfast`), so stiff gains stay stable.

### The arm

`arm.py` loads Seeed's `rebotarm_b601_colored.xml` and adjusts it in memory to match the real arm:

| Motor | ID / host ID | Model | Drives |
|---|---|---|---|
| joint1–3 | 0x01–0x03 / 0x11–0x13 | 4340P (28 N·m) | joints 1–3, 1:1 |
| joint4–6 | 0x04–0x06 / 0x14–0x16 | 4310 (10 N·m) | joints 4–6, 1:1 |
| gripper | 0x07 / 0x17 | 4310 | both fingers through a rack |

**Gripper.** It matches the real arm: 0 rad is closed and −5 rad is fully open, which is 0.0285 m per finger. The −5 rad comes from Seeed's real driver (`_G_ANGLE_OPEN`) and the 0.0285 m from the URDF finger limit. Seeed's MuJoCo model allowed 0.05 m, so the simulator narrows it, and the right finger mirrors the left. Seeed's own driver maps its nominal "0.05 m" command to −2.5 rad, which opens each finger 0.01425 m, on the arm and here.

The table and the three objects are from Seeed's scene.

## Tests

| Test | Covers |
|---|---|
| `tests/test_rebot_protocol.py` | Framing and firmware emulation. No MuJoCo needed. Run `.venv/bin/python -m unittest discover -s tests`. |
| `tests/rebot_motorbridge_client.py` | motorbridge's Python API drives the running bus through the same calls as Seeed's driver: mode switches, MIT streaming, POS_VEL, the gripper's real travel, and sagging when disabled. Run it in the container with `docker compose -f docker/rebot-bus/compose.yaml exec rebot-bus python tests/rebot_motorbridge_client.py`, or locally with the device and status-file paths as arguments (needs `requirements-dev.txt`). |
| `teleop_arm_check --move` (in `trossen-livekit`) | The teleop robot's own C++ driver against the bus. |
| `python -m sims run rebot-arm` | The full robot program over LiveKit: control lease, joint telemetry, joint and gripper moves, home, and parking on shutdown. |

## What the simulator taught us about the real arm

These hold for the real motors too, and the drivers handle them:

- **Mode switches make the arm go limp.** motorbridge's `ensure_mode()` on an enabled motor first sends a zero-gain MIT frame, so the arm goes limp for a moment. Switch modes before enabling.
- **Feedback lags `enable()`.** The cached state says "disabled" until each motor answers again. Wait for fresh feedback before treating a status as a fault.
- **MIT gains alone droop.** Seeed's wrist gains (kp 18) sag about 0.1 rad under gravity without a torque feedforward.

## Limits

- The POS_VEL, VEL and force-position loops are approximations; see the table above. MIT, the teleop driver's mode, is exact.
- Not emulated: the CAN timeout register (9), communication-loss faults, temperatures (feedback reports 30 °C), and bus timing.
- The table objects are physically simulated. Only the scene camera shows their state.
- Seeed's ROS 2 stack (their driver, web console, camera, MoveIt) is not included. Its driver talks motorbridge on a serial device, so it would still run against the bus from a separate ROS install.
