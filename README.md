# robo-sims

Simulated robots for the LiveKit VR teleop stack in `trossen-livekit/`. One command starts any of them, and each joins the controller app's room as a robot you can select and drive.

```sh
python -m sims list                       # what can run
python -m sims run trossen-workbench      # run one
python -m sims run rebot-arm --help       # its options
```

| Simulation | Level | What it is |
|---|---|---|
| `trossen-wxai` | participant | One Trossen WidowX AI arm with a wrist D405 stereo camera and a scene camera. |
| `trossen-workbench` | participant | Two Trossen WidowX Pro arms with ZED X Nano wrist cameras, a ZED X Mini scene camera and a scene camera. |
| `rebot-arm` | hardware | One Seeed reBot B601-DM arm. The real robot program drives an emulated Damiao motor bus. |

Nothing here uses ROS.

## Two levels of simulation

- **Hardware level (preferred).** The simulation emulates the robot's hardware, and the robot program that runs on the real robot joins LiveKit and drives it with its real driver. Only the device path differs. The driver, safety limits, lease and LiveKit code under test are the real ones. `rebot-arm` works this way ([sims/rebot/README.md](sims/rebot/README.md)).
- **Participant level.** The simulation joins LiveKit itself and stands in for the robot program, serving the same RPCs and data tracks. This is for robots whose vendor driver can't be pointed at a simulator. Trossen's `libtrossen_arm` talks to a proprietary arm controller, so the Trossen simulations work this way ([sims/trossen/README.md](sims/trossen/README.md)).

## Setup

Use Python 3.12. Run these from this folder.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m sims fetch-assets          # or name the simulations you want
```

`fetch-assets` downloads the third-party models at pinned commits into `assets/`, keeping only the files the simulations load, unmodified, with each source's `LICENSE` and a `SOURCE` file:

| Assets | Source | License |
|---|---|---|
| `trossen_arm_mujoco` | [TrossenRobotics/trossen_arm_mujoco](https://github.com/TrossenRobotics/trossen_arm_mujoco) `a208dad` (`workbench_assets` branch) | BSD-3-Clause |
| `rebot_b601_dm` | [Yang-Ci/ReBot_Arm_DigitalTwin_DM](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM) `7456d8b` | Apache-2.0 |

`rebot-arm` also needs the teleop robot program built with motorbridge (see `trossen-livekit/docs/development.md`). The default path is `trossen-livekit/robot/build/trossen_livekit_robot`. To use another build, set `TELEOP_ROBOT_BIN` or pass `--robot-program`.

## LiveKit credentials

All the simulations use the same credentials as the controller app:

- They read `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` and `LIVEKIT_ROOM_NAME` from `trossen-livekit/controller/.env.local`. A variable set in the environment takes precedence. To read another file, pass `--env-file`.
- The simulation makes its own token. Its identity is `simulation-<sim>-<scientist>`, for example `simulation-rebot-arm-curie`, with the scientist name taken from the same list as the controller app's `controller-<scientist>` identities. It picks a name nobody in the room is using, so several simulations can share one room. The log shows the identity:

  ```text
  INFO Joining room <room> as simulation-rebot-arm-curie
  ```

- To join with a token you made yourself, set `LIVEKIT_TOKEN` or pass `--token`, together with `LIVEKIT_URL` or `--url`.
- For a hardware-level simulation, the token goes to the robot program in its environment, not on its command line. The API secret stays in the launcher.

In the controller app, select the simulation's identity in the Robot list. Its identity doesn't start with `controller-`, so the app lists it as a robot.

`--check` loads a simulation and prints what it provides without joining LiveKit.

## Layout

| Path | Contents |
|---|---|
| `sims/__main__.py` | The `list`, `fetch-assets` and `run` commands. |
| `sims/registry.py` | The simulations. To add one, write a module with `add_arguments(parser, sim)` and `run(args, sim)`, and register it here. |
| `sims/livekit_auth.py` | Credentials and identity picking, shared by every simulation. |
| `sims/assets.py` | The pinned model sources. |
| `sims/paths.py` | Where the teleop repo, the robot program and the assets are. `TELEOP_REPO`, `TELEOP_ROBOT_BIN` and `SIM_ASSETS_DIR` override the defaults. |
| `sims/trossen/` | The Trossen simulations and their LiveKit probe. |
| `sims/rebot/` | The reBot emulated hardware and its launcher. |
| `tools/serial_bridge.py` | Turns a bus served over TCP back into a local serial device. Standard library only. |
| `docker/rebot-bus/` | The reBot emulated hardware alone, as a container. |
| `tests/` | Unit tests (`.venv/bin/python -m unittest discover -s tests`) and the motorbridge end-to-end client. |

## Planned

These are planned, but not built yet: Trossen Rivet, reBot dual arm, SO-101, a four-wheeled rover, and the rover with a reBot arm. Each new simulation should be hardware level where the robot's drivers allow it.
