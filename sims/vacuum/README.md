# Vacuum robot (`vacuum`, participant level)

A round vacuum robot cleans a furnished 2 x 2 m room. MuJoCo calculates movement and contact. LiveKit carries two video views, driving commands, and robot measurements. The design is in [docs/vacuum-design.md](../../docs/vacuum-design.md). This file describes what exists now and what is still open.

```sh
.venv/bin/python -m sims run vacuum --check
.venv/bin/python -m sims run vacuum --preview overhead.png
.venv/bin/python -m sims run vacuum --identity simulation-vacuum
```

The simulation needs no ROS process and no downloaded 3D shapes. `--check` prints the discovery document, the policy, the floor areas, and every data track without a LiveKit connection.

## Options

| Option | Effect |
| --- | --- |
| `--start normal` | Full battery and an empty bin. This configuration allows a complete run. |
| `--start low_battery` | The battery holds 0.05 Wh. Driving and cleaning stop after about 6 seconds. |
| `--start nearly_full_bin` | The bin holds 0.49 L of 0.5 L. Cleaning stops after 0.25 m². |
| `--auto-start` | Start the timed run on the first driving command. Use it until the browser has a Start run button. |
| `--no-assist` | Turn off the slowing near walls and furniture. |
| `--no-video` | Run physics and telemetry without a graphics context. |
| `--benchmark SECONDS` | Measure physics speed offline. |
| `--route` | Drive the reference route offline and print the result. |

## Room and robot

The room is 2 x 2 m with north at the top. It holds a four-legged chair, a cabinet against the north wall, and a floor-lamp base. The furniture is fixed. The positions are in `room.py`.

The robot is a cylinder that measures 0.34 m across and 0.09 m tall. It has two powered wheel hinges and two low-friction casters. The values in the design table are in `simulation.py`. The chair seat is 0.45 m above the floor and the lower bars are 0.20 m above the floor. The robot top is 0.105 m above the floor. The gap between chair legs is 0.46 m, which is wider than the 0.42 m minimum passage. The robot drives under the chair seat.

Collision shapes and decoration are separate. Decoration has `contype=0`, `conaffinity=0`, and zero mass.

MuJoCo uses the larger friction value of two touching shapes. For this reason the floor friction is low, the wheels carry the grip, and the walls and furniture have low friction.

## Floor and cleaning

One grid of 1 cm squares (200 x 200) drives the floor image, the cleaned-area counter, and the bin fill. Separate maps mark furniture and floor that the brush cannot reach.

The brush footprint is a circle that measures 0.34 m across. At each physics step the simulation cleans along the path between the previous and the current measured position. Cleaning needs a started run, battery energy, and free bin space. A second pass over a clean square adds no dust to the bin. A full bin stops cleaning but not driving.

The planner keeps the robot center 5 mm away from every solid. If the brush reaches a square with 1 cm to spare, the square counts as cleanable. Other squares do not count. For this reason a strip of about 1 cm beside each wall and each piece of furniture counts as unreachable. This is a design choice that avoids millimeter driving. Change `ROBOT_MARGIN` and `REACH_TOLERANCE` in `room.py` to change it.

Areas of the starter room:

| Area | Value |
| --- | --- |
| Exposed floor | 3.80 m² |
| Under furniture contact points | 0.20 m² |
| Cleanable (the brush can reach it) | 3.61 m² |
| Unreachable (the brush cannot reach it) | 0.19 m² |

Completion means 100% of the cleanable squares. Unreachable floor is blue-gray and never counts as vacuumed.

The floor texture is replaced from the grid with `mjr_uploadTexture`. The upload runs on the render thread with its graphics context active. When the grid changes, the upload runs. Otherwise no upload runs.

## Driving

Send `geometry_msgs/msg/Twist` JSON on `/cmd_vel`. The robot reads `linear.x` as forward speed and `angular.z` as yaw rate. Positive yaw turns left. The robot rejects invalid numbers, sideways motion, and any other axis with a nonzero value.

The driving limits are 0.45 m/s forward, 0.25 m/s reverse, 1.8 rad/s yaw, 0.8 m/s² acceleration, and 5 rad/s² turning acceleration. A precision speed of 0.15 m/s is in the policy.

The simulation shapes the command with these limits, converts it to wheel speeds, and lets MuJoCo move the body. If one wheel target is over its limit, both targets shrink by the same factor. Reported motion comes from the simulated body and never from the command.

Driving assist reduces the forward speed near objects. It uses the distance ahead and the stopping time. It never goes below 0.03 m/s, so the robot can still touch a wall. Turning and reverse stay available. When the assist is active, the on-screen display says so.

The watchdog limit is 150 ms. If commands stop, the robot brakes within its limits, drops the controls, and the operator must request control and arm again. A new publication must send neutral before it can move the robot. Handoffs, disconnects, reconnects, role changes, and overloads work the same way.

If the simulation falls more than 100 ms behind, it records an overload, disables driving, and marks the timing trial as invalid. Simulation time never speeds up, and old commands are never replayed.

## Run control

The operator that controls the robot, with controls armed, calls two RPC methods with an empty payload.

| Method | Effect |
| --- | --- |
| `vacuum.start_run` | Start the 60 second timed run and turn on the vacuum. |
| `vacuum.reset_run` | Return to the start pose, clean the floor, refill the bin and battery for the chosen `--start`, and use a new run ID. |

Other users receive an RPC error. A run starts only once. Reset before another attempt.

## Data tracks

Every track uses JSON with a registered JSON Schema. State messages carry `sim_time_s`, `run_id`, and `sequence`. The schema ID of each custom message ends in `V1`.

| Track | Contents | Rate |
| --- | --- | --- |
| `/speed_mps` | Measured forward speed, negative in reverse | 20 Hz |
| `/yaw_rate_rad_s` | Measured yaw rate | 20 Hz |
| `/battery_state` | Remaining and total Wh, percent, volts, watts | 10 Hz |
| `/dust_bin_state` | Used, total, remaining liters, percent, full flag | 10 Hz |
| `/coverage_state` | Cleaned, cleanable, exposed, unreachable m², percent, vacuum state | 10 Hz |
| `/run_state` | Run ID, elapsed, deadline, result, overload, requested motion, assist | 10 Hz |
| `/odom` | Measured position, direction, and body-relative motion | 20 Hz |
| `/base/battery_state` | Shared `sensor_msgs/BatteryState` for the workspace | 1 Hz |

Data tracks are lossy, so each message repeats the complete current state. The sender pushes at most one track per 5 ms tick, because a burst starves some tracks.

Two video tracks publish at 960 x 540 and 30 fps: `overhead` and `onboard`. The overhead view shows the remaining dust, the robot direction arrow, the chair legs, the cleaned percentage, and the elapsed time.

Battery power is 5 W for electronics, 18 W for the vacuum, and up to 12 W for driving. Bin fill is 0.04 L/m². These are initial assumptions and not product data.

## Web workspace

The parent repository holds the browser changes. `web/src/lib/demo/vacuum-adapter.ts` reads the `vacuum.twist.v1` policy and builds Twist commands. It applies a deadband of 0.08 and a squared response, and it sends neutral first. It accepts a third axis (0 or 1) as the precision switch. The tests are in `web/tests/vacuum.test.ts`.

## Tests

```sh
.venv/bin/python -m unittest tests.test_vacuum
```

Run the LiveKit test against a local development server:

```sh
livekit-server --dev --config tests/livekit-dev.yaml
ROBO_SIMS_LIVEKIT_TEST=1 .venv/bin/python -m unittest tests.test_vacuum_livekit -v
ROBO_SIMS_LIVEKIT_TEST=1 ROBO_SIMS_VIDEO_TEST=1 .venv/bin/python -m unittest tests.test_vacuum_livekit -v
```

Run the real-time soak:

```sh
PYTHONPATH=. .venv/bin/python tests/vacuum_soak.py 600
```

## Measurements

Measured on an Apple M4 Pro (`Mac16,7`, macOS 26).

| Item | Result |
| --- | --- |
| Offline physics speed | About 100 times real time |
| Top forward speed | 0.43 m/s with a 0.45 m/s limit |
| Top yaw rate | About 1.6 rad/s with a 1.8 rad/s limit |
| Stopping distance from 0.43 m/s | 0.11 m, which matches the 0.8 m/s² limit |
| Overhead and onboard video | 30 fps each |
| Soak, 10 minutes | Moving one-minute ratio from 0.9999 to 1.0001, no overloads |
| Soak video | 29.98 fps on each view, largest gap between frames 96 ms |

## Open work

The design needs these items, and they are not done.

- The reference route does not meet the 50 second target. The first route cleans about 56% in 60 seconds and about 89% in 90 seconds. Chair legs break each lane into pieces, and the follower slows near them. The test `test_reference_route_cleans_every_declared_square_within_50_seconds` is an expected failure that records this gap. A smaller room, a layout with fewer broken lanes, or a better route planner can close it. Until then the room layout is not accepted.
- No new operator drove the room, so the 60 second target and the camera mapping are untested with people.
- The browser has no Start run and Reset run buttons yet. Call the RPC methods from a client. The precision switch has no control in the browser yet.
- No browser session tested the workspace end to end.
- Latency below 150 ms for 95% of measurements is not measured. The wide-area network is not measured.
- Stopping distance is measured for a command stop in a unit test. It is not recorded for each event (watchdog, lost focus, disconnect, handoff, overload).
- The managed `/try` scene is not added, and nothing is published or deployed.
