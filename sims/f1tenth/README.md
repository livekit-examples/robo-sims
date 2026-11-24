# Formula RC (`f1tenth`, participant level)

A formula-style RC car on the community Spielberg circuit, driven through the
generalized Teleop workspace. This simulation needs no ROS process or robot
driver. It publishes an onboard camera, vehicle state, odometry and simulated
battery state, and accepts speed/steering commands from the active operator.

```sh
.venv/bin/python -m sims fetch-assets f1tenth
.venv/bin/python -m sims run f1tenth --check
.venv/bin/python -m sims run f1tenth --identity simulation-f1tenth
```

Credentials use the same `.env.local` and LiveKit room as the other simulations.
Open the Teleop `/workspace`, choose this participant, open Controls, select
keyboard, joystick or gamepad, request control, and arm the input. Let the input
return to neutral before driving. The default gamepad mapping uses the left
stick: up/down requests forward/reverse speed; left/right requests steering.
Keyboard uses the arrow keys. Neutral applies braking and centers steering.
Changing direction brakes to near rest before reverse torque is applied.

The Teleop web changes in the parent repository are required: old web builds do
not understand the `rc.ackermann.v1` simulation implementation. The speed/lap HUD
appears in the car's control panel. Managed `/try` sessions use the same policy
and include a Formula RC card after the updated worker image and web app are
built. This change does not deploy either service.

## Community scale

`vehicle.json` records the pinned F1TENTH Gym source revision and baseline:
0.58 × 0.31 m footprint, 0.3302 m wheelbase, 3.74 kg, 0.04712 kg·m² yaw inertia,
±0.4189 rad steering, 3.2 rad/s steering-rate bound, and the 9.51 m/s² acceleration
parameter. The generated body is an unbranded formula-style shell within that
footprint, rather than a replica of a particular F1 car.

Spielberg is imported from `f1tenth/f1tenth_racetracks` at
`b95c4eff766f6367d66b310ea20cd2c9563712c0`. Its coordinates are already in scaled
meters; road width is 2.20 m. The supplied centerline closes to **343.32 m**.
Those are the community data's dimensions, not an assertion that its layout is
an exact 1:10 survey of today's circuit. Road rendering, tire surfaces, progress
and barriers derive from the same centerline/widths. No second scale factor is
applied. `fetch-assets` preserves the upstream files, README and GPL-3.0 license
under `assets/f1tenth_racetracks`, together with a pinned `SOURCE` record.

Forward operating speed defaults to **8 m/s**, a provisional teleop cap inside
the community simulator envelope of −5 to +20 m/s. Set `--max-speed 4` for slower
commissioning or up to `--max-speed 20` for controlled testing. Reverse is capped
at the smaller of 5 m/s and the chosen forward cap. The cap is a target-speed
limit; tire forces and acceleration determine measured speed. Physics and video
run in real time; full-size F1 speeds are not divided by ten.

Analog steering in the updated browser uses a progressive curve: one-quarter
stick/joystick requests about 1.5° instead of 6°. Full travel still requests
the community limit of about 24°, so full-lock turns at high speed can exceed
tire grip. The policy declares `steering_input_exponent: 2`; older policies
without this field retain their linear mapping. Direct Ackermann commands remain
physical steering angles in radians. Keyboard steering builds progressively while
left/right is held, taking about 1.25 seconds to reach full input. Short taps give
small turns; releasing the steering key centers more quickly. A direction change
returns through center first. Arming, disarming, changing input mode and losing
focus clear the accumulated keyboard steering. Throttle key release still
requests braking immediately. Holding full lock at high speed can still exceed
grip; this input shaping does not add a stability or track-following assist.

## Dynamics and limits

The planar four-wheel model runs at 200 Hz with wheel integration substeps of
at most 1 ms. It includes individual wheel angular speeds, Ackermann front-wheel
angles, bounded steering response, four-wheel drive with equal drive torque,
nonlinear longitudinal/lateral tire forces sharing a friction budget, approximate
load transfer, drag, rolling resistance and lower grip on grass. Barriers limit
the oriented footprint and remove outward velocity; impacts are approximate.
There are no pit stops, tire wear or fuel management.

Braking is applied as a passive wheel impulse: it can stop a wheel but cannot
drive it backward. Brake demand follows current normal loads, an approximation
of balanced braking that avoids excessive demand on an unloaded rear wheel.
This corrects the earlier reverse-torque braking model, which could produce a
spin during an otherwise mild corner. Reverse propulsion engages near rest.

The `provisional` block in `vehicle.json` identifies additional tire, drivetrain
and actuator assumptions. They are not validated hardware measurements. This is
an RC-inspired handling simulation with a community dimensional baseline. It
does not model suspension, road elevation, curb hops or rollover. Curbs are
visual road-edge markers; outside the paved road, the tire uses grass grip.
Lap timing counts full forward circuits and invalidates laps on leaving the road
or contacting barriers. Off-track recovery uses driving/reverse, without an
automatic reset or track-following assist.

## Transport and stopping

Discovery uses schema version 1 and Portal version 3. The `base.drive` operation
advertises `base.ackermann_drive.v1`, JSON `ackermann_msgs/msg/AckermannDrive` on
`/ackermann_cmd`. Values are meters/second and radians, positive steering left.
Only finite speed/angle values within policy bounds are accepted. Unsupported
nonzero acceleration, jerk and steering-rate fields are rejected; local actuator
limits always apply. Command schema: `backend.ACKERMANN_SCHEMA`.

The gateway checks active-owner identity, role/version, exact command endpoint
and publication schema/encoding. A newly armed publication must send zero speed
and steering before motion. Handoff, reconnect and timeout invalidate extant
command publications. The browser must request control and publish/arm a fresh
input to resume. These rules also reject delayed messages on the old track.

After **150 ms** without an accepted command, the robot brakes and centers
steering, disarms and clears ownership. Malformed/non-owner commands do not
refresh that deadline. Disconnect, track closure, role loss and shutdown also
cancel the drive target. A stop changes actuator targets; it does not set measured
velocity to zero instantaneously. Backend command timeouts use a monotonic clock.
The physics scheduler catches up in bounded fixed steps; more than 100 ms of lag
disarms the car and records an overload rather than replaying a long backlog.

Telemetry at 20 Hz:

| Track | Schema | Contents |
| --- | --- | --- |
| `/vehicle_state` | `robo_sims/msg/VehicleStateV1` | Measured/command speed, steering, yaw, slip, track position, laps, off-track/contact and timeout state. |
| `/odom` | `nav_msgs/msg/Odometry` | Exact simulated pose in `map`, body-frame velocity in `base_link`; zero covariance indicates simulator ground truth. |
| `/speed_mph` | `std_msgs/msg/Float64` | `data`: nonnegative measured ground speed in mph, including lateral motion and reverse. |
| `/gear` | `std_msgs/msg/String` | `data`: `F`, `N` or `R` for the fixed-ratio electric drive state. |
| `/battery_energy` | `robo_sims/msg/BatteryEnergyV1` | `remaining_wh`, `capacity_wh`, and `state_of_charge_percent`. |
| `/steering_angle_deg` | `std_msgs/msg/Float64` | `data`: actual equivalent front steering angle in degrees, positive left. |
| `/yaw_rate_deg_s` | `std_msgs/msg/Float64` | `data`: actual yaw rate in degrees/second, positive left. |
| `/longitudinal_acceleration_g` | `std_msgs/msg/Float64` | `data`: body-forward force acceleration divided by 9.81 m/s². |
| `/lateral_acceleration_g` | `std_msgs/msg/Float64` | `data`: body-left force acceleration divided by 9.81 m/s². |
| `/lap_number` | `std_msgs/msg/Int32` | `data`: number of completed forward laps; lap 0 is the initial lap. |
| `/lap_time_s` | `std_msgs/msg/Float64` | `data`: elapsed simulation seconds since the current lap began. |
| `/last_valid_lap_time_s` | `robo_sims/msg/OptionalFloat64V1` | `data`: most recent valid lap time in seconds; `null` until one exists. Invalid laps retain the prior valid result. |
| `/track_progress_percent` | `std_msgs/msg/Float64` | `data`: position around the current circuit, 0–100%, wrapping at the start line. |
| `onboard` | Video | 90° horizontal field of view; matching fixed `camera_info`. |

Dedicated instrument tracks publish at 20 Hz and appear automatically in the
workspace topic selector. Add a Value or Plot visualization, choose the track,
and select the `data` field. For `/speed_mph`, label the panel “Speed (mph)”.
The conversion uses the exact international mile, 1,609.344 meters; speed is
measured motion, not commanded speed or a full-size F1 equivalent.
Publications are spread across the 50 ms cycle to avoid burst starvation on the
lossy transport. `--check` lists the instrument names, schemas and target rates.

Use a State visualization for `/gear`, because its `data` field is a string.
F/N/R describes the drive state, not a multi-speed racing gearbox. During
braking into a reverse request it retains F until the model permits reverse near
rest; stopped with no speed request reports N. N does not select a separate
coasting control mode. `/last_valid_lap_time_s` is deliberately nullable: a
numeric visualization becomes usable after the first valid lap, rather than
displaying a fabricated zero-second lap. The acceleration tracks report modeled
tire/drag forces; instantaneous barrier impulses are not included.

Battery energy replaces a combustion-fuel gauge for this electric model. The
provisional pack is 11.1 V × 5 Ah = 55.5 Wh. This is a simulator preset, not a
universal community battery specification; the [community FAQ](https://f1tenth.readthedocs.io/en/main/getting_started/faq.html)
includes 3S LiPo packs among its examples. `vehicle.json` also marks 15 W idle
load, 85% drive efficiency, and up to 25 W additional drive loss as provisional.
Energy consumption integrates positive wheel-shaft power plus those estimated
losses in simulation time, including reverse drive. Braking does not recharge
the pack. At zero energy propulsion is inhibited while braking remains available.
A new simulation starts with a full pack; there is no refueling or pit-stop flow.

For a battery Value/Plot, select `remaining_wh` or `state_of_charge_percent`
instead of `data`. The pre-existing `/base/battery_state` voltage now follows the
same pack estimate (12.6 V full to 9.9 V empty); it no longer uses the unrelated
24 V arm fixture. Its simple voltage estimate is not a fitted discharge curve.
The new instruments leave the existing aggregate `/vehicle_state` schema intact.
Dedicated contact/slip and command-quality tracks were not added.

JSON tracks carry explicit schema and frame-encoding metadata through the shared
schema bridge. No video is rendered with `--no-video`; telemetry and controls
still operate. Stable identities preserve workspace bindings.

## Graphics

The scene is generated at startup with no downloaded art: textured asphalt with
painted edge lines, red/white curbs only where the circuit bends, LiveKit
hoardings on the barriers, a grandstand and pit building on the start straight,
a start gantry, low-poly trees, horizon hills, fog and sky haze. The car is an
unbranded lofted formula body with halo, wings, wishbones and slick tires, in
gloss black with LiveKit cyan accents. Road, curbs and barriers use the same
centerline and widths as the physics; the hoarding face sits exactly at the
physics barrier limit. Visuals never change grip or collisions.

Static scenery is merged into a few textured meshes, so a frame has about 80
geoms instead of the previous ~900 boxes. On the 2-core llvmpipe test host the
default view rendered at 960x540 in ~18 ms versus ~19 ms before (full video cycle
21 ms versus 26 ms); 640x360 at 30 fps stays under budget.

| Flag | Values | Effect |
| --- | --- | --- |
| `--camera` | `tcam` (default), `cockpit`, `chase` | Single streamed view. `tcam` is the broadcast camera above the airbox; `chase` is a rigid third-person view. `camera_info` is computed from the selected mount, so metadata stays exact. |
| `--graphics` | `auto` (default), `low`, `medium`, `high` | `low`: no trees, hills or stands. `medium`: full scenery with a soft contact shadow. `high`: adds a real shadow map and 4x MSAA. `auto` probes the OpenGL renderer: `high` on a GPU (Mac, Modal L4) and `medium` on a software rasterizer (OSMesa/llvmpipe, as in the managed Docker image), where a shadow pass more than doubles frame time. |

## Render and measure locally

```sh
.venv/bin/python -m sims run f1tenth --no-video
.venv/bin/python -m sims run f1tenth --preview /tmp/f1tenth.png
.venv/bin/python -m sims run f1tenth --benchmark 10
.venv/bin/python -m sims run f1tenth --benchmark-video 15 --video-stats
.venv/bin/python -m sims run f1tenth --fps 30 --width 640 --height 360
.venv/bin/python -m sims run f1tenth --camera chase --graphics high --preview /tmp/chase.png
```

The default is one 960×540 camera at 60 fps, using the classic offscreen renderer.
Managed sessions render at 640×360 and respect `SIM_SESSION_FPS` (the existing
session-service default is 30). Set it to 60 after measuring the host's capacity.
`--renderer filament` uses the existing batched rendering implementation; on
macOS run it with ordinary Python so the main-thread event pump can operate.
The renderer copies state into its worker-owned data. It never advances vehicle
physics. The physics-only benchmark runs faster than wall time deliberately;
normal teleop and video benchmarks use the wall-paced scheduler.

The video benchmark measures rendering/readback/frame capture with physics,
excluding network and encoding. Actual remote latency and supported concurrent
session count require measurements on the deployment host. The existing pool
limit of 20 does not establish graphics or encoder capacity for 20 cars.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_f1tenth*.py' -v
livekit-server --dev --config tests/livekit-dev.yaml
# In another terminal, against that localhost server only:
ROBO_SIMS_LIVEKIT_TEST=1 ROBO_SIMS_VIDEO_TEST=1 \
  .venv/bin/python -m unittest discover -s tests -p test_f1tenth_livekit.py -v
```

Python tests cover geometry/units, strict command parsing, independent timeout,
braking/reverse, steering, tire-force bounds, surface/collision response, lap-line
oscillation, numerical convergence and camera metadata. The opt-in native test
receives decoded video and exercises discovery, unauthorized commands, driving,
timeout, fresh-publication requirements, handoff and role loss. The web project
adds adapter tests and `tests/browser/racing.spec.ts`; enable the browser test
with `TELEOP_RACING_E2E=1` against the local managed session service and web app.

For the optional ten-minute moving-car rendering soak, run
`PYTHONPATH=. .venv/bin/python tests/f1tenth_soak.py`. This test fails if the
scheduler overloads, even if it recovers. See [measured validation and remaining
limits](../../docs/f1tenth-validation.md), including the two overload stops
observed on the shared development host.
