# 1/10-scale formula teleop simulator: research and design

Date: 2026-11-09. Status: original design proposal, followed by a first implementation.
See [the simulator README](../sims/f1tenth/README.md) for current behavior and
[implementation validation](f1tenth-validation.md) for measured results and remaining limits.
The sections below preserve the design and proposed acceptance criteria; they are not a completion checklist.

The user selected research and design for this pass, with **real 1/10-scale RC-car physics**. The intended addition is an F1-style car operated through the existing Teleop application. The user's scale clarification gives **community conventions priority for car dimensions, track sizing, speeds and related parameters wherever practical**. Pit stops are outside scope.

## Recommendation

Add a participant-level simulation named `f1tenth` with a small vehicle-dynamics backend, a 3D scene rendered using the repository's MuJoCo infrastructure, and LiveKit discovery/control. Use a planar four-wheel handling model with nonlinear tire forces for the first driving release. Use F1TENTH's simpler single-track model as a reference and comparison baseline, rather than treating its default environment as a finished remote-driving simulator.

Use the F1TENTH Gym vehicle dimensions and parameter envelope, together with an existing F1TENTH community circuit at its published scale, as the initial compatibility baseline. Keep the track pack's 2.20 m road-width convention. This supersedes the earlier proposal to enforce exact full-size F1 car-to-track ratios. Use a formula-style visual body within the baseline footprint; label it as a stylized RoboRacer-scale vehicle. Do not divide full-size F1 speeds or physical parameters by ten. Start with one car, one circuit, one forward onboard camera, a speed display, and lap timing. A track overview is optional and lower priority than the onboard view.

This is a proposed engineering choice for this repository, not a claim that a custom model is more accurate than an existing vehicle simulator. Measured handling and operator trials must establish that it is adequate. A planar first release cannot reproduce suspension travel, jumping curbs, banking, or rollover; those require a later 3D dynamics extension.

## 1. Does this exist in the community?

Yes. [RoboRacer, formerly F1TENTH](https://roboracer.ai/), is an established open-source autonomous-racing platform. Its current hardware description uses a Traxxas Slash 4x4 chassis, steering servo, electric motor controller and onboard computer. It is a strong robotics precedent, but its standard vehicle is not a miniature open-wheel F1 car.

Literal 1/10 formula RC cars also exist. For example, [Tamiya's F104 PRO II](https://tamiya.com/english/products/58652/index.html) is a rear-drive electric formula car, with published length 440 mm, width 180 mm, wheelbase 270 mm and tire diameter 64 mm. That establishes a physical formula-car precedent; it does not establish the dynamics of our proposed car or exact proportions of a current full-size F1 car.

Relevant open-source components:

| Project | Useful contribution | Gap for this request |
| --- | --- | --- |
| [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym) | Small vehicle model, control examples and reproducible stepping. | Its stock environment is aimed at autonomy experiments; we still need onboard 3D video, Teleop integration and handling validation. |
| [F1TENTH ROS bridge](https://github.com/f1tenth/f1tenth_gym_ros) | Demonstrates a robotics interface around the simulator. | This repository does not use ROS; adding a ROS runtime is unnecessary for LiveKit commands. |
| [F1TENTH racetracks](https://github.com/f1tenth/f1tenth_racetracks) | Familiar circuit layouts and map formats; selected default for track scale. | Its README says layout is downscaled but width is fixed at 2.20 m. Follow that convention while documenting that it changes full-size proportions. |
| [TUM racetrack database](https://github.com/TUMFTM/racetrack-database) | Original centerlines and separate left/right widths. | GPS/satellite-derived approximations, not surveyed 3D circuits. Elevation and curbs need another source or explicit omission. |

**Selected convention:** Follow the community track pack, including its normalized road width. Exact full-size car-to-track ratios are no longer the default requirement. Do not apply another 0.1 factor to an already scaled map. The pack's [processing description](https://raw.githubusercontent.com/f1tenth/f1tenth_racetracks/main/README.md) documents the width change; the [original database description](https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/README.md) explains the source accuracy limits. An exact-proportion circuit could be a separately named future variant.

## 2. How robotics communities simulate cars

There are several useful levels of detail. Steering geometry alone does not determine tire grip or handling at speed.

| Approach | What it computes | Useful for | Tradeoff here |
| --- | --- | --- | --- |
| Kinematic bicycle | Motion from wheelbase, speed and an equivalent front steering angle; front/rear wheel pairs are collapsed. | Parking, slow driving and basic controller checks. | Cannot by itself reproduce grip loss, understeer or oversteer. |
| Dynamic single-track | Adds sideways motion, yaw dynamics and tire response at two equivalent axles. | Fast autonomy/control experiments. | Efficient baseline; simpler models need extension for limit handling. |
| Four-wheel handling | Separate tire forces and load estimates, chassis longitudinal/lateral/yaw motion. | Steering/braking interaction and RC drivetrain handling. | More parameters to calibrate, but still a small numerical system. |
| Multibody vehicle | 3D chassis, wheel rotation, suspension joints, contacts and tire forces. | Bumps, banking, suspension and rollover. | Greater solver and tuning cost; extra mechanical detail does not supply tire calibration. |
| Detailed tires/terrain | Tire structure deformation and possibly deformable ground. | Tire/terrain engineering. | Unnecessary for the initial paved-circuit teleop task. |

The [F1TENTH dynamics source](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/dynamic_models.py) implements a seven-state dynamic single-track model, steering/acceleration limits, and a kinematic branch below 0.5 m/s. Its lateral equations use a linearized response. Merely selecting that model does not provide a nonlinear combined-slip tire model or establish realistic drifting. Its [vehicle class](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/base_classes.py) supports Euler/RK4 integration, Numba-compiled functions, and a default 0.01 s step.

Robotics frameworks also provide joint-based vehicles: [Gazebo's Ackermann steering system](https://gazebosim.org/api/sim/7/classgz_1_1sim_1_1systems_1_1AckermannSteering.html) configures wheel and steering joints. For higher-fidelity handling, [Chrono::Vehicle](https://api.projectchrono.org/wheeled_tire.html) distinguishes rigid tires, semi-empirical handling tires such as Pacejka/TMeasy, and finite-element tires. Those are different fidelity choices, not interchangeable guarantees of realism.

CARLA is another option when a rich road scene and camera sensors matter. Its [timing documentation](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/) separates simulation time, fixed steps, physics substeps and client synchronization. For this repository, a dedicated CARLA or Chrono runtime would add deployment work; evaluate one if the simpler proposed model fails the handling requirements or 3D vehicle mechanics become necessary.

## 3. Geometry and physical scale

Use SI units throughout: meters, seconds, kilograms, radians, newtons. Prefer a named, pinned community preset over inventing a scale or mixing dimensions from different platforms. These are widely used project conventions, not a single universal standard for every 1/10 RC vehicle or race.

The initial vehicle baseline comes from the executable defaults in [F1TENTH Gym's environment source](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/f110_env.py):

| Parameter | Community simulator baseline | Use in this design |
| --- | --- | --- |
| Vehicle footprint | 0.58 m long × 0.31 m wide | Keep visual envelope and collision footprint consistent. |
| Wheelbase | 0.3302 m | Sum of front/rear center-of-mass offsets, 0.15875 m and 0.17145 m. |
| Mass / yaw inertia | 3.74 kg / 0.04712 kg·m² | Baseline chassis properties. |
| Center-of-mass height | 0.074 m | Baseline for approximate load transfer. |
| Equivalent steering angle | ±0.4189 rad, about ±24° | Mechanical command envelope. |
| Steering rate | ±3.2 rad/s | Baseline actuator rate bound. |
| Acceleration magnitude parameter | 9.51 m/s² | Preserve the baseline constraint and its speed-dependent acceleration behavior when comparing models. |
| Speed bounds | −5 to +20 m/s | Simulator parameter envelope, not a recommended operating speed everywhere or a competition-wide limit. |

Use the community circuit's supplied meter coordinates, widths and map-resolution/origin metadata. The selected track pack uses 1:10 layouts and a fixed 2.20 m road width; do not rescale either again. Generate road visuals, surface queries and barriers from the same imported boundaries. Check that the vehicle-to-road ratio matches those source dimensions. Preserve the source layout's identity/year and approximation limits.

Proposed first asset: the community Spielberg layout at its published scale, initially flat. An illustrative 5 km full-size layout is approximately 500 m after 1:10 downscaling; do not shorten straights to force an arbitrary lap time. Choose an unbranded formula-style visual shell within the 0.58 × 0.31 m footprint, with wheel locations matching the baseline. This is a stylized robotics car, not a dimensional replica of a particular F1 car or Tamiya F104.

Keep gravity at 9.81 m/s² and advance one simulation second per wall-clock second. Angles, camera field of view and image resolution are not divided by ten. Camera mounting dimensions should fit the actual baseline vehicle. Use the community RC speed envelope directly rather than dividing full-size F1 speed by ten or applying a theoretical similarity rule.

For teleop commissioning, propose a separately declared forward speed cap of 8 m/s, adjustable after latency and operator trials. That cap is our provisional operating policy, not a community standard; it must not silently change the vehicle preset or track scale. The +20 m/s baseline ceiling remains available for controlled validation, subject to grip and actuator dynamics. Publish the effective operating limits to the browser. Community raceline speed values are planning references, not guaranteed safe speeds for our tire model or remote operator.

Maintain one versioned vehicle parameter file with the community source revision and all effective overrides. Reuse the preset's tire-response coefficients for the single-track comparison. The proposed nonlinear four-wheel model also needs tire radius, drive/brake torque, differential behavior and nonlinear tire curves; those do not follow automatically from the single-track preset and require source data or explicitly provisional calibration. Start with the community platform's four-wheel-drive architecture rather than introducing rear drive solely because the shell looks like F1. Full-size F1 downforce coefficients must not be transplanted without evidence. The community baseline supports comparability; it does not establish validated formula RC hardware fidelity.

## 4. Recommended vehicle model and driving feel

The driving-release model should integrate chassis position, heading, body-frame longitudinal/lateral velocity, yaw rate, four wheel angular speeds, and steering actuator state. Derive each tire's local contact velocity from chassis velocity, yaw rate and wheel position.

1. Compute left/right front steering angles from an equivalent Ackermann angle. Include finite servo speed and response time.
2. Compute longitudinal slip and lateral slip angle at each tire. Regularize near zero speed and implement a stable stopped/rolling transition; simply dividing by speed is unstable at rest.
3. Compute nonlinear longitudinal/lateral tire forces, limited jointly by available friction. A friction ellipse prevents maximum braking and maximum turning grip from being available simultaneously.
4. Estimate normal loads from static distribution and quasi-static longitudinal/lateral transfer. Clamp loads nonnegative and conserve total normal load. This approximation does not simulate chassis roll or suspension travel.
5. Apply electric drive torque through a documented four-wheel-drive distribution and differential model consistent with the community platform; include braking, rolling resistance and drag. Integrate wheel speeds and chassis dynamics at the fixed step. Any later rear-drive formula variant must use a separate preset.
6. Sample asphalt/curb/grass grip at each tire. Use the same boundaries for the rendered road and physical surface queries.

Tune front/rear grip and response against the community baseline and eventual reference data, including understeer, traction loss and recovery. The car must slow physically under braking and slide physically after saturation. Do not force rear-drive behavior onto the four-wheel-drive baseline. Avoid unlimited yaw rotation, instantaneous velocity assignment, forced track following, or automatic heading alignment.

Keep collision geometry simple and independent of decorative meshes. Use a spatial index of barrier segments plus swept footprint collision checks to prevent crossing barriers between steps. Resolve impact with bounded impulses and friction; test sliding along barriers. Off-track driving should reduce grip and invalidate laps, rather than teleport the car. Curbs initially affect surface grip only; the flat model cannot honestly claim curb-hop dynamics.

For remote feel, prioritize a stable, car-mounted forward view with visible front tires or nose, correct camera height, consistent perspective, and nearby trackside markers. Start at 90° horizontal field of view as a tuning value. Avoid an automatically rotating chase camera as the main operator feed. Add speed, lap time, off-track state and connection quality through telemetry. Audio and steering-wheel force feedback are later additions; neither currently follows from discovery metadata alone.

**Known control compromise:** the existing Ackermann profile commands target speed, not throttle/brake pedals. A bounded speed servo can reproduce RC speed-and-steering teleoperation, but neutral then means controlled braking rather than coasting. Build that behavior explicitly and validate it with operators. If independent throttle, brake and coast become requirements, define a separate executable profile and input adapter; do not reinterpret `AckermannDrive.speed` as throttle.

## 5. Real-time execution and reduced compute

A fixed physics step does not automatically make a simulator real time. The scheduler must complete enough steps to match elapsed wall time, while keeping rendering and networking from delaying control.

Proposed initial rates, subject to measurement:

| Activity | Initial setting | Reason |
| --- | --- | --- |
| Vehicle physics | 200 Hz, `dt = 0.005 s` | Small car/fast steering; compare against 400 Hz for numerical convergence. |
| Browser commands | 50 Hz | Lower input sampling delay than the existing 20 Hz presentation loop. |
| Onboard video | 60 fps at 960×540 | Start with fluid motion; offer 30 fps when measured capacity requires it. |
| Vehicle telemetry | 20 Hz | Adequate for HUD values; physics never waits for telemetry. |
| Track overview | Off by default; optional 10–15 fps | Avoid making secondary video consume the driving-view budget. |

Use a monotonic-clock accumulator: consume elapsed time in fixed 5 ms physics increments and sleep only until the next deadline. Keep rendering on its own worker using immutable/copied state snapshots; interpolate visual pose between recent states when appropriate, without predicting unacknowledged operator input. Capture video and telemetry timestamps from their actual simulation state. Do not wait for a browser tick or network round trip to advance the car.

Bound catch-up work, for example to ten substeps per scheduler pass. Record remaining lag. If the simulator falls more than 100 ms behind, disarm and reset the wall-time baseline with an explicit overload event; do not replay a long backlog of stale drive commands. Do not hide lag by increasing the physics step. Sustained load should lower optional rendering or refuse additional sessions. These thresholds are proposed policy, to be tested.

Reduce cost where it has the least effect on operator feel:

- Integrate a small handling model rather than deforming tires or solving a full suspension mechanism.
- Precompute track meshes, arc length, normals, collision index and surface lookup. Avoid scanning the entire circuit every step.
- Share static geometry and textures; use simple meshes, limited shadows and distant-detail reduction. Rendering quality should not alter physical boundaries.
- Render one primary camera and drop obsolete render requests. Use bounded queues and the newest state, not a growing video backlog.
- Keep only the latest valid control setpoint. Validate commands at reception; integrate the held setpoint locally until it changes or expires.
- Run physics-only checks without rendering; measure rendering, readback, video encoding and transport separately during teleop tests.
- Use hardware video encoding where the actual deployment supports it and measurements show a benefit. Do not assume the current Python/LiveKit path uses it.
- Compile numerical kernels only if profiling identifies a need. Avoid adding GPU physics for one small car without measured benefit.

“Headless” does not mean cameras are free: [CARLA's rendering documentation](https://carla.readthedocs.io/en/latest/adv_rendering_options/) distinguishes off-screen rendering, which still computes images, from no-rendering mode, which cannot supply camera images. The same distinction matters for our server-side onboard video.

At 960×540 RGB and 60 fps, uncompressed pixel production is about 93 MB/s per camera before encoding. This is an arithmetic estimate, not a bandwidth or CPU benchmark. Start with a configurable 4–8 Mbit/s encoded-video budget and inspect actual readability, latency and network behavior. The existing managed pool cap of 20 is not evidence that one host can render 20 driving sessions at 60 fps.

## 6. Teleop integration: findings from the local code

Reviewed the requested `teleop/docs/configuration.md`, the generalized Ackermann example, browser controls/runtime, and this repository's gateway, registry, video and managed-session code.

| Existing file | Verified behavior | Required change |
| --- | --- | --- |
| `../docs/configuration.md` | Discovery schema 1; Portal version 3; robot ownership; `base.ackermann_drive.v1`; JSON commands. | Follow this contract; implement explicit limits, neutral, release and stop behavior. |
| `sims/registry.py` | Pluggable runners; managed module is optional. | Register `f1tenth` after it is runnable. Start participant-level: there is no selected real car driver to emulate. |
| `sims/contract/config.py` | Validates arm/gripper bindings and joint feedback. | Add drive-specific configuration, or split generic discovery from arm bindings. Do not supply fake joints to reuse arm validation. |
| `sims/contract/gateway.py` | Ownership lifecycle is reusable, but commands always parse as `Trajectory`, feedback is JointState, and implementation is `mujoco.position.v2`. | Extract profile dispatch/discovery/feedback hooks and add Ackermann handling. Preserve arm semantics and tests. |
| `sims/contract/gateway.py:_simulate` | Steps 0.01 s and then sleeps 0.01 s. Compute/scheduling time adds drift. | Give the vehicle a monotonic fixed-step scheduler; avoid blindly copying this loop. |
| `sims/video.py`, `sims/trossen/filament.py` | Video capture and worker-owned rendering already exist. | Reuse the capture pattern with a car scene; keep car state independent from renderer state. |
| `../web/src/lib/teleop/controls.ts` | Ackermann adapter exists only in `fixtureAdapters()`, with fixture limits. Live adapters otherwise need installation. | Add a validated car adapter with the car's actual policy. Fixture support does not enable a live robot. |
| `../web/src/lib/demo/trajectory-adapter.ts`, `types.ts` | Simulator opt-in supports arm position implementations/policies. Unknown implementation IDs throw. | Add a distinct versioned car implementation/policy and discovery routing; do not label the car as an arm. |
| `../web/src/components/workspace/control-panel.tsx` | Inputs publish every 50 ms; gamepad uses stick axes; neutral input is required after arming. | Preserve neutral-before-motion; add 20 ms car input sampling and optional trigger mapping with explicit semantics. |
| `../web/src/lib/teleop/controls.ts:tick` | Non-position inputs stop after 250 ms without input. | Retain browser deadman behavior and add an independent vehicle-side timeout. |
| `sims/demo/rules.py`, worker, browser demo session | Managed trials currently allow Trossen scenes; browser policy types are position-only. | Add car policy support and managed registration after direct operation works. Benchmark capacity before enabling hosted trials. |

Proposed minimal discovery, **illustrative, not an enabled configuration**:

```json
{
  "schema_version": 1,
  "label": "1/10 Formula RC",
  "default_video_rate_hz": 60,
  "control": {"scope": "robot", "ownership": "portal_active_operator"},
  "control_interfaces": {
    "base": {
      "label": "Race car",
      "frame_id": "base_link",
      "operations": {
        "drive": {
          "profile": "base.ackermann_drive.v1",
          "command": {
            "track_name": "/ackermann_cmd",
            "msg_type": "ackermann_msgs/msg/AckermannDrive",
            "format": "json"
          }
        }
      }
    }
  }
}
```

This omits `feedback_tracks` deliberately: the profile does not inherently require odometry. Publish `/odom` (`nav_msgs/msg/Odometry`) and `/vehicle_state` (a versioned custom JSON schema) for displays, with speed, steering, yaw rate, slip estimates, lap state and timeout state. Ordinary video publications supply `onboard`; add fixed `camera_info` matching transmitted pixels. Every data track, including the browser's command publication, needs registered schema metadata and explicit JSON frame encoding through the existing schema bridge.

Proposed implementation ID: `rc.ackermann.v1`. Its separately versioned policy must bind the exact interface, operation and command endpoint and provide speed bounds, equivalent steering-angle bounds, acceleration/braking/steering-rate limits, input mapping, refresh rate, command timeout and neutral/release rules. Reuse the same validated policy for manual discovery and managed credentials. This is a new extension to implement on both ends; the name alone enables nothing today.

Command semantics:

- `speed`: signed longitudinal target in m/s; `steering_angle`: virtual front-axle angle in radians, positive left. Follow the [AckermannDrive definition](https://raw.githubusercontent.com/ros-drivers/ackermann_msgs/ros2/msg/AckermannDrive.msg); convert to individual front wheel angles inside the model.
- Validate finite values, payload size and policy bounds. Reject malformed/out-of-range commands without refreshing their expiry. Initially accept omitted/zero optional acceleration, jerk and steering-rate fields; reject unsupported nonzero requests rather than silently pretending to implement them. Internal policy limits always apply.
- Target speed goes through a bounded drive/brake controller. Steering goes through the servo model. Neither is assigned directly to measured state.
- Neutral/release requests zero target speed and centered steering, with bounded braking and centering. No teleporting to zero velocity. A reverse request brakes to near rest before applying reverse torque; document that this is simulated RC recovery behavior.
- Vehicle-side timeout starts at receipt of the last accepted command, using a monotonic clock. Initial threshold: 150 ms at 50 Hz. On timeout, brake, center, clear the target and require a fresh ownership/arming handshake before drive resumes. Test sensitivity to jitter before finalizing the threshold.
- Ownership handoff/release, operator disconnect, role/version loss, command-track closure, gateway reconnect and shutdown cancel drive. Only the current connected Portal v3 owner may command. A robot-owned attribute is not command authorization by itself.
- Arming or reconnecting sends no nonzero command, even with a displaced gamepad. Commands from old publications or the previous owner must not resume motion. Add a command-generation/session mechanism if transport behavior allows queued packets to outlive an ownership transition; verify this explicitly in the integration test.

## 7. Acceptance criteria and implementation sequence

All numerical thresholds below are proposed targets, not results. Record hardware, renderer, dependency versions, scene, resolution and concurrent sessions with every benchmark.

### Phase A: geometry and physics harness

Create the pinned community parameter file, import the community-scaled circuit, and build the vehicle model and deterministic offline stepping. Check footprint, wheelbase, track widths and meter-per-pixel metadata against the imported sources, including a check against accidental double scaling. Compare steady low-speed turning radius with Ackermann geometry. Test acceleration, coast/braking policy, reverse transition, slalom, constant-radius cornering, combined braking/turning, spin recovery and surface changes. Check rest stability, finite state, normal-load conservation and force bounds. Compare 200 Hz against 400 Hz; target under 2% difference in stopping distance and steady turning response for defined maneuvers, with separate treatment of chaotic post-spin paths.

Use the community single-track model as a baseline comparison, not ground truth. To claim hardware fidelity, collect reference RC data: mass and dimensions, steering step/rate, speed versus command, acceleration, stopping distance and constant-radius lateral response. Until then call the result an RC-inspired handling simulation.

### Phase B: direct teleop

Implement drive configuration/handler, policy/schema publication, browser adapter and onboard view. A future command should be `python -m sims run f1tenth --check`, followed by `python -m sims run f1tenth`; these commands do not exist yet. Keep physics-only operation available. Verify startup without external robot model downloads when using generated assets.

Add a local LiveKit end-to-end test for discovery, schema encoding, actual speed/steering response, non-owner rejection, malformed commands, dropped commands, timeout braking, handoff, track closure and reconnect without resumed motion. Verify decoded onboard video and telemetry against the simulated camera and state.

### Phase C: feel and performance

Measure command-receipt-to-physics response, render/encode time, received frame age, browser input-to-visible-response and real-time factor. Do not infer one-way delay from unsynchronized host/browser clocks; use a clock-offset method or an observable input/video loopback measurement.

Initial acceptance targets:

- Ten-minute driving run with no accumulating lag and real-time factor within 0.99–1.01 over rolling one-minute windows.
- Physics batch cost with ample margin below its 5 ms budget; initial p99 target under 1 ms on the stated target host.
- Stable 60 fps primary video when the host can sustain it; make a measured 30 fps fallback explicit. No growing frame queue.
- Controlled local-network p95 input-to-visible-response below 100 ms; assess WAN behavior separately rather than promising that latency everywhere.
- Inject 50/100/150 ms network delay, jitter, burst loss and a stalled render worker. Verify bounded braking and determine the speed at which driving remains controllable.
- At 8 m/s, a 100 ms delay represents 0.8 m of travel. This arithmetic is a reason to test RC-scale speed limits carefully; it is not solved by more accurate tires.
- Have a remote operator perform the same braking points, slalom and recovery tasks with gamepad and keyboard. Tune response from those observations. Numerical stability alone does not establish that driving feels real.

### Phase D: managed sessions

Add `managed.py`, the supported scene ID, browser policy union/adapter routing, demo card and deployment assets. Reuse the current worker lifecycle and expiration machinery. Benchmark 1, 2, 5 and then higher concurrent sessions; constrain admission by measured CPU/GPU/encoder/network capacity, even if the shared coordinator allows 20. Re-run the existing arm tests after gateway refactoring.

Asset imports must record pinned source revisions and notices through `sims/assets.py`. The community track repository contains a [GPL-3.0 license file](https://raw.githubusercontent.com/f1tenth/f1tenth_racetracks/main/LICENSE); choose and document the actual imported source and its notices before packaging. No third-party assets were copied in this research pass.

## Remaining decisions and evidence

The direction is selected: real RC handling, community vehicle dimensions and operating envelope, community track scale/width, participant-level LiveKit integration and no pit stops. The F1TENTH Gym dimensions and limits above are the initial baseline; source revisions, additional physical calibration data, circuit revision and target deployment hardware remain to be selected during Phase A. Every departure from a community preset must be named and justified, including the provisional teleop speed cap and the nonlinear tire model. The proposed flat model deliberately limits mechanical fidelity; if realistic curb impacts, elevation and suspension are required for the first release, move to 3D dynamics before calling the driving model complete. Performance and remote feel remain unmeasured until the prototype and operator trials exist.
