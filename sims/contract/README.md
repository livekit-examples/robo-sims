# Generalized contract gateway

Trossen simulations default to the discovery contract in
`teleop/generalization/contract.md`. One gateway and one trajectory handler serve
both the single-arm and dual-arm scenes. No ROS process is required.

```sh
python -m sims run trossen-wxai --identity simulation-wxai
python -m sims run trossen-workbench --identity simulation-workbench
python -m sims run trossen-workbench --check
```

Use `--protocol legacy` with the original Trossen controller and its binary probe.
Use `--no-video` for contract data/physics without OpenGL. Credentials still come
from the usual `.env.local`/environment. Supplied tokens must allow the robot to
update its own participant attributes. `--identity` must match a supplied token;
otherwise it sets the identity used for token creation. Stable identities allow
the generalized UI to restore saved bindings. Run each identity only once per room.

## Configuration

`configs/*.discovery.json` are public robot descriptions, derived from the
single-arm and two-arm generalization examples. They advertise only the implemented
`joints.trajectory.v1` and `gripper.position.v1` operations. They contain no model paths, device addresses,
camera catalog, or telemetry catalog. `get_robot_config` returns this document
with `default_video_rate_hz` set to the selected `--fps` (default 12).
The gateway adds a `simulation_control` extension identifying
`mujoco.position.v2` and its per-operation position/speed policy.
The manual `/workspace` connection uses this explicit implementation declaration
to install the simulation trajectory adapter; robot labels alone never enable it.
Restart both the simulator and UI after upgrading from a version without this extension.

`configs/*.bindings.json` are local implementation configuration. They reference
the discovery file relative to the binding file, choose the scene/model, and map
each public interface/joint to a backend arm, MuJoCo joint, position actuator,
base body, end-effector site, and maximum joint speed. The model path is relative
to `--assets-dir`. Select a different configuration with `--bindings PATH`.

Endpoints are read only from discovery. Interface IDs and joint names must agree
between the two documents. Startup resolves the bindings against the compiled
model and rejects missing/cross-arm joints, repeated actuators, unsupported
profiles, and excluded mandatory feedback. Joint position bounds come from the
model. The checked-in 0.5 rad/s velocity bound matches the existing simulation's
arm position-control cap; this is a simulation implementation, not hardware
validation of a new vendor controller.

Scene construction, camera optics, and objects remain in `sims/trossen/simulation.py`.
The backend uses its compiled model and rendering but owns contract motion;
it does not translate messages into legacy binary commands. Model families with
different actuators need a backend implementing the same gateway methods.

## Data and ownership

The gateway advertises `lk.portal.role=robot`, `lk.portal.version=3`, and the
robot-owned `lk.portal.active_operator` attribute. Discovery needs no ownership.
`portal.set_active_operator` takes the raw identity or an empty string and returns
an empty string. Compatible operators/observers may request handoff; the last
accepted request wins. Commands are accepted only from the selected, connected
Portal v3 operator. Identity prefixes are not authorization in contract mode.

Ownership change/release, owner disconnect, role/version loss, command-track
closure, gateway reconnect, and shutdown cancel affected trajectories and hold
measured joint positions. The owner pointer persists across operator disconnects,
but motion does not restart. A fresh ownership RPC is required after that operator
reconnects or the gateway reconnects. The browser must still enforce explicit
arming and restrict command subscriptions to the selected robot, as required by
the contract. Robot restart initializes ownership to empty.

Each advertised feedback track publishes `sensor_msgs/msg/JointState` JSON at
20 Hz, with measured joint positions/velocities and an empty `effort` array.
Stamps use simulation time. Feedback can be per-arm or shared: point both
operations at one feedback name to publish the union of their named joints.
Cameras are regular LiveKit video publications, discovered dynamically; stereo
interpretation belongs to client settings.

Both feedback and command tracks require explicit JSON frame encoding and JSON
Schema IDs. Definitions are registered through LiveKit's schema API. The pinned
`livekit==1.2.1` wheel has native schema support without Python convenience
wrappers. `rtc_schema.py` isolates that version-checked bridge, including preservation
of received publication metadata. Missing/incompatible metadata is rejected.
Self-hosted LiveKit must enable `enable_participant_data_blob: true` for schema
storage. Errors stop startup rather than publishing unlabeled JSON.

## Executable trajectory semantics

This is the simulator's documented implementation of the draft profile, to be
matched by the client adapter before enabling its controls:

- Command: `trajectory_msgs/msg/JointTrajectory` JSON on the exact advertised
  endpoint, published with JSON Schema metadata and JSON frame encoding.
- `joint_names` is a non-empty, unique subset of this interface's joint names.
  Array order is interpreted by name. Revolute positions use radians.
- Only position waypoints are supported. `velocities`, `accelerations`, and
  `effort` must be absent or empty. Acceleration/jerk limiting is not implemented.
- `header` is optional. If present, its stamp must be zero and its frame empty.
  Trajectories start on receipt; scheduled execution is not supported.
- `time_from_start` has integer `sec` and `nanosec`, strictly increasing from a
  positive first time to at most 30 seconds. At most 256 points/64 KiB are accepted.
- Targets interpolate linearly from the current commanded position through each
  waypoint. Every segment must respect model position bounds and configured
  speeds. Invalid commands leave the previous trajectory intact.
- A new trajectory replaces the prior trajectory for that interface; joints not
  selected by the new trajectory keep their last commanded target. Other
  interfaces continue independently. At completion the final target is held.
- An empty `points` array explicitly cancels the interface trajectory and holds
  measured position. An all-zero position waypoint is an ordinary move to zero.
- Finite trajectories run to completion without a streaming heartbeat. The
  ownership/track lifecycle above cancels them. Client mode changes/releases
  must send the explicit cancellation, or release ownership.

The six public arm joints exclude the gripper. Each arm also advertises
`operations.gripper_position`, profile `gripper.position.v1`, on its independent
`gripper` channel. Its explicit joint selector uses a single carriage displacement
in meters (0–0.044 m in the pinned models), with a 0.025 m/s target speed cap.
The two slide joints have verified equal-displacement coupling and one position
actuator; this coordinate is not the full jaw gap. A separate gripper trajectory
endpoint accepts position-only single-joint trajectories and empty-points cancellation.
Arm and gripper trajectories run concurrently. Shared feedback contains their named
joint union, while the arm's interface joint list stays at six.

Gripper Open/Close targets come from the model-derived policy. Cancellation holds
measured position; it never opens automatically and does not guarantee preserved
gripping force. Actuator force limits remain those of the model. Contact force,
stall, grasp success, and effort commands are not advertised. Manual discovery and
managed sessions share policy generation, units, limits, and command semantics.

Joint velocity, Cartesian/VR, home, and clear-fault actions still require separate
executable profiles. Legacy mode retains its existing controls.

## Validation

```sh
python -m unittest discover -s tests -v
livekit-server --dev --config tests/livekit-dev.yaml
# In another terminal, against that local server only:
ROBO_SIMS_LIVEKIT_TEST=1 python -m unittest discover -s tests -p test_contract_livekit.py -v
# Also render and receive all configured camera feeds:
ROBO_SIMS_LIVEKIT_TEST=1 ROBO_SIMS_VIDEO_TEST=1 python -m unittest discover -s tests -p test_contract_livekit.py -v
```

The pure tests cover config/profile rejection. Model tests require the dependencies
and fetched Trossen assets. The opt-in native test runs both scenes in isolated
development rooms and checks discovery, JSON metadata/feedback, unauthorized-command
rejection, motion, cancellation, handoff, disconnect/reconnect arming, and version
loss. The video variant checks received decoded frames from every camera.

The reBot emulated bus remains independent of this gateway. Its real C++ robot
program still serves its existing protocol. Generalized reBot discovery/control
must be added to that program's gateway to keep testing the real driver path.

The browser refreshes position targets at 200 ms only after explicit operator input
to tolerate lossy data-track delivery and initial subscription timing. The backend
still completes each accepted finite trajectory without a refresh. Mounting or
arming sends no target; disarm/reconnect clears the browser's target cache. Target
values are requests, not acknowledged action results.
