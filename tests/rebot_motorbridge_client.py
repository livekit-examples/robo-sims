"""End-to-end check: drive the running simulator with motorbridge, as on the real arm.

    python -m sims.rebot.bus --status-file /tmp/rebot-status.json &   # prints its serial device
    python tests/rebot_motorbridge_client.py SERIAL_DEVICE /tmp/rebot-status.json

Defaults to /dev/ttyACM0 and /tmp/rebot-sim-status.json, the paths in the
rebot-bus container (docker/rebot-bus). Uses the same motor
IDs, models and calls as Seeed's driver, so a pass means that driver code works
against the sim unchanged. Exits non-zero on failure.
"""

import json
import sys
import time

from motorbridge import Controller, Mode

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
STATUS = sys.argv[2] if len(sys.argv) > 2 else "/tmp/rebot-sim-status.json"
MOTORS = [
    ("joint1", 0x01, 0x11, "4340P"),
    ("joint2", 0x02, 0x12, "4340P"),
    ("joint3", 0x03, 0x13, "4340P"),
    ("joint4", 0x04, 0x14, "4310"),
    ("joint5", 0x05, 0x15, "4310"),
    ("joint6", 0x06, 0x16, "4310"),
    ("gripper", 0x07, 0x17, "4310"),
]
failures = []


def check(name, actual, expected, tolerance):
    ok = abs(actual - expected) <= tolerance
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {actual:+.4f} (expected {expected:+.4f} +- {tolerance})")
    if not ok:
        failures.append(name)


def states(controller, motors):
    for motor in motors.values():
        motor.request_feedback()
    deadline = time.monotonic() + 0.05
    while time.monotonic() < deadline:
        controller.poll_feedback_once()
    return {name: motor.get_state() for name, motor in motors.items()}


def check_finger(name, expected):
    """Checks the simulated finger travel, when the status file is reachable."""
    try:
        with open(STATUS) as status:
            travel = json.load(status)["joints"]["gripper"]
    except OSError:
        print(f"skip {name}: no status file at {STATUS}")
        return
    check(name, travel, expected, 0.0005)


controller = Controller.from_dm_serial(PORT, 921600)
motors = {name: controller.add_damiao_motor(mid, fid, model) for name, mid, fid, model in MOTORS}

ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# Seeed's MIT gains (arm.yaml). Without gravity feedforward the arm droops a
# little under load, as the real one does.
GAINS = {name: (120.0, 8.0) if name in ("joint1", "joint2", "joint3") else (18.0, 2.0) for name in ARM}
DROOP = 0.15  # rad; the 4310 wrist at kp=18 sags about 0.11 rad.

print("-- startup: set MIT mode while disabled, then enable and stream a hold")
# ensure_mode() on an enabled motor sends a zero-gain MIT frame first, so the
# arm goes limp for the switch. Switching before enabling avoids that.
for name in ARM:
    motors[name].ensure_mode(Mode.MIT, 1000)
motors["gripper"].ensure_mode(Mode.POS_VEL, 1000)
rest = states(controller, motors)
for name, state in rest.items():
    check(f"{name} disabled at power-up", state.status_code, 0, 0)


def stream_mit(goal_of, seconds):
    """Streams MIT commands at 500 Hz, as a teleop loop does."""
    start = time.monotonic()
    while (now := time.monotonic()) < start + seconds:
        fraction = min(1.0, (now - start) / seconds)
        for name in ARM:
            kp, kd = GAINS[name]
            motors[name].send_mit(goal_of(name, fraction), 0.0, kp, kd, 0.0)
        controller.poll_feedback_once()
        time.sleep(0.002)


hold = {name: rest[name].pos for name in ARM}
for name in ARM:
    kp, kd = GAINS[name]
    motors[name].send_mit(hold[name], 0.0, kp, kd, 0.0)  # Seed the hold before enabling.
for motor in motors.values():
    motor.enable()
stream_mit(lambda name, _: hold[name], 1.0)
held = states(controller, motors)
for name in ARM:
    check(f"{name} enabled", held[name].status_code, 1, 0)
    check(f"{name} MIT hold", held[name].pos, hold[name], DROOP)

print("-- MIT: stream an interpolated move")
targets = {"joint1": 0.6, "joint2": -0.8, "joint3": -0.9, "joint4": 0.3, "joint5": -0.4, "joint6": 0.5}
stream_mit(lambda name, f: hold[name] + f * (targets[name] - hold[name]), 2.0)
stream_mit(lambda name, _: targets[name], 0.5)
moved = states(controller, motors)
for name in ARM:
    check(f"{name} MIT position", moved[name].pos, targets[name], DROOP)

print("-- POS_VEL: Seeed's default arm mode; its integrator removes the droop")
for name in ARM:
    motors[name].ensure_mode(Mode.POS_VEL, 1000)
    motors[name].send_pos_vel(targets[name], 2.0)
targets = {"joint1": -0.4, "joint2": -1.2, "joint3": -1.1, "joint4": -0.3, "joint5": 0.4, "joint6": -0.5}
for name, target in targets.items():
    motors[name].send_pos_vel(target, 2.0)
time.sleep(2.5)
final = states(controller, motors)
for name, target in targets.items():
    check(f"{name} POS_VEL position", final[name].pos, target, 0.02)
hold = {name: final[name].pos for name in ARM}

print("-- gripper: 0 rad closed, -5 rad fully open = 0.0285 m per finger")
motors["gripper"].send_pos_vel(-5.0, 3.0)
time.sleep(2.5)
check("gripper motor angle", states(controller, motors)["gripper"].pos, -5.0, 0.05)
check_finger("finger travel (m)", 0.0285)
motors["gripper"].send_pos_vel(-2.5, 3.0)
time.sleep(1.5)
check("half-open gripper motor angle", states(controller, motors)["gripper"].pos, -2.5, 0.05)
check_finger("half-open finger travel (m)", 0.01425)

print("-- disable: no brakes, so the arm sags")
for name in ("joint2", "joint3"):
    motors[name].disable()
time.sleep(1.0)
sagged = states(controller, motors)
check("joint2 status after disable", sagged["joint2"].status_code, 0, 0)
moved = abs(sagged["joint2"].pos - hold["joint2"]) + abs(sagged["joint3"].pos - hold["joint3"])
check("shoulder+elbow moved under gravity (rad, > 0.1)", min(moved, 0.1), 0.1, 0.0)

controller.disable_all()
controller.close()
print("MOTORBRIDGE CLIENT TEST", "FAILED: " + ", ".join(failures) if failures else "PASSED")
sys.exit(1 if failures else 0)
