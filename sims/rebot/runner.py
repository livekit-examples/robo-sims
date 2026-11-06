"""Runs the simulated reBot B601-DM with the real robot program (hardware level).

Starts two processes:

1. the emulated hardware (`sims.rebot.bus`): MuJoCo behind an emulated Damiao
   bus, on a pseudo-terminal that stands in for the arm's /dev/ttyACM0, and a
   rendered scene camera served as MJPEG on localhost, standing in for the
   arm's UVC scene camera;
2. the teleop robot program, `trossen_livekit_robot --platform rebot_b601_dm`,
   with its arm connection pointed at that pseudo-terminal, its scene camera
   at that stream, and a LiveKit token made here.

The robot program is the one that runs on the arm, unchanged, so the driver,
safety limits and LiveKit code under test are the real ones. On Ctrl-C the
robot program parks the arm on the still-running bus, then the bus stops.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from ..livekit_auth import credentials
from ..paths import ROBOT_PROGRAM, RUNTIME_DIR, SIMULATION_DIR
from .bus import DEFAULT_MODEL

LOG = logging.getLogger("sims.rebot")
BUS_START_TIMEOUT = 15.0  # s, MuJoCo loads the meshes first
ROBOT_STOP_TIMEOUT = 20.0  # s, the robot parks the arm before it exits
BUS_STOP_TIMEOUT = 5.0


def add_arguments(parser, sim) -> None:
    parser.add_argument("--robot-program", type=Path, default=ROBOT_PROGRAM,
                        help="teleop robot program, built with motorbridge (default: %(default)s; or TELEOP_ROBOT_BIN)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Seeed MJCF model (default: %(default)s)")
    parser.add_argument("--serial-link", type=Path, default=RUNTIME_DIR / f"{sim.name}-ttyACM0",
                        help="where to put the bus's serial device (default: %(default)s)")
    parser.add_argument("--bus-port", type=int, default=0,
                        help="also serve the bus on this localhost TCP port, for tools/serial_bridge.py (default: off)")
    parser.add_argument("--status-file", type=Path, default=None, help="write the joint state JSON here at 10 Hz")
    parser.add_argument("--no-scene-camera", dest="scene_camera", action="store_false",
                        help="don't render the scene camera; the robot program then runs without one")
    parser.add_argument("--camera-port", type=int, default=0,
                        help="localhost port of the scene camera stream (default: a free one)")
    parser.add_argument("--camera-size", default="640x480", metavar="WxH",
                        help="rendered scene camera size (default: %(default)s, the robot program's output size)")
    parser.add_argument("--camera-fps", type=int, default=30, help="scene camera frame rate (default: %(default)s)")
    parser.add_argument("--robot-arg", action="append", default=[], metavar="ARG",
                        help="extra robot program argument; repeat for each, e.g. --robot-arg=--camera-device "
                             "--robot-arg=scene=/dev/video0")


def _die_with_parent() -> None:
    """Child setup: get SIGTERM if this launcher dies, so no simulator is left behind."""
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _scene_camera_url(args) -> str | None:
    """The stream the robot program opens as its scene camera, or None when it uses none or its own."""
    if not args.scene_camera or "--camera-device" in args.robot_arg:
        return None
    return f"http://127.0.0.1:{args.camera_port}/scene.mjpg"


def _start_bus(args) -> subprocess.Popen:
    command = [sys.executable, "-m", "sims.rebot.bus", "--model", str(args.model),
               "--serial-link", str(args.serial_link), "--tcp-port", str(args.bus_port)]
    if _scene_camera_url(args):
        width, height = args.camera_width, args.camera_height
        command += ["--camera-port", str(args.camera_port), "--camera-width", str(width),
                    "--camera-height", str(height), "--camera-fps", str(args.camera_fps)]
    if args.status_file:
        command += ["--status-file", str(args.status_file)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SIMULATION_DIR), os.environ.get("PYTHONPATH")]))}
    if args.serial_link.is_symlink():
        args.serial_link.unlink()
    # Its own session, so a Ctrl-C in the terminal reaches only the robot
    # program, which then parks the arm on the bus before the bus stops.
    bus = subprocess.Popen(command, env=env, start_new_session=True, preexec_fn=_die_with_parent)
    deadline = time.monotonic() + BUS_START_TIMEOUT
    while not args.serial_link.exists():
        if bus.poll() is not None:
            raise SystemExit(f"the emulated bus exited with status {bus.returncode}")
        if time.monotonic() > deadline:
            bus.terminate()
            raise SystemExit(f"the emulated bus did not create {args.serial_link}")
        time.sleep(0.05)
    return bus


def _robot_command(args, sim) -> list[str]:
    command = [str(args.robot_program), "--platform", sim.options["platform"],
               "--arm-connection", str(args.serial_link), "--controller-prefix", args.controller_prefix]
    url = _scene_camera_url(args)
    if url:
        command += ["--camera-device", f"scene={url}"]
    elif "--camera-device" not in args.robot_arg:
        command += ["--no-camera", "scene"]
    return command + args.robot_arg


def _stop(process: subprocess.Popen, sig: int, timeout: float, name: str) -> None:
    if process.poll() is not None:
        return
    process.send_signal(sig)
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        LOG.warning("%s did not stop in %.0f s; killing it", name, timeout)
        process.kill()
        process.wait()


def run(args, sim) -> int:
    try:
        args.camera_width, args.camera_height = (int(n) for n in args.camera_size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--camera-size takes WxH, for example 640x480, not {args.camera_size!r}") from None
    if args.camera_width < 2 or args.camera_height < 2 or args.camera_width % 2 or args.camera_height % 2:
        raise SystemExit("--camera-size needs an even width and height")
    if args.camera_fps < 1:
        raise SystemExit("--camera-fps must be positive")
    if _scene_camera_url(args) and not args.camera_port:
        args.camera_port = _free_port()
    if not args.model.is_file():
        raise SystemExit(f"{args.model} does not exist. Run: python -m sims fetch-assets {sim.name}")
    program_ready = os.access(args.robot_program, os.X_OK)
    if args.check:
        from .arm import MOTORS, SimulatedArm

        arm = SimulatedArm(args.model)
        print(f"model: {args.model} (physics step {arm.dt * 1000:.1f} ms)")
        for spec in MOTORS:
            print(f"  {spec.name:<8} motor 0x{spec.motor_id:02x}, feedback 0x{spec.feedback_id:02x}, "
                  f"Damiao {spec.model:<5} -> {spec.joint}")
        print(f"robot program: {args.robot_program} ({'found' if program_ready else 'MISSING'})")
        print("robot command: " + " ".join(_robot_command(args, sim)))
        return 0 if program_ready else 1
    if not program_ready:
        raise SystemExit(f"{args.robot_program} is not an executable. Build the teleop robot program with "
                         f"motorbridge (see trossen-livekit/docs/development.md), or pass --robot-program.")

    creds = asyncio.run(credentials(args, sim.name))
    if creds.identity:
        LOG.info("Joining room %s as %s", creds.room, creds.identity)
    else:
        LOG.info("Joining with the token from --token or LIVEKIT_TOKEN")

    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    bus = robot = None
    try:
        bus = _start_bus(args)
        LOG.info("Emulated bus on %s", args.serial_link)
        # The token goes in the environment, not on the command line, where
        # other users could read it.
        robot_env = {**os.environ, "LIVEKIT_URL": creds.url, "LIVEKIT_TOKEN": creds.token}
        robot = subprocess.Popen(_robot_command(args, sim), env=robot_env, preexec_fn=_die_with_parent)
        while not stopping and robot.poll() is None and bus.poll() is None:
            time.sleep(0.1)
        if bus.poll() is not None and robot.poll() is None:
            LOG.error("The emulated bus exited with status %s; stopping the robot program", bus.returncode)
    finally:
        if robot is not None:
            _stop(robot, signal.SIGINT, ROBOT_STOP_TIMEOUT, "the robot program")
        if bus is not None:
            _stop(bus, signal.SIGTERM, BUS_STOP_TIMEOUT, "the emulated bus")
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if robot is not None and robot.returncode not in (0, -signal.SIGINT):
        return robot.returncode if robot.returncode > 0 else 1
    return 0 if bus is None or bus.returncode in (0, -signal.SIGTERM) else 1
