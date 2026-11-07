"""Runs a Trossen MuJoCo scene as a LiveKit robot participant (participant level).

Trossen's arm driver, libtrossen_arm, talks to a proprietary arm controller,
so this simulation stands in for the robot program. The default generalized
contract uses shared JSON trajectory handlers and per-scene bindings. The
legacy mode serves the original binary protocol, control lease, and robot timing.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from livekit import rtc

from ..assets import SOURCES
from ..livekit_auth import credentials
from .protocol import encode_joint_state, parse_command
from .simulation import TrossenSimulation


LOG = logging.getLogger("sims.trossen")
ASSETS = SOURCES["trossen_arm_mujoco"].path / "assets"
COMMANDS = ("joint_control", "position", "velocity", "pose")
SUPPORTED = ("joint_position", "joint_velocity", "cartesian_position",
             "cartesian_velocity", "cartesian_pose", "home", "clear_faults")
# The arm labels of the robot presets in trossen-livekit/robot/src/config.cpp.
ARM_LABELS = {"arm": "Arm", "left": "Left arm", "right": "Right arm"}


def data_tracks(arm_id: str) -> dict[str, str]:
    prefix = f"trossen.teleop.arm.{arm_id}."
    return {
        "joints": prefix + "joints.v1",
        "joint_control": prefix + "joint-control.v1",
        "position": prefix + "position.v1",
        "velocity": prefix + "velocity.v1",
        "pose": prefix + "pose.v1",
    }


def manifest(sim: TrossenSimulation) -> dict:
    videos = [
        {"id": feed_id, "label": feed.label, "name": feed.name, "role": feed.role,
         "layout": "stereo_sbs" if feed.stereo else "mono",
         **({"arm_id": feed.arm_id} if feed.arm_id else {})}
        for feed_id, feed in sim.feeds.items()
    ]
    arms = []
    for arm_id in sim.arms:
        arm_videos = [feed_id for feed_id, feed in sim.feeds.items()
                      if feed.stereo and feed.arm_id in (arm_id, None)]
        arms.append({"id": arm_id, "label": ARM_LABELS[arm_id], "model": sim.arm_model,
                     "control_enabled": True, "joint_count": 7, "gripper_index": 6,
                     "supported_commands": list(SUPPORTED), "data_tracks": data_tracks(arm_id),
                     "video_track_ids": arm_videos})
    return {"schema_version": 1, "platform": sim.platform, "mobile_base": False,
            "lift_present": False, "arms": arms, "video_tracks": videos}


class SimParticipant:
    def __init__(self, sim: TrossenSimulation, url: str, token: str, fps: int, prefix: str):
        self.sim = sim
        self.url = url
        self.token = token
        self.fps = fps
        self.prefix = prefix
        self.room = rtc.Room()
        self.stopping = asyncio.Event()
        self.controller: str | None = None
        self.robot_identity = ""
        self.joint_tracks: dict[str, rtc.LocalDataTrack] = {}
        self.video_sources: dict[str, rtc.VideoSource] = {}
        self.video_tracks: dict[str, rtc.LocalVideoTrack] = {}
        self.command_tracks = {
            data_tracks(arm_id)[kind]: (arm_id, kind)
            for arm_id in sim.arms for kind in COMMANDS
        }
        self.readers: dict[str, tuple[str, asyncio.Task]] = {}
        self.last_sequences: dict[tuple[str, str], tuple[int, int]] = {}
        self.tasks: list[asyncio.Task] = []
        self._bind_events()

    def _bind_events(self) -> None:
        @self.room.on("data_track_published")
        def on_track(track: rtc.RemoteDataTrack) -> None:
            if track.info.name not in self.command_tracks:
                return
            task = asyncio.create_task(self._read_commands(track))
            self.readers[track.info.sid] = (track.publisher_identity, task)

        @self.room.on("data_track_unpublished")
        def on_unpublished(sid: str) -> None:
            reader = self.readers.pop(sid, None)
            if reader:
                reader[1].cancel()

        @self.room.on("participant_disconnected")
        def on_left(participant: rtc.RemoteParticipant) -> None:
            for sid, (identity, task) in list(self.readers.items()):
                if identity == participant.identity:
                    task.cancel()
                    self.readers.pop(sid, None)
            if self.controller == participant.identity:
                LOG.info("Controller %s left; platform control released", self.controller)
                self.controller = None
                self.sim.stop_all_motion()

        @self.room.on("disconnected")
        def on_disconnected(*_args) -> None:
            self.stopping.set()

    async def _read_commands(self, track: rtc.RemoteDataTrack) -> None:
        arm_id, track_kind = self.command_tracks[track.info.name]
        kind = "joint" if track_kind == "joint_control" else track_kind
        stream = track.subscribe(buffer_size=16)
        try:
            async for frame in stream:
                if track.publisher_identity != self.controller:
                    continue
                try:
                    command = parse_command(frame.payload, self.robot_identity, kind)
                except (ValueError, UnicodeDecodeError) as error:
                    LOG.warning("Ignored %s frame: %s", track.info.name, error)
                    continue
                key = (track.publisher_identity, track.info.name)
                last = self.last_sequences.get(key)
                if last and last[0] == command.session and command.sequence <= last[1]:
                    continue
                self.last_sequences[key] = (command.session, command.sequence)
                self.sim.apply(arm_id, command)
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    async def _publish(self) -> None:
        participant = self.room.local_participant
        for arm_id in self.sim.arms:
            self.joint_tracks[arm_id] = await participant.publish_data_track(
                name=data_tracks(arm_id)["joints"])
        for camera_id, feed in self.sim.feeds.items():
            source = rtc.VideoSource(*self.sim.render_size(camera_id))
            track = rtc.LocalVideoTrack.create_video_track(feed.name, source)
            await participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
            self.video_sources[camera_id] = source
            self.video_tracks[camera_id] = track
        response = json.dumps(manifest(self.sim), separators=(",", ":"))

        @participant.register_rpc_method("get_robot_config")
        def get_config(invocation: rtc.RpcInvocationData) -> str:
            if invocation.payload:
                raise ValueError("get_robot_config takes an empty payload")
            return response

        @participant.register_rpc_method("acquire_control")
        def acquire(invocation: rtc.RpcInvocationData) -> str:
            caller = invocation.caller_identity
            if not caller.startswith(self.prefix):
                return "false"
            if self.controller is None:
                self.controller = caller
                LOG.info("Controller %s acquired the whole platform", caller)
            return "true" if self.controller == caller else "false"

        @participant.register_rpc_method("release_control")
        def release(invocation: rtc.RpcInvocationData) -> str:
            if invocation.caller_identity == self.controller:
                LOG.info("Controller %s released the whole platform", self.controller)
                self.controller = None
                self.sim.stop_all_motion()
            return "true"

    async def _simulate(self) -> None:
        while not self.stopping.is_set():
            self.sim.step(0.01)
            await asyncio.sleep(0.01)

    async def _telemetry(self) -> None:
        sequence = {arm_id: 0 for arm_id in self.sim.arms}
        while not self.stopping.is_set():
            for arm_id, track in self.joint_tracks.items():
                positions, limits = self.sim.joint_state(arm_id)
                sequence[arm_id] += 1
                try:
                    track.try_push(rtc.DataTrackFrame(
                        payload=encode_joint_state(positions, limits, sequence[arm_id])))
                except rtc.PushFrameError as error:
                    LOG.debug("Joint telemetry dropped for %s: %s", arm_id, error)
            await asyncio.sleep(0.1)

    async def _video(self) -> None:
        # OpenGL rendering takes most of each frame. Run it on one worker thread
        # so the physics, commands, and telemetry keep running on this loop.
        loop = asyncio.get_running_loop()
        period = 1.0 / self.fps
        next_frame = loop.time()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-render") as executor:
            try:
                # Inside the try, so that a failure part way through still
                # closes the contexts that were created, on this thread.
                await loop.run_in_executor(executor, self.sim.open_renderers)
                while not self.stopping.is_set():
                    for camera_id, source in self.video_sources.items():
                        renderer = self.sim.prepare(camera_id)
                        pixels = await loop.run_in_executor(executor, renderer.render)
                        source.capture_frame(rtc.VideoFrame(
                            pixels.shape[1], pixels.shape[0], rtc.VideoBufferType.RGB24,
                            pixels.tobytes()))
                    next_frame = max(next_frame + period, loop.time())
                    await asyncio.sleep(next_frame - loop.time())
            finally:
                # The contexts belong to the render thread, so close them there.
                await asyncio.shield(loop.run_in_executor(executor, self.sim.close))

    async def run(self) -> None:
        try:
            await self.room.connect(self.url, self.token, rtc.RoomOptions(auto_subscribe=False))
            self.robot_identity = self.room.local_participant.identity
            await self._publish()
            LOG.info("Simulation online as %s with %d arms and %d cameras", self.robot_identity,
                     len(self.sim.arms), len(self.video_sources))
            self.tasks = [asyncio.create_task(loop()) for loop in
                          (self._simulate, self._telemetry, self._video)]
            for task in self.tasks:
                task.add_done_callback(self._background_done)
            await self.stopping.wait()
            for task in self.tasks:
                if task.done() and not task.cancelled():
                    task.result()
        finally:
            for task in self.tasks:
                task.cancel()
            for _, task in self.readers.values():
                task.cancel()
            await asyncio.gather(*self.tasks, *(task for _, task in self.readers.values()),
                                 return_exceptions=True)
            self.sim.close()
            await self.room.disconnect()

    def _background_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            LOG.error("Simulation task failed", exc_info=task.exception())
            self.stopping.set()


def add_arguments(parser, sim) -> None:
    from ..contract.config import CONFIGS

    parser.add_argument("--protocol", choices=("contract", "legacy"), default="contract",
                        help="generalized JSON contract (default) or original binary controller protocol")
    parser.add_argument("--bindings", type=Path, default=CONFIGS / f"{sim.name}.bindings.json",
                        help="local MuJoCo bindings and reference to the public discovery document")
    parser.add_argument("--no-video", action="store_true", help="run contract physics and data without rendering video")
    parser.add_argument("--assets-dir", type=Path, default=Path(os.environ.get("TROSSEN_MUJOCO_ASSETS") or ASSETS),
                        help="Trossen MuJoCo assets directory (default: %(default)s)")
    parser.add_argument("--width", type=int, default=320, help="width of each wrist camera eye")
    parser.add_argument("--height", type=int, default=240, help="height of each camera eye")
    parser.add_argument("--scene-width", type=int, default=960)
    parser.add_argument("--scene-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=12)


async def _run(args, sim_spec) -> int:
    if not args.assets_dir.is_dir():
        raise SystemExit(f"{args.assets_dir} does not exist. Run: python -m sims fetch-assets {sim_spec.name}")
    sizes = (args.width, args.height, args.scene_width, args.scene_height)
    if any(size < 2 or size % 2 for size in sizes) or args.fps < 1:
        raise SystemExit("width and height must be positive even numbers, and fps must be positive")
    config = None
    model_path = None
    if args.protocol == "contract":
        from ..contract.config import Configuration

        config = Configuration.load(args.bindings, args.fps)
        if config.bindings["scene"] != sim_spec.options["scene"]:
            raise ValueError("bindings scene does not match the selected simulation")
        model_path = args.assets_dir / config.bindings["model"]
    elif args.no_video:
        raise ValueError("--no-video is available with --protocol contract")
    sim = TrossenSimulation(args.assets_dir, sim_spec.options["scene"], args.width, args.height,
                            args.scene_width, args.scene_height, model_path=model_path)
    try:
        return await _serve(args, sim_spec, sim, config)
    finally:
        sim.close()


async def _serve(args, sim_spec, sim, config) -> int:
    backend = None
    if config is not None:
        from ..contract.mujoco_backend import MujocoBackend

        backend = MujocoBackend(sim, config)
        for keys in config.feedback_tracks.values():
            backend.feedback(keys)  # Validate shared joint-name selection before connecting.
    if args.check:
        print(json.dumps(config.discovery if config else manifest(sim), indent=2))
        return 0
    creds = await credentials(args, sim_spec.name)
    if creds.identity:
        LOG.info("Joining room %s as %s", creds.room, creds.identity)
    else:
        LOG.info("Joining with the token from --token or LIVEKIT_TOKEN")
    if config is not None:
        from ..contract.gateway import ContractGateway

        bridge = ContractGateway(backend, config, creds.url, creds.token, args.fps, video=not args.no_video)
    else:
        bridge = SimParticipant(sim, creds.url, creds.token, args.fps, args.controller_prefix)
    loop = asyncio.get_running_loop()
    for interrupt in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(interrupt, bridge.stopping.set)
        except NotImplementedError:
            pass
    await bridge.run()
    return 0


def run(args, sim_spec) -> int:
    try:
        return asyncio.run(_run(args, sim_spec))
    finally:
        gc.collect()
