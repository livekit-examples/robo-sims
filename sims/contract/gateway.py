"""Shared LiveKit contract gateway. No robot/model-specific names or command encoders."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from livekit import rtc

from .config import Configuration, JOINT_STATE
from .rtc_schema import SchemaRoom, matches_json, publish_json
from .schemas import JOINT_STATE_SCHEMA
from .trajectory import Trajectory

LOG = logging.getLogger("sims.contract")
ROLE, VERSION, OWNER = "lk.portal.role", "lk.portal.version", "lk.portal.active_operator"


class ContractGateway:
    def __init__(self, backend, config: Configuration, url: str, token: str, fps: int,
                 *, video: bool = True):
        self.backend, self.config = backend, config
        self.url, self.token, self.fps = url, token, fps
        self.video = video
        self.room = SchemaRoom()
        self.stopping = asyncio.Event()
        self.ready = asyncio.Event()
        self.active_operator = ""
        self.armed = False
        self.ownership_lock = asyncio.Lock()
        self.readers: dict[str, tuple[object, asyncio.Task]] = {}
        self.sources: dict[tuple[str, str], str] = {}
        self.tasks: list[asyncio.Task] = []
        self.feedback_tracks = {}
        self.video_sources = {}
        self._bind_events()

    def _compatible(self, identity, roles=("operator",)):
        peer = self.room.remote_participants.get(identity)
        return (peer is not None and peer.attributes.get(ROLE) in roles
                and peer.attributes.get(VERSION) == "3")

    def _accepts(self, identity):
        return self.armed and identity == self.active_operator and self._compatible(identity)

    def _disarm(self):
        self.armed = False
        self.backend.hold()

    def _bind_events(self):
        @self.room.on("data_track_published")
        def published(track):
            endpoint = self.config.command_tracks.get(track.info.name)
            if endpoint is None or track.info.sid in self.readers:
                return
            if not matches_json(track, endpoint[1]["command"]["msg_type"]):
                LOG.warning("Ignored command publication %s: expected JointTrajectory JSON metadata", track.info.name)
                return
            self.sources[track.publisher_identity, track.info.name] = track.info.sid
            task = asyncio.create_task(self._read_commands(track, endpoint[0]))
            self.readers[track.info.sid] = (track, task)
            task.add_done_callback(self._reader_done)

        @self.room.on("data_track_unpublished")
        def unpublished(sid):
            self._stop_reader(sid)

        @self.room.on("participant_disconnected")
        def departed(peer):
            if peer.identity == self.active_operator:
                # Portal ownership persists; motion and connection arming do not.
                self._disarm()
            for sid, (track, _) in list(self.readers.items()):
                if track.publisher_identity == peer.identity:
                    self._stop_reader(sid)

        @self.room.on("participant_attributes_changed")
        def changed(_attributes, peer):
            if peer.identity == self.active_operator and not self._compatible(peer.identity):
                self._disarm()

        @self.room.on("reconnecting")
        def reconnecting():
            self._disarm()
            for sid in list(self.readers):
                self._stop_reader(sid)

        @self.room.on("disconnected")
        def disconnected(*_args):
            self._disarm()
            self.stopping.set()

    def _hold_source(self, track):
        key = (track.publisher_identity, track.info.name)
        if self.sources.get(key) == track.info.sid:
            self.sources.pop(key, None)
            if track.publisher_identity == self.active_operator:
                self.backend.hold(self.config.command_tracks[track.info.name][0])

    def _stop_reader(self, sid):
        reader = self.readers.get(sid)
        if reader:
            self._hold_source(reader[0])
            reader[1].cancel()

    def _reader_done(self, task):
        if not task.cancelled() and task.exception():
            LOG.error("Command subscription failed: %s", task.exception())

    async def _read_commands(self, track, interface_id):
        stream = None
        try:
            stream = track.subscribe(buffer_size=1)
            async for frame in stream:
                if not self._accepts(track.publisher_identity) or self.sources.get(
                    (track.publisher_identity, track.info.name)
                ) != track.info.sid:
                    continue
                try:
                    command = Trajectory.parse(frame.payload,
                        self.config.discovery["control_interfaces"][interface_id]["joint_names"])
                    self.backend.submit(interface_id, command)
                except (ValueError, UnicodeDecodeError, KeyError, TypeError, OverflowError) as error:
                    LOG.warning("Rejected %s command: %s", track.info.name, error)
        finally:
            self._hold_source(track)
            self.readers.pop(track.info.sid, None)
            if stream is not None:
                await stream.aclose()

    async def _set_owner(self, invocation):
        caller = invocation.caller_identity
        # Participant attributes may arrive just after the first RPC.
        for _ in range(30):
            if self._compatible(caller, ("operator", "observer")):
                break
            await asyncio.sleep(0.05)
        else:
            raise rtc.RpcError(2003, "Caller must be a Portal v3 operator or observer")
        target = invocation.payload
        if not isinstance(target, str) or len(target.encode("utf-8")) > 255:
            raise rtc.RpcError(2003, "Invalid operator identity")
        async with self.ownership_lock:
            self._disarm()
            # Keep receive gate closed until the attribute write succeeds.
            try:
                await self.room.local_participant.set_attributes({OWNER: target})
            except Exception as error:
                raise rtc.RpcError(2002, "Failed to update active operator") from error
            self.active_operator = target
            self.armed = bool(target and self._compatible(target))
        return ""

    async def _publish(self):
        participant = self.room.local_participant
        await participant.set_attributes({ROLE: "robot", VERSION: "3", OWNER: ""})
        response = json.dumps(self.config.discovery, allow_nan=False, separators=(",", ":"))

        @participant.register_rpc_method("get_robot_config")
        def get_config(invocation):
            if invocation.payload != "":
                raise rtc.RpcError(2000, "get_robot_config takes an empty payload")
            return response

        participant.register_rpc_method("portal.set_active_operator", self._set_owner)
        for name in self.config.feedback_tracks:
            self.feedback_tracks[name] = await publish_json(participant, name, JOINT_STATE, JOINT_STATE_SCHEMA)
        if self.video:
            for key, feed in self.backend.feeds.items():
                source = rtc.VideoSource(*self.backend.render_size(key))
                track = rtc.LocalVideoTrack.create_video_track(feed.name, source)
                await participant.publish_track(track, rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_CAMERA,
                    video_encoding=rtc.VideoEncoding(max_framerate=self.fps, max_bitrate=4_000_000)))
                self.video_sources[key] = source

    async def _simulate(self):
        while True:
            self.backend.step(0.01)
            await asyncio.sleep(0.01)

    async def _telemetry(self):
        while True:
            for name, track in self.feedback_tracks.items():
                value = self.backend.feedback(self.config.feedback_tracks[name])
                try:
                    track.try_push(rtc.DataTrackFrame(payload=json.dumps(
                        value, allow_nan=False, separators=(",", ":")).encode()))
                except rtc.PushFrameError as error:
                    LOG.debug("Feedback dropped: %s", error)
            await asyncio.sleep(0.05)

    async def _video(self):
        loop = asyncio.get_running_loop()
        period = 1 / self.fps
        next_frame = loop.time()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="contract-render") as executor:
            try:
                await loop.run_in_executor(executor, self.backend.open_renderers)
                while True:
                    for key, source in self.video_sources.items():
                        renderer = self.backend.prepare(key)
                        pixels = await loop.run_in_executor(executor, renderer.render)
                        source.capture_frame(rtc.VideoFrame(pixels.shape[1], pixels.shape[0],
                            rtc.VideoBufferType.RGB24, pixels.tobytes()))
                    next_frame = max(next_frame + period, loop.time())
                    await asyncio.sleep(next_frame - loop.time())
            finally:
                await asyncio.shield(loop.run_in_executor(executor, self.backend.close))

    def _background_done(self, task):
        if not task.cancelled():
            self.stopping.set()
            if task.exception():
                LOG.error("Gateway task failed: %s", task.exception())

    async def run(self):
        try:
            await self.room.connect(self.url, self.token, rtc.RoomOptions(auto_subscribe=False))
            await self._publish()
            LOG.info("Contract gateway online as %s", self.room.local_participant.identity)
            loops = [self._simulate, self._telemetry] + ([self._video] if self.video_sources else [])
            self.tasks = [asyncio.create_task(loop()) for loop in loops]
            for task in self.tasks:
                task.add_done_callback(self._background_done)
            self.ready.set()
            await self.stopping.wait()
            for task in self.tasks:
                if task.done() and not task.cancelled():
                    task.result()
        finally:
            self._disarm()
            for task in self.tasks:
                task.cancel()
            readers = [task for _, task in self.readers.values()]
            for task in readers:
                task.cancel()
            await asyncio.gather(*self.tasks, *readers, return_exceptions=True)
            self.backend.close()
            await self.room.disconnect()
