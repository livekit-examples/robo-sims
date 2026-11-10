"""Ackermann participant using the shared Portal ownership and video lifecycle."""
import asyncio
import json
import logging

from livekit import rtc

from ..contract.gateway import ContractGateway, OWNER
from ..contract.battery import PERIOD
from ..contract.rtc_schema import define_json_schema, publish_json
from ..contract.schemas import schema_for
from .backend import ACKERMANN_SCHEMA, COMMAND, ODOM_TRACK, STATE_TRACK, STATE

LOG = logging.getLogger("sims.f1tenth")


class DriveGateway(ContractGateway):
    def __init__(self, backend, url, token, fps=60, **kwargs):
        self.invalid_sources = set()
        self.neutral_sources = set()
        self.overloads = 0
        super().__init__(backend, backend.config, url, token, fps, **kwargs)

    def _disarm(self):
        # A handoff, timeout or reconnect invalidates every extant publication.
        # A newly armed browser must publish a fresh track, then send neutral.
        self.invalid_sources.update(self.readers)
        self.neutral_sources.clear()
        super()._disarm()

    def _receive_command(self, track, interface_id, operation, payload):
        if track.info.sid in self.invalid_sources:
            return
        command = self.backend.parse(payload)
        if track.info.sid not in self.neutral_sources:
            if command.speed != 0 or command.steering != 0:
                raise ValueError("a fresh drive publication must send neutral before motion")
            self.neutral_sources.add(track.info.sid)
        self.backend.submit(payload)

    def discovery(self):
        return {**self.config.discovery, "simulation_control": {
            "implementation": "rc.ackermann.v1", "policy": self.backend.policy()}}

    async def _publish_feedback(self, participant):
        # Browser publishers can reference this schema ID without defining it.
        await define_json_schema(participant, COMMAND, ACKERMANN_SCHEMA)
        for name, msg_type, sample in ((STATE_TRACK, STATE, self.backend.state()),
                                     (ODOM_TRACK, "nav_msgs/msg/Odometry", self.backend.odometry())):
            schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema_for(sample)}
            self.feedback_tracks[name] = await publish_json(participant, name, msg_type, schema)
        for name, (msg_type, schema) in self.backend.instrument_tracks().items():
            self.feedback_tracks[name] = await publish_json(participant, name, msg_type, schema)

    def _feedback_value(self, name):
        if name == STATE_TRACK:
            return self.backend.state()
        if name == ODOM_TRACK:
            return self.backend.odometry()
        return self.backend.instrument_value(name)

    async def _telemetry(self):
        # Spread publications across the 50 ms cycle. Pushing every instrument
        # in one burst can starve earlier tracks in the lossy native sender.
        loop = asyncio.get_running_loop()
        slot = 0.05 / len(self.feedback_tracks)
        next_frame = loop.time()
        while True:
            for name, track in self.feedback_tracks.items():
                payload = json.dumps(self._feedback_value(name), allow_nan=False, separators=(",", ":")).encode()
                try:
                    track.try_push(rtc.DataTrackFrame(payload=payload))
                except rtc.PushFrameError as error:
                    LOG.debug("Instrument %s dropped: %s", name, error)
                next_frame = max(next_frame + slot, loop.time())
                await asyncio.sleep(next_frame - loop.time())
            self._feedback_ready = True
            self._mark_ready()

    async def _battery(self):
        # Match the estimated RC pack rather than the shared arm's 24 V fixture.
        while True:
            payload = json.dumps(self.backend.battery_state(), allow_nan=False, separators=(",", ":")).encode()
            for track in self.battery_tracks.values():
                try:
                    track.try_push(rtc.DataTrackFrame(payload=payload))
                except rtc.PushFrameError as error:
                    LOG.debug("Battery state dropped: %s", error)
            await asyncio.sleep(PERIOD)

    async def _expire_drive(self, reason):
        async with self.ownership_lock:
            self._disarm()
            self.active_operator = ""
            if self.room.isconnected():
                await self.room.local_participant.set_attributes({OWNER: ""})
        LOG.warning("Drive disarmed: %s; request control and arm a fresh input", reason)

    async def _simulate(self):
        loop = asyncio.get_running_loop()
        dt = 0.005
        deadline = loop.time()
        while True:
            if self.backend.watchdog():
                await self._expire_drive("command timeout")
            now = loop.time()
            if now - deadline > 0.1:
                self.overloads += 1
                await self._expire_drive("physics scheduler overload")
                deadline = loop.time()
            count = 0
            while deadline <= now and count < 10:
                self.backend.step(dt)
                deadline += dt
                count += 1
            await asyncio.sleep(max(0, deadline - loop.time()))
