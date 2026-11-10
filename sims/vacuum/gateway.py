"""Vacuum participant: shared Portal ownership and video lifecycle plus differential-drive control."""
import asyncio
import json
import logging

from livekit import rtc

from ..contract.battery import PERIOD
from ..contract.gateway import ContractGateway, OWNER
from ..contract.rtc_schema import define_json_schema, publish_json
from .backend import COMMAND, KIND, TWIST_SCHEMA
from .simulation import PHYSICS_DT

LOG = logging.getLogger("sims.vacuum")
START_RUN, RESET_RUN = "vacuum.start_run", "vacuum.reset_run"
TELEMETRY_TICK = 0.005     # 200 ticks per second share the 100 messages per second
OVERLOAD_LAG = 0.1         # seconds the physics may fall behind before driving stops


class VacuumGateway(ContractGateway):
    def __init__(self, backend, url, token, fps=30, **kwargs):
        self.invalid_sources = set()
        self.neutral_sources = set()
        self.overloads = 0
        super().__init__(backend, backend.config, url, token, fps, **kwargs)

    def _disarm(self):
        # A handoff, timeout or reconnect invalidates every existing publication.
        # A newly armed browser must publish a fresh track, then send neutral.
        self.invalid_sources.update(self.readers)
        self.neutral_sources.clear()
        super()._disarm()

    def _receive_command(self, track, interface_id, operation, payload):
        if track.info.sid in self.invalid_sources:
            return
        command = self.backend.parse(payload)
        if track.info.sid not in self.neutral_sources:
            if command.v != 0 or command.omega != 0:
                raise ValueError("a fresh drive publication must send neutral before motion")
            self.neutral_sources.add(track.info.sid)
        self.backend.submit(payload)

    def discovery(self):
        return {**self.config.discovery, "simulation_control": {"implementation": KIND, "policy": self.backend.policy()}}

    async def _set_owner(self, invocation):
        result = await super()._set_owner(invocation)
        if self.armed:
            self.backend.sim.enable_driving()
        return result

    async def _publish(self):
        await super()._publish()
        participant = self.room.local_participant
        participant.register_rpc_method(START_RUN, self._start_run)
        participant.register_rpc_method(RESET_RUN, self._reset_run)

    def _require_operator(self, invocation, method):
        if invocation.payload != "":
            raise rtc.RpcError(2000, f"{method} takes an empty payload")
        if not (self.armed and invocation.caller_identity == self.active_operator and not self.expired):
            raise rtc.RpcError(2003, f"{method} needs the operator that controls the robot, with controls armed")

    async def _start_run(self, invocation):
        self._require_operator(invocation, START_RUN)
        try:
            self.backend.start_run()
        except ValueError as error:
            raise rtc.RpcError(2001, str(error)) from error
        return ""

    async def _reset_run(self, invocation):
        self._require_operator(invocation, RESET_RUN)
        self.backend.reset_run()
        return ""

    async def _publish_feedback(self, participant):
        # Browser publishers can reference this schema ID without defining it.
        await define_json_schema(participant, COMMAND, TWIST_SCHEMA)
        self.rates = {}
        for name, (msg_type, schema, rate) in self.backend.instrument_tracks().items():
            self.feedback_tracks[name] = await publish_json(participant, name, msg_type, schema)
            self.rates[name] = rate

    def _feedback_value(self, name):
        return self.backend.instrument_value(name)

    async def _telemetry(self):
        # Every track repeats its complete current state, because data tracks are lossy. The native
        # sender can starve tracks that are pushed in one burst, so at most one track goes out per tick.
        loop = asyncio.get_running_loop()
        due = {name: loop.time() for name in self.feedback_tracks}
        next_tick = loop.time()
        while True:
            now = loop.time()
            ready = [name for name in self.feedback_tracks if due[name] <= now]
            if ready:
                name = min(ready, key=due.get)
                due[name] = max(due[name] + 1 / self.rates[name], now - 1 / self.rates[name])
                payload = json.dumps(self._feedback_value(name), allow_nan=False, separators=(",", ":")).encode()
                try:
                    self.feedback_tracks[name].try_push(rtc.DataTrackFrame(payload=payload))
                except rtc.PushFrameError as error:
                    LOG.debug("Track %s dropped: %s", name, error)
            self._feedback_ready = True
            self._mark_ready()
            next_tick = max(next_tick + TELEMETRY_TICK, loop.time())
            await asyncio.sleep(next_tick - loop.time())

    async def _battery(self):
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
        """Fixed physics steps on a monotonic clock. A long delay stops driving; time never speeds up."""
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while True:
            if self.backend.watchdog():
                await self._expire_drive("command timeout")
            now = loop.time()
            if now - deadline > OVERLOAD_LAG:
                self.overloads += 1
                self.backend.record_overload()
                await self._expire_drive("physics scheduler overload")
                deadline = loop.time()  # drop the missed steps instead of replaying them
            count = 0
            while deadline <= now and count < 10:
                self.backend.step(PHYSICS_DT)
                deadline += PHYSICS_DT
                count += 1
            await asyncio.sleep(max(0, deadline - loop.time()))
