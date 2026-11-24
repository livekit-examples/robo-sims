"""Opt-in transport test for the vacuum participant against a local development LiveKit server.

    livekit-server --dev --config tests/livekit-dev.yaml
    ROBO_SIMS_LIVEKIT_TEST=1 python -m unittest tests.test_vacuum_livekit -v
Add ROBO_SIMS_VIDEO_TEST=1 to also decode both video views.
"""
import asyncio
import json
import logging
import os
import unittest
import uuid


@unittest.skipUnless(os.environ.get("ROBO_SIMS_LIVEKIT_TEST") == "1", "opt-in local LiveKit integration")
class VacuumTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_tracks_driving_ownership_run_control_and_expiry(self):
        logging.basicConfig(level=logging.WARNING)
        from livekit import api, rtc
        from sims.contract.gateway import OWNER, ROLE, VERSION
        from sims.contract.rtc_schema import SchemaRoom, matches_json, publish_json
        from sims.vacuum.backend import COMMAND, COMMAND_TRACK, TWIST_SCHEMA, Backend
        from sims.vacuum.gateway import RESET_RUN, START_RUN, VacuumGateway

        url = "ws://127.0.0.1:17880"
        room_name = "vacuum-test-" + uuid.uuid4().hex

        def token(identity, role):
            return (api.AccessToken("devkey", "secret").with_identity(identity)
                    .with_attributes({ROLE: role, VERSION: "3"})
                    .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True,
                                                 can_publish_data=True, can_update_own_metadata=True)).to_jwt())

        async def eventually(predicate, timeout=8):
            async def wait():
                while not predicate():
                    await asyncio.sleep(0.02)
            await asyncio.wait_for(wait(), timeout)

        video = os.environ.get("ROBO_SIMS_VIDEO_TEST") == "1"
        backend = Backend(video=video, width=320, height=180, fps=30)
        gateway = VacuumGateway(backend, url, token("robot", "robot"), 30, video=video)
        running = asyncio.create_task(gateway.run())
        definitions = backend.instrument_tracks()
        values, counts, metadata, frames, peers, readers = {}, {}, [], {}, [], []

        async def connect(identity):
            peer = SchemaRoom()
            peers.append(peer)

            @peer.on("data_track_published")
            def data_published(track):
                if track.publisher_identity != "robot" or track.info.name not in definitions:
                    return
                metadata.append(matches_json(track, definitions[track.info.name][0]))

                async def read():
                    stream = track.subscribe(buffer_size=1)
                    try:
                        async for frame in stream:
                            values[identity, track.info.name] = json.loads(frame.payload)
                            counts[identity, track.info.name] = counts.get((identity, track.info.name), 0) + 1
                    finally:
                        await stream.aclose()
                readers.append(asyncio.create_task(read()))

            @peer.on("track_published")
            def published(pub, participant):
                if participant.identity == "robot":
                    pub.set_subscribed(True)

            @peer.on("track_subscribed")
            def subscribed(track, pub, participant):
                if participant.identity != "robot" or track.kind != rtc.TrackKind.KIND_VIDEO:
                    return

                async def read_video():
                    stream = rtc.VideoStream(track)
                    try:
                        async for event in stream:
                            frames[track.name] = (event.frame.width, event.frame.height)
                    finally:
                        await stream.aclose()
                readers.append(asyncio.create_task(read_video()))

            await peer.connect(url, token(identity, "operator"), rtc.RoomOptions(auto_subscribe=False))
            await peer.local_participant.set_attributes({ROLE: "operator", VERSION: "3"})
            for participant in peer.remote_participants.values():
                for publication in participant.track_publications.values():
                    published(publication, participant)
            return peer

        async def rpc(peer, method, payload=""):
            return await peer.local_participant.perform_rpc(destination_identity="robot", method=method,
                                                            payload=payload, response_timeout=3)

        async def request(peer):
            self.assertEqual(await rpc(peer, "portal.set_active_operator", peer.local_participant.identity), "")

        async def publication(peer):
            return await publish_json(peer.local_participant, COMMAND_TRACK, COMMAND, TWIST_SCHEMA)

        async def send(track, speed, seconds=0.25, yaw=0.0):
            deadline = asyncio.get_running_loop().time() + seconds
            payload = json.dumps({"linear": {"x": speed, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": yaw}}).encode()
            while asyncio.get_running_loop().time() < deadline:
                track.try_push(rtc.DataTrackFrame(payload=payload))
                await asyncio.sleep(0.02)

        try:
            await eventually(lambda: gateway.ready.is_set() or running.done(), 20)
            if running.done():
                await running
            first, second = await connect("alice"), await connect("bob")
            await eventually(lambda: all(("alice", name) in values for name in definitions))
            self.assertTrue(all(metadata) and len(metadata) >= 2 * len(definitions))
            run = values["alice", "/run_state"]
            self.assertEqual((run["state"], run["deadline_s"], run["trial_valid"]), ("idle", 60.0, True))
            self.assertIn("run_id", values["alice", "/coverage_state"])
            discovery = json.loads(await rpc(first, "get_robot_config"))
            self.assertEqual(discovery["simulation_control"]["implementation"], "vacuum.twist.v1")
            self.assertEqual(discovery["simulation_control"]["policy"], backend.policy())
            if video:
                await eventually(lambda: {"overhead", "onboard"} <= frames.keys(), 20)
                self.assertEqual(frames["overhead"], (320, 180))

            # Other users cannot drive, start or reset.
            stranger = await publication(second)
            await send(stranger, 0.3)
            self.assertEqual(backend.sim.requested.v, 0)
            await eventually(lambda: gateway._compatible("alice") and gateway._compatible("bob"))
            for method in (START_RUN, RESET_RUN):
                with self.assertRaises(rtc.RpcError):
                    await rpc(second, method)
            with self.assertRaises(rtc.RpcError):  # not even the owner may start before arming
                await rpc(first, START_RUN)

            await request(first)
            await stranger.unpublish()
            one = await publication(first)
            await send(one, 0.3)  # motion before neutral is rejected
            self.assertEqual(backend.sim.requested.v, 0)
            await send(one, 0)
            with self.assertRaises(rtc.RpcError):
                await rpc(second, START_RUN)
            self.assertEqual(await rpc(first, START_RUN), "")
            await eventually(lambda: backend.sim.state == "running")
            before = {name: counts.get(("alice", name), 0) for name in definitions}
            await send(one, 0.3, seconds=1.0)
            self.assertGreater(backend.sim.velocity()[0], 0.1)
            self.assertGreater(values["alice", "/speed_mps"]["data"], 0.1)
            self.assertEqual(values["alice", "/run_state"]["state"], "running")
            self.assertGreater(values["alice", "/coverage_state"]["cleaned_m2"], 0)
            self.assertLess(values["alice", "/battery_state"]["remaining_wh"], 30)
            self.assertGreater(values["alice", "/dust_bin_state"]["used_l"], 0)
            for name, (_, _, rate) in definitions.items():  # lossy, but no track may starve
                minimum = max(3, int(rate * 0.5 * 0.5))
                self.assertGreaterEqual(counts.get(("alice", name), 0) - before[name], minimum, name)

            # The robot measures expiry; no stop message is sent. Controls must be armed again.
            await eventually(lambda: backend.timed_out and not gateway.armed)
            self.assertEqual(gateway.active_operator, "")
            await eventually(lambda: abs(backend.sim.velocity()[0]) < 0.02, 6)
            with self.assertRaises(rtc.RpcError):
                await rpc(first, RESET_RUN)
            await request(first)
            await send(one, 0.3)  # the old publication stays invalid
            self.assertEqual(backend.sim.requested.v, 0)
            await one.unpublish()
            one = await publication(first)
            await send(one, 0)
            await send(one, 0.2)
            self.assertEqual(backend.sim.requested.v, 0.2)
            with self.assertRaises(rtc.RpcError):  # the run already started
                await rpc(first, START_RUN)
            self.assertEqual(await rpc(first, RESET_RUN), "")
            self.assertEqual((backend.sim.state, backend.sim.requested.v), ("idle", 0))

            # Handoff stops the first operator; a role change stops the second.
            await request(second)
            two = await publication(second)
            await send(two, 0)
            await asyncio.gather(send(one, 0.3), send(two, 0.1))
            self.assertEqual(backend.sim.requested.v, 0.1)
            await second.local_participant.set_attributes({VERSION: "2"})
            await eventually(lambda: not gateway.armed)
            self.assertEqual(backend.sim.requested.v, 0)
            await second.local_participant.set_attributes({VERSION: "3"})
            await send(two, 0.3)
            self.assertEqual(backend.sim.requested.v, 0)

            # A disconnect stops driving and a rejoined identity must start over.
            await request(first)
            await one.unpublish()
            one = await publication(first)
            await send(one, 0)
            await send(one, 0.2)
            await first.disconnect()
            await eventually(lambda: not gateway.armed)
            self.assertEqual(backend.sim.requested.v, 0)
            rejoined = await connect("alice")
            again = await publication(rejoined)
            await send(again, 0.3)
            self.assertEqual(backend.sim.requested.v, 0)
            self.assertFalse(gateway.armed)
            # Portal ownership persists across a reconnect, but arming does not.
            self.assertEqual(gateway.room.local_participant.attributes.get(OWNER, ""), "alice")
        finally:
            gateway.stopping.set()
            await asyncio.wait_for(running, 10)
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            for peer in peers:
                await peer.disconnect()
            backend.close()
            async with api.LiveKitAPI(url, "devkey", "secret") as lk:
                await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))


if __name__ == "__main__":
    unittest.main()
