"""Opt-in native transport integration against a local LiveKit dev server.

    livekit-server --dev --config tests/livekit-dev.yaml
    ROBO_SIMS_LIVEKIT_TEST=1 python -m unittest discover -s tests -p test_contract_livekit.py -v

Uses only development credentials and an isolated room. No cloud credentials.
"""

import asyncio
import json
import os
import unittest
import uuid


@unittest.skipUnless(os.environ.get("ROBO_SIMS_LIVEKIT_TEST") == "1", "opt-in local LiveKit integration")
class LiveKitContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_and_dual_gateway(self):
        from livekit import api, rtc
        from sims.assets import SOURCES
        from sims.contract.config import CONFIGS, Configuration, JOINT_STATE, JOINT_TRAJECTORY
        from sims.contract.gateway import ContractGateway, OWNER, ROLE, VERSION
        from sims.contract.mujoco_backend import MujocoBackend
        from sims.contract.rtc_schema import SchemaRoom, matches_json, publish_json
        from sims.contract.schemas import TRAJECTORY_SCHEMA
        from sims.trossen.simulation import TrossenSimulation

        url = "ws://127.0.0.1:17880"

        async def eventually(predicate, timeout=8):
            async def wait():
                while not predicate():
                    await asyncio.sleep(0.02)
            await asyncio.wait_for(wait(), timeout)

        for sim_name in ("trossen-wxai", "trossen-workbench"):
            with self.subTest(sim=sim_name):
                room_name = "contract-test-" + uuid.uuid4().hex

                def token(identity, role):
                    return (api.AccessToken("devkey", "secret").with_identity(identity)
                            .with_attributes({ROLE: role, VERSION: "3"})
                            .with_grants(api.VideoGrants(room_join=True, room=room_name,
                                can_publish=True, can_subscribe=True, can_publish_data=True,
                                can_update_own_metadata=True)).to_jwt())

                config = Configuration.load(CONFIGS / f"{sim_name}.bindings.json", 12)
                assets = SOURCES["trossen_arm_mujoco"].path / "assets"
                video = os.environ.get("ROBO_SIMS_VIDEO_TEST") == "1"
                scene = TrossenSimulation(assets, config.bindings["scene"], 64, 48, 128, 96,
                                         model_path=assets / config.bindings["model"])
                backend = MujocoBackend(scene, config)
                gateway = ContractGateway(backend, config, url, token("robot", "robot"), 12, video=video)
                running = asyncio.create_task(gateway.run())
                peers, readers, states, video_frames = [], [], {}, set()

                async def connect(identity, role="operator"):
                    peer = SchemaRoom()
                    peers.append(peer)

                    @peer.on("track_published")
                    def video_published(publication, participant):
                        if participant.identity == "robot":
                            publication.set_subscribed(True)

                    @peer.on("track_subscribed")
                    def video_subscribed(track, publication, participant):
                        if participant.identity != "robot" or track.kind != rtc.TrackKind.KIND_VIDEO:
                            return

                        async def read_video():
                            stream = rtc.VideoStream(track)
                            try:
                                async for event in stream:
                                    self.assertGreater(event.frame.width, 0)
                                    video_frames.add((identity, publication.name))
                            finally:
                                await stream.aclose()
                        readers.append(asyncio.create_task(read_video()))

                    @peer.on("data_track_published")
                    def published(track):
                        if track.publisher_identity != "robot":
                            return
                        self.assertTrue(matches_json(track, JOINT_STATE))

                        async def read():
                            stream = track.subscribe(buffer_size=1)
                            try:
                                async for frame in stream:
                                    states[identity, track.info.name] = json.loads(frame.payload)
                            finally:
                                await stream.aclose()
                        readers.append(asyncio.create_task(read()))

                    await peer.connect(url, token(identity, role), rtc.RoomOptions(auto_subscribe=False))
                    for participant in peer.remote_participants.values():
                        for publication in participant.track_publications.values():
                            video_published(publication, participant)
                    return peer

                async def rpc(peer, method, payload=""):
                    return await peer.local_participant.perform_rpc(destination_identity="robot",
                        method=method, payload=payload, response_timeout=3)

                interface_id = next(iter(config.discovery["control_interfaces"]))
                interface = config.discovery["control_interfaces"][interface_id]
                operation = interface["operations"]["joint_position"]
                endpoint = operation["command"]["track_name"]
                feedback = operation["feedback_tracks"][0]
                joint_name = interface["joint_names"][0]

                def push(track, value, *, cancel=False):
                    message = {"joint_names": [joint_name], "points": [] if cancel else [
                        {"positions": [value], "time_from_start": {"sec": 2, "nanosec": 0}}]}
                    track.try_push(rtc.DataTrackFrame(payload=json.dumps(message).encode()))

                try:
                    await eventually(lambda: gateway.ready.is_set() or running.done())
                    if running.done():
                        await running
                    first = await connect("operator-one")
                    second = await connect("operator-two")
                    await eventually(lambda: ("operator-one", feedback) in states)
                    if video:
                        await eventually(lambda: all(("operator-one", feed.name) in video_frames
                                                     for feed in scene.feeds.values()), timeout=20)
                    self.assertEqual(json.loads(await rpc(first, "get_robot_config")), config.discovery)
                    self.assertFalse(gateway.armed)
                    self.assertEqual(first.remote_participants["robot"].attributes[ROLE], "robot")
                    self.assertEqual(first.remote_participants["robot"].attributes[VERSION], "3")

                    one = await publish_json(first.local_participant, endpoint, JOINT_TRAJECTORY, TRAJECTORY_SCHEMA)
                    two = await publish_json(second.local_participant, endpoint, JOINT_TRAJECTORY, TRAJECTORY_SCHEMA)
                    await eventually(lambda: len(gateway.readers) == 2)
                    initial = states["operator-one", feedback]["position"][0]
                    push(one, initial + 0.2)
                    await asyncio.sleep(0.1)
                    self.assertFalse(backend.motions)

                    self.assertEqual(await rpc(first, "portal.set_active_operator", "operator-one"), "")
                    await eventually(lambda: first.remote_participants["robot"].attributes.get(OWNER) == "operator-one")
                    push(two, initial - 0.2)
                    await asyncio.sleep(0.1)
                    self.assertFalse(backend.motions)
                    push(one, initial + 0.2)
                    await eventually(lambda: interface_id in backend.motions)
                    await eventually(lambda: states["operator-one", feedback]["position"][0] > initial + 0.03)
                    self.assertEqual(set(backend.motions), {interface_id})

                    # Last accepted ownership request wins and cancels the old trajectory.
                    await rpc(second, "portal.set_active_operator", "operator-two")
                    self.assertFalse(backend.motions)
                    push(one, initial + 0.3)
                    await asyncio.sleep(0.1)
                    self.assertFalse(backend.motions)
                    push(two, initial - 0.2)
                    await eventually(lambda: interface_id in backend.motions)
                    push(two, 0, cancel=True)
                    await eventually(lambda: not backend.motions)
                    push(two, initial + 0.2)
                    await eventually(lambda: bool(backend.motions))
                    await two.unpublish()
                    await eventually(lambda: not backend.motions and not any(
                        track.info.sid == two.info.sid for track, _ in gateway.readers.values()))
                    two = await publish_json(second.local_participant, endpoint, JOINT_TRAJECTORY, TRAJECTORY_SCHEMA)
                    await eventually(lambda: any(track.info.sid == two.info.sid
                                                 for track, _ in gateway.readers.values()))

                    # Wrong/missing metadata never reaches the controller.
                    third = await connect("operator-three")
                    plain = await third.local_participant.publish_data_track(name=endpoint)
                    await rpc(third, "portal.set_active_operator", "operator-three")
                    push(plain, initial + 0.2)
                    await asyncio.sleep(0.15)
                    self.assertFalse(backend.motions)

                    await rpc(second, "portal.set_active_operator", "operator-two")
                    push(two, initial + 0.2)
                    await eventually(lambda: bool(backend.motions))
                    await second.disconnect()
                    await eventually(lambda: not gateway.armed and not backend.motions)
                    self.assertEqual(gateway.active_operator, "operator-two")
                    self.assertEqual(gateway.room.local_participant.attributes[OWNER], "operator-two")

                    rejoined = await connect("operator-two")
                    rejoined_track = await publish_json(rejoined.local_participant, endpoint,
                                                        JOINT_TRAJECTORY, TRAJECTORY_SCHEMA)
                    await eventually(lambda: any(track.publisher_identity == "operator-two"
                                                 for track, _ in gateway.readers.values()))
                    push(rejoined_track, initial + 0.2)
                    await asyncio.sleep(0.1)
                    self.assertFalse(backend.motions)
                    await rpc(rejoined, "portal.set_active_operator", "operator-two")
                    push(rejoined_track, initial + 0.2)
                    await eventually(lambda: bool(backend.motions))
                    await rejoined.local_participant.set_attributes({VERSION: "2"})
                    await eventually(lambda: not gateway.armed and not backend.motions)

                    observer = await connect("observer", "observer")
                    await eventually(lambda: gateway._compatible("observer", ("observer",)))
                    await eventually(lambda: ("observer", feedback) in states)
                    await rpc(observer, "portal.set_active_operator", "operator-one")
                    # An incompatible peer cannot take or clear ownership.
                    with self.assertRaises(rtc.RpcError):
                        await rpc(rejoined, "portal.set_active_operator", "operator-two")
                    self.assertEqual(gateway.active_operator, "operator-one")
                    await rpc(first, "portal.set_active_operator", "")
                    self.assertEqual(gateway.active_operator, "")
                    self.assertFalse(gateway.armed)
                finally:
                    gateway.stopping.set()
                    await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 10)
                    for peer in peers:
                        if peer.isconnected():
                            await peer.disconnect()
                    for reader in readers:
                        reader.cancel()
                    await asyncio.gather(*readers, return_exceptions=True)
                    scene.close()
