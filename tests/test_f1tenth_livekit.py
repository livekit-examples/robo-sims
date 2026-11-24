"""Opt-in transport and decoded onboard video against localhost dev credentials."""
import asyncio
import json
import os
import unittest
import uuid
import logging


@unittest.skipUnless(os.environ.get('ROBO_SIMS_LIVEKIT_TEST') == '1', 'opt-in local LiveKit integration')
class DrivingTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_drive_watchdog_handoff_and_video(self):
        logging.basicConfig(level=logging.DEBUG if os.environ.get('ROBO_SIMS_DEBUG') else logging.WARNING)
        from livekit import api, rtc
        from sims.f1tenth.backend import Backend, COMMAND, COMMAND_TRACK, ACKERMANN_SCHEMA, STATE_TRACK, STATE
        from sims.f1tenth.gateway import DriveGateway
        from sims.contract.gateway import ROLE, VERSION, OWNER
        from sims.contract.rtc_schema import SchemaRoom, publish_json, matches_json
        from sims.f1tenth.telemetry import SPEED_TRACK, GEAR_TRACK, BATTERY_TRACK, LAST_LAP_TRACK

        url = 'ws://127.0.0.1:17880'
        room_name = 'driving-test-' + uuid.uuid4().hex
        def token(identity, role):
            return (api.AccessToken('devkey', 'secret').with_identity(identity)
                .with_attributes({ROLE: role, VERSION: '3'})
                .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True,
                    can_subscribe=True, can_publish_data=True, can_update_own_metadata=True)).to_jwt())

        async def eventually(predicate, timeout=8):
            async def wait():
                while not predicate():
                    await asyncio.sleep(0.02)
            await asyncio.wait_for(wait(), timeout)

        video = os.environ.get('ROBO_SIMS_VIDEO_TEST') == '1'
        backend = Backend(video=video, width=320, height=180, fps=30)
        gateway = DriveGateway(backend, url, token('robot', 'robot'), 30, video=video)
        running = asyncio.create_task(gateway.run())
        peers, readers, messages, instruments, frames, metadata = [], [], {}, {}, [], []
        instrument_counts = {}

        async def connect(identity):
            peer = SchemaRoom()
            peers.append(peer)
            @peer.on('data_track_published')
            def data_published(track):
                if track.publisher_identity != 'robot':
                    return
                definitions = backend.instrument_tracks()
                if track.info.name != STATE_TRACK and track.info.name not in definitions:
                    return
                msg_type = STATE if track.info.name == STATE_TRACK else definitions[track.info.name][0]
                metadata.append(matches_json(track, msg_type))
                async def read():
                    stream = track.subscribe(buffer_size=1)
                    try:
                        async for frame in stream:
                            value = json.loads(frame.payload)
                            if track.info.name == STATE_TRACK:
                                messages[identity] = value
                            else:
                                instruments[identity, track.info.name] = value
                                key = (identity, track.info.name)
                                instrument_counts[key] = instrument_counts.get(key, 0) + 1
                    finally:
                        await stream.aclose()
                readers.append(asyncio.create_task(read()))
            @peer.on('track_published')
            def published(pub, participant):
                if participant.identity == 'robot':
                    pub.set_subscribed(True)
            @peer.on('track_subscribed')
            def subscribed(track, pub, participant):
                if participant.identity != 'robot' or track.kind != rtc.TrackKind.KIND_VIDEO:
                    return
                async def read_video():
                    stream = rtc.VideoStream(track)
                    try:
                        async for event in stream:
                            frames.append((event.frame.width, event.frame.height))
                    finally:
                        await stream.aclose()
                readers.append(asyncio.create_task(read_video()))
            await peer.connect(url, token(identity, 'operator'), rtc.RoomOptions(auto_subscribe=False))
            # Match the browser transport, which announces its role after connecting.
            await peer.local_participant.set_attributes({ROLE: 'operator', VERSION: '3'})
            for participant in peer.remote_participants.values():
                for publication in participant.track_publications.values():
                    published(publication, participant)
            return peer

        async def request(peer):
            result = await peer.local_participant.perform_rpc(destination_identity='robot',
                method='portal.set_active_operator', payload=peer.local_participant.identity, response_timeout=3)
            self.assertEqual(result, '')

        async def publication(peer):
            return await publish_json(peer.local_participant, COMMAND_TRACK, COMMAND, ACKERMANN_SCHEMA)

        async def send(track, speed, seconds=0.25, steering=0):
            deadline = asyncio.get_running_loop().time() + seconds
            while asyncio.get_running_loop().time() < deadline:
                track.try_push(rtc.DataTrackFrame(payload=json.dumps({'speed': speed, 'steering_angle': steering}).encode()))
                await asyncio.sleep(0.02)

        try:
            await eventually(lambda: gateway.ready.is_set() or running.done(), 20)
            if running.done():
                await running
            first = await connect('alice')
            second = await connect('bob')
            await eventually(lambda: 'alice' in messages)
            await eventually(lambda: all(('alice', name) in instruments for name in backend.instrument_tracks()))
            self.assertTrue(all(metadata))
            self.assertEqual(instruments['alice', SPEED_TRACK], {'data': 0})
            self.assertEqual(instruments['alice', GEAR_TRACK], {'data': 'N'})
            self.assertEqual(instruments['alice', LAST_LAP_TRACK], {'data': None})
            initial_energy = instruments['alice', BATTERY_TRACK]['remaining_wh']
            self.assertAlmostEqual(instruments['alice', BATTERY_TRACK]['capacity_wh'], 55.5)
            response = await first.local_participant.perform_rpc(destination_identity='robot',
                method='get_robot_config', payload='', response_timeout=3)
            discovery = json.loads(response)
            self.assertEqual(discovery['simulation_control']['implementation'], 'rc.ackermann.v1')
            self.assertEqual(discovery['simulation_control']['policy'], backend.policy())
            if video:
                await eventually(lambda: bool(frames), 20)
                self.assertEqual(frames[0], (320, 180))
            unauthorized = await publication(second)
            await send(unauthorized, 4)
            self.assertEqual(backend.car.vx, 0)

            await eventually(lambda: gateway._compatible('alice'))
            await request(first)
            one = await publication(first)
            await send(one, 0)
            before_counts = {name: instrument_counts.get(('alice', name), 0)
                             for name in backend.instrument_tracks()}
            await send(one, 4, seconds=0.8)
            self.assertGreater(backend.car.vx, 1)
            self.assertGreater(messages['alice']['speed'], 0.5)
            self.assertGreater(instruments['alice', SPEED_TRACK]['data'], 1,
                {"samples": {name: count for (peer, name), count in instrument_counts.items() if peer == 'alice'},
                 "values": {name: value for (peer, name), value in instruments.items() if peer == 'alice'},
                 "backend_speed": backend.instrument_value(SPEED_TRACK)})
            self.assertEqual(instruments['alice', GEAR_TRACK], {'data': 'F'})
            self.assertLess(instruments['alice', BATTERY_TRACK]['remaining_wh'], initial_energy)
            # The channel is lossy, but no instrument may be starved by the others.
            # Require at least half of the nominal 16 samples over this interval.
            for name in backend.instrument_tracks():
                self.assertGreaterEqual(instrument_counts.get(('alice', name), 0) - before_counts[name], 8, name)
            # Expiry is measured by the robot; no browser stop message is sent.
            await eventually(lambda: backend.timed_out and not gateway.armed)
            self.assertEqual(gateway.active_operator, '')
            await eventually(lambda: abs(backend.car.vx) < 0.1)
            await request(first)
            await send(one, 4)  # The previous publication is invalid even after reacquisition.
            self.assertEqual(backend.car.target.speed, 0)
            await one.unpublish()
            one = await publication(first)
            await send(one, 0)
            await send(one, 2)
            self.assertEqual(backend.car.target.speed, 2)

            await request(second)
            await unauthorized.unpublish()
            two = await publication(second)
            await send(two, 0)
            await asyncio.gather(send(one, 4), send(two, 1))
            self.assertEqual(backend.car.target.speed, 1)
            await second.local_participant.set_attributes({VERSION: '2'})
            await eventually(lambda: not gateway.armed)
            self.assertEqual(backend.car.target.speed, 0)
            await second.local_participant.set_attributes({VERSION: '3'})
            await send(two, 4)
            self.assertEqual(backend.car.target.speed, 0)
            self.assertFalse(gateway.armed)

            await request(first)
            await one.unpublish()
            one = await publication(first)
            await send(one, 0)
            await send(one, 2)
            await one.unpublish()
            await eventually(lambda: backend.car.target.speed == 0)
            one = await publication(first)
            await send(one, 0)
            await send(one, 2)
            await first.disconnect()
            await eventually(lambda: not gateway.armed)
            self.assertEqual(backend.car.target.speed, 0)
            rejoined = await connect('alice')
            rejoined_track = await publication(rejoined)
            await send(rejoined_track, 4)
            self.assertEqual(backend.car.target.speed, 0)
            self.assertFalse(gateway.armed)
        finally:
            gateway.stopping.set()
            await asyncio.wait_for(running, 10)
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            for peer in peers:
                await peer.disconnect()
            backend.close()
            async with api.LiveKitAPI(url, 'devkey', 'secret') as lk:
                await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
