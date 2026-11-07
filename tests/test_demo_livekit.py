"""Real managed-provider acceptance. Opt in against tests/livekit-dev.yaml only."""
import asyncio
import json
import os
from pathlib import Path
import secrets
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('ROBO_SIMS_DEMO_TEST') == '1', 'opt-in managed provider integration')
class ManagedProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_managed_video_motion_and_cleanup(self):
        from aiohttp.test_utils import TestClient, TestServer
        from livekit import api, rtc
        from sims.demo.supervisor import Supervisor, application
        from sims.demo.provider import DockerProvider
        from sims.contract.rtc_schema import SchemaRoom, publish_json
        from sims.contract.schemas import TRAJECTORY_SCHEMA
        async def eventually(predicate, timeout=45):
            async def wait():
                while not predicate():
                    await asyncio.sleep(0.05)
            await asyncio.wait_for(wait(), timeout)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            provider = DockerProvider('teleop-simulation:local') if os.environ.get('ROBO_SIMS_DEMO_PROVIDER') == 'docker' else None
            key = secrets.token_hex(32)
            supervisor = Supervisor(Path(directory), key, 'ws://127.0.0.1:17880', 'devkey', 'secret', provider=provider)
            async with TestClient(TestServer(application(supervisor))) as client:
                headers = {'Authorization': f'Bearer {key}', 'X-Demo-Owner': secrets.token_hex(16)}
                choices = [s for s in supervisor.catalog() if s['available']]
                self.assertTrue(choices)
                for simulation in choices:
                    with self.subTest(simulation=simulation['id']):
                        response = await client.post('/sessions', headers=headers, json={
                            'simulation_id': simulation['id'], 'request_id': secrets.token_hex(16)})
                        self.assertEqual(response.status, 202)
                        run = await response.json()
                        identifier = run['id']
                        def record(): return supervisor.owned(identifier, headers['X-Demo-Owner'])
                        await eventually(lambda: record()['state'] != 'starting')
                        self.assertEqual(record()['state'], 'ready', record())
                        response = await client.post(f'/sessions/{identifier}/credentials', headers=headers, json={'mode': 'control'})
                        credentials = await response.json()
                        peer = SchemaRoom()
                        states, videos, readers = {}, set(), []
                        @peer.on('data_track_published')
                        def data(track):
                            async def read():
                                stream = track.subscribe(buffer_size=1)
                                try:
                                    async for frame in stream:
                                        states[track.info.name] = json.loads(frame.payload)
                                finally: await stream.aclose()
                            readers.append(asyncio.create_task(read()))
                        @peer.on('track_published')
                        def published(publication, participant):
                            publication.set_subscribed(True)
                        @peer.on('track_subscribed')
                        def subscribed(track, publication, participant):
                            if track.kind != rtc.TrackKind.KIND_VIDEO: return
                            async def read():
                                stream = rtc.VideoStream(track)
                                try:
                                    async for event in stream:
                                        if event.frame.width: videos.add(publication.name)
                                finally: await stream.aclose()
                            readers.append(asyncio.create_task(read()))
                        try:
                            await peer.connect(credentials['server_url'], credentials['token'])
                            await peer.local_participant.set_attributes({'lk.portal.role': 'operator', 'lk.portal.version': '3'})
                            for participant in peer.remote_participants.values():
                                for publication in participant.track_publications.values():
                                    publication.set_subscribed(True)
                            async def rpc(method, payload=''):
                                return await peer.local_participant.perform_rpc(destination_identity=credentials['robot_identity'], method=method, payload=payload)
                            discovery = json.loads(await rpc('get_robot_config'))
                            interface_id = next(iter(discovery['control_interfaces']))
                            interface = discovery['control_interfaces'][interface_id]
                            operation = next(iter(interface['operations'].values()))
                            feedback = operation['feedback_tracks'][0]
                            await eventually(lambda: feedback in states and videos, timeout=20)
                            name = interface['joint_names'][0]
                            state = states[feedback]
                            initial = state['position'][state['name'].index(name)]
                            await rpc('portal.set_active_operator', peer.local_participant.identity)
                            track = await publish_json(peer.local_participant, operation['command']['track_name'],
                                                       operation['command']['msg_type'], TRAJECTORY_SCHEMA)
                            message = json.dumps({'joint_names': [name], 'points': [
                                {'positions': [initial + .1], 'time_from_start': {'sec': 2, 'nanosec': 0}}]}).encode()
                            async def send():
                                while True:
                                    track.try_push(rtc.DataTrackFrame(payload=message))
                                    await asyncio.sleep(.2)
                            sender = asyncio.create_task(send())
                            try:
                                await eventually(lambda: states[feedback]['position'][states[feedback]['name'].index(name)] > initial + .02, timeout=8)
                            finally:
                                sender.cancel()
                                await asyncio.gather(sender, return_exceptions=True)
                        finally:
                            for reader in readers: reader.cancel()
                            await asyncio.gather(*readers, return_exceptions=True)
                            await peer.disconnect()
                            await client.delete(f'/sessions/{identifier}', headers=headers)
                        await eventually(lambda: record()['state'] == 'stopped', timeout=12)
                        self.assertFalse(supervisor.provider.status(record()))
                        rooms = await supervisor.lk.room.list_rooms(api.ListRoomsRequest(names=[record()['room']]))
                        self.assertFalse(rooms.rooms)
