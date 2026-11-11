"""Shared-room queue, ownership and cleanup tests with controlled time."""
import asyncio
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import jwt
from livekit import api, rtc

from sims.demo.hosted_store import ModalStore
from sims.demo.shared import SharedService
from sims.demo.shared_rules import (ACTIVITY, CONTROL_ROBOT, GENERATION, KIND, QUEUED,
    QUEUE_ROBOT, UNTIL, Problem, activity, sign, verify)
from sims.demo.live_queue import ControlQueue
from sims.contract.gateway import ContractGateway


class Dictionary:
    def __init__(self): self.values = {}
    def get(self, key, default): return copy.deepcopy(self.values.get(key, default))
    def put(self, key, value): self.values[key] = copy.deepcopy(value)


class Room:
    def __init__(self):
        self.peers = {}
        self.removed = []
        self.updates = []
        self.fail_revoke = False
    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass
    async def participants(self): return copy.deepcopy(list(self.peers.values()))
    async def update(self, identity, attributes, publishing=None):
        if publishing is False and self.fail_revoke: raise ConnectionError('uncertain')
        if identity not in self.peers: return None
        peer = self.peers[identity]
        peer.attributes.update(attributes)
        if publishing is not None:
            peer.permission.CopyFrom(api.ParticipantPermission(can_subscribe=True,
                can_publish=publishing, can_publish_data=publishing))
        self.updates.append((identity, attributes, publishing))
        return copy.deepcopy(peer)
    async def remove(self, identity):
        self.removed.append(identity)
        self.peers.pop(identity, None)
    def add(self, identity, attrs=None):
        self.peers[identity] = api.ParticipantInfo(identity=identity, attributes=attrs or {})
        return self.peers[identity]


class SharedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.time = 1000000
        self.clock = patch('time.time', side_effect=lambda: self.time/1000)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.room = Room()
        self.store = ModalStore(Dictionary(), 'test', writable=True)
        self.env = dict(SIM_SUPERVISOR_KEY='k'*32, LIVEKIT_URL='ws://localhost',
            LIVEKIT_API_KEY='key', LIVEKIT_API_SECRET='s'*32)
        self.service = SharedService(self.store, self.env,
            [dict(id=name, available=True) for name in ('f1tenth', 'vacuum')], lambda *_: self.room)

    def launch(self, name='f1tenth'):
        public = self.service.watch('owner-aaaaaaaaaaaa', name)
        self.service.dispatch(SimpleNamespace(submit=lambda *_: 'call'))
        record = self.service.read(public['id'])
        run = self.service.claim(record['id'], record['dispatch_key'], 'call')
        self.service.report(run['id'], 'Bearer '+run['worker_key'], dict(state='ready',
            ready_at=self.time, expires_at=self.time+21600000, policy={}))
        self.room.add(run['robot_identity'])
        return self.service.read(run['id'])

    async def viewer(self, run, name):
        self.room.add(name, {KIND: 'viewer'})
        await ControlQueue(self.room, run).engage(name, 'initial-event-aaaa')

    async def test_shared_singleton_and_distinct_types(self):
        first = self.service.watch('owner-aaaaaaaaaaaa', 'f1tenth')
        other = self.service.watch('owner-bbbbbbbbbbbb', 'f1tenth')
        self.assertEqual(first['id'], other['id'])
        second = self.service.watch('owner-bbbbbbbbbbbb', 'vacuum')
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual({r['room_name'] for r in self.store.records()}, {'teleop-simulations'})
        self.assertEqual(len(self.store.records()), 2)

    async def test_room_credentials_are_spectator_and_proofs_are_bound(self):
        credential = self.service.room_credentials('owner-aaaaaaaaaaaa')
        claims = jwt.decode(credential['token'], options={'verify_signature': False})
        self.assertFalse(claims['video']['canPublish'])
        self.assertFalse(claims['video']['canPublishData'])
        self.assertFalse(claims['video']['canUpdateOwnMetadata'])
        self.assertEqual(claims['video']['room'], 'teleop-simulations')
        with self.assertRaises(Problem):
            self.service.room_identity(credential['proof'], 'owner-bbbbbbbbbbbb')
        with self.assertRaises(Problem): verify('wrong', credential['proof'])

    async def test_fifo_two_queues_and_no_cross_queue_release(self):
        first, second = self.launch(), self.launch('vacuum')
        await self.viewer(first, 'a'); await self.viewer(first, 'b'); await self.viewer(second, 'c')
        q1, q2 = ControlQueue(self.room, first), ControlQueue(self.room, second)
        self.assertEqual((await q1.action('a', 'join'))['controller']['identity'], 'a')
        state = await q1.action('b', 'join')
        self.assertEqual(state['queue_position'], 1)
        self.assertEqual(state['estimated_wait_seconds'], 60)
        self.assertEqual((await q2.action('c', 'join'))['controller']['identity'], 'c')
        await q1.action('c', 'release')
        self.assertEqual(self.room.peers['c'].attributes[CONTROL_ROBOT], second['robot_identity'])
        self.time += 61000
        state = await q1.reconcile('b')
        self.assertEqual(state['controller']['identity'], 'b')
        self.assertFalse(self.room.peers['a'].permission.can_publish)
        self.assertTrue(self.room.peers['c'].permission.can_publish)

    async def test_uncontested_renewal_does_not_extend_engagement(self):
        run = self.launch(); await self.viewer(run, 'a')
        q = ControlQueue(self.room, run)
        await q.action('a', 'join')
        original = activity(self.room.peers['a'])
        self.time += 61000
        state = await q.reconcile()
        self.assertEqual(state['controller']['until'], self.time+60000)
        self.assertEqual(activity(self.room.peers['a']), original)
        self.time += 60000
        self.assertIsNone((await q.reconcile())['controller'])
        self.assertNotIn('a', self.room.peers)

    async def test_cleanup_one_world_keeps_other_worker_viewer_and_turn(self):
        one, two = self.launch(), self.launch('vacuum')
        await self.viewer(one, 'a')
        await ControlQueue(self.room, two).engage('a', 'second-event-aaaa')
        await ControlQueue(self.room, two).action('a', 'join')
        await self.service.cleanup(one['id'])
        self.assertNotIn(one['robot_identity'], self.room.peers)
        self.assertIn(two['robot_identity'], self.room.peers)
        self.assertIn('a', self.room.peers)
        self.assertEqual(self.room.peers['a'].attributes[CONTROL_ROBOT], two['robot_identity'])
        self.assertTrue(self.room.peers['a'].permission.can_publish)
        self.assertNotIn(one['id'], activity(self.room.peers['a']))

    async def test_uncertain_revoke_does_not_grant_next_visitor(self):
        run = self.launch(); await self.viewer(run, 'a'); await self.viewer(run, 'b')
        q = ControlQueue(self.room, run)
        await q.action('a', 'join'); await q.action('b', 'join')
        self.time += 61000; self.room.fail_revoke = True
        with self.assertRaises(ConnectionError): await q.reconcile()
        self.assertFalse(self.room.peers['b'].permission.can_publish)

    async def test_one_participant_cannot_join_two_queues(self):
        one, two = self.launch(), self.launch('vacuum')
        await self.viewer(one, 'a'); await ControlQueue(self.room, two).engage('a', 'second-event-aaaa')
        await ControlQueue(self.room, one).action('a', 'join')
        with self.assertRaises(Problem): await ControlQueue(self.room, two).action('a', 'join')

    async def test_idle_clock_is_per_instance_and_stopping_cannot_revive(self):
        one, two = self.launch(), self.launch('vacuum')
        self.time += 119000
        self.service.touch(two['id'], self.time)
        self.time += 1000
        self.service.deadlines()
        self.assertEqual(self.service.read(one['id'])['state'], 'stopping')
        self.assertEqual(self.service.read(two['id'])['state'], 'ready')
        with self.assertRaises(Problem): self.service.touch(one['id'], self.time)
        self.assertEqual(self.service.watch('owner-bbbbbbbbbbbb', 'f1tenth')['id'], one['id'])
        self.service.finish(one['id'])
        self.assertNotEqual(self.service.watch('owner-bbbbbbbbbbbb', 'f1tenth')['id'], one['id'])

    async def test_duplicate_claim_lost_ack_and_drain(self):
        public = self.service.watch('owner-aaaaaaaaaaaa', 'f1tenth')
        calls = []
        def submit(*args): calls.append(args); raise ConnectionError('lost')
        with self.assertLogs('sims.demo.shared', level='ERROR'):
            self.service.dispatch(SimpleNamespace(submit=submit))
        self.service.dispatch(SimpleNamespace(submit=submit))
        self.assertEqual(len(calls), 1)
        run = self.service.read(public['id'])
        self.assertIsNotNone(self.service.claim(run['id'], run['dispatch_key'], 'recovered', 'attempt'))
        self.assertIsNone(self.service.claim(run['id'], run['dispatch_key'], 'replayed', 'other'))
        self.service.admission(True)
        with self.assertRaises(Problem): self.service.watch('owner-aaaaaaaaaaaa', 'vacuum')

    async def test_replayed_activity_event_does_not_refresh_deadline(self):
        run = self.launch(); await self.viewer(run, 'a')
        self.time += 50000
        at = await ControlQueue(self.room, run).engage('a', 'initial-event-aaaa')
        self.assertEqual(at, run['ready_at'])

    async def test_constant_targets_and_neutral_frames_do_not_count_as_activity(self):
        gateway = ContractGateway.__new__(ContractGateway)
        gateway._last_commands = {}
        gateway.last_activity = None
        command = dict(joint_names=['joint'], points=[dict(positions=[1], time_from_start={'sec': 1})])
        gateway._record_activity('a', json.dumps(command), 'arm')
        gateway._record_activity('a', json.dumps(command), 'gripper')
        last = gateway.last_activity
        command['points'][0]['time_from_start']['sec'] = 2
        for source in ('arm', 'gripper'):
            gateway._record_activity('a', json.dumps(command), source)
        gateway._record_activity('a', '{"speed":0,"steering_angle":0}', 'drive')
        gateway._record_activity('a', '{"points":[]}', 'arm')
        self.assertEqual(gateway.last_activity, last)
        command['points'][0]['positions'] = [2]
        gateway._record_activity('a', json.dumps(command), 'arm')
        self.assertNotEqual(gateway.last_activity, last)

    async def test_full_reconnect_restores_activity_without_extending_it(self):
        run = self.launch()
        owner = 'owner-aaaaaaaaaaaa'
        credential = self.service.room_credentials(owner)
        self.room.add(credential['identity'], {KIND: 'viewer'})
        issued = await self.service.credentials(run['id'], owner, dict(proof=credential['proof']))
        self.time += 50000
        self.room.peers[credential['identity']].attributes.clear()
        self.room.peers[credential['identity']].attributes[KIND] = 'viewer'
        resumed = await self.service.credentials(run['id'], owner, dict(proof=issued['proof'], resume=True))
        self.assertEqual(issued['engaged_until'], resumed['engaged_until'])
        self.time += 70001
        with self.assertRaises(Problem):
            await self.service.credentials(run['id'], owner, dict(proof=issued['proof'], resume=True))

    async def test_worker_rejects_other_target_expired_grant_and_arbitrary_owner(self):
        run = self.launch()
        peer = SimpleNamespace(identity='a', attributes={CONTROL_ROBOT: 'other',
            GENERATION: run['id'], UNTIL: str(self.time+60000)},
            permissions=api.ParticipantPermission(can_publish=True, can_publish_data=True))
        gateway = ContractGateway.__new__(ContractGateway)
        gateway.shared_instance = run['id']; gateway._grant_clocks = {}
        gateway.room = SimpleNamespace(local_participant=SimpleNamespace(identity=run['robot_identity']),
                                       remote_participants={'a':peer})
        gateway.expired = False; gateway.active_operator = 'a'; gateway.armed = True
        self.assertFalse(gateway._accepts('a'))
        peer.attributes[CONTROL_ROBOT] = run['robot_identity']
        self.assertTrue(gateway._accepts('a'))
        with self.assertRaises(rtc.RpcError):
            await gateway._set_owner(SimpleNamespace(caller_identity='a', payload='someone-else'))
        self.time += 61000
        # Real monotonic time has not moved in this wall-clock test, so force its bound.
        gateway._grant_clocks = {('a', int(peer.attributes[UNTIL])): 0}
        self.assertFalse(gateway._accepts('a'))
