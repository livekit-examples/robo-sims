"""Shared 20-slot admission, FIFO queue and recovery without cloud compute."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import threading
import unittest
from unittest.mock import patch
import jwt

from sims.demo.hosted import HostedService, Problem
from sims.demo.rules import SUPPORTED, SessionConfig, now, room_candidates, token, worker_ttl
from sims.demo.queue import RUNNING


class MemoryStore:
    def __init__(self):
        self.runs, self.rooms = {}, {}
        self.lock = threading.RLock()
    @contextmanager
    def transaction(self):
        with self.lock:
            before = copy.deepcopy((self.runs, self.rooms))
            try:
                yield self
            except BaseException:
                self.runs, self.rooms = before
                raise
    def records(self): return copy.deepcopy(list(self.runs.values()))
    def get(self, identifier): return copy.deepcopy(self.runs.get(identifier))
    def save(self, run): self.runs[run['id']] = copy.deepcopy(run)
    def reserve(self, name, identifier, until):
        if name in self.rooms: return False
        self.rooms[name] = identifier
        return True
    def prune(self, *_): pass


class Provider:
    def __init__(self): self.calls = []
    def submit(self, identifier, key):
        self.calls.append((identifier, key))
        return 'fc-' + identifier


class HostedTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.env = dict(SIM_SUPERVISOR_KEY='k'*32, SIM_IMAGE_VERSION='test-v1', LIVEKIT_URL='wss://example.test',
                        LIVEKIT_API_KEY='key', LIVEKIT_API_SECRET='s'*32)
        self.service = HostedService(self.store, self.env, [dict(id=s, available=True) for s in SUPPORTED])
        self.provider = Provider()
    def launch(self, owner='owner-aaaaaaaaaaaa', request='request-aaaaaaaaaa', simulation=SUPPORTED[0]):
        return self.service.launch(owner, dict(simulation_id=simulation, request_id=request), 'launcher-aaaaaaaa')
    def claim(self, run):
        self.service.dispatch(self.provider)
        record = self.store.get(run['id'])
        return self.service.claim(run['id'], record['dispatch_key'], 'fc-'+run['id'])
    def ready(self, run):
        record = self.store.get(run['id'])
        event = dict(state='ready', ready_at=now(), expires_at=now() + record['duration_seconds']*1000, policy={})
        self.service.report(run['id'], 'Bearer ' + record['worker_key'], event)
        return event
    def problem(self, status, action):
        with self.assertRaises(Problem) as caught: action()
        self.assertEqual(caught.exception.status, status)

    def test_forty_computers_all_four_scenes_share_twenty_slots(self):
        with ThreadPoolExecutor(12) as pool:
            runs = list(pool.map(lambda i: self.launch(f'owner-{i:016d}', simulation=SUPPORTED[i%4]), range(40)))
        self.service.dispatch(self.provider)
        records = self.store.records()
        self.assertEqual(sum(r['state'] == 'starting' for r in records), 20)
        self.assertEqual(sum(r['state'] == 'queued' for r in records), 20)
        self.assertEqual(len(self.provider.calls), 20)
        self.assertEqual({r['simulation_id'] for r in records}, set(SUPPORTED))
        for run in runs:
            if self.store.get(run['id'])['state'] == 'queued':
                result = self.service.session(run['id'], self.store.get(run['id'])['owner'])
                self.assertIn(result['queue_position'], range(1,21))
                self.assertGreater(result['estimated_wait_seconds'],0)
        self.service.dispatch(self.provider)
        self.assertEqual(len(self.provider.calls),20)

    def test_concurrent_duplicate_requests_keep_one_place_and_one_call(self):
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda _: self.launch(), range(20)))
        self.assertEqual(len({r['id'] for r in results}),1)
        self.service.dispatch(self.provider)
        self.service.dispatch(self.provider)
        self.assertEqual(len(self.provider.calls),1)
        self.problem(409, lambda: self.launch(simulation=SUPPORTED[1]))

    def test_one_waiting_trial_per_visitor_and_owner_credentials(self):
        run = self.launch()
        self.problem(409, lambda: self.launch(request='request-bbbbbbbbbb'))
        self.problem(404, lambda: self.service.session(run['id'],'another-owner-aaaa'))
        self.problem(410, lambda: self.service.credentials(run['id'],'owner-aaaaaaaaaaaa',{'mode':'control'}))
        self.claim(run); self.ready(run)
        credentials = self.service.credentials(run['id'],'owner-aaaaaaaaaaaa',{'mode':'control'})
        claims = jwt.decode(credentials['token'], 's'*32, algorithms=['HS256'],options={'verify_aud':False})
        self.assertEqual(claims['video']['room'],credentials['room_name'])
        self.assertEqual(credentials['room_name'],self.store.get(run['id'])['room_name'])
        self.problem(404, lambda: self.service.credentials(run['id'],'another-owner-aaaa',{'mode':'control'}))
        self.problem(400, lambda: self.service.credentials(run['id'],'owner-aaaaaaaaaaaa',{'mode':'control','room_name':'other'}))

    def test_fifo_moves_forward_after_cleanup_and_cancel(self):
        self.service.capacity=1
        first=self.launch()
        self.claim(first)
        second=self.launch('owner-bbbbbbbbbbbb')
        third=self.launch('owner-cccccccccccc',simulation=SUPPORTED[3])
        self.assertEqual(third['queue_position'],2)
        self.service.session(second['id'],'owner-bbbbbbbbbbbb',stopping=True)
        self.assertEqual(self.service.session(third['id'],'owner-cccccccccccc')['queue_position'],1)
        self.service.session(first['id'],'owner-aaaaaaaaaaaa',stopping=True)
        self.assertEqual(self.service.dispatch(self.provider),[])
        self.service.finish(first['id'],0)
        self.assertEqual(self.service.dispatch(self.provider),[third['id']])
        self.assertEqual(self.store.get(second['id'])['state'],'stopped')

    def test_estimate_uses_remaining_deadlines_and_waiting_durations(self):
        self.service.capacity=1
        first=self.launch(); self.claim(first)
        event=self.ready(first)
        second=self.launch('owner-bbbbbbbbbbbb')
        third=self.launch('owner-cccccccccccc')
        with patch('sims.demo.rules.now',return_value=event['expires_at']-10000):
            self.assertEqual(self.service.session(second['id'],'owner-bbbbbbbbbbbb')['estimated_wait_seconds'],15)
            self.assertEqual(self.service.session(third['id'],'owner-cccccccccccc')['estimated_wait_seconds'],125)

    def test_duration_is_stored_readiness_idempotent_and_late_events_cannot_revive(self):
        self.service.config=SessionConfig(7,24)
        run=self.launch(); self.claim(run)
        self.service.config=SessionConfig(120,30)
        event=self.ready(run)
        record=self.store.get(run['id'])
        event.update(ready_at=event['ready_at']+1000,expires_at=event['expires_at']+1000)
        self.service.report(run['id'],'Bearer '+record['worker_key'],event)
        self.assertEqual(self.store.get(run['id'])['expires_at'],record['expires_at'])
        self.assertEqual(record['duration_seconds'],7)
        self.service.session(run['id'],'owner-aaaaaaaaaaaa',stopping=True)
        self.assertTrue(self.service.report(run['id'],'Bearer '+record['worker_key'],event)['stop'])
        self.problem(401,lambda:self.service.report(run['id'],'Bearer wrong',event))

    def test_queue_timeout_does_not_run_physics(self):
        run=self.launch()
        record=self.store.get(run['id'])
        with patch('sims.demo.hosted.now',return_value=record['queue_expires_at']+1):
            self.service.deadlines()
            self.assertEqual(self.service.dispatch(self.provider),[])
        self.assertEqual(self.store.get(run['id'])['state'],'expired')
        self.assertEqual(self.provider.calls,[])

    def test_delayed_or_duplicate_function_claim_cannot_start_simulation(self):
        run=self.launch(); self.service.dispatch(self.provider)
        record=self.store.get(run['id'])
        self.assertIsNone(self.service.claim(run['id'],'wrong','fc-wrong'))
        self.assertIsNotNone(self.service.claim(run['id'],record['dispatch_key'],'fc-right'))
        self.assertIsNone(self.service.claim(run['id'],record['dispatch_key'],'fc-duplicate'))
        late=self.launch('owner-bbbbbbbbbbbb'); self.service.dispatch(self.provider)
        record=self.store.get(late['id'])
        with patch('sims.demo.hosted.now',return_value=record['start_deadline']+1):
            self.assertIsNone(self.service.claim(late['id'],record['dispatch_key'],'fc-late'))

    def test_lost_spawn_ack_is_not_resubmitted_and_worker_recovers_call_id(self):
        run=self.launch()
        original=self.provider.submit
        def lost(*args): original(*args); raise OSError('lost ACK')
        self.provider.submit=lost
        with self.assertLogs('sims.demo.hosted',level='ERROR'):
            self.service.dispatch(self.provider)
        self.service.dispatch(self.provider)
        self.assertEqual(len(self.provider.calls),1)
        record=self.store.get(run['id'])
        self.service.claim(run['id'],record['dispatch_key'],'fc-recovered')
        self.assertEqual(self.store.get(run['id'])['call_id'],'fc-recovered')

    def test_collisions_retry_and_request_retry_preserves_room(self):
        self.store.rooms['curie-00']='other'
        with patch('sims.demo.hosted.room_candidates',return_value=iter(['curie-00','bohr-0a'])):
            run=self.launch()
        self.assertEqual(self.store.get(run['id'])['room_name'],'bohr-0a')
        self.assertEqual(run['id'],self.launch()['id'])
        with patch('sims.demo.hosted.room_candidates',return_value=iter(['curie-00'])):
            self.problem(503,lambda:self.launch('owner-bbbbbbbbbbbb'))

    def test_queue_limit_hourly_budget_and_scoped_launcher_stop(self):
        self.service.queue_limit=1
        run=self.launch()
        self.problem(429,lambda:self.launch('owner-bbbbbbbbbbbb'))
        self.service.stop_launcher('launcher-unrelated')
        self.assertEqual(self.store.get(run['id'])['state'],'queued')
        self.service.stop_launcher('launcher-aaaaaaaa')
        self.assertEqual(self.store.get(run['id'])['state'],'stopped')
        self.service.hourly=1
        self.problem(429,lambda:self.launch('owner-bbbbbbbbbbbb'))

    def test_client_cannot_override_configuration_and_service_auth(self):
        self.problem(400,lambda:self.service.launch('owner-aaaaaaaaaaaa',dict(
            simulation_id=SUPPORTED[0],request_id='request-aaaaaaaaaa',fps=1000)))
        self.problem(401,lambda:self.service.authorize('Bearer wrong','owner-aaaaaaaaaaaa'))
        self.assertEqual(HostedService(self.store,{**self.env,'SIM_SESSION_CAPACITY':'999'},[]).capacity,20)


class RuleTests(unittest.TestCase):
    def test_limits_and_two_digit_hash(self):
        for values in ({'SIM_SESSION_DURATION_SECONDS':'nan'},{'SIM_SESSION_FPS':'inf'},
                       {'SIM_SESSION_FPS':'1.5'},{'SIM_SESSION_DURATION_SECONDS':'0'},
                       {'SIM_SESSION_FPS':'61'},{'SIM_SESSION_DURATION_SECONDS':'3601'}):
            with self.assertRaises(ValueError): SessionConfig.from_env(values)
        self.assertEqual(SessionConfig.from_env({}),SessionConfig(60,30))
        names=list(room_candidates('stable-session'))
        self.assertEqual(names,list(room_candidates('stable-session')))
        for name in names: self.assertRegex(name,r'^[a-z]+-[0-9a-f]{2}$')
    def test_long_worker_token_matches_stored_room_and_duration(self):
        run=dict(room_name='curie-00',duration_seconds=3600,startup_seconds=45,cleanup_seconds=5)
        claims=jwt.decode(token(run,'robot','robot',worker_ttl(run),'key','s'*32),options={'verify_signature':False})
        self.assertGreater(claims['exp']-claims['nbf'],3600)
        self.assertEqual(claims['video']['room'],'curie-00')
