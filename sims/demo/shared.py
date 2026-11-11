"""Shared instances and LiveKit queues for Modal and local execution."""
import hmac
import json
import logging
import secrets
import uuid
from datetime import timedelta

from livekit import api

from .hosted import stop
from .live_queue import ControlQueue, LiveRoom
from .rules import ACTIVE, KEY_PATTERN, SUPPORTED, accept_ready, now, worker_ttl
from .shared_rules import (ACTIVITY, KIND, Problem, SharedConfig, activity, sign, verify)

LOG = logging.getLogger(__name__)
PENDING = ('queued', *ACTIVE)


def public(run):
    return {key: run.get(key) for key in ('id', 'simulation_id', 'robot_identity', 'state',
        'created_at', 'ready_at', 'expires_at', 'idle_deadline', 'reason', 'fps', 'idle_seconds',
        'turn_seconds', 'policy', 'generation', 'discovery')}


class SharedService:
    def __init__(self, store, env, catalog, room_factory=LiveRoom):
        self.store, self.env, self.catalog = store, env, catalog
        self.config = SharedConfig.from_env(env)
        self.capacity = len(SUPPORTED)
        self.key = env['SIM_SUPERVISOR_KEY']
        self.room_name = env.get('SIM_LIVEKIT_ROOM_NAME', env.get('SIM_MODAL_APP', 'teleop')+'-simulations')
        if len(self.key) < 32 or not self.room_name or len(self.room_name) > 128:
            raise ValueError('Configure a server key and a shared simulation room.')
        self.room_factory = room_factory

    def room(self):
        return self.room_factory(self.env, self.room_name)

    def authorize(self, authorization, owner):
        if not hmac.compare_digest(authorization, f'Bearer {self.key}') or not KEY_PATTERN.fullmatch(owner):
            raise Problem(401, 'Unauthorized.')

    def read(self, identifier, ready=False):
        run = next((r for r in self.store.records() if r['id'] == identifier), None)
        if run is None:
            raise Problem(404, 'Simulation instance not found.')
        if ready and (run['state'] != 'ready' or now() >= min(run['expires_at'], run['idle_deadline'])):
            raise Problem(410, 'This simulation stopped. Choose Watch to start it again.')
        return run

    def simulations(self):
        runs = self.store.records()
        return [{**item, 'turn_seconds': self.config.turn, 'idle_seconds': self.config.idle,
            'instance': next((public(r) for r in reversed(runs)
                              if r['simulation_id'] == item['id'] and r['state'] in PENDING), None)}
                for item in self.catalog]

    def instance(self, identifier):
        return public(self.read(identifier))

    async def catalog_status(self):
        result = self.simulations()
        if any(item['instance'] and item['instance']['state'] == 'ready' for item in result):
            from .shared_rules import QUEUE_ROBOT, engaged
            async with self.room() as room:
                peers = await room.participants()
            for item in result:
                run = item['instance']
                if run and run['state'] == 'ready':
                    record = self.read(run['id'])
                    run['viewer_count'] = sum(engaged(p, record, now()) for p in peers)
                    run['queue_length'] = sum(p.attributes.get(QUEUE_ROBOT) == run['robot_identity'] for p in peers)
        return result

    def watch(self, owner, simulation):
        if simulation not in SUPPORTED:
            raise Problem(400, 'Unknown simulation.')
        with self.store.transaction() as tx:
            if tx.snapshot.get('admissions_paused'):
                raise Problem(503, 'Simulation updates are in progress. Try again shortly.')
            runs = tx.records()
            current = next((r for r in runs if r['simulation_id'] == simulation and r['state'] in PENDING), None)
            if current:
                return public(current)
            item = next(c for c in self.catalog if c['id'] == simulation)
            if not item['available']:
                raise Problem(409, item['reason'])
            recent = [r for r in runs if r['created_at'] > now()-3600000]
            if len(recent) >= int(self.env.get('SIM_LAUNCHES_PER_HOUR', 120)) or sum(
                    r['owner'] == owner for r in recent) >= int(self.env.get('SIM_VISITOR_LAUNCHES_PER_HOUR', 6)):
                raise Problem(429, 'The launch limit has been reached. Try again later.')
            identifier = uuid.uuid4().hex
            run = dict(id=identifier, generation=identifier, owner=owner, request_id=identifier,
                simulation_id=simulation, state='queued', created_at=now(), room_name=self.room_name,
                robot_identity=f'simulation-{simulation}-{identifier}', worker_key=secrets.token_hex(32),
                image_version=self.env.get('SIM_IMAGE_VERSION', 'local'), expires_at=None,
                token_until=now()+worker_ttl(self.config.fields())*1000+30000, **self.config.fields())
            tx.save(run)
        return public(run)

    def dispatch(self, provider):
        with self.store.transaction() as tx:
            selected = [r for r in tx.records() if r['state'] == 'queued']
            for run in selected:
                run.update(state='starting', admitted_at=now(), start_deadline=now()+
                    (run['startup_seconds']+30)*1000, dispatch_key=secrets.token_hex(16))
                tx.save(run)
        for run in selected:
            try:
                call = provider.submit(run['id'], run['dispatch_key'])
                with self.store.transaction() as tx:
                    current = tx.get(run['id'])
                    current['call_id'] = call
                    tx.save(current)
            except Exception:
                LOG.exception('instance=%s dispatch acknowledgement lost; no resubmission', run['id'])
        return [r['id'] for r in selected]

    def claim(self, identifier, dispatch_key, call_id, attempt=None):
        with self.store.transaction() as tx:
            run = tx.get(identifier)
            if not run or run['state'] != 'starting' or run.get('started_at') or \
                    run.get('dispatch_key') != dispatch_key or now() >= run['start_deadline']:
                return None
            run.update(started_at=now(), call_id=call_id, attempt_id=attempt or uuid.uuid4().hex,
                       hard_end=now()+worker_ttl(run)*1000)
            run['token_until'] = run['hard_end']+30000
            tx.save(run)
            return run

    def report(self, identifier, authorization, event):
        with self.store.transaction() as tx:
            run = tx.get(identifier)
            if not run or not hmac.compare_digest(authorization, 'Bearer '+run['worker_key']):
                raise Problem(401, 'Unauthorized worker.')
            if run['state'] not in ('starting', 'ready'):
                return dict(stop=True)
            if run['state'] == 'starting' and now() >= run['start_deadline']:
                stop(run, 'failed', 'Simulation startup timed out.')
            elif event.get('state') == 'ready':
                if accept_ready(run, event):
                    run.update(last_engaged_at=run['ready_at'], idle_deadline=run['ready_at']+run['idle_seconds']*1000)
                    run['discovery'] = event.get('discovery')
            elif event.get('state') in ('failed', 'stopped', 'expired'):
                stop(run, event['state'], 'Simulation execution failed.' if event['state'] == 'failed' else None)
            run['last_report_at'] = now()
            tx.save(run)
        return dict(stop=run['state'] == 'stopping', idle_deadline=run.get('idle_deadline'))

    def touch(self, identifier, at):
        with self.store.transaction() as tx:
            run = tx.get(identifier)
            if run['state'] != 'ready' or now() >= min(run['idle_deadline'], run['expires_at']):
                raise Problem(410, 'The simulation stopped due to inactivity.')
            run['last_engaged_at'] = max(run['last_engaged_at'], at)
            run['idle_deadline'] = run['last_engaged_at'] + run['idle_seconds']*1000
            tx.save(run)

    def room_credentials(self, owner):
        identity = 'demo-view-'+uuid.uuid4().hex
        credential = (api.AccessToken(self.env['LIVEKIT_API_KEY'], self.env['LIVEKIT_API_SECRET'])
            .with_identity(identity).with_ttl(timedelta(minutes=10))
            .with_attributes({KIND: 'viewer', 'lk.portal.role': 'observer', 'lk.portal.version': '3'})
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name, can_subscribe=True,
                can_publish=False, can_publish_data=False, can_update_own_metadata=False)).to_jwt())
        proof = sign(self.key, scope='room', identity=identity, room=self.room_name, owner=owner,
                     until=now()+21600000)
        return dict(identity=identity, token=credential, server_url=self.env['LIVEKIT_URL'],
                    room_name=self.room_name, proof=proof)

    def room_identity(self, proof, owner):
        return verify(self.key, proof, scope='room', room=self.room_name, owner=owner)['identity']

    def simulation_identity(self, proof, owner, run):
        return verify(self.key, proof, scope='instance', room=self.room_name, owner=owner,
                      instance=run['id'], generation=run['id'])['identity']

    def simulation_proof(self, run, owner, identity, at):
        return sign(self.key, scope='instance', room=self.room_name, owner=owner, identity=identity,
            instance=run['id'], generation=run['id'], until=run['expires_at'],
            engaged_until=at+run['idle_seconds']*1000)

    async def credentials(self, identifier, owner, body):
        run = self.read(identifier, ready=True)
        if body.get('resume'):
            identity = self.simulation_identity(body.get('proof'), owner, run)
            claims = verify(self.key, body['proof'])
            until = claims.get('engaged_until', 0)
            if until <= now():
                raise Problem(410, 'Continue watching to return.')
            at = until-run['idle_seconds']*1000
            async with self.room() as room:
                peer = await ControlQueue(room, run).viewer(identity)
                values = activity(peer)
                previous = values.get(identifier, {})
                at = max(at, previous.get('at', 0))
                values[identifier] = dict(generation=identifier, at=at, event='resume')
                await room.update(identity, {ACTIVITY: json.dumps(values, separators=(',', ':'))})
        else:
            identity = self.room_identity(body.get('proof'), owner)
            event = body.get('event_id', uuid.uuid4().hex)
            if not isinstance(event, str) or not KEY_PATTERN.fullmatch(event):
                raise Problem(400, 'Invalid activity event.')
            async with self.room() as room:
                at = await ControlQueue(room, run).engage(identity, event)
            self.touch(identifier, at)
        return dict(**public(self.read(identifier)), proof=self.simulation_proof(run, owner, identity, at),
                    identity=identity, engaged_until=at+run['idle_seconds']*1000)

    async def queue(self, identifier, owner, proof, action=None, resume=False):
        run = self.read(identifier, ready=True)
        identity = self.simulation_identity(proof, owner, run)
        async with self.room() as room:
            queue = ControlQueue(room, run)
            result = await queue.action(identity, action, engage_join=not resume) if action else await queue.reconcile(identity)
        if action == 'join' and not resume and result.get('engaged_until'):
            self.touch(identifier, result['engaged_until']-run['idle_seconds']*1000)
        if result.get('engaged_until'):
            result['proof'] = self.simulation_proof(run, owner, identity,
                result['engaged_until']-run['idle_seconds']*1000)
        return result

    async def engagement(self, identifier, owner, body):
        run = self.read(identifier, ready=True)
        identity = self.simulation_identity(body.get('proof'), owner, run)
        event = body.get('event_id')
        if not isinstance(event, str) or not KEY_PATTERN.fullmatch(event):
            raise Problem(400, 'Invalid activity event.')
        async with self.room() as room:
            at = await ControlQueue(room, run).engage(identity, event)
        self.touch(identifier, at)
        return dict(engaged_until=at+run['idle_seconds']*1000, server_time=now(),
                    proof=self.simulation_proof(run, owner, identity, at))

    async def heartbeat(self, identifier, authorization, event):
        result = self.report(identifier, authorization, event)
        run = self.read(identifier)
        if run['state'] == 'ready':
            async with self.room() as room:
                queue = ControlQueue(room, run)
                command = event.get('activity')
                if isinstance(command, dict) and isinstance(command.get('identity'), str):
                    try:
                        at = await queue.engage(command['identity'], command.get('event'), worker=True)
                        if at:
                            self.touch(identifier, at)
                    except Problem as error:
                        if error.status != 409:
                            raise
                await queue.reconcile()
            result['idle_deadline'] = self.read(identifier)['idle_deadline']
        self.deadlines()
        result['stop'] = self.read(identifier)['state'] not in ('starting', 'ready')
        return result

    async def cleanup(self, identifier):
        run = self.read(identifier)
        async with self.room() as room:
            await ControlQueue(room, run).reconcile(stopping=True)
            await room.remove(run['robot_identity'])

    def finish(self, identifier, code=None, failed=False):
        with self.store.transaction() as tx:
            run = tx.get(identifier)
            if not run or run.get('cleaned_at'):
                return
            terminal = run.get('terminal', 'failed' if failed else 'expired')
            run.update(state=terminal, cleaned_at=now(), exit_code=code)
            tx.save(run)

    def deadlines(self):
        with self.store.transaction() as tx:
            for run in tx.records():
                if run['state'] == 'starting' and now() >= run['start_deadline']:
                    stop(run, 'failed', 'Simulation startup timed out.')
                elif run['state'] == 'ready' and now() >= min(run['idle_deadline'], run['expires_at']):
                    stop(run, 'expired', 'Simulation paused due to inactivity.' if
                         now() >= run['idle_deadline'] else 'The simulation reached its runtime limit.')
                tx.save(run)
            tx.prune(now()-86400000, now())

    def admission(self, paused):
        with self.store.transaction() as tx:
            tx.snapshot['admissions_paused'] = paused
            if paused:
                for run in tx.records():
                    stop(run, 'stopped', 'Simulation updates are in progress.')
                    tx.save(run)
        return dict(paused=paused)

    async def handle(self, action, *args):
        if action == 'simulations':
            return await self.catalog_status()
        if action in ('credentials', 'queue', 'engagement', 'heartbeat', 'cleanup'):
            return await getattr(self, action)(*args)
        return getattr(self, action)(*args)
