"""Modal session rules. Mutations are serialized by the coordinator Function."""
import hmac
import logging
import math
import secrets
import uuid

from .rules import (ACTIVE, SUPPORTED, KEY_PATTERN, SessionConfig, accept_ready,
                    now, public, room_candidates, token, worker_ttl, bounded)

from .queue import HOSTED_CAPACITY, PENDING, ordered

LOG = logging.getLogger(__name__)


class Problem(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def stop(run, terminal='stopped', reason=None):
    if run['state'] == 'queued':
        run.update(state=terminal, reason=reason, cleaned_at=now())
    elif run['state'] in ('starting', 'ready'):
        run.update(state='stopping', terminal=terminal, reason=reason, stop_at=now())
        LOG.info('session=%s stopping terminal=%s', run['id'], terminal)


class HostedService:
    def __init__(self, store, env, catalog):
        self.store, self.env, self.catalog = store, env, catalog
        self.config = SessionConfig.from_env(env)
        self.capacity = HOSTED_CAPACITY
        self.queue_limit = bounded(env.get('SIM_QUEUE_LIMIT', 200), 'queue limit', 1, 1000, True)
        self.queue_seconds = bounded(env.get('SIM_QUEUE_TIMEOUT_SECONDS', 900), 'queue timeout', 30, 3600, True)
        self.hourly = bounded(env.get('SIM_LAUNCHES_PER_HOUR', 120), 'hourly launches', 1, 10000, True)
        self.visitor_hourly = bounded(env.get('SIM_VISITOR_LAUNCHES_PER_HOUR', 20), 'visitor launches', 1, 100, True)
        self.key = env['SIM_SUPERVISOR_KEY']
        if len(self.key) < 32:
            raise ValueError('SIM_SUPERVISOR_KEY must have at least 32 characters.')

    def authorize(self, authorization, owner):
        if not hmac.compare_digest(authorization, f'Bearer {self.key}') or not KEY_PATTERN.fullmatch(owner):
            raise Problem(401, 'Unauthorized.')

    def owned(self, tx, identifier, owner):
        run = tx.get(identifier)
        if not run or run['owner'] != owner:
            raise Problem(404, 'Simulation session not found.')
        return run

    def launch(self, owner, body, launcher=None):
        if set(body) != {'simulation_id', 'request_id'}:
            raise Problem(400, 'Only simulation_id and request_id are accepted.')
        simulation, request = body['simulation_id'], body['request_id']
        if simulation not in SUPPORTED or not isinstance(request, str) or not KEY_PATTERN.fullmatch(request):
            raise Problem(400, 'Invalid simulation or request ID.')
        with self.store.transaction() as tx:
            runs = tx.records()
            previous = next((r for r in runs if r['owner'] == owner and r['request_id'] == request), None)
            if previous:
                if previous['simulation_id'] != simulation:
                    raise Problem(409, 'This request ID belongs to another simulation.')
                return public(previous, runs, self.capacity)
            item = next(c for c in self.catalog if c['id'] == simulation)
            if not item['available']:
                raise Problem(409, item['reason'])
            if sum(r['state'] == 'queued' for r in runs) >= self.queue_limit:
                raise Problem(429, 'The simulation queue is full. Try again shortly.')
            if any(r['owner'] == owner and r['state'] in PENDING for r in runs):
                raise Problem(409, 'Stop your active simulation before starting another.')
            recent = [r for r in runs if r['created_at'] > now() - 3600000]
            if len(recent) >= self.hourly or sum(r['owner'] == owner for r in recent) >= self.visitor_hourly:
                raise Problem(429, 'The launch limit has been reached. Try again later.')
            identifier = uuid.uuid4().hex
            run = dict(id=identifier, owner=owner, request_id=request, simulation_id=simulation,
                       state='queued', created_at=now(), queued_at=now(),
                       queue_expires_at=now() + self.queue_seconds * 1000,
                       queue_sequence=max((r.get('queue_sequence', 0) for r in runs), default=0) + 1,
                       expires_at=None, reason=None,
                       robot_identity=f'simulation-{identifier}', launcher=launcher,
                       worker_key=secrets.token_hex(32), image_version=self.env['SIM_IMAGE_VERSION'],
                       **self.config.fields())
            run['token_until'] = run['queue_expires_at'] + worker_ttl(run) * 1000 + 30000
            for name in room_candidates(identifier):
                if tx.reserve(name, identifier, run['token_until']):
                    run['room_name'] = name
                    break
            else:
                raise Problem(503, 'No trial room name is available.')
            tx.save(run)  # Durable outbox: reconciler also picks up un-dispatched starts.
        LOG.info('session=%s reserved room=%s', identifier, run['room_name'])
        result = public(run, self.store.records(), self.capacity)
        LOG.info('session=%s queued position=%s estimated_wait_seconds=%s', identifier,
                 result['queue_position'], result['estimated_wait_seconds'])
        return result

    def list(self, owner):
        runs = self.store.records()
        return [public(r, runs, self.capacity) for r in sorted(runs, key=lambda r: r['created_at'], reverse=True)
                if r['owner'] == owner][:30]

    def read_owned(self, identifier, owner):
        run = next((r for r in self.store.records() if r['id'] == identifier and r['owner'] == owner), None)
        if not run:
            raise Problem(404, 'Simulation session not found.')
        return run

    def session(self, identifier, owner, stopping=False):
        if stopping:
            with self.store.transaction() as tx:
                run = self.owned(tx, identifier, owner)
                stop(run)
                tx.save(run)
        else:
            run = self.read_owned(identifier, owner)
        return public(run, self.store.records(), self.capacity)

    def credentials(self, identifier, owner, body):
        run = self.read_owned(identifier, owner)
        if run['state'] != 'ready' or now() >= run['expires_at']:
            raise Problem(410, 'This simulation is not ready or its session ended.')
        if set(body) != {'mode'} or body['mode'] not in ('preview', 'control'):
            raise Problem(400, 'Choose preview or control mode.')
        mode = body['mode']
        identity = f"demo-{'view' if mode == 'preview' else 'op'}-{identifier}-{uuid.uuid4().hex}"
        ttl = max(1, math.ceil((run['expires_at'] - now()) / 1000))
        credential = token(run, identity, 'observer' if mode == 'preview' else 'operator', ttl,
                           self.env['LIVEKIT_API_KEY'], self.env['LIVEKIT_API_SECRET'])
        return dict(server_url=self.env['LIVEKIT_URL'], token=credential, room_name=run['room_name'],
                    robot_identity=run['robot_identity'], expires_at=run['expires_at'], mode=mode,
                    duration_seconds=run['duration_seconds'], policy=run['policy'] if mode == 'control' else None)

    def report(self, identifier, authorization, event):
        with self.store.transaction() as tx:
            run = tx.get(identifier)
            if not run or not hmac.compare_digest(authorization, f"Bearer {run['worker_key']}"):
                raise Problem(401, 'Unauthorized worker.')
            if run['state'] not in ('starting', 'ready'):
                return {'stop': True}
            if run['state'] == 'starting' and now() > run.get('start_deadline', 0):
                stop(run, 'failed', 'Simulation startup timed out.')
                tx.save(run)
                return {'stop': True}
            state = event.get('state')
            if state == 'ready':
                if accept_ready(run, event):
                    LOG.info('session=%s ready expires_at=%s', identifier, run['expires_at'])
            elif state in ('stopped', 'expired', 'failed'):
                stop(run, state, 'Simulation execution failed.' if state == 'failed' else None)
            elif state != 'starting':
                raise Problem(400, 'Invalid worker event.')
            run['last_report_at'] = now()
            tx.save(run)
            return {'stop': run['state'] == 'stopping'}

    def stop_launcher(self, launcher):
        if not KEY_PATTERN.fullmatch(launcher):
            raise Problem(400, 'Invalid launcher ID.')
        with self.store.transaction() as tx:
            for run in tx.records():
                if run.get('launcher') == launcher:
                    stop(run)
                    tx.save(run)

    def dispatch(self, provider):
        """Reserve one shared pool before submitting calls; never re-submit an ambiguous call."""
        with self.store.transaction() as tx:
            runs = tx.records()
            for run in ordered(runs):
                if now() >= run['queue_expires_at']:
                    stop(run, 'expired', 'Your place in the queue expired. Launch again.')
                    tx.save(run)
            free = max(0, self.capacity - sum(r['state'] in ACTIVE for r in runs))
            selected = ordered(runs)[:free]
            for run in selected:
                run.update(state='starting', admitted_at=now(), start_deadline=now() +
                           (run['startup_seconds'] + 30) * 1000, dispatch_key=secrets.token_hex(16))
                tx.save(run)
        for run in selected:
            # State is committed first. A lost acknowledgment is recovered by the worker's claim.
            try:
                call_id = provider.submit(run['id'], run['dispatch_key'])
                with self.store.transaction() as tx:
                    current = tx.get(run['id'])
                    current['call_id'] = call_id
                    tx.save(current)
                LOG.info('session=%s admitted call=%s', run['id'], call_id)
            except Exception:
                LOG.exception('session=%s dispatch acknowledgment lost; will not submit twice', run['id'])
        return [r['id'] for r in selected]

    def claim(self, identifier, dispatch_key, call_id):
        """A delayed or duplicate function invocation cannot start another simulation."""
        with self.store.transaction() as tx:
            run = tx.get(identifier)
            if not run or run['state'] != 'starting' or run.get('started_at') or \
                    run.get('dispatch_key') != dispatch_key or now() >= run['start_deadline']:
                return None
            run.update(started_at=now(), call_id=call_id)
            run['hard_end'] = now() + worker_ttl(run) * 1000
            run['token_until'] = max(run['token_until'], run['hard_end'] + 30000)
            tx.save(run)
            return run

    def finish(self, identifier, code=None, failed=False):
        """Called only after the worker process is gone and the room is removed."""
        with self.store.transaction() as tx:
            run = tx.get(identifier)
            if not run or run.get('cleaned_at'):
                return
            if failed:
                stop(run, 'failed', 'Simulation startup or execution failed.')
                run['terminal'] = 'failed'
            elif run['state'] in ('starting', 'ready'):
                expired = run.get('expires_at') is not None and now() >= run['expires_at']
                stop(run, 'expired' if expired else 'failed', None if expired else 'The simulation worker exited.')
            run.update(state=run.get('terminal', 'stopped'), cleaned_at=now(), exit_code=code)
            run.pop('policy', None)
            tx.save(run)
        LOG.info('session=%s cleaned state=%s', identifier, run['state'])

    def deadlines(self):
        with self.store.transaction() as tx:
            for run in tx.records():
                if run['state'] == 'queued' and now() >= run['queue_expires_at']:
                    stop(run, 'expired', 'Your place in the queue expired. Launch again.')
                elif run['state'] == 'starting' and now() >= run['start_deadline']:
                    stop(run, 'failed', 'Simulation startup timed out.')
                elif run['state'] == 'ready' and now() >= run['expires_at']:
                    stop(run, 'expired')
                tx.save(run)
            tx.prune(now() - 86400000, now())
