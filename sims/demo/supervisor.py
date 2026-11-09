"""Local HTTP supervisor with persisted sessions and isolated workers."""
import argparse
import asyncio
import fcntl
import hmac
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

from aiohttp import web, ClientTimeout
from livekit import api

from ..assets import SOURCES
from ..livekit_auth import read_env_file
from ..registry import SIMULATIONS
from .files import read_json, write_json
from .provider import NativeProvider, DockerProvider

from .rules import (ACTIVE, KEY_PATTERN, SUPPORTED, SessionConfig, room_candidates,
                    public, token, worker_ttl, accept_ready, bounded)
from .queue import PENDING, ordered
LOG = logging.getLogger(__name__)


def now():
    return int(time.time() * 1000)


class Supervisor:
    def __init__(self, directory: Path, key: str, url: str, api_key: str, api_secret: str,
                 room_prefix='teleop-demo', capacity=2, provider=None, config=None):
        if len(key) < 32:
            raise ValueError('SIM_SUPERVISOR_KEY must contain at least 32 characters.')
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.lock = (self.directory / 'supervisor.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(self.directory / 'sessions.sqlite3')
        self.db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, owner TEXT, request TEXT, data TEXT, UNIQUE(owner,request))')
        self.key, self.url, self.api_key, self.api_secret = key, url, api_key, api_secret
        self.room_prefix, self.capacity = room_prefix, bounded(capacity, 'capacity', 1, 100, True)
        self.hourly = 30
        self.visitor_hourly = 6
        self.config = config or SessionConfig()
        self.launches = set()
        self.db.execute('CREATE TABLE IF NOT EXISTS rooms (name TEXT PRIMARY KEY, session TEXT UNIQUE)')
        # Keep legacy reservations too; never recycle a room with valid tokens.
        for run in self.records():
            name = run.get('room_name', run.get('room'))
            self.db.execute('INSERT OR IGNORE INTO rooms VALUES (?,?)', (name, run['id']))
        self.db.commit()
        self.provider = provider or NativeProvider()
        self.lk = None
        self.guard = asyncio.Lock()
        self.task = None
        self.closing = False

    def records(self, owner=None):
        rows = self.db.execute('SELECT data FROM runs' + (' WHERE owner=?' if owner else ''),
                               (owner,) if owner else ())
        return [json.loads(row[0]) for row in rows]

    def save(self, run):
        self.db.execute('INSERT INTO runs VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data',
                        (run['id'], run['owner'], run['request_id'], json.dumps(run)))
        self.db.commit()

    def owned(self, identifier, owner):
        row = self.db.execute('SELECT data FROM runs WHERE id=? AND owner=?', (identifier, owner)).fetchone()
        if row is None:
            raise web.HTTPNotFound(text='Simulation session not found.')
        return json.loads(row[0])

    def public(self, run):
        return public(run, self.records(), self.capacity)

    def catalog(self):
        result = []
        for spec in (SIMULATIONS[name] for name in SUPPORTED):
            reason = None
            if not spec.managed_module:
                reason = 'Generalized control is not available for this simulation.'
            elif not all(SOURCES[name].installed() for name in spec.assets):
                reason = 'Model assets are missing. Run the asset download command.'
            result.append({'id': spec.name, 'summary': spec.summary, 'icon': spec.icon,
                           'available': reason is None, 'reason': reason,
                           'duration_seconds': self.config.duration, 'fps': self.config.fps})
        return result

    def token(self, run, identity, role, ttl):
        return token(run, identity, role, ttl, self.api_key, self.api_secret)

    def reserve_room(self, identifier):
        for name in room_candidates(identifier):
            try:
                self.db.execute('INSERT INTO rooms VALUES (?,?)', (name, identifier))
                self.db.commit()
                return name
            except sqlite3.IntegrityError:
                self.db.rollback()
        raise web.HTTPServiceUnavailable(text='No trial room name is available.')

    async def launch(self, owner, simulation, request_id):
        if simulation not in SUPPORTED:
            raise web.HTTPBadRequest(text='Unknown simulation.')
        if not KEY_PATTERN.fullmatch(request_id):
            raise web.HTTPBadRequest(text='A request ID with 16 to 100 letters, digits or dashes is required.')
        async with self.guard:
            runs = self.records()
            previous = next((r for r in runs if r['owner'] == owner and r['request_id'] == request_id), None)
            if previous:
                if previous['simulation_id'] != simulation:
                    raise web.HTTPConflict(text='This request ID belongs to another simulation.')
                return previous
            item = next(c for c in self.catalog() if c['id'] == simulation)
            if not item['available']:
                raise web.HTTPConflict(text=item['reason'])
            if sum(r['state'] == 'queued' for r in runs) >= 200:
                raise web.HTTPTooManyRequests(text='The simulation queue is full. Try again shortly.')
            if any(r['owner'] == owner and r['state'] in PENDING for r in runs):
                raise web.HTTPConflict(text='Stop your active simulation before starting another.')
            recent = [r for r in runs if r['created_at'] > now() - 3600000]
            if len(recent) >= self.hourly or sum(r['owner'] == owner for r in recent) >= self.visitor_hourly:
                raise web.HTTPTooManyRequests(text='The launch limit has been reached. Try again later.')
            identifier = uuid.uuid4().hex
            run = dict(id=identifier, owner=owner, request_id=request_id, simulation_id=simulation,
                       room_name=self.reserve_room(identifier), robot_identity=f'simulation-{identifier}',
                       state='queued', created_at=now(), queued_at=now(), queue_expires_at=now()+900000,
                       queue_sequence=max((r.get('queue_sequence', 0) for r in runs), default=0)+1,
                       expires_at=None, pid=None,
                       reason=None, **self.config.fields())
            run['room'] = run['room_name']  # Compatibility with old local records and tools.
            self.save(run)
            directory = self.directory / identifier
            directory.mkdir(mode=0o700)
            self.dispatch()
            return self.owned(identifier, owner)

    def dispatch(self):
        if self.closing:
            return
        runs = self.records()
        free = max(0, self.capacity - sum(r['state'] in ACTIVE for r in runs))
        for run in ordered(runs)[:free]:
            run.update(state='starting', admitted_at=now())
            self.save(run)
            task = asyncio.create_task(self.start_worker(run, self.directory / run['id']))
            self.launches.add(task)
            task.add_done_callback(self.launches.discard)
            LOG.info('session=%s admitted from queue', run['id'])

    async def start_worker(self, run, directory):
        try:
            existing = await self.lk.room.list_rooms(api.ListRoomsRequest(names=[run['room_name']]))
            if existing.rooms:
                raise RuntimeError('Reserved room is already in use.')
            await self.lk.room.create_room(api.CreateRoomRequest(name=run['room_name'], empty_timeout=10))
            current = self.owned(run['id'], run['owner'])
            current['room_created'] = True
            self.save(current)
            run = current
            if run['state'] != 'starting':
                return
            token = self.token(run, run['robot_identity'], 'robot', worker_ttl(run))
            pid = await self.provider.start(run, directory, token, self.url, run['cleanup_seconds'])
            run = self.owned(run['id'], run['owner'])
            run['pid'] = pid
            if self.closing:
                self.begin_stop(run, 'stopped')
            self.save(run)
        except Exception as error:
            LOG.exception('session=%s simulation startup failed', run['id'])
            run = self.owned(run['id'], run['owner'])
            self.begin_stop(run, 'failed', 'Simulation startup failed. Make sure that LiveKit and the model assets are available.')

    def begin_stop(self, run, terminal='stopped', reason=None):
        if run['state'] == 'queued':
            run.update(state=terminal, reason=reason)
            self.save(run)
            return
        if run['state'] not in ACTIVE or run['state'] == 'stopping':
            return
        run.update(state='stopping', terminal=terminal, stop_at=now(), reason=reason or run.get('reason'))
        write_json(self.directory / run['id'] / 'control.json', {'stop': True})
        self.save(run)

    async def tick(self):
        for run in self.records():
            if run['state'] == 'queued':
                if now() >= run.get('queue_expires_at', now()+1):
                    self.begin_stop(run, 'expired', 'Your place in the queue expired. Launch again.')
                continue
            if run['state'] not in ACTIVE:
                continue
            event = read_json(self.directory / run['id'] / 'event.json')
            alive = bool(run.get('pid')) and self.provider.status(run)
            if run['state'] == 'starting':
                if event.get('state') == 'ready':
                    accept_ready(run, event)
                    self.save(run)
                elif now() - run.get('admitted_at', run['created_at']) > (run.get('startup_seconds', 45) + 2) * 1000:
                    self.begin_stop(run, 'failed', 'Simulation startup timed out.')
            if run['state'] == 'ready' and now() >= run['expires_at']:
                self.begin_stop(run, 'expired')
            if run['state'] in ('starting', 'ready') and (event.get('state') in ('failed', 'stopped', 'expired') or (run.get('pid') and not alive)):
                # Do not return raw worker errors to clients. The local log keeps details.
                terminal = 'expired' if event.get('state') == 'expired' else 'failed'
                self.begin_stop(run, terminal, None if terminal == 'expired' else 'The simulation stopped. Read its worker log for details.')
            if run['state'] == 'stopping':
                if alive and now() - run['stop_at'] >= run['cleanup_seconds'] * 1000:
                    self.provider.stop(run)
                try:
                    if run.get('room_created', 'room_name' not in run):
                        await self.lk.room.delete_room(api.DeleteRoomRequest(room=run.get('room_name', run['room'])))
                except Exception as error:
                    if getattr(error, 'code', '') != 'not_found':
                        continue  # Retry room cleanup even if the worker already exited.
                if not (run.get('pid') and self.provider.status(run)) and not any(
                        not t.done() for t in self.launches):
                    run['state'] = run['terminal']
                    child = getattr(self.provider, 'children', {}).pop(run['id'], None)
                    run['exit_code'] = child.returncode if child else None
                    self.save(run)
        self.dispatch()

    async def lifecycle(self):
        self.launches = set()
        self.lk = api.LiveKitAPI(self.url, self.api_key, self.api_secret, timeout=ClientTimeout(total=3))
        for run in self.records():
            if run['state'] in PENDING:
                if run['state'] == 'stopping':
                    write_json(self.directory / run['id'] / 'control.json', {'stop': True})
                else:
                    self.begin_stop(run, 'stopped', 'The local supervisor restarted.')
        async def monitor():
            while True:
                try:
                    await self.tick()
                except Exception:
                    LOG.exception('Simulation reconciliation failed')
                await asyncio.sleep(0.2)
        self.task = asyncio.create_task(monitor())

    async def close(self):
        self.closing = True
        await asyncio.gather(*self.launches, return_exceptions=True)
        for run in self.records():
            self.begin_stop(run)
        end = time.monotonic() + 8
        while any(r['state'] in ACTIVE for r in self.records()) and time.monotonic() < end:
            await self.tick()
            await asyncio.sleep(0.1)
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        await self.lk.aclose()
        self.db.close()
        self.lock.close()


def application(supervisor):
    @web.middleware
    async def access(request, handler):
        if not hmac.compare_digest(request.headers.get('Authorization', ''), f'Bearer {supervisor.key}'):
            raise web.HTTPUnauthorized()
        owner = request.headers.get('X-Demo-Owner', '')
        if not KEY_PATTERN.fullmatch(owner):
            raise web.HTTPUnauthorized()
        request['owner'] = owner
        try:
            return await handler(request)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text='Invalid request.')

    app = web.Application(middlewares=[access], client_max_size=8192)
    async def catalog(request):
        return web.json_response(supervisor.catalog())
    async def sessions(request):
        if request.method == 'GET':
            return web.json_response([supervisor.public(r) for r in supervisor.records(request['owner'])][-30:][::-1])
        body = await request.json()
        if set(body) != {'simulation_id', 'request_id'}:
            raise web.HTTPBadRequest(text='Only simulation_id and request_id are accepted.')
        run = await supervisor.launch(request['owner'], body['simulation_id'], body['request_id'])
        return web.json_response(supervisor.public(run), status=202)
    async def session(request):
        run = supervisor.owned(request.match_info['id'], request['owner'])
        if request.method == 'DELETE':
            supervisor.begin_stop(run)
        return web.json_response(supervisor.public(run))
    async def credentials(request):
        run = supervisor.owned(request.match_info['id'], request['owner'])
        if run['state'] != 'ready' or now() >= run['expires_at']:
            raise web.HTTPGone(text='This simulation is not ready or its session ended.')
        body = await request.json()
        mode = body.get('mode')
        if set(body) != {'mode'} or mode not in ('preview', 'control'):
            raise web.HTTPBadRequest(text='Choose preview or control mode.')
        role = 'observer' if mode == 'preview' else 'operator'
        identity = f"demo-{'view' if mode == 'preview' else 'op'}-{run['id']}-{uuid.uuid4().hex}"
        ttl = max(1, math.ceil((run['expires_at'] - now()) / 1000))
        return web.json_response(dict(server_url=supervisor.url, token=supervisor.token(run, identity, role, ttl),
            robot_identity=run['robot_identity'], room_name=run.get('room_name', run['room']),
            duration_seconds=run.get('duration_seconds', 60), expires_at=run['expires_at'], mode=mode,
            policy=run['policy'] if mode == 'control' else None), headers={'Cache-Control': 'no-store'})
    app.router.add_get('/health', lambda _: web.json_response({'ok': True}))
    app.router.add_get('/simulations', catalog)
    app.router.add_get('/sessions', sessions)
    app.router.add_post('/sessions', sessions)
    app.router.add_get('/sessions/{id}', session)
    app.router.add_delete('/sessions/{id}', session)
    app.router.add_post('/sessions/{id}/credentials', credentials)
    async def startup(_): await supervisor.lifecycle()
    async def shutdown(_): await supervisor.close()
    app.on_startup.append(startup)
    app.on_cleanup.append(shutdown)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=Path('.demo'))
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--provider', choices=('native', 'docker'), default='native')
    parser.add_argument('--image', default='teleop-simulation:local')
    args = parser.parse_args()
    values = {**read_env_file(args.env_file), **os.environ}
    supervisor = Supervisor(args.directory, values['SIM_SUPERVISOR_KEY'], values['LIVEKIT_URL'],
                            values['LIVEKIT_API_KEY'], values['LIVEKIT_API_SECRET'],
                            values.get('LIVEKIT_ROOM_NAME', 'teleop-demo'),
                            provider=DockerProvider(args.image) if args.provider == 'docker' else None,
                            config=SessionConfig.from_env(values),
                            capacity=int(values.get('SIM_SESSION_CAPACITY', 2)))
    supervisor.hourly = bounded(values.get('SIM_LAUNCHES_PER_HOUR', 30), 'hourly launches', 1, 10000, True)
    supervisor.visitor_hourly = bounded(values.get('SIM_VISITOR_LAUNCHES_PER_HOUR', 6), 'visitor launches', 1, 100, True)
    web.run_app(application(supervisor), host='127.0.0.1', port=args.port, access_log=None)


if __name__ == '__main__':
    main()
