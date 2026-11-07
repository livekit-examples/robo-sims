"""Local HTTP supervisor with persisted sessions and isolated workers."""
import argparse
import asyncio
from datetime import timedelta
import fcntl
import hmac
import json
import logging
import math
import os
from pathlib import Path
import re
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

ACTIVE = ('starting', 'ready', 'stopping')
KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{16,100}$')
LOG = logging.getLogger(__name__)


def now():
    return int(time.time() * 1000)


class Supervisor:
    def __init__(self, directory: Path, key: str, url: str, api_key: str, api_secret: str,
                 room_prefix='teleop-demo', capacity=2, provider=None):
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
        self.room_prefix, self.capacity = room_prefix, capacity
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

    @staticmethod
    def public(run):
        return {k: run.get(k) for k in ('id', 'simulation_id', 'state', 'created_at', 'ready_at',
                                       'expires_at', 'reason', 'exit_code')}

    def catalog(self):
        result = []
        for spec in SIMULATIONS.values():
            reason = None
            if not spec.managed_module:
                reason = 'Generalized control is not available for this simulation.'
            elif not all(SOURCES[name].installed() for name in spec.assets):
                reason = 'Model assets are missing. Run the asset download command.'
            result.append({'id': spec.name, 'summary': spec.summary, 'icon': spec.icon,
                           'available': reason is None, 'reason': reason})
        return result

    def token(self, run, identity, role, ttl):
        return (api.AccessToken(self.api_key, self.api_secret).with_identity(identity)
                .with_attributes({'lk.portal.role': role, 'lk.portal.version': '3'})
                .with_ttl(timedelta(seconds=ttl))
                .with_grants(api.VideoGrants(room_join=True, room=run['room'], can_subscribe=True,
                    can_publish=role != 'observer', can_publish_data=True, can_update_own_metadata=True)).to_jwt())

    async def launch(self, owner, simulation, request_id):
        if simulation not in SIMULATIONS:
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
            if sum(r['state'] in ACTIVE for r in runs) >= self.capacity:
                raise web.HTTPTooManyRequests(text='All local simulation slots are in use.')
            if any(r['owner'] == owner and r['state'] == 'starting' for r in runs):
                raise web.HTTPConflict(text='Wait for your pending simulation to start.')
            identifier = uuid.uuid4().hex
            run = dict(id=identifier, owner=owner, request_id=request_id, simulation_id=simulation,
                       room=f'{self.room_prefix}-{identifier}', robot_identity=f'simulation-{identifier}',
                       state='starting', created_at=now(), expires_at=None, pid=None,
                       reason=None, cleanup_seconds=SIMULATIONS[simulation].cleanup_seconds)
            self.save(run)
            directory = self.directory / identifier
            directory.mkdir(mode=0o700)
            # Track startup tasks so shutdown cannot miss an in-flight launch.
            task = asyncio.create_task(self.start_worker(run, directory))
            self.launches.add(task)
            task.add_done_callback(self.launches.discard)
            return run

    async def start_worker(self, run, directory):
        try:
            await self.lk.room.create_room(api.CreateRoomRequest(name=run['room'], empty_timeout=10))
            token = self.token(run, run['robot_identity'], 'robot', 120)
            pid = await self.provider.start(run, directory, token, self.url, run['cleanup_seconds'])
            run = self.owned(run['id'], run['owner'])
            run['pid'] = pid
            if self.closing:
                self.begin_stop(run, 'stopped')
            self.save(run)
        except Exception as error:
            LOG.warning('Simulation startup failed (%s)', type(error).__name__)
            self.begin_stop(run, 'failed', 'Simulation startup failed. Make sure that LiveKit and the model assets are available.')

    def begin_stop(self, run, terminal='stopped', reason=None):
        if run['state'] not in ACTIVE or run['state'] == 'stopping':
            return
        run.update(state='stopping', terminal=terminal, stop_at=now(), reason=reason or run.get('reason'))
        write_json(self.directory / run['id'] / 'control.json', {'stop': True})
        self.save(run)

    async def tick(self):
        for run in self.records():
            if run['state'] not in ACTIVE:
                continue
            event = read_json(self.directory / run['id'] / 'event.json')
            alive = bool(run.get('pid')) and self.provider.status(run)
            if run['state'] == 'starting':
                if event.get('state') == 'ready':
                    run.update(state='ready', ready_at=event['ready_at'], expires_at=event['expires_at'],
                               policy=event['policy'])
                    self.save(run)
                elif now() - run['created_at'] > 47_000:
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
                    await self.lk.room.delete_room(api.DeleteRoomRequest(room=run['room']))
                except Exception as error:
                    if getattr(error, 'code', '') != 'not_found':
                        continue  # Retry room cleanup even if the worker already exited.
                if not (run.get('pid') and self.provider.status(run)):
                    run['state'] = run['terminal']
                    child = getattr(self.provider, 'children', {}).pop(run['id'], None)
                    run['exit_code'] = child.returncode if child else None
                    self.save(run)

    async def lifecycle(self):
        self.launches = set()
        self.lk = api.LiveKitAPI(self.url, self.api_key, self.api_secret, timeout=ClientTimeout(total=3))
        for run in self.records():
            if run['state'] in ACTIVE:
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
        if mode not in ('preview', 'control'):
            raise web.HTTPBadRequest(text='Choose preview or control mode.')
        role = 'observer' if mode == 'preview' else 'operator'
        identity = f"demo-{'view' if mode == 'preview' else 'op'}-{run['id']}-{uuid.uuid4().hex}"
        ttl = max(1, math.ceil((run['expires_at'] - now()) / 1000))
        return web.json_response(dict(server_url=supervisor.url, token=supervisor.token(run, identity, role, ttl),
            robot_identity=run['robot_identity'], expires_at=run['expires_at'], mode=mode,
            policy=run['policy'] if mode == 'control' else None), headers={'Cache-Control': 'no-store'})
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
                            provider=DockerProvider(args.image) if args.provider == 'docker' else None)
    web.run_app(application(supervisor), host='127.0.0.1', port=args.port, access_log=None)


if __name__ == '__main__':
    main()
