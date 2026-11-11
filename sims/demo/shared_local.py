"""Run the same shared-room service locally with native or Docker workers."""
import argparse
import asyncio
import fcntl
import json
import logging
import os
from pathlib import Path
import sqlite3
import time

from aiohttp import web

from ..livekit_auth import read_env_file
from .check_image import catalog
from .files import read_json, write_json
from .hosted_store import ModalStore
from .provider import NativeProvider, DockerProvider
from .rules import token, worker_ttl
from .shared import SharedService
from .shared_http import resolve
from .shared_rules import Problem

LOG = logging.getLogger(__name__)


class LocalDictionary:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (directory / 'shared.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(directory / 'shared.sqlite3')
        self.db.execute('CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, value TEXT)')

    def get(self, name, default=None):
        row = self.db.execute('SELECT value FROM snapshots WHERE name=?', (name,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, name, value):
        self.db.execute('INSERT INTO snapshots VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value',
                        (name, json.dumps(value)))
        self.db.commit()


class LocalService:
    def __init__(self, directory, env, provider=None):
        self.directory, self.env = Path(directory), env
        self.dictionary = LocalDictionary(self.directory)
        self.service = SharedService(ModalStore(self.dictionary, 'shared-v1', writable=True), env, catalog())
        self.provider = provider or NativeProvider()
        self.guard = asyncio.Lock()
        self.jobs = {}
        self.closing = False

    def submit(self, identifier, dispatch_key):
        self.jobs[identifier] = asyncio.create_task(self.execute(identifier, dispatch_key))
        return 'local-'+identifier

    async def execute(self, identifier, dispatch_key):
        async with self.guard:
            run = self.service.claim(identifier, dispatch_key, 'local-'+identifier)
        if run is None:
            return
        directory = self.directory / identifier
        directory.mkdir(mode=0o700, exist_ok=True)
        failed, code = False, None
        try:
            async with self.service.room() as room:
                await room.ensure()
            credential = token(run, run['robot_identity'], 'robot', worker_ttl(run),
                               self.env['LIVEKIT_API_KEY'], self.env['LIVEKIT_API_SECRET'])
            pid = await self.provider.start(run, directory, credential, self.env['LIVEKIT_URL'], run['cleanup_seconds'])
            async with self.guard:
                with self.service.store.transaction() as tx:
                    current = tx.get(identifier)
                    current['pid'] = run['pid'] = pid
                    tx.save(current)
            end = time.monotonic()+worker_ttl(run)
            while self.provider.status(run):
                if time.monotonic() >= end:
                    failed = True
                    break
                try:
                    async with self.guard:
                        response = await self.service.heartbeat(identifier, 'Bearer '+run['worker_key'],
                                                               read_json(directory / 'event.json'))
                    write_json(directory / 'control.json', response)
                except Exception as error:
                    LOG.warning('instance=%s heartbeat failed: %s', identifier, error)
                await asyncio.sleep(.5)
            self.provider.stop(run)
            child = getattr(self.provider, 'children', {}).get(identifier)
            if child:
                await child.wait()
                code = child.returncode
        except Exception:
            LOG.exception('instance=%s failed', identifier)
            failed = True
        finally:
            if run.get('pid'):
                self.provider.stop(run)
                child = getattr(self.provider, 'children', {}).get(identifier)
                if child:
                    await child.wait()
            try:
                async with self.guard:
                    await self.service.cleanup(identifier)
                    self.service.finish(identifier, code, failed)
            except Exception:
                LOG.exception('instance=%s awaits cleanup recovery', identifier)

    async def monitor(self):
        while True:
            try:
                async with self.guard:
                    self.service.deadlines()
                    for run in self.service.store.records():
                        if run['state'] not in ('starting', 'ready', 'stopping'):
                            continue
                        job = self.jobs.get(run['id'])
                        if job and not job.done():
                            if run['state'] == 'stopping':
                                write_json(self.directory / run['id'] / 'control.json', {'stop': True})
                                if time.time()*1000-run['stop_at'] > run['cleanup_seconds']*1000:
                                    self.provider.stop(run)
                            continue
                        if run.get('pid') and self.provider.status(run):
                            self.provider.stop(run)
                            continue
                        # A restarted local process cannot still initialize an unclaimed worker.
                        await self.service.cleanup(run['id'])
                        self.service.finish(run['id'], failed=run['state'] != 'stopping')
                    if not self.closing:
                        self.service.dispatch(self)
            except Exception as error:
                LOG.warning('Shared simulation recovery failed: %s', error)
                await asyncio.sleep(2)
            await asyncio.sleep(.5)

    async def close(self):
        self.closing = True
        async with self.guard:
            self.service.admission(True)
        try:
            await asyncio.wait_for(asyncio.gather(*self.jobs.values(), return_exceptions=True), 12)
        finally:
            self.monitor_task.cancel()
            await asyncio.gather(self.monitor_task, return_exceptions=True)
            self.dictionary.db.close()
            self.dictionary.lock.close()


def application(local):
    @web.middleware
    async def access(request, handler):
        try:
            local.service.authorize(request.headers.get('Authorization', ''), request.headers.get('X-Demo-Owner', ''))
            result = await handler(request)
            result.headers['Cache-Control'] = 'no-store'
            return result
        except Problem as error:
            return web.Response(text=error.message, status=error.status)
        except (ValueError, TypeError):
            return web.Response(text='Invalid request.', status=400)
    app = web.Application(middlewares=[access], client_max_size=8192)

    async def endpoint(request):
        if request.path == '/health':
            return web.json_response(dict(ok=True, shared=True, simulation_capacity=7))
        body = await request.json() if request.can_read_body else {}
        action, args = resolve(request.method, request.path.lstrip('/'), request.query,
            body, request.headers.get('X-Sim-Proof', ''), request.headers.get('X-Demo-Owner', ''))
        async with local.guard:
            result = await local.service.handle(action, *args)
            if action == 'watch':
                local.service.dispatch(local)
        return web.json_response(result)
    app.router.add_route('*', '/{path:.*}', endpoint)
    async def start(_):
        # Resume admission after an intentional local launcher restart.
        local.service.admission(False)
        local.monitor_task = asyncio.create_task(local.monitor())
    async def close(_):
        await local.close()
    app.on_startup.append(start)
    app.on_cleanup.append(close)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=Path('.demo'))
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--provider', choices=('native', 'docker'), default='native')
    parser.add_argument('--image', default='teleop-simulation:local')
    args = parser.parse_args()
    env = {**read_env_file(args.env_file), **os.environ}
    logging.basicConfig(level=logging.INFO)
    local = LocalService(args.directory, env, DockerProvider(args.image) if args.provider == 'docker' else None)
    web.run_app(application(local), host='127.0.0.1', port=args.port, access_log=None)
