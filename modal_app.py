"""Deploy: modal deploy modal_app.py (run from robo-sims)."""
import os
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent
APP_NAME = os.environ.get('SIM_MODAL_APP', 'teleop-trials-dev')
IMAGE_VERSION = os.environ.get('SIM_IMAGE_VERSION', 'development')
app = modal.App(APP_NAME)
secret = modal.Secret.from_name(os.environ.get('SIM_MODAL_SECRET', 'teleop-trials-dev'))

# Build before traffic arrives. All scene types use one capped worker Function.
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('libegl1', 'libgl1', 'libopengl0', 'libglib2.0-0', 'libosmesa6')
    .pip_install_from_requirements(ROOT / 'requirements.txt')
    .pip_install('fastapi==0.162.2')
    .run_commands("mkdir -p /usr/share/glvnd/egl_vendor.d && printf '%s' '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    .env({'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl', 'PYTHONUNBUFFERED': '1',
          'NVIDIA_DRIVER_CAPABILITIES': 'all', 'SIM_IMAGE_VERSION': IMAGE_VERSION})
    .workdir('/app')
    .add_local_dir(ROOT / 'sims', '/app/sims', copy=True, ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'assets', '/app/assets', copy=True)
    .run_commands('python -m sims.demo.check_image'))


def service(writable=False):
    from sims.demo.hosted import HostedService
    from sims.demo.hosted_store import ModalStore
    from sims.demo.check_image import catalog
    from urllib.parse import urlparse
    parsed = urlparse(os.environ['LIVEKIT_URL'])
    if parsed.scheme != 'wss' or not parsed.hostname or parsed.hostname in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('LIVEKIT_URL must be a reachable cloud URL.')
    dictionary = modal.Dict.from_name(os.environ.get('SIM_STATE_DICT', APP_NAME + '-sessions'), create_if_missing=True)
    store = ModalStore(dictionary, 'sessions', writable=writable)
    return HostedService(store, os.environ, catalog())


class ModalProvider:
    def __init__(self):
        self.env = os.environ

    def room(self, run, deleting=False):
        import asyncio
        import json
        from aiohttp import ClientTimeout
        from livekit import api
        async def operation():
            async with api.LiveKitAPI(self.env['LIVEKIT_URL'], self.env['LIVEKIT_API_KEY'],
                    self.env['LIVEKIT_API_SECRET'], timeout=ClientTimeout(total=4)) as lk:
                found = (await lk.room.list_rooms(api.ListRoomsRequest(names=[run['room_name']]))).rooms
                if found and found[0].metadata != json.dumps({'session': run['id']}):
                    if deleting:
                        return  # Never delete another service's room.
                    raise RuntimeError('Reserved room is already in use.')
                if deleting:
                    if found:
                        await lk.room.delete_room(api.DeleteRoomRequest(room=run['room_name']))
                elif not found:
                    await lk.room.create_room(api.CreateRoomRequest(name=run['room_name'],
                        metadata=json.dumps({'session': run['id']}), empty_timeout=90))
        asyncio.run(operation())

    def ensure_room(self, run):
        self.room(run)

    def delete_room(self, run):
        self.room(run, deleting=True)

    def submit(self, identifier, dispatch_key):
        return run_simulation.spawn(identifier, dispatch_key).object_id


@app.function(image=image, secrets=[secret], max_containers=1, timeout=60, retries=0)
def coordinate(action: str, *args):
    """Only writer. Keep one synchronous input/container; never add input concurrency."""
    import logging
    logging.basicConfig(level=logging.INFO)
    from sims.demo.hosted import Problem
    svc = service(writable=True)
    try:
        if action == 'launch':
            result = svc.launch(*args)
        elif action == 'stop':
            result = svc.session(*args, stopping=True)
        elif action == 'launcher':
            result = svc.stop_launcher(*args)
        elif action == 'report':
            result = svc.report(*args)
        elif action == 'claim':
            result = svc.claim(*args)
        elif action == 'finish':
            result = svc.finish(*args)
        elif action == 'dispatch':
            svc.deadlines()
            result = svc.dispatch(ModalProvider())
        elif action == 'deadlines':
            result = svc.deadlines()
        else:
            raise ValueError('Unknown coordinator action.')
        return {'result': result}
    except Problem as error:
        return {'error': error.message, 'status': error.status}


def command(action, *args):
    from sims.demo.hosted import Problem
    response = coordinate.remote(action, *args)
    if 'error' in response:
        raise Problem(response['status'], response['error'])
    return response['result']


def request_dispatch():
    import logging
    try:
        coordinate.spawn('dispatch')
    except Exception:
        logging.exception('Dispatch unavailable; scheduled recovery will retry.')


# This is ONE shared pool for ALL scenes, browsers and visitors. Do not split by
# scene or enable concurrent inputs: one invocation owns one simulation's lifetime.
@app.function(image=image, secrets=[secret], gpu='L4', cpu=2, memory=4096,
              max_containers=20, timeout=4200, retries=0, scaledown_window=2)
def run_simulation(identifier: str, dispatch_key: str):
    import asyncio
    import logging
    import tempfile
    import time
    from sims.demo.files import read_json, write_json
    from sims.demo.provider import NativeProvider
    from sims.demo.rules import token, worker_ttl
    logging.basicConfig(level=logging.INFO)
    call_id = modal.current_function_call_id()
    run = command('claim', identifier, dispatch_key, call_id)
    if run is None:
        return {'code': None, 'skipped': True}  # No room or physics for stale/duplicate calls.

    run['stream_logs'] = True  # Child stdout/stderr remain available as FunctionCall logs.
    logging.info('session=%s function=%s starting simulation=%s', identifier, call_id, run['simulation_id'])

    async def execute():
        provider = NativeProvider()
        room = ModalProvider()
        failed, code, cleaned = False, None, False
        try:
            await asyncio.to_thread(room.ensure_room, run)
            with tempfile.TemporaryDirectory(prefix='trial-') as directory:
                path = Path(directory)
                credential = token(run, run['robot_identity'], 'robot', worker_ttl(run),
                                   os.environ['LIVEKIT_API_KEY'], os.environ['LIVEKIT_API_SECRET'])
                run['pid'] = await provider.start(run, path, credential, os.environ['LIVEKIT_URL'], run['cleanup_seconds'])
                child = provider.children[identifier]
                end = time.monotonic() + worker_ttl(run)
                last_event = None
                while True:
                    event = read_json(path / 'event.json')
                    if event and event != last_event:
                        try:
                            response = await asyncio.to_thread(command, 'report', identifier,
                                'Bearer ' + run['worker_key'], event)
                            last_event = event
                            if response['stop']:
                                write_json(path / 'control.json', {'stop': True})
                        except Exception:
                            logging.exception('session=%s report unavailable; child deadline remains active', identifier)
                    if child.returncode is not None:
                        code = child.returncode
                        break
                    if time.monotonic() >= end:
                        failed = True
                        break
                    try:
                        current = next(r for r in service().store.records() if r['id'] == identifier)
                        if current['state'] not in ('starting', 'ready'):
                            write_json(path / 'control.json', {'stop': True})
                    except Exception:
                        logging.warning('session=%s stop-state unavailable', identifier)
                    await asyncio.sleep(.5)
                provider.stop(run)
                await child.wait()
                if code is None:
                    code = child.returncode
        except BaseException:
            failed = True
            logging.exception('session=%s simulation function failed', identifier)
        finally:
            # Kill/wait before freeing the logical slot. Never leave physics in a reused container.
            if run.get('pid'):
                provider.stop(run)
                await provider.children[identifier].wait()
            try:
                await asyncio.to_thread(room.delete_room, run)
                cleaned = True
            except Exception:
                logging.exception('session=%s room cleanup will be retried by sweep', identifier)
            if cleaned:
                await asyncio.to_thread(command, 'finish', identifier, code, failed)
                request_dispatch()
        return {'code': code, 'failed': failed}
    return asyncio.run(execute())


@app.function(image=image, secrets=[secret], schedule=modal.Period(minutes=1), timeout=120, retries=0)
def sweep():
    import logging
    import time
    from sims.demo.queue import RUNNING
    # Poll during this bounded invocation for responsive admission if a worker is lost.
    end = time.monotonic() + 50
    while True:
        command('deadlines')
        svc = service()
        for run in svc.store.records():
            if run['state'] not in RUNNING:
                continue
            try:
                if run.get('call_id'):
                    call = modal.FunctionCall.from_id(run['call_id'])
                    try:
                        code = call.get(timeout=0)
                        done = True
                    except TimeoutError:
                        code, done = None, False
                    except (modal.exception.FunctionTimeoutError, modal.exception.RemoteError, modal.exception.OutputExpiredError):
                        code, done = None, True
                    except Exception:
                        code, done = None, False  # An RPC failure does not prove execution stopped.
                    if not done and run['state'] == 'stopping' and time.time()*1000 - run['stop_at'] > run['cleanup_seconds']*1000:
                        call.cancel(terminate_containers=True)
                        done = True  # Explicitly terminate this single-input container before freeing its slot.
                else:
                    # A lost spawn ACK is never retried. A late worker must claim before starting.
                    done = time.time()*1000 >= run['start_deadline']
                    code = None
                if done:
                    ModalProvider().delete_room(run)
                    result = code if isinstance(code, dict) else {'code': code}
                    command('finish', run['id'], result.get('code'), result.get('failed', run['state'] != 'stopping'))
            except Exception:
                logging.exception('session=%s cleanup retry failed', run['id'])
        command('dispatch')
        if time.monotonic() >= end:
            return
        time.sleep(2)


@app.function(image=image, secrets=[secret], timeout=60)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    import logging
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel, ConfigDict
    from sims.demo.hosted import Problem
    logging.basicConfig(level=logging.INFO)
    svc = service()
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    class Launch(BaseModel):
        model_config = ConfigDict(extra='forbid')
        simulation_id: str
        request_id: str

    class Credentials(BaseModel):
        model_config = ConfigDict(extra='forbid')
        mode: str

    # Local classes cannot be resolved as forward references by FastAPI.
    def owner(request):
        identity = request.headers.get('X-Demo-Owner', '')
        svc.authorize(request.headers.get('Authorization', ''), identity)
        return identity

    @web.exception_handler(Problem)
    async def problem(_request, error):
        return PlainTextResponse(error.message, status_code=error.status)

    @web.exception_handler(ValueError)
    async def invalid(_request, _error):
        return PlainTextResponse('Invalid request.', status_code=400)

    @web.middleware('http')
    async def no_cache(request, call_next):
        try:
            length = int(request.headers.get('content-length', '0'))
        except ValueError:
            return PlainTextResponse('Invalid content length.', status_code=400)
        if length > 8192:
            return PlainTextResponse('Request too large.', status_code=413)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @web.get('/health')
    def health(request: Request):
        owner(request)
        svc.store.records()
        return {'ok': True, 'image_version': IMAGE_VERSION, 'simulation_capacity': 20}

    @web.get('/simulations')
    def simulations(request: Request):
        owner(request)
        return [{**item, 'duration_seconds': svc.config.duration, 'fps': svc.config.fps} for item in svc.catalog]

    @web.get('/sessions')
    def sessions(request: Request):
        return svc.list(owner(request))

    @web.post('/sessions', status_code=202)
    def launch(request: Request, body: Launch):
        run = command('launch', owner(request), body.model_dump(), request.headers.get('X-Demo-Launcher'))
        request_dispatch()
        return run

    @web.get('/sessions/{identifier}')
    def session(identifier: str, request: Request):
        return svc.session(identifier, owner(request))

    @web.delete('/sessions/{identifier}')
    def stop_session(identifier: str, request: Request):
        result = command('stop', identifier, owner(request))
        request_dispatch()
        return result

    @web.post('/sessions/{identifier}/credentials')
    def credentials(identifier: str, request: Request, body: Credentials):
        return svc.credentials(identifier, owner(request), body.model_dump())

    @web.post('/workers/{identifier}/events')
    def event(identifier: str, request: Request, body: dict):
        return command('report', identifier, request.headers.get('Authorization', ''), body)

    @web.delete('/launchers/{identifier}/sessions')
    def stop_launcher(identifier: str, request: Request):
        owner(request)
        command('launcher', identifier)
        request_dispatch()
        return {'ok': True}

    return web
