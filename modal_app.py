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
          'NVIDIA_DRIVER_CAPABILITIES': 'all', 'SIM_IMAGE_VERSION': IMAGE_VERSION, 'SIM_MODAL_APP': APP_NAME,
          'SIM_MODAL_SECRET': os.environ.get('SIM_MODAL_SECRET', 'teleop-trials-dev')})
    .workdir('/app')
    .add_local_dir(ROOT / 'sims', '/app/sims', copy=True, ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'assets', '/app/assets', copy=True)
    .run_commands('python -m sims.demo.check_image'))


def service(writable=False):
    from sims.demo.shared import SharedService
    from sims.demo.hosted_store import ModalStore
    from sims.demo.check_image import catalog
    from urllib.parse import urlparse
    parsed = urlparse(os.environ['LIVEKIT_URL'])
    if parsed.scheme != 'wss' or not parsed.hostname or parsed.hostname in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('LIVEKIT_URL must be a reachable cloud URL.')
    dictionary = modal.Dict.from_name(os.environ.get('SIM_STATE_DICT', APP_NAME + '-sessions'), create_if_missing=True)
    return SharedService(ModalStore(dictionary, 'shared-v1', writable=writable), os.environ, catalog())


class ModalProvider:
    def submit(self, identifier, dispatch_key):
        return run_simulation.spawn(identifier, dispatch_key).object_id


@app.function(image=image, secrets=[secret], max_containers=1, timeout=120, retries=0)
def coordinate(action: str, *args):
    """Only writer. Never enable concurrent inputs or overlapping coordinator deployments."""
    import asyncio
    import logging
    from sims.demo.shared_rules import Problem
    logging.basicConfig(level=logging.INFO)
    svc = service(writable=True)
    try:
        if action == 'dispatch':
            svc.deadlines()
            result = svc.dispatch(ModalProvider())
        else:
            result = asyncio.run(svc.handle(action, *args))
        return {'result': result}
    except Problem as error:
        return {'error': error.message, 'status': error.status}


def command(action, *args):
    from sims.demo.shared_rules import Problem
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


# No warm workers. Each invocation owns one instance until process cleanup.
@app.function(image=image, secrets=[secret], gpu='L4', cpu=2, memory=4096,
              max_containers=7, timeout=22200, retries=0, scaledown_window=2)
def run_simulation(identifier: str, dispatch_key: str):
    import asyncio
    import logging
    import tempfile
    import time
    import uuid
    from sims.demo.files import read_json, write_json
    from sims.demo.provider import NativeProvider
    from sims.demo.rules import token, worker_ttl
    logging.basicConfig(level=logging.INFO)
    run = command('claim', identifier, dispatch_key, modal.current_function_call_id(), uuid.uuid4().hex)
    if run is None:
        return {'code': None, 'skipped': True}
    run['stream_logs'] = True

    async def execute():
        provider = NativeProvider()
        failed, code = False, None
        try:
            async with service().room() as room:
                await room.ensure()
            with tempfile.TemporaryDirectory(prefix='shared-sim-') as directory:
                path = Path(directory)
                credential = token(run, run['robot_identity'], 'robot', worker_ttl(run),
                                   os.environ['LIVEKIT_API_KEY'], os.environ['LIVEKIT_API_SECRET'])
                run['pid'] = await provider.start(run, path, credential, os.environ['LIVEKIT_URL'], run['cleanup_seconds'])
                child = provider.children[identifier]
                end = time.monotonic() + worker_ttl(run)
                next_report = 0
                while child.returncode is None:
                    if time.monotonic() >= end:
                        failed = True
                        break
                    if time.monotonic() >= next_report:
                        try:
                            response = await asyncio.to_thread(command, 'heartbeat', identifier,
                                'Bearer ' + run['worker_key'], read_json(path / 'event.json'))
                            write_json(path / 'control.json', response)
                        except Exception:
                            logging.exception('instance=%s heartbeat failed; child deadlines remain active', identifier)
                        next_report = time.monotonic() + 2
                    await asyncio.sleep(.2)
                provider.stop(run)
                await child.wait()
                code = child.returncode
        except BaseException:
            failed = True
            logging.exception('instance=%s execution failed', identifier)
        finally:
            if run.get('pid'):
                provider.stop(run)
                await provider.children[identifier].wait()
            try:
                await asyncio.to_thread(command, 'cleanup', identifier)
                await asyncio.to_thread(command, 'finish', identifier, code, failed)
            except Exception:
                logging.exception('instance=%s cleanup awaits scheduled recovery', identifier)
        return {'code': code, 'failed': failed}
    return asyncio.run(execute())


@app.function(image=image, secrets=[secret], schedule=modal.Period(minutes=1), timeout=120, retries=0)
def sweep():
    import logging
    import time
    command('deadlines')
    for run in service().store.records():
        if run['state'] not in ('starting', 'ready', 'stopping'):
            continue
        try:
            if run.get('call_id'):
                call = modal.FunctionCall.from_id(run['call_id'])
                try:
                    result = call.get(timeout=0)
                    done = True
                except TimeoutError:
                    result, done = None, False
                except (modal.exception.FunctionTimeoutError, modal.exception.RemoteError, modal.exception.OutputExpiredError):
                    result, done = None, True
                except Exception:
                    continue  # Unavailable status is not evidence of termination.
                if not done and run['state'] == 'stopping' and time.time()*1000-run['stop_at'] > run['cleanup_seconds']*1000:
                    call.cancel(terminate_containers=True)
                    continue  # A later pass must establish termination before releasing the slot.
            else:
                done = time.time()*1000 >= run['start_deadline']
                result = None
            if done:
                command('cleanup', run['id'])
                result = result if isinstance(result, dict) else {}
                command('finish', run['id'], result.get('code'), result.get('failed', run['state'] != 'stopping'))
        except Exception:
            logging.exception('instance=%s cleanup retry failed', run['id'])
    command('dispatch')


@app.function(image=image, secrets=[secret], timeout=120)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse
    from sims.demo.shared_rules import Problem
    from sims.demo.shared_http import resolve
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @web.exception_handler(Problem)
    async def problem(_request, error):
        return PlainTextResponse(error.message, status_code=error.status)

    async def endpoint(request):
        import asyncio
        svc = service()
        owner = request.headers.get('X-Demo-Owner', '')
        svc.authorize(request.headers.get('Authorization', ''), owner)
        raw = await request.body()
        if len(raw) > 8192:
            raise Problem(413, 'Request too large.')
        if request.url.path == '/health':
            svc.store.records()
            return {'ok': True, 'image_version': IMAGE_VERSION, 'simulation_capacity': 7, 'shared': True}
        try:
            body = await request.json() if raw else {}
            action, args = resolve(request.method, request.url.path.lstrip('/'), request.query_params,
                                   body, request.headers.get('X-Sim-Proof', ''), owner)
        except (ValueError, TypeError):
            raise Problem(400, 'Invalid request.') from None
        if action in ('simulations', 'instance', 'room_credentials'):
            result = await svc.handle(action, *args)
        else:
            result = await asyncio.to_thread(command, action, *args)
        if action == 'watch':
            request_dispatch()
        return result

    # FastAPI needs the actual class, not a local forward-reference annotation.
    endpoint.__annotations__['request'] = Request
    web.add_api_route('/{path:path}', endpoint, methods=['GET', 'POST'])
    @web.middleware('http')
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response
    return web
