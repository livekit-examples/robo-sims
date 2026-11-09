"""Isolated worker with a startup bound and a deadline independent of its parent."""
import argparse
import asyncio
import importlib
import base64
import json
import logging
import os
import signal
import time
from pathlib import Path

from ..registry import SIMULATIONS
from .files import read_json, write_json
from .rules import SessionConfig

LOG = logging.getLogger(__name__)


def verify_room():
    # The service signs the token. This check catches configuration mismatches.
    payload = os.environ["LIVEKIT_TOKEN"].split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    if claims["video"]["room"] != os.environ["LIVEKIT_ROOM_NAME"]:
        raise ValueError("Worker room does not match its token.")


class Reports:
    def __init__(self, directory, identifier):
        self.directory, self.identifier = directory, identifier
        self.url = os.environ.get("SIM_WORKER_API_URL")
        self.key = os.environ.get("SIM_WORKER_KEY")
        self.event = {"state": "starting"}

    def update(self, event):
        self.event = event
        LOG.info("session=%s event=%s", self.identifier, event["state"])
        if not self.url:
            write_json(self.directory / "event.json", event)

    async def exchange(self):
        if not self.url:
            return bool(read_json(self.directory / "control.json").get("stop"))
        from aiohttp import ClientSession, ClientTimeout
        async with ClientSession(timeout=ClientTimeout(total=2)) as client:
            async with client.post(f"{self.url.rstrip(chr(47))}/workers/{self.identifier}/events",
                    json=self.event, headers={"Authorization": f"Bearer {self.key}"}) as response:
                response.raise_for_status()
                return bool((await response.json()).get("stop"))

    async def monitor(self, gateway):
        while True:
            try:
                if await self.exchange():
                    gateway.stopping.set()
            except Exception:
                LOG.warning("session=%s report unavailable", self.identifier)
            await asyncio.sleep(.5)


async def serve(args):
    directory = Path(args.directory)
    running = None
    reporting = None
    reports = Reports(directory, args.id)
    verify_room()
    gateway = None
    reason = 'stopped'
    try:
        spec = SIMULATIONS[args.simulation]
        if not spec.managed_module:
            raise ValueError('This simulation does not support managed discovery.')
        module = importlib.import_module(spec.managed_module)
        gateway, policy = module.create(spec, os.environ['LIVEKIT_URL'], os.environ['LIVEKIT_TOKEN'],
                                        f'demo-op-{args.id}-', fps=args.fps)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, gateway.stopping.set)
        reporting = asyncio.create_task(reports.monitor(gateway))
        running = asyncio.create_task(gateway.run())
        ready = asyncio.create_task(gateway.ready.wait())
        done, _ = await asyncio.wait((ready, running), timeout=args.startup, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            await running
            raise RuntimeError('Simulation stopped before readiness.')
        if ready not in done:
            ready.cancel()
            raise TimeoutError('Simulation startup timed out.')
        ready_at = int(time.time() * 1000)
        expires_at = ready_at + round(args.duration * 1000)
        deadline = loop.time() + args.duration
        reports.update({'state': 'ready', 'ready_at': ready_at,
                                             'expires_at': expires_at, 'policy': policy})
        while not running.done():
            if loop.time() >= deadline:
                reason = 'expired'
                break
            if gateway.stopping.is_set():
                break
            await asyncio.sleep(min(0.1, max(0, deadline - loop.time())))
        gateway.expired = True
        gateway._disarm()
        gateway.stopping.set()
        await asyncio.wait_for(asyncio.shield(running), args.cleanup)
        reports.update({'state': reason})
        try:
            await reports.exchange()
        except Exception:
            pass
    except BaseException as error:
        if gateway:
            gateway.expired = True
            gateway._disarm()
            gateway.stopping.set()
        if running and not running.done():
            try:
                await asyncio.wait_for(asyncio.shield(running), args.cleanup)
            except BaseException:
                pass
        reports.update({'state': 'failed'})
        LOG.exception('session=%s worker failed', args.id)
        try:
            await reports.exchange()
        except Exception:
            pass
        raise
    finally:
        if reporting:
            reporting.cancel()
            await asyncio.gather(reporting, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--simulation', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--startup', type=float, default=45)
    parser.add_argument('--cleanup', type=float, default=5)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()
    try:
        SessionConfig(args.duration, args.fps, args.startup, args.cleanup)
    except ValueError as error:
        parser.error(str(error))
    Path(args.directory).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    if args.log:
        stream = (Path(args.directory) / 'worker.log').open('ab', buffering=0)
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    # Bound native/rendering failures even if the asyncio loop or cleanup stalls.
    def hard_limit(_sig, _frame):
        os._exit(124)
    signal.signal(signal.SIGALRM, hard_limit)
    signal.alarm(int(args.startup + args.duration + args.cleanup + 2))
    asyncio.run(serve(args))


if __name__ == '__main__':
    main()
