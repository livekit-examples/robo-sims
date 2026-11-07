"""Isolated worker with a startup bound and a deadline independent of its parent."""
import argparse
import asyncio
import importlib
import os
import signal
import time
from pathlib import Path

from ..registry import SIMULATIONS
from .files import read_json, write_json


async def serve(args):
    directory = Path(args.directory)
    running = None
    gateway = None
    reason = 'stopped'
    try:
        spec = SIMULATIONS[args.simulation]
        if not spec.managed_module:
            raise ValueError('This simulation does not support managed discovery.')
        module = importlib.import_module(spec.managed_module)
        gateway, policy = module.create(spec, os.environ['LIVEKIT_URL'], os.environ['LIVEKIT_TOKEN'],
                                        f'demo-op-{args.id}-')
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, gateway.stopping.set)
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
        expires_at = ready_at + 60_000
        deadline = loop.time() + 60
        write_json(directory / 'event.json', {'state': 'ready', 'ready_at': ready_at,
                                             'expires_at': expires_at, 'policy': policy})
        while not running.done():
            if loop.time() >= deadline:
                reason = 'expired'
                break
            if read_json(directory / 'control.json').get('stop'):
                break
            await asyncio.sleep(min(0.1, max(0, deadline - loop.time())))
        gateway.expired = True
        gateway._disarm()
        gateway.stopping.set()
        await asyncio.wait_for(asyncio.shield(running), args.cleanup)
        write_json(directory / 'event.json', {'state': reason})
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
        write_json(directory / 'event.json', {'state': 'failed', 'reason': str(error) or type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--simulation', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--startup', type=float, default=45)
    parser.add_argument('--cleanup', type=float, default=5)
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()
    if args.log:
        stream = (Path(args.directory) / 'worker.log').open('ab', buffering=0)
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    # Bound native/rendering failures even if the asyncio loop or cleanup stalls.
    def hard_limit(_sig, _frame):
        os._exit(124)
    signal.signal(signal.SIGALRM, hard_limit)
    signal.alarm(int(args.startup + 60 + args.cleanup + 2))
    asyncio.run(serve(args))


if __name__ == '__main__':
    main()
