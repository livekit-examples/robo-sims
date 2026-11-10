"""CLI runner, offline checks and previews for the vacuum participant."""
import asyncio
import json
import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backend import Backend
from .simulation import START_CONFIGS

LOG = logging.getLogger("sims.vacuum")


def add_arguments(parser, _sim):
    parser.add_argument("--start", choices=sorted(START_CONFIGS), default="normal",
                        help="starting battery and bin: normal, low_battery or nearly_full_bin (default: normal)")
    parser.add_argument("--auto-start", action="store_true",
                        help="start the timed run on the first driving command, for clients without a Start run button")
    parser.add_argument("--no-assist", action="store_true", help="turn off the slowing near walls and furniture")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--no-video", action="store_true", help="physics and telemetry without a graphics context")
    parser.add_argument("--video-stats", action="store_true")
    parser.add_argument("--benchmark", type=float, metavar="SECONDS", help="measure offline physics; no credentials or video")
    parser.add_argument("--route", action="store_true", help="drive the reference route offline and print the result")
    parser.add_argument("--preview", type=Path, metavar="PNG", help="save the overhead view without joining LiveKit")


def run(args, spec):
    return asyncio.run(serve(args, spec))


def check_report(backend) -> dict:
    sim = backend.sim
    areas = sim.floor.areas_m2()
    return {"discovery": backend.config.discovery | {"simulation_control": {
                "implementation": "vacuum.twist.v1", "policy": backend.policy()}},
            "simulation": {"room_m": [2.0, 2.0], "grid_cells": [200, 200], "cell_m": 0.01, "physics_hz": 200,
                           "areas_m2": areas, "start_configs": START_CONFIGS,
                           "data_tracks": {name: {"msg_type": kind, "format": "json", "rate_hz": rate}
                                           for name, (kind, _, rate) in backend.instrument_tracks().items()}}}


async def serve(args, spec):
    from .gateway import VacuumGateway
    if args.benchmark is not None and not 0 < args.benchmark <= 3600:
        raise ValueError("benchmark seconds must be finite and in (0, 3600]")
    if args.no_video and args.preview:
        raise ValueError("preview cannot combine with --no-video")
    video = not (args.no_video or args.benchmark is not None or args.route)
    backend = Backend(fps=args.fps, width=args.width, height=args.height, video=video or bool(args.preview),
                      start=args.start, assist=not args.no_assist, auto_start=args.auto_start)
    try:
        gateway = VacuumGateway(backend, "", "", args.fps, video=not args.no_video, video_stats=args.video_stats)
        if args.check:
            print(json.dumps(check_report(backend), indent=2))
            return 0
        if args.benchmark is not None:
            from .simulation import Command
            backend.start_run()
            backend.sim.requested = Command(0.3, 0.4)
            steps = round(args.benchmark / 0.005)
            started = time.perf_counter()
            for _ in range(steps):
                backend.step()
            elapsed = time.perf_counter() - started
            print(json.dumps({"simulation_seconds": backend.sim.time, "compute_seconds": elapsed,
                              "real_time_factor": backend.sim.time / elapsed}))
            return 0
        if args.route:
            from .route import drive_route
            print(json.dumps(drive_route(backend.sim)))
            return 0
        if args.preview:
            from PIL import Image
            with ThreadPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, backend.open_renderers)
                try:
                    pixels = await loop.run_in_executor(executor, backend.prepare("overhead").render)
                    args.preview.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pixels).save(args.preview)
                finally:
                    await loop.run_in_executor(executor, backend.close)
            print(f"Saved {args.preview}")
            return 0
        from ..livekit_auth import credentials
        creds = await credentials(args, spec.name)
        gateway.url, gateway.token = creds.url, creds.token
        LOG.info("Joining room %s as %s", creds.room, creds.identity or "supplied token identity")
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, gateway.stopping.set)
        await gateway.run()
        return 0
    finally:
        backend.close()
