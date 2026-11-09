"""CLI runner, offline benchmarks and previews for the RC racing participant."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import signal
import sys
import time

from .backend import Backend

LOG = logging.getLogger("sims.f1tenth")


def add_arguments(parser, _sim):
    parser.add_argument("--max-speed", type=float, default=8, help="forward operating cap in m/s, at most 20 (default: 8)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--renderer", choices=("classic", "filament"), default="classic")
    parser.add_argument("--camera", choices=("tcam", "cockpit", "chase"), default="tcam",
                        help="streamed view: broadcast T-cam (default), low cockpit, or rigid chase camera")
    parser.add_argument("--graphics", choices=("auto", "low", "medium", "high"), default="auto",
                        help="auto: high on a GPU, medium on a software rasterizer (default)")
    parser.add_argument("--no-video", action="store_true", help="physics/telemetry without a graphics context")
    parser.add_argument("--video-stats", action="store_true")
    parser.add_argument("--benchmark", type=float, metavar="SECONDS", help="measure offline physics; no credentials or video needed")
    parser.add_argument("--benchmark-video", type=float, metavar="SECONDS", help="measure physics/render/capture, without a LiveKit room")
    parser.add_argument("--preview", type=Path, metavar="PNG", help="save an onboard frame without joining LiveKit")


def run(args, spec):
    return asyncio.run(serve(args, spec))


async def serve(args, spec):
    from .gateway import DriveGateway
    for value in (args.benchmark, args.benchmark_video):
        if value is not None and not 0 < value < 3601:
            raise ValueError("benchmark seconds must be finite and in (0, 3600]")
    if args.no_video and (args.preview or args.benchmark_video):
        raise ValueError("preview/video benchmark cannot combine with --no-video")
    backend = Backend(max_speed=args.max_speed, fps=args.fps, width=args.width, height=args.height,
                      renderer=args.renderer, video=not (args.no_video or args.benchmark is not None),
                      camera=args.camera, graphics=args.graphics)
    events = None
    try:
        if args.renderer == "filament" and sys.platform == "darwin" and backend.scene and not args.check:
            from ..trossen.filament import service_macos_events
            events = asyncio.create_task(service_macos_events())
        gateway = DriveGateway(backend, "", "", args.fps, video=not args.no_video, video_stats=args.video_stats)
        if args.check:
            print(json.dumps({**gateway.discovery(), "simulation": {
                "track": "Spielberg", "track_length_m": backend.car.track.length,
                "vehicle": backend.car.p, "instrument_tracks": {
                    name: {"msg_type": kind, "format": "json", "rate_hz": 20}
                    for name, (kind, _) in backend.instrument_tracks().items()}}}, indent=2))
            return 0
        if args.benchmark is not None:
            from .dynamics import Drive
            elapsed = []
            for _ in range(max(1, round(args.benchmark / 0.005))):
                backend.car.target = Drive(min(4, args.max_speed), 0)
                started = time.perf_counter()
                backend.step()
                elapsed.append(time.perf_counter() - started)
            import numpy as np
            print(json.dumps({"simulation_seconds": backend.car.time, "compute_seconds": sum(elapsed),
                "real_time_factor": backend.car.time / sum(elapsed), "step_p99_ms": float(np.percentile(elapsed, 99) * 1000)}))
            return 0
        if args.preview:
            from PIL import Image
            with ThreadPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, backend.open_renderers)
                try:
                    if backend.batch_rendering:
                        batch = backend.prepare_batch(("onboard",))
                        pixels = (await loop.run_in_executor(executor, batch.render))["onboard"]
                    else:
                        frame = backend.prepare("onboard")
                        pixels = await loop.run_in_executor(executor, frame.render)
                    args.preview.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pixels).save(args.preview)
                finally:
                    await loop.run_in_executor(executor, backend.close)
            print(f"Saved {args.preview}")
            return 0
        if args.benchmark_video is not None:
            from livekit import rtc
            gateway.video_stats = True
            gateway.video_sources = {"onboard": rtc.VideoSource(args.width, args.height)}
            tasks = [asyncio.create_task(gateway._simulate()), asyncio.create_task(gateway._video())]
            try:
                done, _ = await asyncio.wait(tasks, timeout=args.benchmark_video, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
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
        if events:
            events.cancel()
            await asyncio.gather(events, return_exceptions=True)
