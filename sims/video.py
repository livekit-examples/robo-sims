"""Capture one camera cycle with either per-feed or batched rendering."""

import asyncio
import time

from livekit import rtc


async def capture_video(backend, sources, executor, stats=None):
    loop = asyncio.get_running_loop()
    if getattr(backend, "batch_rendering", False) is True:
        started = time.perf_counter()
        batch = backend.prepare_batch(tuple(sources))
        prepared = time.perf_counter()
        frames = await loop.run_in_executor(executor, batch.render)
        rendered = time.perf_counter()
        if stats:
            stats.camera("all cameras (batch)", prepared - started, rendered - prepared, 0)
        for key, source in sources.items():
            started = time.perf_counter()
            pixels = frames[key]
            source.capture_frame(rtc.VideoFrame(
                pixels.shape[1], pixels.shape[0], rtc.VideoBufferType.RGB24, pixels.tobytes()))
            if stats:
                stats.camera(key, 0, 0, time.perf_counter() - started)
    else:
        for key, source in sources.items():
            started = time.perf_counter()
            renderer = backend.prepare(key)
            prepared = time.perf_counter()
            pixels = await loop.run_in_executor(executor, renderer.render)
            rendered = time.perf_counter()
            source.capture_frame(rtc.VideoFrame(
                pixels.shape[1], pixels.shape[0], rtc.VideoBufferType.RGB24, pixels.tobytes()))
            if stats:
                stats.camera(key, prepared - started, rendered - prepared, time.perf_counter() - rendered)
