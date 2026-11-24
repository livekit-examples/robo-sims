"""Opt-in 10 minute 60 fps driving soak. No network, encoding, or credentials.

    PYTHONPATH=. python tests/f1tenth_soak.py

The path follower is test input, not a simulator driving assist.
"""
import asyncio
import json
import logging
import math
import time

import numpy as np
from livekit import rtc

from sims.f1tenth.backend import Backend
from sims.f1tenth.dynamics import Drive
from sims.f1tenth.gateway import DriveGateway


async def main():
    backend = Backend(width=960, height=540)
    gateway = DriveGateway(backend, "", "", 60, video_stats=True)
    gateway.video_sources = {"onboard": rtc.VideoSource(960, 540)}
    gateway._feedback_ready = True
    simulation = asyncio.create_task(gateway._simulate())
    video = asyncio.create_task(gateway._video())
    await asyncio.wait_for(gateway.ready.wait(), 30)
    started, sim_started = time.monotonic(), backend.car.time

    async def drive():
        while True:
            car, track = backend.car, backend.car.track
            lookahead = 1 + abs(car.vx) * 0.18
            distance = (car.progress + lookahead) % track.length
            j = np.searchsorted(track.arc, distance, side="right") - 1
            point = track.points[j] + track.tangents[j] * (distance - track.arc[j])
            alpha = math.remainder(math.atan2(point[1] - car.y, point[0] - car.x) - car.yaw, 2 * math.pi)
            angle = math.atan2(2 * (car.p["lf"] + car.p["lr"]) * math.sin(alpha), lookahead)
            car.target = Drive(3, min(0.4189, max(-0.4189, angle)))
            await asyncio.sleep(0.02)

    driving = asyncio.create_task(drive())
    timer = asyncio.create_task(asyncio.sleep(600))
    try:
        done, _ = await asyncio.wait([simulation, video, timer], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        elapsed = time.monotonic() - started
        result = {"wall_seconds": elapsed, "simulation_seconds": backend.car.time - sim_started,
                  "real_time_factor": (backend.car.time - sim_started) / elapsed,
                  "overloads": gateway.overloads, "laps": backend.car.laps,
                  "last_valid_lap_seconds": backend.car.last_lap_seconds}
        print(json.dumps(result), flush=True)
        if timer not in done or not 0.99 < result["real_time_factor"] < 1.01 \
                or gateway.overloads or backend.car.laps < 1 or backend.car.last_lap_seconds is None:
            raise AssertionError(f"Driving soak failed: {result}")
    finally:
        for task in (simulation, video, driving, timer):
            task.cancel()
        await asyncio.gather(simulation, video, driving, timer, return_exceptions=True)
        backend.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
