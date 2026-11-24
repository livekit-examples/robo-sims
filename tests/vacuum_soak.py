"""Opt-in 10 minute real-time soak: physics, telemetry loop and 960 x 540 / 30 fps overhead plus onboard video.

    PYTHONPATH=. python tests/vacuum_soak.py [SECONDS]

No network, video encoding or credentials. The wanderer below is test input, not a driving assist.
It reports the ratio of simulation time to real time over moving one-minute windows, the video frame
rate and the largest gap between frames, then fails if the ratio leaves 0.99 to 1.01 or an overload occurs.
"""
import asyncio
import json
import logging
import platform
import sys
import time

from livekit import rtc

from sims.vacuum.backend import Backend
from sims.vacuum.gateway import VacuumGateway
from sims.vacuum.simulation import Command

WINDOW_S = 60.0


async def main(seconds):
    backend = Backend(width=960, height=540, fps=30)
    gateway = VacuumGateway(backend, "", "", 30)
    frames = {"overhead": [], "onboard": []}
    gateway.video_sources = {key: rtc.VideoSource(960, 540) for key in frames}
    for key, source in gateway.video_sources.items():
        original = source.capture_frame

        def counted(frame, _key=key, _original=original):
            frames[_key].append(time.monotonic())
            return _original(frame)
        source.capture_frame = counted
    gateway._feedback_ready = True
    tasks = [asyncio.create_task(gateway._simulate()), asyncio.create_task(gateway._video())]
    await asyncio.wait_for(gateway.ready.wait(), 30)
    sim = backend.sim
    sim.start_run()
    started, sim_started = time.monotonic(), sim.uptime
    samples = [(started, sim_started)]  # (wall, uptime): uptime keeps counting across resets

    async def wander():
        turn = 0.0
        while True:
            if sim.state != "running":
                backend.reset_run()
                sim.start_run()
            clearance = sim.clearance
            turn = 1.6 if clearance < 0.3 else 0.25
            sim.requested = Command(0.45, turn)
            gateway.backend.last_command = None  # the watchdog is exercised by the transport test
            await asyncio.sleep(0.02)

    tasks.append(asyncio.create_task(wander()))
    try:
        while time.monotonic() - started < seconds:
            await asyncio.sleep(5)
            samples.append((time.monotonic(), sim.uptime))
            for task in tasks[:2]:
                if task.done():
                    task.result()
        wall = time.monotonic() - started
        ratios = []
        for i, (t0, s0) in enumerate(samples):
            later = [(t, s) for t, s in samples[i:] if t - t0 >= WINDOW_S]
            if later:
                t1, s1 = later[0]
                ratios.append((s1 - s0) / (t1 - t0))
        gaps = [b - a for times in frames.values() for a, b in zip(times, times[1:])]
        result = {"computer": f"{platform.machine()} {platform.platform()}", "wall_seconds": wall,
                  "simulation_seconds": sim.uptime - sim_started,
                  "overall_ratio": (sim.uptime - sim_started) / wall,
                  "one_minute_ratio_min": min(ratios) if ratios else None,
                  "one_minute_ratio_max": max(ratios) if ratios else None,
                  "overhead_fps": len(frames["overhead"]) / wall, "onboard_fps": len(frames["onboard"]) / wall,
                  "max_frame_gap_ms": max(gaps) * 1000 if gaps else None, "overloads": gateway.overloads}
        print(json.dumps(result), flush=True)
        if gateway.overloads or not ratios or not 0.99 < min(ratios) <= max(ratios) < 1.01:
            raise AssertionError(f"Soak failed: {result}")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        backend.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 600.0))
