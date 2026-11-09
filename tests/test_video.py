"""FPS propagation and source pacing without a GPU, credentials, or server."""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from sims.__main__ import parser
from sims.contract.config import Configuration
from sims.contract.gateway import ContractGateway
from sims.registry import SIMULATIONS
from sims.trossen import managed, runner


class VideoTests(unittest.IsolatedAsyncioTestCase):
    async def test_managed_rate_reaches_discovery_and_every_encoder(self):
        for fps in (12, 30):
            with self.subTest(fps=fps):
                feeds = {name: SimpleNamespace(name=name) for name in ('left', 'center', 'right', 'scene')}
                backend = Mock(feeds=feeds, render_size=lambda _: (4, 2))
                backend.policy.return_value = {}
                participant = Mock(set_attributes=AsyncMock(), publish_track=AsyncMock())
                participant.register_rpc_method.side_effect = lambda *_: lambda f: f
                room = Mock(local_participant=participant)
                with (patch.object(managed, 'TrossenSimulation'),
                      patch.object(managed, 'MujocoBackend', return_value=backend),
                      patch('sims.contract.gateway.SchemaRoom', return_value=room)):
                    options = {} if fps == 12 else {'fps': fps}
                    gateway, _ = managed.create(SIMULATIONS['trossen-workbench-whack'], '', '', 'demo-op-', **options)
                self.assertEqual(gateway.config.discovery['default_video_rate_hz'], fps)
                self.assertEqual(gateway.fps, fps)
                with (patch('sims.contract.gateway.publish_json', AsyncMock()),
                      patch('sims.contract.gateway.rtc.LocalVideoTrack.create_video_track', Mock())):
                    await gateway._publish()
                self.assertEqual(participant.publish_track.await_count, 4)
                for call in participant.publish_track.await_args_list:
                    self.assertEqual(call.args[1].video_encoding.max_framerate, fps)

    async def test_cli_rate_reaches_discovery_runner_and_every_encoder(self):
        for name in (name for name in SIMULATIONS if name.startswith("trossen-")):
            with self.subTest(simulation=name):
                args = parser(name).parse_args(["run", name, "--fps", "30", "--video-stats"])
                config = Configuration.load(args.bindings, args.fps)
                self.assertEqual(config.discovery["default_video_rate_hz"], 30)
                sim = SimpleNamespace(
                    feeds={key: SimpleNamespace(name=key) for key in ("left", "center", "right", "scene")},
                    render_size=lambda _: (4, 2),
                    model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=8, shadowsize=8192))))
                backend = Mock(feeds=sim.feeds, render_size=sim.render_size)
                backend.policy.return_value = {}
                participant = Mock(set_attributes=AsyncMock(), publish_track=AsyncMock())
                participant.register_rpc_method.side_effect = lambda *_: lambda f: f
                room = Mock(local_participant=participant)
                with patch("sims.contract.gateway.SchemaRoom", return_value=room):
                    gateway = ContractGateway(backend, config, "", "", args.fps)
                gateway.run = AsyncMock()
                with (patch.object(runner, "credentials", AsyncMock(return_value=SimpleNamespace(
                        url="", token="", identity=None))),
                      patch("sims.contract.mujoco_backend.MujocoBackend", return_value=backend),
                      patch("sims.contract.gateway.ContractGateway", return_value=gateway) as factory):
                    await runner._serve(args, SIMULATIONS[name], sim, config)
                self.assertEqual(factory.call_args.args[4], 30)
                self.assertTrue(factory.call_args.kwargs["video_stats"])
                with (patch("sims.contract.gateway.publish_json", AsyncMock()),
                      patch("sims.contract.gateway.rtc.LocalVideoTrack.create_video_track", Mock())):
                    await gateway._publish()
                self.assertEqual(participant.publish_track.await_count, 4)
                for call in participant.publish_track.await_args_list:
                    self.assertEqual(call.args[1].video_encoding.max_framerate, 30)

    async def test_both_video_loops_can_submit_thirty_fps_when_rendering_is_fast(self):
        for protocol in ("contract", "legacy"):
            with self.subTest(protocol=protocol):
                frames = []
                ready = asyncio.Event()
                pixels = np.zeros((2, 4, 3), dtype=np.uint8)
                scene = Mock()
                scene.prepare.return_value.render.return_value = pixels

                def capture(_):
                    frames.append(time.perf_counter())
                    if len(frames) >= 30:
                        ready.set()

                # Use the production loops, avoiding construction of a room.
                bridge_type = ContractGateway if protocol == "contract" else runner.SimParticipant
                bridge = object.__new__(bridge_type)
                bridge.fps, bridge.video_stats = 30, False
                bridge.backend = bridge.sim = scene
                bridge.stopping = asyncio.Event()
                bridge._video_ready = set()
                bridge._mark_ready = Mock()
                bridge.video_sources = {"camera": SimpleNamespace(capture_frame=capture)}
                task = asyncio.create_task(bridge._video())
                try:
                    await asyncio.wait_for(ready.wait(), 4)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                rate = (len(frames) - 1) / (frames[-1] - frames[0])
                self.assertGreater(rate, 20, "a hidden 12 FPS cap would fail this check")
                self.assertLess(rate, 36)
                scene.close.assert_called_once()

    async def test_benchmark_does_not_fetch_credentials_and_propagates_render_failure(self):
        args = parser("trossen-workbench").parse_args([
            "run", "trossen-workbench", "--fps", "30", "--benchmark-video", "0.1"])
        config = Configuration.load(args.bindings, args.fps)
        scene = Mock(feeds={"scene": SimpleNamespace(name="scene")})
        scene.render_size.return_value = (4, 2)
        scene.model.vis.quality.offsamples = 8
        scene.model.vis.quality.shadowsize = 8192
        backend = Mock()
        backend.open_renderers.side_effect = RuntimeError("GPU unavailable")
        with (patch.object(runner, "credentials", AsyncMock()) as credentials,
              patch("sims.contract.mujoco_backend.MujocoBackend", return_value=backend)):
            with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
                await runner._serve(args, SIMULATIONS[args.sim], scene, config)
            credentials.assert_not_called()
        backend.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
