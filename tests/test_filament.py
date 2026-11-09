"""Filament camera math plus opt-in native GPU checks.

ROBO_SIMS_FILAMENT_TEST=1 python -m unittest discover -s tests -p test_filament.py -v
"""

import asyncio
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import renderer
import numpy as np

from sims.contract.config import CONFIGS, Configuration
from sims.trossen.filament import FilamentFrames, service_macos_events
from sims.trossen.runner import ASSETS
from sims.trossen.simulation import TrossenSimulation


class FilamentCameraTests(unittest.TestCase):
    def test_batch_preserves_stereo_baselines_projection_and_snapshot(self):
        scene = TrossenSimulation(ASSETS, "workbench")
        self.addCleanup(scene.close)
        batch = object.__new__(FilamentFrames)
        batch.model, batch.data = copy.copy(scene.model), mujoco.MjData(scene.model)
        batch.mjrf, batch.api = mjrf, renderer
        batch.feeds, batch.fog = scene.feeds, False
        batch.lights, batch.renderables, batch.renderer = Mock(), Mock(), Mock()
        batch.width, batch.height = 960, 1440
        batch.rects = {"left_wrist": (0, 1200, 768, 240), "center": (0, 960, 768, 240),
                       "right_wrist": (0, 720, 768, 240), "scene": (0, 0, 960, 720)}
        batch.camera_ids = {key: scene.model.camera(feed.camera).id for key, feed in scene.feeds.items()}
        batch.pixels = {key: object() for key in scene.feeds}
        batch.prepare(scene.model, scene.data, tuple(scene.feeds))
        before = batch.data.qpos.copy()
        scene.data.qpos[0] += .1
        np.testing.assert_array_equal(batch.data.qpos, before)
        self.assertEqual(batch.render(), batch.pixels)
        views = batch.renderer.render.call_args.args[0]
        self.assertEqual(len(views), 7)
        offset = 0
        for key, feed in scene.feeds.items():
            count = 2 if feed.stereo else 1
            eyes = views[offset:offset + count]
            camera_id = batch.camera_ids[key]
            np.testing.assert_allclose(eyes[0].camera.pos, batch.data.cam_xpos[camera_id], atol=1e-6)
            if feed.stereo:
                expected = feed.stereo.baseline * batch.data.cam_xmat[camera_id].reshape(3, 3)[:, 0]
                np.testing.assert_allclose(eyes[1].camera.pos - eyes[0].camera.pos, expected, atol=1e-6)
                self.assertAlmostEqual(eyes[0].viewport[2] * batch.width, scene.eye_size(key)[0])
            self.assertGreater(eyes[0].camera.frustum_top, eyes[0].camera.frustum_bottom)
            self.assertTrue(all(view.target == "cameras" for view in eyes))
            offset += count


@unittest.skipUnless(os.environ.get("ROBO_SIMS_FILAMENT_TEST") == "1", "opt-in native Filament GPU test")
class FilamentGpuTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_trossen_scenes_images_motion_and_game_colors(self):
        events = asyncio.create_task(service_macos_events()) if sys.platform == "darwin" else None
        try:
            for name in ("trossen-wxai", "trossen-workbench", "trossen-wxai-whack", "trossen-workbench-whack"):
                with self.subTest(simulation=name):
                    config = Configuration.load(CONFIGS / f"{name}.bindings.json", 30)
                    scene = TrossenSimulation(ASSETS, config.bindings["scene"], 160, 120, 480, 360,
                        model_path=ASSETS / config.bindings["model"], renderer="filament")
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        try:
                            await loop.run_in_executor(executor, scene.open_renderers)

                            async def capture():
                                batch = scene.prepare_batch(tuple(scene.feeds))
                                frames = await loop.run_in_executor(executor, batch.render)
                                return {key: pixels.copy() for key, pixels in frames.items()}

                            await capture()  # Shader warmup.
                            before = await capture()
                            for key, pixels in before.items():
                                self.assertEqual(pixels.dtype, np.uint8)
                                self.assertEqual(pixels.shape, (*scene.render_size(key)[::-1], 3))
                                self.assertGreater(pixels.std(), 2, f"blank {name}/{key}")
                                if scene.feeds[key].stereo:
                                    left, right = np.split(pixels.astype(float), 2, axis=1)
                                    self.assertGreater(np.abs(left - right).mean(), .05)
                            output = os.environ.get("ROBO_SIMS_FILAMENT_OUTPUT")
                            if output:
                                from PIL import Image
                                Path(output).mkdir(parents=True, exist_ok=True)
                                for key, pixels in before.items():
                                    Image.fromarray(pixels).save(Path(output) / f"{name}-{key}.png")
                            scene.data.qpos[0] += .3
                            mujoco.mj_forward(scene.model, scene.data)
                            after = await capture()
                            self.assertGreater(np.abs(after["scene"].astype(float) - before["scene"]).mean(), .01)
                            if scene.game:
                                for mole in scene.game.moles:
                                    scene.data.mocap_pos[mole.mocap, 2] = scene.game.layout.ball_up_z
                                    scene.model.geom_rgba[mole.geom] = [1, 0, 0, 1]
                                mujoco.mj_forward(scene.model, scene.data)
                                red = await capture()
                                for mole in scene.game.moles:
                                    scene.model.geom_rgba[mole.geom] = [0, 1, 0, 1]
                                green = await capture()
                                self.assertGreater(np.abs(red["scene"].astype(float) - green["scene"]).mean(), .01)
                                scene.game._light("blue" if scene.game.target != "blue" else "violet")
                                lamp = await capture()
                                self.assertGreater(np.abs(lamp["scene"].astype(float) - green["scene"]).mean(), .005)
                        finally:
                            await loop.run_in_executor(executor, scene.close)
        finally:
            if events:
                events.cancel()
                await asyncio.gather(events, return_exceptions=True)
