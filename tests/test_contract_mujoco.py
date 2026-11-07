"""Actual model checks; skipped when MuJoCo or pinned assets are unavailable."""

import copy
import importlib.util
import unittest

from sims.assets import SOURCES
from sims.contract.config import CONFIGS, Configuration
from sims.contract.trajectory import Trajectory


@unittest.skipUnless(importlib.util.find_spec("mujoco") and SOURCES["trossen_arm_mujoco"].installed(),
                     "install requirements and fetch Trossen assets")
class MujocoContractTests(unittest.TestCase):
    def backend(self, name):
        from sims.contract.mujoco_backend import MujocoBackend
        from sims.trossen.simulation import TrossenSimulation
        config = Configuration.load(CONFIGS / f"{name}.bindings.json", 12)
        assets = SOURCES["trossen_arm_mujoco"].path / "assets"
        scene = TrossenSimulation(assets, config.bindings["scene"], model_path=assets / config.bindings["model"])
        self.addCleanup(scene.close)
        return MujocoBackend(scene, config), config

    def test_both_models_resolve_and_publish_measured_named_state(self):
        for name in ("trossen-wxai", "trossen-wxai-whack", "trossen-workbench"):
            with self.subTest(name=name):
                backend, config = self.backend(name)
                for keys in config.feedback_tracks.values():
                    state = backend.feedback(keys)
                    self.assertEqual(len(state["name"]), 6)
                    self.assertEqual(len(state["position"]), 6)
                    self.assertEqual(len(state["velocity"]), 6)
                    self.assertEqual(state["effort"], [])

    def test_dual_arm_independent_motion_and_cancel(self):
        backend, _ = self.backend("trossen-workbench")
        left, right = "left_arm", "right_arm"
        initial_left = backend.feedback([left])["position"][0]
        initial_right = backend.feedback([right])["position"][0]
        backend.submit(left, Trajectory(("left_arm_joint_1",), (1.0,), ((initial_left + 0.2,),)))
        backend.submit(right, Trajectory(("right_arm_joint_1",), (2.0,), ((initial_right - 0.2,),)))
        for _ in range(50):
            backend.step()
        self.assertGreater(backend.feedback([left])["position"][0], initial_left + 0.04)
        self.assertLess(backend.feedback([right])["position"][0], initial_right - 0.015)
        backend.submit(left, Trajectory(("left_arm_joint_1",), (), ()))
        self.assertNotIn(left, backend.motions)
        self.assertIn(right, backend.motions)
        stopped = backend.targets[backend.joints[left]["left_arm_joint_1"].actuator]
        for _ in range(180):
            backend.step()
        self.assertAlmostEqual(backend.feedback([left])["position"][0], stopped, delta=0.025)
        self.assertAlmostEqual(backend.feedback([right])["position"][0], initial_right - 0.2, delta=0.025)
        backend.hold()
        self.assertFalse(backend.motions)

    def test_bad_command_does_not_replace_running_trajectory(self):
        backend, _ = self.backend("trossen-wxai")
        name = "arm_joint_1"
        initial = backend.feedback(["arm"])["position"][0]
        backend.submit("arm", Trajectory((name,), (1.0,), ((initial + 0.1,),)))
        old = backend.motions["arm"]
        with self.assertRaises(ValueError):
            backend.submit("arm", Trajectory((name,), (0.01,), ((initial + 1,),)))
        self.assertIs(backend.motions["arm"], old)

    def test_shared_feedback_contains_both_arms_with_exact_names(self):
        backend, config = self.backend("trossen-workbench")
        state = backend.feedback(["left_arm", "right_arm"])
        expected = [name for interface in config.discovery["control_interfaces"].values()
                    for name in interface["joint_names"]]
        self.assertEqual(state["name"], expected)
        self.assertEqual(len(state["position"]), 12)

    def test_binding_cannot_cross_arm_or_point_at_missing_joint(self):
        from sims.contract.mujoco_backend import MujocoBackend
        backend, config = self.backend("trossen-workbench")
        for name in ("right_joint_0", "missing_joint"):
            broken = copy.deepcopy(config)
            broken.bindings["control_interfaces"]["left_arm"]["joint_map"]["left_arm_joint_1"]["joint"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                MujocoBackend(backend.scene, broken)

    def test_whack_mallet_snaps_to_gripper_through_contract_backend(self):
        import numpy as np
        backend, config = self.backend("trossen-wxai-whack")
        scene = backend.scene
        game, arm = scene.game, scene.arms["arm"]
        grip = scene.data.site_xpos[game.grip_site].copy()
        goals = scene._inverse_kinematics(arm, grip + [0, 0, 0.01], np.eye(3), 200)
        names = config.discovery["control_interfaces"]["arm"]["joint_names"]
        backend.submit("arm", Trajectory(tuple(names), (3.0,), (tuple(float(g) for g in goals),)))
        for _ in range(400):
            backend.step()
        self.assertTrue(game.held)
        ee = scene.data.site_xpos[arm.site_id]
        self.assertLess(np.linalg.norm(scene.data.xpos[game.hammer_body] - ee), 0.01)

    def test_whack_scoring(self):
        backend, _ = self.backend("trossen-wxai-whack")
        game, data = backend.scene.game, backend.scene.data
        mole = game.moles[0]
        wrong = next(c for c in ("violet", "white", "blue") if c != game.target)
        mole.color, mole.phase = wrong, "up"
        game._whack(mole, data, 0.0)
        self.assertEqual((game.score, game.misses), (0, 1))  # never below zero
        for expected in (1, 2):
            mole.color, mole.phase, mole.hit = game.target, "up", False
            game._whack(mole, data, 0.0)
            self.assertEqual(game.score, expected)
        self.assertEqual(game.state()["hits"], 2)
