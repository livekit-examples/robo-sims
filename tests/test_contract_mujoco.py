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
    def test_gripper_moves_coupled_fingers_without_replacing_arm_or_other_gripper(self):
        for model in ("trossen-wxai", "trossen-workbench"):
            with self.subTest(model=model):
                backend, config = self.backend(model)
                arm = next(iter(config.discovery["control_interfaces"]))
                key = (arm, "gripper")
                name, joint = next(iter(backend.joints[key].items()))
                start = backend.feedback([key])["position"][0]
                goal = joint.limits[1] if start < 0.022 else joint.limits[0]
                arm_name, arm_joint = next(iter(backend.joints[arm].items()))
                arm_start = backend.feedback([arm])["position"][0]
                backend.submit(arm, Trajectory((arm_name,), (8.0,), ((arm_start + 0.1,),)))
                backend.submit(key, Trajectory((name,), (3.0,), ((goal,),)))
                other = next((k for k in backend.joints if isinstance(k, tuple) and k != key), None)
                if other:
                    other_name = next(iter(backend.joints[other]))
                    backend.submit(other, Trajectory((other_name,), (8.0,), ((0.03,),)))
                for _ in range(200):
                    backend.step()
                moved = backend.feedback([key])["position"][0]
                self.assertGreater(abs(moved - start), 0.01)
                binding = config.bindings["control_interfaces"][arm]["gripper"]
                import mujoco
                coupled = backend.model.joint(binding["coupled_joint"]).id
                self.assertAlmostEqual(moved, backend.data.qpos[backend.model.jnt_qposadr[coupled]], delta=0.001)
                before = backend.motions[key]
                with self.assertRaises(ValueError):
                    backend.submit(key, Trajectory((name,), (0.01,), ((0.5,),)))
                self.assertIs(backend.motions[key], before)
                backend.submit(key, Trajectory((name,), (), ()))
                self.assertNotIn(key, backend.motions)
                self.assertIn(arm, backend.motions)
                if other:
                    self.assertIn(other, backend.motions)
                self.assertAlmostEqual(backend.targets[joint.actuator], moved, delta=1e-8)
                policy = backend.policy()["operations"][arm]["gripper_position"]
                self.assertEqual(policy["units"], "m")
                self.assertEqual(policy["joint_names"], [name])
                backend.hold()
                self.assertFalse(backend.motions)

    def test_gripper_rejects_wrong_arm_actuator_or_missing_coupling(self):
        from sims.contract.mujoco_backend import MujocoBackend
        backend, config = self.backend("trossen-workbench")
        for field, value in (("actuator", "right_gripper"), ("joint", "left_joint_0"),
                             ("coupled_joint", "right_right_carriage_joint")):
            broken = copy.deepcopy(config)
            broken.bindings["control_interfaces"]["left_arm"]["gripper"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                MujocoBackend(backend.scene, broken)

    def backend(self, name):
        from sims.contract.mujoco_backend import MujocoBackend
        from sims.trossen.simulation import TrossenSimulation
        config = Configuration.load(CONFIGS / f"{name}.bindings.json", 12)
        assets = SOURCES["trossen_arm_mujoco"].path / "assets"
        scene = TrossenSimulation(assets, config.bindings["scene"], model_path=assets / config.bindings["model"])
        self.addCleanup(scene.close)
        return MujocoBackend(scene, config), config

    def test_both_models_resolve_and_publish_measured_named_state(self):
        for name in ("trossen-wxai", "trossen-wxai-whack", "trossen-workbench", "trossen-workbench-whack"):
            with self.subTest(name=name):
                backend, config = self.backend(name)
                for keys in config.feedback_tracks.values():
                    state = backend.feedback(keys)
                    self.assertEqual(len(state["name"]), 7)
                    self.assertEqual(len(state["position"]), 7)
                    self.assertEqual(len(state["velocity"]), 7)
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

    def _grab_mallet(self, backend, config, interface, arm_id, hammer_index):
        import numpy as np
        scene = backend.scene
        arm, hammer = scene.arms[arm_id], scene.game.hammers[hammer_index]
        grip = scene.data.site_xpos[hammer.grip_site].copy()
        goals = scene._inverse_kinematics(arm, grip + [0, 0, 0.01], np.eye(3), 300)
        names = config.discovery["control_interfaces"][interface]["joint_names"]
        start = [backend.feedback([interface])["position"][i] for i in range(len(names))]
        duration = max(3.0, 1.2 * max(abs(g - p) for g, p in zip(goals, start)) / 0.5)
        backend.submit(interface, Trajectory(tuple(names), (duration,), (tuple(float(g) for g in goals),)))
        for _ in range(int(duration * 100) + 100):
            backend.step()
        self.assertIsNotNone(hammer.held_by)
        ee = scene.data.site_xpos[arm.site_id]
        self.assertLess(np.linalg.norm(scene.data.xpos[hammer.body] - ee), 0.01)

    def test_whack_mallet_snaps_to_gripper_through_contract_backend(self):
        backend, config = self.backend("trossen-wxai-whack")
        self._grab_mallet(backend, config, "arm", "arm", 0)
        self.assertTrue(backend.scene.game.held)

    def test_workbench_whack_each_arm_picks_up_a_mallet(self):
        backend, config = self.backend("trossen-workbench-whack")
        game = backend.scene.game
        self.assertEqual((len(game.moles), len(game.hammers)), (5, 2))
        self._grab_mallet(backend, config, "left_arm", "left", 0)
        self._grab_mallet(backend, config, "right_arm", "right", 1)
        self.assertEqual(game.state()["hammers"], ["left", "right"])

    def test_workbench_whack_either_mallet_scores(self):
        import mujoco
        from sims.trossen import whack
        backend, _ = self.backend("trossen-workbench-whack")
        scene = backend.scene
        model, data, game = scene.model, scene.data, scene.game
        head_x = whack.HANDLE_LENGTH - whack.GRIP_FROM_END
        for expected, (hammer, mole) in enumerate(zip(game.hammers, game.moles[:2]), start=1):
            # Stand the mallet upright with its head face resting into an up ball.
            mole.color, mole.phase, mole.hit = game.target, "up", False
            data.mocap_pos[mole.mocap, 2] = game.layout.ball_up_z
            ball = data.mocap_pos[mole.mocap]
            data.qpos[hammer.qpos:hammer.qpos + 3] = [
                ball[0] - head_x, ball[1], ball[2] + whack.BALL_RADIUS + whack.HEAD_HALF_LENGTH - 0.005]
            data.qpos[hammer.qpos + 3:hammer.qpos + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            game._check_hits(data, float(data.time))
            self.assertEqual(game.score, expected)

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
