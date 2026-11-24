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
    def test_stereo_calibration_matches_render_sizes_optical_axes_and_mounts(self):
        import math
        import mujoco
        import numpy as np
        for name in ("trossen-wxai", "trossen-wxai-whack", "trossen-workbench", "trossen-workbench-whack"):
            with self.subTest(model=name):
                backend, config = self.backend(name)
                scene = backend.scene
                info = config.discovery["camera_info"]
                self.assertEqual(set(info), {key for key, feed in scene.feeds.items() if feed.stereo})
                for key, camera in info.items():
                    feed = scene.feeds[key]
                    camera_id = scene.model.camera(feed.camera).id
                    parent_id = int(scene.model.cam_bodyid[camera_id])
                    width, height = scene.eye_size(key)
                    self.assertEqual(camera["track_name"], feed.name)
                    self.assertEqual((camera["width"], camera["height"]), scene.render_size(key))
                    world_rotation = scene.data.cam_xmat[camera_id].reshape(3, 3)
                    parent_rotation = scene.data.xmat[parent_id].reshape(3, 3)
                    for index, eye in enumerate(("left", "right")):
                        view = camera["views"][eye]
                        self.assertEqual(view["image_region"], {
                            "x": index * width, "y": 0, "width": width, "height": height})
                        intrinsic = view["intrinsics"]
                        focal = height / (2 * math.tan(math.radians(feed.stereo.fovy) / 2))
                        self.assertAlmostEqual(intrinsic["fx"], focal, delta=1e-4)
                        self.assertAlmostEqual(intrinsic["fy"], focal, delta=1e-4)
                        self.assertEqual((intrinsic["cx"], intrinsic["cy"]), (width / 2, height / 2))
                        self.assertEqual(intrinsic["distortion_model"], "none")
                        extrinsic = view["extrinsics"]
                        self.assertEqual(extrinsic["parent_frame_id"], scene.model.body(parent_id).name)
                        q = np.asarray(extrinsic["rotation_xyzw"])[[3, 0, 1, 2]]
                        rotation = np.zeros(9)
                        mujoco.mju_quat2Mat(rotation, q)
                        expected = world_rotation @ np.diag([1, -1, -1])
                        np.testing.assert_allclose(parent_rotation @ rotation.reshape(3, 3), expected, atol=1e-8)
                        position = scene.data.xpos[parent_id] + parent_rotation @ extrinsic["translation"]
                        np.testing.assert_allclose(position, scene.data.cam_xpos[camera_id] +
                            index * feed.stereo.baseline * world_rotation[:, 0], atol=1e-8)
                    left, right = (camera["views"][eye]["extrinsics"]["translation"] for eye in ("left", "right"))
                    self.assertAlmostEqual(np.linalg.norm(np.asarray(right) - left), feed.stereo.baseline)
                # Moving an arm must not turn fixed mount-relative metadata into a startup world pose.
                snapshot = copy.deepcopy(info)
                scene.data.qpos[0] += 0.2
                mujoco.mj_forward(scene.model, scene.data)
                self.assertEqual(scene.camera_info(), snapshot)

    def test_stereo_intrinsics_follow_nondefault_render_resolution(self):
        from sims.trossen.simulation import TrossenSimulation
        assets = SOURCES["trossen_arm_mujoco"].path / "assets"
        first = TrossenSimulation(assets, "single_wxai", width=320, height=240)
        second = TrossenSimulation(assets, "single_wxai", width=640, height=480)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        a, b = (s.camera_info()["arm_wrist"] for s in (first, second))
        self.assertEqual((a["width"], a["height"]), (640, 240))
        self.assertEqual((b["width"], b["height"]), (1280, 480))
        for eye in ("left", "right"):
            for value in ("fx", "fy", "cx", "cy"):
                self.assertAlmostEqual(b["views"][eye]["intrinsics"][value], 2 * a["views"][eye]["intrinsics"][value])

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

    def test_cartesian_pose_feedback_and_inverse_kinematics_follow_the_base_frame(self):
        import numpy as np
        from sims.contract.pose import PoseTarget
        for name, arm in (("trossen-wxai", "arm"), ("trossen-workbench", "right_arm")):
            with self.subTest(model=name):
                backend, config = self.backend(name)
                self.assertIn(f"/{arm}/end_effector_pose", config.pose_tracks)
                for _ in range(5):
                    backend.step()
                start = backend.pose_feedback(arm)
                self.assertGreater(sum(v * v for v in start["pose"]["position"].values()), 0.01)
                self.assertEqual(start["header"]["frame_id"], config.discovery["control_interfaces"][arm]["frame_id"])
                position = start["pose"]["position"]
                quaternion = start["pose"]["orientation"]
                self.assertAlmostEqual(sum(v * v for v in quaternion.values()), 1, places=6)
                target = PoseTarget((position["x"] + 0.03, position["y"], position["z"] + 0.02),
                                    (quaternion["x"], quaternion["y"], quaternion["z"], quaternion["w"]))
                backend.submit_pose(arm, target)
                self.assertIn(arm, backend.motions)
                for _ in range(300):
                    backend.step()
                end = backend.pose_feedback(arm)["pose"]["position"]
                self.assertAlmostEqual(end["x"], position["x"] + 0.03, delta=0.01)
                self.assertAlmostEqual(end["z"], position["z"] + 0.02, delta=0.01)
                self.assertAlmostEqual(end["y"], position["y"], delta=0.01)
                # Holding the measured pose is accepted and leaves the arm in place.
                backend.submit_pose(arm, PoseTarget(
                    tuple(end[a] for a in "xyz"),
                    tuple(backend.pose_feedback(arm)["pose"]["orientation"][a] for a in "xyzw")))
                for _ in range(50):
                    backend.step()
                held = backend.pose_feedback(arm)["pose"]["position"]
                self.assertAlmostEqual(np.linalg.norm([held[a] - end[a] for a in "xyz"]), 0, delta=0.005)

    def test_cartesian_policy_shares_the_arm_joints(self):
        backend, _ = self.backend("trossen-workbench")
        policy = backend.policy()["operations"]["left_arm"]
        self.assertEqual(policy["cartesian_pose"]["profile"], "arm.cartesian_pose.v1")
        self.assertEqual(policy["cartesian_pose"]["joint_names"], policy["joint_position"]["joint_names"])
        self.assertEqual(policy["cartesian_pose"]["control_channel"], "primary")

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

    def _grab_mallet(self, backend, config, interface, arm_id, hammer_index, pitch=0.0):
        import numpy as np
        # `pitch` tilts the gripper down about y. The Workbench mallets are too far out to reach level.
        rotation = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        scene = backend.scene
        arm, hammer = scene.arms[arm_id], scene.game.hammers[hammer_index]
        grip = scene.data.site_xpos[hammer.grip_site].copy()
        goals = scene._inverse_kinematics(arm, grip + [0, 0, 0.01], rotation, 300)
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
        import numpy as np
        backend, config = self.backend("trossen-workbench-whack")
        game = backend.scene.game
        self.assertEqual((len(game.moles), len(game.hammers)), (5, 2))
        self._grab_mallet(backend, config, "left_arm", "left", 0, pitch=np.pi / 4)
        self._grab_mallet(backend, config, "right_arm", "right", 1, pitch=np.pi / 4)
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
