"""Cluttered table: the scene, its bindings, the discovery document, and hand tracking control."""

import importlib.util
import json
import unittest

from sims.assets import SOURCES
from sims.contract.config import CONFIGS, Configuration
from sims.registry import SIMULATIONS

READY = bool(importlib.util.find_spec("mujoco")) and SOURCES["trossen_arm_mujoco"].installed()


def make_backend(fps=12):
    from sims.contract.mujoco_backend import MujocoBackend
    from sims.trossen.simulation import TrossenSimulation

    config = Configuration.load(CONFIGS / "cluttered-table.bindings.json", fps)
    assets = SOURCES["trossen_arm_mujoco"].path / "assets"
    scene = TrossenSimulation(assets, "cluttered_table", 160, 120, 480, 360, renderer="classic")
    return MujocoBackend(scene, config), config


class RegistryTests(unittest.TestCase):
    def test_simulation_is_registered_for_the_cli_and_managed_runs(self):
        spec = SIMULATIONS["cluttered-table"]
        self.assertEqual(spec.options, {"scene": "cluttered_table"})
        self.assertEqual(spec.managed_module, "sims.cluttered_table.managed")
        from sims.demo.rules import SUPPORTED
        self.assertIn("cluttered-table", SUPPORTED)

    def test_discovery_lists_hand_tracking_first_for_both_arms(self):
        config = Configuration.load(CONFIGS / "cluttered-table.bindings.json", 12)
        interfaces = config.discovery["control_interfaces"]
        self.assertEqual(set(interfaces), {"left_arm", "right_arm"})
        for interface in interfaces.values():
            pose = interface["operations"]["cartesian_pose"]
            self.assertNotIn("presentations", pose)
            self.assertEqual(set(interface["operations"]), {"joint_position", "gripper_position", "cartesian_pose"})
            # The app defaults to the first operation of the primary channel.
            self.assertEqual(next(iter(interface["operations"])), "cartesian_pose")
        self.assertEqual(config.discovery["default_video_rate_hz"], 12)

    def test_discovery_has_nothing_private(self):
        text = (CONFIGS / "cluttered-table.discovery.json").read_text()
        for private in ("office.xml", "robocasa", "model", "/Users", "token"):
            self.assertNotIn(private, text)

    def test_legacy_protocol_is_rejected(self):
        from argparse import Namespace
        import asyncio
        from sims.trossen import runner

        args = Namespace(assets_dir=SOURCES["trossen_arm_mujoco"].path / "assets", width=320, height=240,
                         scene_width=960, scene_height=720, fps=12, benchmark_video=None, no_video=False,
                         check=True, protocol="legacy")
        if not args.assets_dir.is_dir():
            self.skipTest("fetch Trossen assets")
        with self.assertRaises(SystemExit) as caught:
            asyncio.run(runner._run(args, SIMULATIONS["cluttered-table"]))
        self.assertIn("only --protocol contract", str(caught.exception))

    def test_non_empty_get_robot_config_payload_is_refused_by_the_gateway(self):
        # The gateway serves every contract robot. This pins the empty-payload rule it applies here.
        source = (CONFIGS.parent / "gateway.py").read_text()
        self.assertIn('invocation.payload != ""', source)
        self.assertIn("get_robot_config takes an empty payload", source)


@unittest.skipUnless(READY, "install requirements and fetch Trossen assets")
class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend, cls.config = make_backend()

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_backend_accepts_both_arms_and_grippers(self):
        self.assertEqual(set(self.backend.joints), {"left_arm", "right_arm", ("left_arm", "gripper"),
                                                    ("right_arm", "gripper")})
        for key in ("left_arm", "right_arm"):
            self.assertEqual(len(self.backend.joints[key]), 6)

    def test_cameras_are_the_four_feeds(self):
        names = {feed.name for feed in self.backend.feeds.values()}
        self.assertEqual(names, {"overhead_camera", "left_wrist_camera", "right_wrist_camera", "scene_camera"})

    def test_policy_has_cartesian_operations_for_both_arms(self):
        policy = self.backend.policy()
        self.assertEqual(policy["version"], 2)
        for interface in ("left_arm", "right_arm"):
            self.assertEqual(policy["operations"][interface]["cartesian_pose"]["profile"], "arm.cartesian_pose.v1")

    def test_gateway_discovery_declares_the_simulation_control_extension(self):
        from sims.contract.gateway import ContractGateway

        import asyncio

        async def build():
            return ContractGateway(self.backend, self.config, "", "", 12).discovery()

        document = asyncio.run(build())
        self.assertEqual(document["simulation_control"]["implementation"], "mujoco.position.v2")
        json.dumps(document, allow_nan=False)

    def test_tool_frame_has_z_along_the_fingers_and_x_across_the_jaws(self):
        import mujoco
        import numpy as np

        model, data = self.backend.model, self.backend.data
        for side in ("left", "right"):
            site = model.site(f"{side}_ee_site").id
            rotation = data.site_xmat[site].reshape(3, 3)
            fingers = data.site_xpos[site] - data.xpos[model.site_bodyid[site]]
            fingers /= np.linalg.norm(fingers)
            self.assertGreater(float(rotation[:, 2] @ fingers), 0.99)
            left = data.xpos[model.body(f"{side}_carriage_left").id]
            right = data.xpos[model.body(f"{side}_carriage_right").id]
            across = (left - right) / np.linalg.norm(left - right)
            self.assertGreater(abs(float(rotation[:, 0] @ across)), 0.99)
            self.assertIsNotNone(mujoco)

    def test_props_stay_on_the_table(self):
        import mujoco
        import numpy as np

        model, data = self.backend.model, self.backend.data
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.assertGreaterEqual(len(free), 12)
        start = np.array([data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3] for j in free])
        for _ in range(200):
            self.backend.step(0.01)
        end = np.array([data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3] for j in free])
        self.assertTrue(np.isfinite(end).all())
        self.assertTrue((end[:, 2] > -0.02).all(), "a prop fell through the table")
        self.assertLess(float(np.abs(end[:, :2] - start[:, :2]).max()), 0.15, "a prop moved off its spot")

    def test_pose_commands_reach_poses_over_the_table_and_hold(self):
        import numpy as np
        from sims.contract.pose import parse_pose

        backend = self.backend
        for key in ("left_arm", "right_arm"):
            feedback = backend.pose_feedback(key)
            frame = feedback["header"]["frame_id"]
            pose = feedback["pose"]
            wanted = {"x": pose["position"]["x"] + 0.03, "y": pose["position"]["y"], "z": pose["position"]["z"] - 0.03}
            payload = json.dumps({"header": {"frame_id": frame},
                                  "pose": {"position": wanted, "orientation": pose["orientation"]}}).encode()
            backend.submit_pose(key, parse_pose(payload, frame))
            for _ in range(400):
                backend.step(0.01)
            got = backend.pose_feedback(key)["pose"]["position"]
            error = np.linalg.norm([got[a] - wanted[a] for a in "xyz"])
            self.assertLess(float(error), 0.02, f"{key} missed its pose by {error:.3f} m")


if __name__ == "__main__":
    unittest.main()
