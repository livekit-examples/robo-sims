"""Contract configuration and trajectory validation; standard library only."""

import copy
import json
import unittest

from sims.contract.config import CONFIGS, Configuration, load_json
from sims.contract.trajectory import Trajectory


def trajectory(names=("arm_joint_1",), points=((1, (0.1,)),)):
    return json.dumps({"joint_names": list(names), "points": [
        {"positions": list(values), "time_from_start": {"sec": int(at),
         "nanosec": round((at - int(at)) * 1e9)}} for at, values in points
    ]}).encode()


class ConfigurationTests(unittest.TestCase):
    def test_all_trossen_configs_use_one_feedback_list_for_arm_and_gripper(self):
        for path in CONFIGS.glob("*.bindings.json"):
            with self.subTest(config=path.name):
                config = Configuration.load(path, 12)
                for interface in config.discovery["control_interfaces"].values():
                    for operation in interface["operations"].values():
                        self.assertEqual(len(operation["feedback_tracks"]), 1)

    def test_position_profiles_reject_missing_empty_or_multiple_feedback_tracks(self):
        original = Configuration.load(CONFIGS / "trossen-wxai.bindings.json", 12)
        for operation_id in ("joint_position", "gripper_position"):
            for feedback in (None, [], ["/one", "/two"]):
                config = copy.deepcopy(original)
                operation = config.discovery["control_interfaces"]["arm"]["operations"][operation_id]
                if feedback is None:
                    del operation["feedback_tracks"]
                else:
                    operation["feedback_tracks"] = feedback
                with self.subTest(operation=operation_id, feedback=feedback), self.assertRaises(ValueError):
                    config.validate()

    def test_gripper_requires_explicit_separate_bound_joint_channel_and_endpoint(self):
        original = Configuration.load(CONFIGS / "trossen-wxai.bindings.json", 12)
        for field, value in (("joint_names", ["arm_joint_1"]), ("control_channel", "primary"),
                             ("joint_names", []), ("control_channel", "bad:channel")):
            config = copy.deepcopy(original)
            config.discovery["control_interfaces"]["arm"]["operations"]["gripper_position"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                config.validate()
        gripper = original.discovery["control_interfaces"]["arm"]["operations"]["gripper_position"]
        command = Trajectory.parse(trajectory(gripper["joint_names"], ((3, (0.04,)),)), gripper["joint_names"])
        self.assertEqual(command.positions, ((0.04,),))
        with self.assertRaises(ValueError):
            Trajectory.parse(trajectory(), gripper["joint_names"])

    def test_single_and_dual_share_the_same_profile_and_endpoint_contract(self):
        single = Configuration.load(CONFIGS / "trossen-wxai.bindings.json", 17)
        dual = Configuration.load(CONFIGS / "trossen-workbench.bindings.json", 12)
        self.assertEqual(single.discovery["default_video_rate_hz"], 17)
        self.assertEqual(set(single.discovery["control_interfaces"]), {"arm"})
        self.assertEqual(set(dual.discovery["control_interfaces"]), {"left_arm", "right_arm"})
        for config in (single, dual):
            self.assertNotIn("video_tracks", config.discovery)
            for interface in config.discovery["control_interfaces"].values():
                self.assertEqual(len(interface["joint_names"]), 6)
                self.assertEqual(set(interface["operations"]), {"joint_position", "gripper_position"})
                self.assertEqual(interface["operations"]["joint_position"]["profile"], "joints.trajectory.v1")

    def test_rejects_duplicate_keys_and_non_finite_json(self):
        for raw in ('{"a":1,"a":2}', '{"points":NaN}', '{"x":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_json(raw)

    def test_rejects_unbound_or_unimplemented_controls(self):
        original = Configuration.load(CONFIGS / "trossen-wxai.bindings.json", 12)
        for mutation in ("joint", "profile", "feedback", "legacy", "version", "speed", "exclude"):
            config = copy.deepcopy(original)
            interface = config.discovery["control_interfaces"]["arm"]
            op = interface["operations"]["joint_position"]
            if mutation == "joint":
                interface["joint_names"].append("invented")
            elif mutation == "profile":
                op["profile"] = "arm.cartesian_pose.v1"
            elif mutation == "feedback":
                op["feedback_tracks"] = []
            elif mutation == "legacy":
                config.discovery["topics"] = {}
            elif mutation == "version":
                config.discovery["schema_version"] = True
            elif mutation == "speed":
                config.bindings["control_interfaces"]["arm"]["joint_map"]["arm_joint_1"]["max_velocity"] = -1
            else:
                config.discovery["exclude_data_tracks"] = ["/arm/.*"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                config.validate()

    def test_shared_feedback_catalog_is_derived_from_operations(self):
        config = Configuration.load(CONFIGS / "trossen-workbench.bindings.json", 12)
        for interface in config.discovery["control_interfaces"].values():
            for op in interface["operations"].values():
                op["feedback_tracks"] = ["/joint_states"]
        config.validate()
        self.assertEqual(config.feedback_tracks, {"/joint_states": ["left_arm", ("left_arm", "gripper"), "right_arm", ("right_arm", "gripper")]})


class TrajectoryTests(unittest.TestCase):
    def test_timed_waypoints_interpolate_and_finish(self):
        command = Trajectory.parse(trajectory(points=((1, (0.2,)), (2, (-0.2,)))), ["arm_joint_1"])
        command.validate_motion([0], [(-1, 1)], [0.5])
        self.assertAlmostEqual(command.sample(0.5, [0])[0], 0.1)
        self.assertAlmostEqual(command.sample(1.5, [0])[0], 0)
        self.assertAlmostEqual(command.sample(10, [0])[0], -0.2)

    def test_empty_points_is_explicit_cancel_not_a_zero_pose(self):
        self.assertEqual(Trajectory.parse(trajectory(points=()), ["arm_joint_1"]).times, ())
        self.assertEqual(Trajectory.parse(trajectory(points=((1, (0,)),)), ["arm_joint_1"]).positions, ((0.0,),))

    def test_joint_names_select_instead_of_array_order(self):
        command = Trajectory.parse(trajectory(("b", "a"), ((2, (0.2, -0.1)),)), ["a", "b"])
        self.assertEqual(command.names, ("b", "a"))
        self.assertEqual(command.sample(2, [0, 0]), (0.2, -0.1))

    def test_rejects_bad_shapes_units_times_and_names(self):
        valid = json.loads(trajectory())
        messages = [None, [], {**valid, "joint_names": ["unknown"]},
                    {**valid, "joint_names": ["arm_joint_1", "arm_joint_1"]},
                    {**valid, "header": {"stamp": {"sec": 1, "nanosec": 0}}}]
        for field, value in (("positions", [True]), ("positions", [1e309]),
                             ("positions", []), ("velocities", [0]),
                             ("time_from_start", {"sec": 0, "nanosec": 0}),
                             ("time_from_start", {"sec": -1, "nanosec": 0}),
                             ("time_from_start", {"sec": 31, "nanosec": 0})):
            message = copy.deepcopy(valid)
            message["points"][0][field] = value
            messages.append(message)
        for message in messages:
            with self.subTest(message=message), self.assertRaises((ValueError, TypeError)):
                Trajectory.parse(json.dumps(message).encode(), ["arm_joint_1"])

    def test_rejects_limits_and_fast_motion_atomically(self):
        command = Trajectory.parse(trajectory(points=((1, (0.6,)),)), ["arm_joint_1"])
        with self.assertRaises(ValueError):
            command.validate_motion([0], [(-1, 1)], [0.5])
        with self.assertRaises(ValueError):
            command.validate_motion([0], [(-0.5, 0.5)], [1])


if __name__ == "__main__":
    unittest.main()
