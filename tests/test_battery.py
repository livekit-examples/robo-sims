import unittest

from sims.contract.battery import battery_state, battery_track


class BatteryTest(unittest.TestCase):
    def test_voltage_is_24_minus_small_noise(self):
        for _ in range(500):
            state = battery_state()
            self.assertGreaterEqual(state["voltage"], 23.75)
            self.assertLessEqual(state["voltage"], 23.99)
            self.assertTrue(state["present"])

    def test_track_name(self):
        self.assertEqual(battery_track("left_arm"), "/left_arm/battery_state")


if __name__ == "__main__":
    unittest.main()
