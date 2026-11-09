"""RC handling, community scale, watchdog, schema and ownership behavior."""
import asyncio
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import numpy as np

from sims.f1tenth.backend import Backend
from sims.f1tenth.dynamics import Car, Drive
from sims.f1tenth.track import Track


def fixture_track(directory):
    # A large closed rectangle; the first straight leaves room for brake tests.
    points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    rows = []
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        for t in np.linspace(0, 1, 100, endpoint=False):
            rows.append([*(a * (1-t) + b*t), 1.1, 1.1])
    path = Path(directory) / "track.csv"
    np.savetxt(path, rows, delimiter=",")
    return path


class RacingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = fixture_track(self.directory.name)
        self.now = 0.0
        self.backend = Backend(video=False, track_path=self.path, clock=lambda: self.now)
        self.car = self.backend.car

    def advance(self, seconds, drive):
        self.car.target = drive
        for _ in range(round(seconds / 0.005)):
            self.car.step()

    def test_community_dimensions_units_and_policy(self):
        self.assertEqual(self.car.p['length'], 0.58)
        self.assertEqual(self.car.p['width'], 0.31)
        self.assertAlmostEqual(self.car.p['lf'] + self.car.p['lr'], 0.3302)
        self.assertAlmostEqual(self.car.track.length, 400)
        self.assertAlmostEqual(self.car.track.project([10, 0.5]).half_width, 1.1)
        self.assertEqual(self.backend.policy()['max_speed'], 8)
        self.assertEqual(self.backend.policy()['command_timeout_ms'], 150)
        self.assertEqual(self.backend.policy()['steering_input_exponent'], 2)
        json.dumps(self.backend.state(), allow_nan=False)
        json.dumps(self.backend.odometry(), allow_nan=False)

    def test_mph_track_reports_ground_motion_including_reverse_and_slide(self):
        from sims.f1tenth.telemetry import SPEED_TRACK, FLOAT64
        self.assertEqual(self.backend.instrument_tracks()[SPEED_TRACK][0], FLOAT64)
        self.car.vx, self.car.vy = 3, 4
        self.assertAlmostEqual(self.backend.instrument_value(SPEED_TRACK)['data'], 11.184681460272)
        self.car.vx, self.car.vy = -5, 0
        self.assertAlmostEqual(self.backend.instrument_value(SPEED_TRACK)['data'], 11.184681460272)
        self.car.vx = self.car.vy = 0
        self.car.target = Drive(8, 0)
        self.assertEqual(self.backend.instrument_value(SPEED_TRACK), {'data': 0})
        with self.assertRaises(KeyError):
            self.backend.instrument_value('/not_an_instrument')

    def test_only_approved_instrument_tracks_and_units(self):
        expected = {'/speed_mph', '/gear', '/battery_energy', '/steering_angle_deg', '/yaw_rate_deg_s',
                    '/longitudinal_acceleration_g', '/lateral_acceleration_g', '/lap_number',
                    '/lap_time_s', '/last_valid_lap_time_s', '/track_progress_percent'}
        self.assertEqual(set(self.backend.instrument_tracks()), expected)
        self.car.steering, self.car.yaw_rate = math.pi / 6, math.pi / 2
        self.car.ax, self.car.ay = 9.81, -4.905
        self.car.progress = self.car.track.length / 4
        for name, number in (('/steering_angle_deg', 30), ('/yaw_rate_deg_s', 90),
                             ('/longitudinal_acceleration_g', 1), ('/lateral_acceleration_g', -0.5),
                             ('/track_progress_percent', 25)):
            self.assertAlmostEqual(self.backend.instrument_value(name)['data'], number)
        self.assertEqual(self.backend.instrument_value('/last_valid_lap_time_s'), {'data': None})
        for name in expected:
            json.dumps(self.backend.instrument_value(name), allow_nan=False)

    def test_fixed_ratio_drive_state_and_brake_before_reverse(self):
        self.assertEqual(self.backend.instrument_value('/gear'), {'data': 'N'})
        self.advance(1, Drive(4, 0))
        self.assertEqual(self.backend.instrument_value('/gear'), {'data': 'F'})
        self.car.target = Drive(-2, 0)
        self.car.step()
        self.assertGreater(self.car.vx, 0.15)
        self.assertEqual(self.car.gear, 'F')
        self.advance(2, Drive(-2, 0))
        self.assertLess(self.car.vx, -1)
        self.assertEqual(self.car.gear, 'R')
        self.advance(2, Drive(0, 0))
        self.assertEqual(self.car.gear, 'N')

    def test_battery_energy_uses_simulation_time_load_and_wh_units(self):
        from sims.f1tenth.energy import Battery
        battery = Battery(self.car.tuning)
        self.assertAlmostEqual(battery.capacity_wh, 55.5)
        initial = battery.remaining_wh
        battery.step(3600, 10, 0)
        self.assertAlmostEqual(initial - battery.remaining_wh, 15 + 10 / 0.85)
        # There is no regeneration: negative shaft power cannot recharge a pack.
        remaining = battery.remaining_wh
        battery.step(1, -500, 0)
        self.assertAlmostEqual(remaining - battery.remaining_wh, 15 / 3600)
        battery.step(100000, 1000, 1)
        self.assertEqual(battery.state()['remaining_wh'], 0)
        self.assertEqual(battery.state()['state_of_charge_percent'], 0)

        self.advance(1, Drive(0, 0))
        idle_used = self.car.battery.capacity_wh - self.car.battery.remaining_wh
        self.assertAlmostEqual(idle_used, 15 / 3600)
        before = self.car.battery.remaining_wh
        self.advance(1, Drive(4, 0))
        self.assertGreater(before - self.car.battery.remaining_wh, idle_used)
        state = self.backend.instrument_value('/battery_energy')
        self.assertGreater(state['state_of_charge_percent'], 0)
        self.assertLess(state['state_of_charge_percent'], 100)
        self.assertGreater(self.backend.battery_state()['voltage'], 9.9)
        self.assertLessEqual(self.backend.battery_state()['voltage'], 12.6 + 1e-9)

    def test_empty_battery_removes_propulsion_but_allows_braking(self):
        self.car.battery.remaining_wh = 0
        self.advance(0.5, Drive(4, 0))
        self.assertEqual(self.car.vx, 0)
        self.assertEqual(self.car.battery.remaining_wh, 0)
        self.car.vx = 3
        self.car.wheels[:] = 3 / self.car.tuning['tire_radius']
        self.advance(2, Drive(0, 0))
        self.assertLess(abs(self.car.vx), 0.1)

    def test_invalid_lap_retains_previous_valid_lap_time(self):
        self.car.last_lap_seconds = 42.0
        self.car.time = 80
        self.car.lap_started = 10
        self.car.lap_valid = False
        self.car.last_progress = self.car.track.length * 0.99
        self.car.lap_distance = self.car.track.length
        self.car._lap()
        self.assertEqual(self.car.laps, 1)
        self.assertEqual(self.backend.instrument_value('/last_valid_lap_time_s'), {'data': 42})
        self.car.time = 100
        self.car.last_progress = self.car.track.length * 0.99
        self.car.lap_distance = self.car.track.length
        self.car._lap()
        self.assertEqual(self.backend.instrument_value('/last_valid_lap_time_s'), {'data': 20})

    def test_invalid_commands_do_not_extend_watchdog_or_replace_target(self):
        self.backend.submit(b'{"speed":4,"steering_angle":0}')
        first = self.backend.last_command
        self.now += 0.1
        for payload in (b'{"speed":NaN,"steering_angle":0}', b'{"speed":true,"steering_angle":0}',
                        b'{"speed":9,"steering_angle":0}', b'{"speed":0,"steering_angle":1}',
                        b'{"speed":0,"steering_angle":0,"acceleration":1}',
                        b'{"speed":0,"speed":1,"steering_angle":0}', b'[]', b'{}', b'x' * 4097):
            with self.subTest(payload=payload[:100]), self.assertRaises(ValueError):
                self.backend.submit(payload)
        self.assertEqual(self.backend.last_command, first)
        self.assertEqual(self.car.target.speed, 4)
        self.now = 0.151
        self.assertTrue(self.backend.watchdog())
        self.assertEqual(self.car.target, Drive(0, 0))
        self.assertTrue(self.backend.state()['command_timed_out'])

    def test_rest_acceleration_braking_and_reverse(self):
        self.advance(0.5, Drive(0, 0))
        self.assertEqual(self.car.vx, 0)
        self.advance(2, Drive(4, 0))
        self.assertGreater(self.car.vx, 3.5)
        before = self.car.x
        velocity = self.car.vx
        self.car.target = Drive(0, 0)
        self.car.step()
        self.assertGreater(self.car.vx, 0)
        self.assertLess(abs(self.car.vx - velocity), 0.2)
        self.advance(2, Drive(0, 0))
        self.assertLess(abs(self.car.vx), 0.1)
        self.assertGreater(self.car.x, before)
        self.advance(2, Drive(-2, 0))
        self.assertLess(self.car.vx, -1.5)

    def test_braking_in_mild_fast_turn_does_not_reverse_wheels_or_spin(self):
        # Isolate braking from walls/grass: about 17 mph, 2.9 degree steering.
        self.car._surfaces = lambda: None
        self.car._barriers = lambda: None
        self.car._lap = lambda: None
        self.advance(3, Drive(8, 0))
        self.advance(0.5, Drive(8, 0.05))
        start_heading = self.car.yaw
        self.car.target = Drive(0, 0.05)
        peak_yaw = peak_sideslip = 0.0
        for _ in range(600):
            self.car.step()
            peak_yaw = max(peak_yaw, abs(self.car.yaw_rate))
            if self.car.vx > 0.5:
                self.assertGreaterEqual(float(self.car.wheels.min()), 0)
                peak_sideslip = max(peak_sideslip, abs(math.atan2(self.car.vy, self.car.vx)))
        self.assertLess(peak_sideslip, math.radians(10))
        self.assertLess(peak_yaw, math.radians(90))
        self.assertLess(abs(math.remainder(self.car.yaw - start_heading, 2 * math.pi)), math.pi / 2)
        self.assertEqual(self.car.vx, 0)
        self.assertEqual(self.car.vy, 0)

    def test_steering_rate_ackermann_and_friction_budget(self):
        self.car.target = Drive(3, 0.4)
        self.car.step()
        self.assertLessEqual(self.car.steering, 3.2 * 0.005 + 1e-9)
        self.advance(0.8, Drive(3, 0.3))
        self.assertGreater(self.car.yaw_rate, 0)
        left, right, _, _ = self.car.wheel_angles()
        self.assertGreater(left, right)
        self.assertTrue(np.isfinite(self.car.forces).all())
        self.assertAlmostEqual(sum(self.car.loads), 3.74 * 9.81)
        self.assertTrue(np.all(np.linalg.norm(self.car.forces, axis=1) <= self.car.grip * self.car.loads + 1e-8))

    def test_low_speed_turning_radius_matches_geometry(self):
        # Isolate handling from the narrow road boundary for a complete circle.
        self.car._surfaces = lambda: None
        self.car._barriers = lambda: None
        self.advance(5, Drive(0.5, 0.2))
        measured = abs(self.car.vx / self.car.yaw_rate)
        expected = (self.car.p['lf'] + self.car.p['lr']) / math.tan(0.2)
        self.assertLess(abs(measured - expected) / expected, 0.02)

    def test_grass_and_barrier_response(self):
        self.car.x, self.car.y = 10, 1.5
        self.car.step()
        self.assertTrue(self.car.off_track)
        self.assertFalse(self.car.lap_valid)
        self.assertLess(max(self.car.grip), self.car.p['mu'])
        self.car.y, self.car.vy = 2.2, 5
        for _ in range(30):
            self.car.step()
        road = self.car.track.project([self.car.x, self.car.y])
        self.assertLess(abs(road.lateral), 2.3)
        self.assertLess(abs(self.car.vy), 1)

    def test_cannot_count_laps_by_oscillating_at_start(self):
        for _ in range(10):
            self.car.x, self.car.y = 0, 0.1
            self.car._lap()
            self.car.x, self.car.y = 0.1, 0
            self.car._lap()
        self.assertEqual(self.car.laps, 0)

    def test_numerical_convergence_for_straight_stopping(self):
        distances = []
        for dt in (0.005, 0.0025):
            car = Car(Track(self.path))
            for target, duration in ((Drive(5, 0), 2), (Drive(0, 0), 2)):
                car.target = target
                for _ in range(round(duration / dt)):
                    car.step(dt)
            distances.append(car.x)
        self.assertLess(abs(distances[0] - distances[1]) / distances[1], 0.02)

    def test_render_model_and_calibration_without_graphics_context(self):
        from sims.f1tenth.scene import Scene
        from sims.contract.camera_info import validate_camera_info
        scene = Scene(self.car, 320, 180)
        validate_camera_info(scene.camera_info())
        self.assertEqual(scene.model.ncam, 2)
        self.assertEqual(scene.model.nmocap, 1)
        scene.close()

    def test_scene_presets_keep_draw_calls_low_and_metadata_exact(self):
        import mujoco
        from sims.f1tenth.scene import CAMERAS, Scene
        from sims.contract.camera_info import validate_camera_info
        for camera in CAMERAS:
            for graphics in ("low", "medium", "high"):
                scene = Scene(self.car, 320, 180, camera=camera, graphics=graphics)
                info = scene.camera_info()
                validate_camera_info(info)
                extrinsics = info["front"]["views"]["mono"]["extrinsics"]
                # Optical +Z (forward) must match the MuJoCo camera's -Z in base_link.
                x, y, z, w = extrinsics["rotation_xyzw"]
                optical = np.zeros(9)
                mujoco.mju_quat2Mat(optical, np.array([w, x, y, z]))
                camera_mat = np.zeros(9)
                mujoco.mju_quat2Mat(camera_mat, scene.model.cam_quat[scene.camera_id])
                np.testing.assert_allclose(optical.reshape(3, 3)[:, 2], -camera_mat.reshape(3, 3)[:, 2], atol=1e-6)
                self.assertLess(scene.model.ngeom, 120)  # merged scenery, not per-segment boxes
                scene.close()
        with self.assertRaises(ValueError):
            Scene(self.car, 320, 180, camera="drone")
        with self.assertRaises(ValueError):
            Scene(self.car, 320, 180, graphics="ultra")


class RacingGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_publication_and_neutral_required_after_handoff(self):
        from sims.f1tenth.gateway import DriveGateway
        from sims.contract.gateway import ROLE, VERSION
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend(video=False, track_path=fixture_track(directory))
            gateway = DriveGateway(backend, '', '', video=False)
            peer = SimpleNamespace(attributes={ROLE: 'operator', VERSION: '3'})
            gateway.room = SimpleNamespace(remote_participants={'alice': peer, 'bob': peer}, isconnected=lambda: True,
                local_participant=SimpleNamespace(set_attributes=AsyncMock()))
            old = SimpleNamespace(info=SimpleNamespace(sid='old'))
            gateway.readers['old'] = (old, None)
            await gateway._set_owner(SimpleNamespace(caller_identity='alice', payload='alice'))
            self.assertTrue(gateway._accepts('alice'))
            self.assertFalse(gateway._accepts('bob'))
            gateway._receive_command(old, 'base', {}, b'{"speed":4,"steering_angle":0}')
            self.assertEqual(backend.car.target.speed, 0)
            new = SimpleNamespace(info=SimpleNamespace(sid='new'))
            with self.assertRaises(ValueError):
                gateway._receive_command(new, 'base', {}, b'{"speed":4,"steering_angle":0}')
            gateway._receive_command(new, 'base', {}, b'{"speed":0,"steering_angle":0}')
            gateway._receive_command(new, 'base', {}, b'{"speed":4,"steering_angle":0}')
            self.assertEqual(backend.car.target.speed, 4)
            await gateway._expire_drive('test timeout')
            self.assertFalse(gateway._accepts('alice'))
            self.assertEqual(backend.car.target.speed, 0)
            self.assertEqual(gateway.active_operator, '')
