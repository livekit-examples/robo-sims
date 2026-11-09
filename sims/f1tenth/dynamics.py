"""Planar four-wheel RC handling. No rendering, networking, or wall clock."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from .energy import Battery

PARAMETERS = Path(__file__).with_name("vehicle.json")


@dataclass(frozen=True)
class Drive:
    speed: float
    steering: float


class Car:
    def __init__(self, track):
        self.p = json.loads(PARAMETERS.read_text())
        self.tuning = self.p["provisional"]
        self.battery = Battery(self.tuning)
        self.gear = "N"
        self.track = track
        self.x, self.y, self.yaw = track.spawn()
        self.vx = self.vy = self.yaw_rate = self.steering = 0.0
        self.wheels = np.zeros(4)
        self.slip = np.zeros(4)
        self.loads = np.zeros(4)
        self.forces = np.zeros((4, 2))
        self.ax = self.ay = self.time = 0.0
        self.target = Drive(0, 0)
        w = self.tuning["track_width"] / 2
        self.contacts = np.array([[self.p["lf"], w], [self.p["lf"], -w],
                                  [-self.p["lr"], w], [-self.p["lr"], -w]])
        self.grip = np.full(4, self.p["mu"])
        self.off_track = self.collision = False
        self.progress = 0.0
        self.laps = 0
        self.lap_valid = True
        self.last_lap_seconds = None
        self.lap_started = 0.0
        self.lap_distance = 0.0
        self.last_progress = track.project([self.x, self.y]).progress

    def wheel_angles(self):
        delta = self.steering
        if abs(delta) < 1e-8:
            return np.zeros(4)
        length = self.p["lf"] + self.p["lr"]
        radius = length / math.tan(delta)
        half = self.tuning["track_width"] / 2
        return np.array([math.atan(length / (radius - half)),
                         math.atan(length / (radius + half)), 0, 0])

    def _surfaces(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rotation = np.array([[c, -s], [s, c]])
        world = self.contacts @ rotation.T + [self.x, self.y]
        self.off_track = False
        for i, point in enumerate(world):
            road = self.track.project(point)
            outside = abs(road.lateral) > road.half_width
            self.off_track |= outside
            self.grip[i] = self.p["mu"] * (self.tuning["grass_grip_ratio"] if outside else 1)
        if self.off_track:
            self.lap_valid = False

    def step(self, dt=0.005):
        if not math.isfinite(dt) or not 0 < dt <= 0.02:
            raise ValueError("physics step must be positive and at most 0.02 seconds")
        self._surfaces()
        # Explicit wheel dynamics need finer steps than the external scheduler.
        count = math.ceil(dt / 0.001)
        for _ in range(count):
            self._integrate(dt / count)
        self._barriers()
        self._lap()

    def _integrate(self, dt):
        p, t = self.p, self.tuning
        error = self.target.steering - self.steering
        self.steering += float(np.clip(error / t["steering_response_seconds"],
                                      -p["max_steering_rate"], p["max_steering_rate"])) * dt
        angles = self.wheel_angles()
        co, si = np.cos(angles), np.sin(angles)
        ux = self.vx - self.yaw_rate * self.contacts[:, 1]
        uy = self.vy + self.yaw_rate * self.contacts[:, 0]
        longitudinal = co * ux + si * uy
        lateral = -si * ux + co * uy
        raw_slip = (self.wheels * t["tire_radius"] - longitudinal) / np.maximum(np.abs(longitudinal), 0.5)
        self.slip += (np.clip(raw_slip, -3, 3) - self.slip) * (1 - math.exp(-dt / t["slip_response_seconds"]))
        alpha = np.arctan2(lateral, np.maximum(np.abs(longitudinal), 0.4))
        length = p["lf"] + p["lr"]
        front = (p["mass"] * 9.81 * p["lr"] - p["mass"] * self.ax * p["cg_height"]) / length
        front = float(np.clip(front, 0, p["mass"] * 9.81))
        rear = p["mass"] * 9.81 - front
        transfer = p["mass"] * self.ay * p["cg_height"] / t["track_width"]
        self.loads[:] = [front / 2 - transfer * p["lr"] / length,
                         front / 2 + transfer * p["lr"] / length,
                         rear / 2 - transfer * p["lf"] / length,
                         rear / 2 + transfer * p["lf"] / length]
        self.loads[:] = np.maximum(self.loads, 0)
        self.loads *= p["mass"] * 9.81 / np.sum(self.loads)
        capacity = self.grip * self.loads
        stiffness = np.array([p["cornering_front"]] * 2 + [p["cornering_rear"]] * 2)
        fx = capacity * np.tanh(t["longitudinal_stiffness"] * self.slip / self.grip)
        fy = -capacity * np.tanh(stiffness * alpha / self.grip)
        scale = np.maximum(1, np.hypot(fx, fy) / np.maximum(capacity, 1e-9))
        fx, fy = fx / scale, fy / scale
        self.forces[:, 0], self.forces[:, 1] = fx, fy
        requested = self.target.speed
        # Change direction only after braking to rest.
        if requested * self.vx < 0 and abs(self.vx) > 0.15:
            requested = 0
        # A direction indicator for the fixed-ratio electric drive. Retain the
        # current direction during braking; reverse engages only near rest.
        direction = requested if requested != 0 else self.vx if abs(self.vx) > 0.06 else 0
        self.gear = "F" if direction > 0 else "R" if direction < 0 else "N"
        acceleration = float(np.clip(5 * (requested - self.vx), -p["max_acceleration"], p["max_acceleration"]))
        if acceleration * self.vx > 0 and abs(self.vx) > p["speed_switch"]:
            acceleration = float(np.clip(acceleration, -p["max_acceleration"] * p["speed_switch"] / abs(self.vx),
                                         p["max_acceleration"] * p["speed_switch"] / abs(self.vx)))
        torque = np.full(4, acceleration * p["mass"] * t["tire_radius"] / 4)
        braking = requested == 0 or (requested * self.vx >= 0 and acceleration * self.vx < 0)
        if self.battery.remaining_wh <= 0 and not braking:
            torque[:] = 0
        max_torque = p["max_acceleration"] * p["mass"] * t["tire_radius"] / 4
        drive_load = abs(float(torque[0])) / max_torque if not braking else 0
        positive_power = 0.0 if braking else float(np.maximum(torque * self.wheels, 0).sum())
        self.battery.step(dt, positive_power, drive_load)
        # An open differential per axle permits different cornering wheel speeds.
        free_wheels = self.wheels - (fx * t["tire_radius"] + 0.0003 * self.wheels) / t["wheel_inertia"] * dt
        if braking:
            # Braking is a passive impulse, not reverse motor torque. It may
            # stop a wheel but must never drive it through zero. Distribute the
            # requested braking by current normal load so an unloaded rear
            # wheel does not receive the same brake demand as a loaded front.
            brake_torque = abs(acceleration) * t["tire_radius"] * self.loads / 9.81
            reduction = brake_torque * dt / t["wheel_inertia"]
            self.wheels[:] = np.sign(free_wheels) * np.maximum(0, np.abs(free_wheels) - reduction)
        else:
            self.wheels[:] = free_wheels + torque / t["wheel_inertia"] * dt
        body_fx, body_fy = co * fx - si * fy, si * fx + co * fy
        drag = t["drag_coefficient"] * self.vx * abs(self.vx)
        rolling = t["rolling_resistance"] * p["mass"] * 9.81 * math.tanh(self.vx * 5)
        self.ax = float((sum(body_fx) - drag - rolling) / p["mass"])
        self.ay = float(sum(body_fy) / p["mass"])
        dvx = self.ax + self.yaw_rate * self.vy
        dvy = self.ay - self.yaw_rate * self.vx
        moment = float(sum(self.contacts[:, 0] * body_fy - self.contacts[:, 1] * body_fx))
        self.vx += dvx * dt
        self.vy += dvy * dt
        self.yaw_rate += moment / p["yaw_inertia"] * dt
        self.yaw = math.remainder(self.yaw + self.yaw_rate * dt, 2 * math.pi)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * self.vx - s * self.vy) * dt
        self.y += (s * self.vx + c * self.vy) * dt
        if requested == 0 and math.hypot(self.vx, self.vy) < 0.06 and abs(self.yaw_rate) < 0.15:
            self.vx = self.vy = self.yaw_rate = 0
            self.wheels[:] = self.slip[:] = 0
        self.time += dt

    def _barriers(self):
        road = self.track.project([self.x, self.y])
        heading = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        side = np.array([-heading[1], heading[0]])
        extent = abs(heading @ road.normal) * self.p["length"] / 2 + abs(side @ road.normal) * self.p["width"] / 2
        limit = road.half_width + self.track.barrier_runoff - extent
        self.collision = bool(abs(road.lateral) > limit)
        if self.collision:
            normal = road.normal * (1 if road.lateral > 0 else -1)
            self.x, self.y = np.array([self.x, self.y]) - normal * (abs(road.lateral) - limit)
            velocity = heading * self.vx + side * self.vy
            outward = float(velocity @ normal)
            if outward > 0:
                velocity -= 1.05 * outward * normal
            self.vx, self.vy = float(velocity @ heading), float(velocity @ side)
            self.yaw_rate *= 0.7
            self.lap_valid = False

    def _lap(self):
        progress = self.track.project([self.x, self.y]).progress
        delta = (progress - self.last_progress + self.track.length / 2) % self.track.length - self.track.length / 2
        self.lap_distance += delta
        # Crossing the start line counts only after a full forward circuit.
        if self.last_progress > self.track.length * 0.9 and progress < self.track.length * 0.1 \
                and self.lap_distance > self.track.length * 0.95:
            self.laps += 1
            if self.lap_valid:
                self.last_lap_seconds = self.time - self.lap_started
            self.lap_started = self.time
            self.lap_distance = 0
            self.lap_valid = not self.off_track
        self.last_progress = self.progress = progress

    def state(self):
        return {"simulation_time": self.time, "speed": self.vx, "lateral_speed": self.vy,
                "steering_angle": self.steering, "yaw_rate": self.yaw_rate,
                "position": [self.x, self.y], "heading": self.yaw,
                "slip_ratio": self.slip.tolist(), "off_track": self.off_track,
                "collision": self.collision, "lap": self.laps,
                "lap_valid": self.lap_valid, "lap_seconds": self.time - self.lap_started,
                "last_lap_seconds": self.last_lap_seconds, "progress_m": self.progress,
                "track_length_m": self.track.length}
