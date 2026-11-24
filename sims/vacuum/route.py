"""Reference route: a collision-free coverage path used to test the room. It is not automatic steering.

The route sweeps lanes, then visits any squares the lanes missed. A waypoint follower turns the path
into the same `Command` values an operator sends, so the simulation sees ordinary driving.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from . import room
from .room import CELL, N, ROBOT_MARGIN, ROBOT_RADIUS
from .simulation import MAX_FORWARD, MAX_YAW_RATE, Command, Vacuum

TRAVEL_MARGIN = 0.04      # keep this far from solids while travelling between lanes
WALL_CONTACT = ROBOT_MARGIN  # the smallest clearance the planner allows
LANE_MARGIN = 0.03        # lane pieces keep this far from furniture
LANE_SPACING = 0.31       # a lane cleans a 0.32 m wide band with the reach tolerance
LOOKAHEAD = 0.12


def _free_mask(margin: float, wall_margin: float | None = None) -> np.ndarray:
    """Floor where the robot center clears furniture by `margin` and walls by `wall_margin`."""
    X, Y = room.cell_centers()
    walls = room.HALF - np.maximum(np.abs(X), np.abs(Y))
    wall_margin = margin if wall_margin is None else wall_margin
    return (room.furniture_distance(X, Y) >= ROBOT_RADIUS + margin) & (walls >= ROBOT_RADIUS + wall_margin)


def _cell(p):
    return room.cell_of(p[0], p[1])


def _center(cell):
    return (-room.HALF + (cell[1] + 0.5) * CELL, -room.HALF + (cell[0] + 0.5) * CELL)


def plan_path(start, goal, free: np.ndarray):
    """Shortest 8-connected path between two free cells, then reduced to straight, clear segments."""
    a, b = _cell(start), _cell(goal)
    if not free[a] or not free[b]:
        raise ValueError("path ends must be on free floor")
    previous = {a: None}
    queue = deque([a])
    while queue:
        cell = queue.popleft()
        if cell == b:
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nxt = (cell[0] + dr, cell[1] + dc)
                if (dr or dc) and 0 <= nxt[0] < N and 0 <= nxt[1] < N and free[nxt] and nxt not in previous:
                    if dr and dc and not (free[cell[0] + dr, cell[1]] and free[cell[0], cell[1] + dc]):
                        continue
                    previous[nxt] = cell
                    queue.append(nxt)
    if b not in previous:
        raise ValueError("no path between the cells")
    cells = []
    while b is not None:
        cells.append(b)
        b = previous[b]
    cells.reverse()
    points = [_center(c) for c in cells]
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not _clear(points[i], points[j], free):
            j -= 1
        out.append(points[j])
        i = j
    return out


def _clear(p, q, free):
    steps = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / (CELL / 2)))
    for t in np.linspace(0, 1, steps):
        if not free[_cell((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))]:
            return False
    return True


def lane_waypoints(start, ys=None):
    """Lane sweeps along x, alternating direction, split wherever furniture interrupts a lane."""
    contact = _free_mask(WALL_CONTACT)
    travel = _free_mask(TRAVEL_MARGIN, TRAVEL_MARGIN)
    lane = _free_mask(LANE_MARGIN, ROBOT_MARGIN)
    reach = room.HALF - ROBOT_RADIUS - ROBOT_MARGIN
    waypoints = [start[:2]]
    strict = set()
    heading_east = True
    lanes = math.ceil(2 * reach / LANE_SPACING) + 1
    for y in (np.linspace(-reach, reach, lanes) if ys is None else ys):
        row = room.cell_of(0, y)[0]
        line = lane[row]
        runs, begin = [], None
        for col in range(N + 1):
            on = col < N and line[col]
            if on and begin is None:
                begin = col
            elif not on and begin is not None:
                runs.append((begin, col - 1))
                begin = None
        runs = [r for r in runs if r[1] - r[0] > 2]
        if not heading_east:
            runs.reverse()
        for first, last in runs:
            x0, x1 = _center((row, first))[0], _center((row, last))[0]
            # The wall limit is exact; the grid center sits up to half a cell inside it.
            # A piece that ends at a wall runs on to the exact wall limit; others stay off the furniture.
            x0 = -reach if x0 < -reach + 0.04 else x0
            x1 = reach if x1 > reach - 0.04 else x1
            ends = (x0, x1) if heading_east else (x1, x0)
            for x in ends:
                waypoints.append((x, float(y)))
                strict.add((x, float(y)))
        heading_east = not heading_east
    # Join the pieces with clear paths.
    path = [waypoints[0]]
    for target in waypoints[1:]:
        here = path[-1]
        if math.hypot(target[0] - here[0], target[1] - here[1]) < 1e-6:
            continue
        try:
            a, b = _snap(here, travel), _snap(target, travel)
            points = plan_path(a, b, travel)[1:]
        except ValueError:  # a pocket only the contact map connects
            points = plan_path(_snap(here, contact), _snap(target, contact), contact)[1:]
        path.extend(points)
        path[-1] = target if _near_free(target, contact) else path[-1]
    return path, strict


def _snap(point, free):
    cell = _cell(point)
    if free[cell]:
        return _center(cell)
    best, best_d = None, 1e9
    r0, c0 = cell
    for dr in range(-12, 13):
        for dc in range(-12, 13):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < N and 0 <= c < N and free[r, c] and dr * dr + dc * dc < best_d:
                best, best_d = (r, c), dr * dr + dc * dc
    if best is None:
        raise ValueError("no free floor near the point")
    return _center(best)


def _near_free(point, free):
    return bool(free[_cell(point)])


class Follower:
    """Waypoint follower. `nearest_dirty` supplies cleanup targets after the lanes end."""

    def __init__(self, sim: Vacuum, speed=0.42, tolerance=0.012, ys=None):
        self.sim, self.speed, self.tolerance = sim, speed, tolerance
        path, self.strict = lane_waypoints(room.SPAWN, ys)
        self.path = path[1:]
        self.travel = _free_mask(TRAVEL_MARGIN, TRAVEL_MARGIN)
        self.contact = _free_mask(WALL_CONTACT)
        self.cleanup = False
        self.done = False
        self.skipped = set()

    def _next_cleanup(self, x, y):
        dirty = self.sim.floor.dirty.copy()
        for cell in self.skipped:
            dirty[cell] = False
        if not dirty.any():
            return None
        rows, cols = np.nonzero(dirty)
        cx = -room.HALF + (cols + 0.5) * CELL
        cy = -room.HALF + (rows + 0.5) * CELL
        free_r, free_c = np.nonzero(self.contact)
        fx = -room.HALF + (free_c + 0.5) * CELL
        fy = -room.HALF + (free_r + 0.5) * CELL
        for k in np.argsort(np.hypot(cx - x, cy - y))[:25]:
            target = (float(cx[k]), float(cy[k]))
            # Closest free robot center that still reaches the dirty square with the tolerance to spare.
            inside = np.hypot(fx - target[0], fy - target[1]) <= room.BRUSH_RADIUS - room.REACH_TOLERANCE / 2
            if not inside.any():
                self.skipped.add((int(rows[k]), int(cols[k])))
                continue
            d = np.hypot(fx - x, fy - y)
            d[~inside] = 1e9
            j = int(np.argmin(d))
            goal = (float(fx[j]), float(fy[j]))
            for mask in (self.travel, self.contact):
                try:
                    path = plan_path(_snap((x, y), mask), _snap(goal, mask), mask)[1:]
                except ValueError:
                    continue
                return path + [goal]
            self.skipped.add((int(rows[k]), int(cols[k])))
        return "skip"

    def __call__(self) -> Command:
        sim = self.sim
        x, y, yaw = sim.position()
        while self.path and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < (
                self.tolerance if self.path[0] in self.strict or len(self.path) == 1 else 0.05):
            self.path.pop(0)
        if not self.path:
            if sim.floor.complete or sim.state != "running":
                self.done = True
                return Command()
            new = self._next_cleanup(x, y)
            if new == "skip":
                return Command()
            if new is None:
                self.done = True
                return Command()
            self.path = new
            self.cleanup = True
        # Aim at the first waypoint that is at least LOOKAHEAD away, or at a strict waypoint that
        # must be reached. Cutting a corner by a few centimeters stays inside TRAVEL_MARGIN.
        target = self.path[-1]
        for point in self.path:
            if point in self.strict or math.hypot(point[0] - x, point[1] - y) >= LOOKAHEAD:
                target = point
                break
        tx, ty = target
        distance = math.hypot(tx - x, ty - y)
        want = (tx - x, ty - y)
        # Slide along solids: near one, drop the part of the heading that points into it.
        clearance = float(room.solid_distance(np.array(x), np.array(y))) - ROBOT_RADIUS
        if clearance < 0.04 and distance > 0.06:
            eps = 0.004
            gx = float(room.solid_distance(np.array(x + eps), np.array(y)) - room.solid_distance(np.array(x - eps), np.array(y)))
            gy = float(room.solid_distance(np.array(x), np.array(y + eps)) - room.solid_distance(np.array(x), np.array(y - eps)))
            norm = math.hypot(gx, gy) or 1.0
            gx, gy = gx / norm, gy / norm
            into = want[0] * gx + want[1] * gy
            if into < 0:
                want = (want[0] - into * gx, want[1] - into * gy)
        error = math.atan2(want[1], want[0]) - yaw
        error = math.atan2(math.sin(error), math.cos(error))
        omega = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, 4.0 * error))
        v = self.speed * max(0.0, math.cos(error)) ** 2
        if target in self.strict or target is self.path[-1]:
            v = min(v, math.sqrt(2 * 0.7 * distance) + 0.02)
        if abs(error) > 0.9:
            v = 0.0
        return Command(min(v, MAX_FORWARD), omega)


def drive_route(sim: Vacuum, limit_s=50.0, control_hz=50, start_run=True, record=None, ys=None):
    """Drive the reference route offline. Returns a result dict."""
    follower = Follower(sim, ys=ys)
    if start_run:
        sim.start_run()
    ticks = round(200 / control_hz)
    started = sim.time
    while sim.state == "running" and sim.time - started < limit_s + 15 and not follower.done:
        sim.requested = follower()
        for _ in range(ticks):
            sim.step()
        if record is not None:
            record(sim)
    return {"state": sim.state, "result": sim.result, "elapsed_s": sim.elapsed, "percent": sim.floor.percent,
            "remaining_squares": int(sim.floor.dirty.sum()), "cleanup": follower.cleanup}
