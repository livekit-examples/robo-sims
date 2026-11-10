"""Room layout and the 1 cm floor grid. No MuJoCo, rendering or networking.

Coordinates are meters. The room spans [-HALF, HALF] on both axes, north is +Y,
and the grid maps cell (row, col) to the point
x = -HALF + (col + 0.5) * CELL, y = -HALF + (row + 0.5) * CELL.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

HALF = 1.0                    # half the room side: a 2 x 2 m room
CELL = 0.01                   # grid square side in meters
N = round(2 * HALF / CELL)    # 200 squares per side
ROBOT_RADIUS = 0.17           # 0.34 m across
ROBOT_HEIGHT = 0.09
BRUSH_RADIUS = 0.17           # simplified circular cleaning footprint, 0.34 m across
SPAWN = (-0.82, -0.82, 0.0)   # x, y, yaw
MIN_PASSAGE = 0.42
# Positioning slack: the robot center must clear every solid by ROBOT_MARGIN, and a square counts as
# cleanable only if the brush reaches it with REACH_TOLERANCE to spare. A 1 cm strip beside walls and
# furniture is therefore reported as unreachable instead of demanding millimeter driving.
ROBOT_MARGIN = 0.005
REACH_TOLERANCE = 0.01

WALL_THICKNESS = 0.05
WALL_HEIGHT = 0.3

# Fixed furniture. Only the floor contact points are solid for the robot:
# the chair seat and bars are higher than the robot, so it can drive under them.
CHAIR_CENTER = (-0.05, 0.0)
CHAIR_LEG_SPACING = 0.50      # leg center to leg center
CHAIR_LEG_RADIUS = 0.02
CHAIR_SEAT_BOTTOM = 0.45      # a full-size chair seat, far above the 0.105 m robot top
CHAIR_BAR_Z = 0.20            # lower bars, also above the robot
GROUND_CLEARANCE = 0.015
ROBOT_TOP = GROUND_CLEARANCE + ROBOT_HEIGHT
CABINET_CENTER = (-0.25, 0.80)  # against the north wall
CABINET_HALF = (0.25, 0.15)
CABINET_HEIGHT = 0.4
LAMP_CENTER = (0.5, -0.5)
LAMP_RADIUS = 0.12
LAMP_HEIGHT = 0.05


def chair_legs() -> list[tuple[float, float]]:
    h = CHAIR_LEG_SPACING / 2
    cx, cy = CHAIR_CENTER
    return [(cx + sx * h, cy + sy * h) for sx in (-1, 1) for sy in (-1, 1)]


def cell_centers() -> tuple[np.ndarray, np.ndarray]:
    axis = -HALF + (np.arange(N) + 0.5) * CELL
    return np.meshgrid(axis, axis)  # X[row, col], Y[row, col]


def _circle(X, Y, center, radius):
    return np.hypot(X - center[0], Y - center[1]) - radius


def _box(X, Y, center, half):
    qx, qy = np.abs(X - center[0]) - half[0], np.abs(Y - center[1]) - half[1]
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def furniture_distance(X, Y) -> np.ndarray:
    """Distance to the nearest piece of furniture touching the floor. Negative inside."""
    parts = [_circle(X, Y, leg, CHAIR_LEG_RADIUS) for leg in chair_legs()]
    parts.append(_box(X, Y, CABINET_CENTER, CABINET_HALF))
    parts.append(_circle(X, Y, LAMP_CENTER, LAMP_RADIUS))
    return np.min(parts, axis=0)


def solid_distance(X, Y) -> np.ndarray:
    """Distance to the nearest wall or furniture. Negative inside a solid or past a wall."""
    walls = HALF - np.maximum(np.abs(X), np.abs(Y))
    return np.minimum(walls, furniture_distance(X, Y))


def _disk_offsets(radius: float) -> list[tuple[int, int]]:
    reach = int(np.ceil(radius / CELL))
    return [(dr, dc) for dr in range(-reach, reach + 1) for dc in range(-reach, reach + 1)
            if np.hypot(dr, dc) * CELL <= radius + 1e-9]


def disk_mask(radius: float = BRUSH_RADIUS) -> np.ndarray:
    reach = int(np.ceil(radius / CELL))
    size = 2 * reach + 1
    index = np.arange(size) - reach
    return np.hypot(*np.meshgrid(index, index)) * CELL <= radius + 1e-9


def dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    out = np.zeros_like(mask)
    for dr, dc in _disk_offsets(radius):
        src = mask[max(0, -dr):N - max(0, dr), max(0, -dc):N - max(0, dc)]
        out[max(0, dr):N - max(0, -dr) if dr < 0 else N, max(0, dc):N - max(0, -dc) if dc < 0 else N] |= src
    return out


def cell_of(x: float, y: float) -> tuple[int, int]:
    return int((y + HALF) // CELL), int((x + HALF) // CELL)


@dataclass(frozen=True)
class FloorMaps:
    exposed: np.ndarray       # floor inside the room, not under a leg, cabinet or lamp
    furniture: np.ndarray     # floor covered by furniture contact
    centers: np.ndarray       # positions the robot center can reach from the start
    cleanable: np.ndarray     # exposed floor the brush can reach
    unreachable: np.ndarray   # exposed floor the brush cannot reach


def build_maps(spawn=SPAWN) -> FloorMaps:
    X, Y = cell_centers()
    furniture = furniture_distance(X, Y) <= 0
    exposed = ~furniture
    free = solid_distance(X, Y) >= ROBOT_RADIUS + ROBOT_MARGIN
    start = cell_of(spawn[0], spawn[1])
    if not free[start]:
        raise ValueError("the robot start position is not free")
    reached = np.zeros_like(free)
    reached[start] = True
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < N and 0 <= nc < N and free[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                queue.append((nr, nc))
    cleanable = dilate(reached, BRUSH_RADIUS - REACH_TOLERANCE) & exposed
    return FloorMaps(exposed, furniture, reached, cleanable, exposed & ~cleanable)


def forward_clearance(x: float, y: float, yaw: float, reach: float = 1.2) -> float:
    """Free distance ahead of the front edge, across the robot's width. Walls and floor-level furniture only."""
    s = np.arange(ROBOT_RADIUS, ROBOT_RADIUS + reach, 0.01)
    lateral = np.linspace(-ROBOT_RADIUS, ROBOT_RADIUS, 9)
    c, sn = np.cos(yaw), np.sin(yaw)
    px = x + c * s[:, None] - sn * lateral[None, :]
    py = y + sn * s[:, None] + c * lateral[None, :]
    hit = (solid_distance(px, py) <= 0).any(axis=1)
    if not hit.any():
        return reach
    return float(s[np.argmax(hit)] - ROBOT_RADIUS)
