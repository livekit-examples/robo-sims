"""Floor cleaning state: one 1 cm grid drives the image, the cleaned-area counter and the bin."""
from __future__ import annotations

import math

import numpy as np

from . import room
from .room import BRUSH_RADIUS, CELL, N

# One-time constants for the floor image (RGB).
BASE = np.array([196, 172, 138], dtype=float)         # clean floor
DUST = np.array([205, 125, 20], dtype=float)          # amber dust tint
DUST_STRENGTH = 0.62
FURNITURE = np.array([62, 62, 70], dtype=float)       # floor under furniture contact points
UNREACHABLE = np.array([104, 124, 156], dtype=float)  # exposed floor the brush cannot reach
LINE = 0.93                                           # faint 0.5 m lines make motion visible
DUST_LITERS_PER_M2 = 0.04


class Floor:
    def __init__(self, maps: room.FloorMaps | None = None):
        self.maps = maps or room.build_maps()
        self.X, self.Y = room.cell_centers()
        self.reach = int(math.ceil(BRUSH_RADIUS / CELL)) + 1
        self.version = 0
        self.reset()

    def reset(self):
        self.dirty = self.maps.cleanable.copy()
        self.cleaned_cells = 0
        self.version += 1

    @property
    def cleanable_cells(self) -> int:
        return int(self.maps.cleanable.sum())

    def areas_m2(self) -> dict:
        unit = CELL * CELL
        return {"cleaned_m2": self.cleaned_cells * unit, "cleanable_m2": self.cleanable_cells * unit,
                "exposed_m2": int(self.maps.exposed.sum()) * unit,
                "unreachable_m2": int(self.maps.unreachable.sum()) * unit}

    @property
    def percent(self) -> float:
        return 100.0 * self.cleaned_cells / max(1, self.cleanable_cells)

    @property
    def complete(self) -> bool:
        return self.cleaned_cells >= self.cleanable_cells

    def stamp(self, x: float, y: float, allowed: int) -> int:
        """Clean dirty cleanable squares inside the footprint at (x, y). Returns squares cleaned, at most `allowed`."""
        if allowed <= 0:
            return 0
        r0, c0 = room.cell_of(x, y)
        r_lo, r_hi = max(0, r0 - self.reach), min(N, r0 + self.reach + 1)
        c_lo, c_hi = max(0, c0 - self.reach), min(N, c0 + self.reach + 1)
        if r_lo >= r_hi or c_lo >= c_hi:
            return 0
        window = (np.hypot(self.X[r_lo:r_hi, c_lo:c_hi] - x, self.Y[r_lo:r_hi, c_lo:c_hi] - y)
                  <= BRUSH_RADIUS + 1e-9) & self.dirty[r_lo:r_hi, c_lo:c_hi]
        hits = np.flatnonzero(window)
        if hits.size == 0:
            return 0
        hits = hits[:allowed]
        rows, cols = np.divmod(hits, c_hi - c_lo)
        self.dirty[r_lo + rows, c_lo + cols] = False
        self.cleaned_cells += int(hits.size)
        self.version += 1
        return int(hits.size)

    def sweep(self, start, end, allowed: int) -> int:
        """Clean along the path from start to end, so fast turns and contacts leave no gaps."""
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        samples = max(1, math.ceil(distance / (CELL / 2)))
        total = 0
        for i in range(1, samples + 1):
            t = i / samples
            total += self.stamp(start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t,
                                allowed - total)
        return total

    def image(self) -> np.ndarray:
        """RGB uint8, shape (N, N, 3), row 0 at the south edge (y = -1)."""
        img = np.empty((N, N, 3), dtype=float)
        img[:] = BASE
        index = np.arange(N)
        lines = (index % 50 == 0)
        img[lines, :, :] *= LINE
        img[:, lines, :] *= LINE
        exposed_dirty = self.maps.exposed & ~self.maps.cleanable
        img[exposed_dirty] = UNREACHABLE
        img[self.dirty] = BASE * (1 - DUST_STRENGTH) + DUST * DUST_STRENGTH
        img[self.maps.furniture] = FURNITURE
        return img.astype(np.uint8)


class Bin:
    def __init__(self, capacity_l=0.5, used_l=0.0):
        self.capacity_l = capacity_l
        self.used_l = used_l

    @property
    def remaining_l(self) -> float:
        return max(0.0, self.capacity_l - self.used_l)

    @property
    def full(self) -> bool:
        # Full means not even one more square fits; this also absorbs float rounding.
        return self.remaining_l < DUST_LITERS_PER_M2 * CELL * CELL / 2

    def cells_allowed(self) -> int:
        per_cell = DUST_LITERS_PER_M2 * CELL * CELL
        return int(self.remaining_l / per_cell + 1e-6)

    def add_cells(self, count: int):
        self.used_l = min(self.capacity_l, self.used_l + count * DUST_LITERS_PER_M2 * CELL * CELL)

    def state(self) -> dict:
        return {"used_l": self.used_l, "total_l": self.capacity_l, "remaining_l": self.remaining_l,
                "fill_percent": 100 * self.used_l / self.capacity_l, "full": self.full}


class Battery:
    """Simulation-time energy accounting. All power values are initial assumptions, not product data."""
    ELECTRONICS_W = 5.0
    VACUUM_W = 18.0
    DRIVE_W_AT_LIMIT = 8.0
    TURN_W_AT_LIMIT = 4.0
    NOMINAL_V = 14.4

    def __init__(self, capacity_wh=30.0, remaining_wh=None):
        self.capacity_wh = capacity_wh
        self.remaining_wh = capacity_wh if remaining_wh is None else remaining_wh
        self.power_w = 0.0

    def step(self, dt, speed_fraction, yaw_fraction, vacuum_on):
        if self.remaining_wh <= 0:
            self.power_w = 0.0
            return
        self.power_w = max(0.0, self.ELECTRONICS_W + (self.VACUUM_W if vacuum_on else 0.0)
                           + self.DRIVE_W_AT_LIMIT * min(1.0, abs(speed_fraction))
                           + self.TURN_W_AT_LIMIT * min(1.0, abs(yaw_fraction)))
        self.remaining_wh = max(0.0, self.remaining_wh - self.power_w * dt / 3600)

    @property
    def empty(self) -> bool:
        return self.remaining_wh <= 0.0

    def voltage(self) -> float:
        # A linear state-of-charge estimate for 4 cells at 3.0 to 4.2 V, not a fitted discharge curve.
        return 12.0 + 4.8 * self.remaining_wh / self.capacity_wh

    def state(self) -> dict:
        return {"remaining_wh": self.remaining_wh, "total_wh": self.capacity_wh,
                "charge_percent": 100 * self.remaining_wh / self.capacity_wh,
                "voltage_v": self.voltage(), "power_w": self.power_w}
