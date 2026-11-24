"""One geometry source for rendering, tire surfaces, collisions and lap progress."""
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


@dataclass(frozen=True)
class Projection:
    point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    lateral: float
    progress: float
    half_width: float


class Track:
    barrier_runoff = 1.2

    def __init__(self, path: Path):
        rows = np.loadtxt(path, delimiter=",", comments="#")
        if rows.ndim != 2 or rows.shape[1] != 4 or len(rows) < 10 or not np.isfinite(rows).all():
            raise ValueError("track needs finite x/y/right-width/left-width rows in meters")
        if np.any(rows[:, 2:] <= 0):
            raise ValueError("track widths must be positive")
        if np.linalg.norm(rows[0, :2] - rows[-1, :2]) < 1e-6:
            rows = rows[:-1]
        self.rows = rows
        self.points = rows[:, :2]
        self.delta = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(self.delta, axis=1)
        if np.any(self.lengths < 1e-6):
            raise ValueError("track contains repeated points")
        self.tangents = self.delta / self.lengths[:, None]
        self.normals = np.column_stack((-self.tangents[:, 1], self.tangents[:, 0]))
        self.arc = np.r_[0, np.cumsum(self.lengths)]
        self.length = float(self.arc[-1])
        self.cell_size = 4.0
        self.cells = {}
        # Include the entire road/runoff and vehicle footprint in the lookup.
        for i, (p, q) in enumerate(zip(self.points, np.roll(self.points, -1, axis=0))):
            margin = float(max(rows[i, 2:])) + self.barrier_runoff + 1
            low = np.floor((np.minimum(p, q) - margin) / self.cell_size).astype(int)
            high = np.floor((np.maximum(p, q) + margin) / self.cell_size).astype(int)
            for x in range(low[0], high[0] + 1):
                for y in range(low[1], high[1] + 1):
                    self.cells.setdefault((x, y), []).append(i)

    def project(self, position) -> Projection:
        p = np.asarray(position)
        key = tuple(np.floor(p / self.cell_size).astype(int))
        ids = np.asarray(self.cells.get(key, range(len(self.points))), dtype=int)
        delta = p - self.points[ids]
        fraction = np.clip(np.sum(delta * self.delta[ids], axis=1) / self.lengths[ids] ** 2, 0, 1)
        closest = self.points[ids] + fraction[:, None] * self.delta[ids]
        choice = int(np.argmin(np.sum((closest - p) ** 2, axis=1)))
        i = ids[choice]
        lateral = float(np.dot(p - closest[choice], self.normals[i]))
        j = (i + 1) % len(self.points)
        side = 3 if lateral >= 0 else 2
        width = float((1 - fraction[choice]) * self.rows[i, side] + fraction[choice] * self.rows[j, side])
        return Projection(closest[choice], self.tangents[i], self.normals[i], lateral,
                          float(self.arc[i] + fraction[choice] * self.lengths[i]), width)

    def spawn(self):
        return (*self.points[0], math.atan2(self.tangents[0, 1], self.tangents[0, 0]))
