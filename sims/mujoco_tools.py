"""MuJoCo helpers shared by the simulations."""

from __future__ import annotations

import os
import sys

# Render offscreen through EGL on Linux so no X display is needed. This must
# be set before mujoco is imported.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def look_at(position: tuple[float, float, float],
            target: tuple[float, float, float]) -> list[float]:
    """The quaternion of a camera at `position` that looks at `target`, with +z up."""
    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    return quaternion.tolist()
