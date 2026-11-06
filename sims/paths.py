"""Where the simulations find their models, credentials and robot program."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = SIMULATION_DIR.parent

# The teleop repository: its controller app holds the LiveKit credentials and
# the identity names, and its robot program drives hardware-level simulations.
TELEOP_REPO = Path(os.environ.get("TELEOP_REPO") or WORKSPACE / "trossen-livekit")
CONTROLLER_ENV = TELEOP_REPO / "controller" / ".env.local"
SCIENTIST_NAMES_FILE = TELEOP_REPO / "controller" / "app" / "api" / "credentials" / "scientist-names.json"
ROBOT_PROGRAM = Path(os.environ.get("TELEOP_ROBOT_BIN") or TELEOP_REPO / "robot" / "build" / "trossen_livekit_robot")

# Third-party models, fetched by `python -m sims fetch-assets`.
ASSETS_DIR = Path(os.environ.get("SIM_ASSETS_DIR") or SIMULATION_DIR / "assets")

# Pseudo-terminals and other per-run files.
RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / f"sims-{os.getuid()}"
