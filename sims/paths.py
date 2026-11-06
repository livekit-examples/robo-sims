"""Where the simulations find their models, credentials and robot program."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = SIMULATION_DIR.parent

# LiveKit credentials of the simulations (see .env.example).
ENV_FILE = SIMULATION_DIR / ".env.local"



def _teleop_repo() -> Path:
    """The teleop repository: its controller app holds the identity names, and
    its robot program drives hardware-level simulations.

    This repository is usually its robo-sims/ submodule; otherwise it is
    looked for beside this one, as trossen-livekit/.
    """
    if os.environ.get("TELEOP_REPO"):
        return Path(os.environ["TELEOP_REPO"])
    if (WORKSPACE / "robot" / "CMakeLists.txt").is_file() and (WORKSPACE / "controller").is_dir():
        return WORKSPACE
    return WORKSPACE / "trossen-livekit"


TELEOP_REPO = _teleop_repo()
SCIENTIST_NAMES_FILE = TELEOP_REPO / "controller" / "app" / "api" / "credentials" / "scientist-names.json"
ROBOT_PROGRAM = Path(os.environ.get("TELEOP_ROBOT_BIN") or TELEOP_REPO / "robot" / "build" / "trossen_livekit_robot")

# Third-party models, fetched by `python -m sims fetch-assets`.
ASSETS_DIR = Path(os.environ.get("SIM_ASSETS_DIR") or SIMULATION_DIR / "assets")

# Pseudo-terminals and other per-run files.
RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / f"sims-{os.getuid()}"
