"""The simulations `python -m sims run` can start.

A simulation works at one of two levels:

- participant: the simulation joins LiveKit itself and speaks the teleop
  protocol, standing in for the robot program. Used where the vendor's arm
  driver can't be pointed at a simulator (Trossen's libtrossen_arm).
- hardware: the simulation emulates the robot's hardware, and the real robot
  program, with its real driver, joins LiveKit and drives it. Preferred, as
  the code under test is then the code that runs on the robot.

To add a simulation, write a module with `add_arguments(parser, sim)` and
`run(args, sim) -> int`, and register it here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType


@dataclass(frozen=True)
class Simulation:
    name: str
    summary: str
    level: str  # "participant" or "hardware"
    module: str
    assets: tuple[str, ...]
    options: dict = field(default_factory=dict)
    managed_module: str | None = None
    icon: str = "arm"
    cleanup_seconds: float = 5

    def runner(self) -> ModuleType:
        # Imported on use, so `list` and `fetch-assets` need no MuJoCo or LiveKit.
        return importlib.import_module(self.module)


SIMULATIONS = {
    sim.name: sim
    for sim in (
        Simulation(
            "f1tenth",
            "",
            "participant",
            "sims.f1tenth.runner",
            ("f1tenth_racetracks",),
            managed_module="sims.f1tenth.managed",
            icon="car",
        ),
        Simulation(
            "vacuum",
            "",
            "participant",
            "sims.vacuum.runner",
            (),
            managed_module="sims.vacuum.managed",
            icon="vacuum",
        ),
        Simulation(
            "trossen-wxai",
            "",
            "participant",
            "sims.trossen.runner",
            ("trossen_arm_mujoco",),
            {"scene": "single_wxai"},
            "sims.trossen.managed",
        ),
        Simulation(
            "trossen-wxai-whack",
            "",
            "participant",
            "sims.trossen.runner",
            ("trossen_arm_mujoco",),
            {"scene": "whack_wxai"},
            "sims.trossen.managed",
        ),
        Simulation(
            "trossen-workbench",
            "",
            "participant",
            "sims.trossen.runner",
            ("trossen_arm_mujoco",),
            {"scene": "workbench"},
            "sims.trossen.managed",
            "dual-arm",
        ),
        Simulation(
            "trossen-workbench-whack",
            "",
            "participant",
            "sims.trossen.runner",
            ("trossen_arm_mujoco",),
            {"scene": "workbench_whack"},
            "sims.trossen.managed",
            "dual-arm",
        ),
        Simulation(
            "cluttered-table",
            "",
            "participant",
            "sims.trossen.runner",
            ("trossen_arm_mujoco",),
            {"scene": "cluttered_table"},
            "sims.cluttered_table.managed",
            "dual-arm",
        ),
        Simulation(
            "rebot-arm",
            "",
            "hardware",
            "sims.rebot.runner",
            ("rebot_b601_dm",),
            {"platform": "rebot_b601_dm"},
        ),
    )
}
