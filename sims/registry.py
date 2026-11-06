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

    def runner(self) -> ModuleType:
        # Imported on use, so `list` and `fetch-assets` need no MuJoCo or LiveKit.
        return importlib.import_module(self.module)


SIMULATIONS = {
    sim.name: sim
    for sim in (
        Simulation("trossen-wxai", "one Trossen WidowX AI arm with a wrist D405 and a scene camera",
                   "participant", "sims.trossen.runner", ("trossen_arm_mujoco",), {"scene": "single_wxai"}),
        Simulation("trossen-workbench", "two Trossen WidowX Pro arms with ZED X wrist and scene cameras",
                   "participant", "sims.trossen.runner", ("trossen_arm_mujoco",), {"scene": "workbench"}),
        Simulation("rebot-arm", "one Seeed reBot B601-DM arm and a scene camera on emulated hardware, driven by the robot program",
                   "hardware", "sims.rebot.runner", ("rebot_b601_dm",), {"platform": "rebot_b601_dm"}),
    )
}
