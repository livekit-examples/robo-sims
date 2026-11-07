"""Runs a simulated robot.

    python -m sims list
    python -m sims fetch-assets [SIM ...]
    python -m sims run SIM [--check] [options]      # python -m sims run SIM --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import assets
from .paths import ENV_FILE
from .registry import SIMULATIONS


def add_livekit_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LiveKit")
    group.add_argument("--env-file", type=Path, default=ENV_FILE,
                       help="LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET and LIVEKIT_ROOM_NAME "
                            "(default: %(default)s). Variables in the environment take precedence.")
    group.add_argument("--url", default=None, help="LiveKit URL (default: LIVEKIT_URL)")
    group.add_argument("--token", default=None,
                       help="join with this token instead of creating one (default: LIVEKIT_TOKEN)")
    group.add_argument("--identity", default=None,
                       help="stable robot identity for saved controller bindings (default: pick a free scientist name)")
    group.add_argument("--controller-prefix", default="controller-",
                       help="operator identity prefix for legacy/hardware runners (default: %(default)s)")


def parser(selected: str | None = None) -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m sims", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the simulations")
    fetch = commands.add_parser("fetch-assets", help="download the models of the simulations")
    fetch.add_argument("sims", nargs="*", metavar="SIM", help="simulations to fetch for (default: all)")
    fetch.add_argument("--force", action="store_true", help="fetch again even if present")
    run = commands.add_parser("run", help="run one simulation")
    sims = run.add_subparsers(dest="sim", required=True, metavar="SIM")
    for sim in SIMULATIONS.values():
        if selected is not None and sim.name != selected:
            continue
        sub = sims.add_parser(sim.name, help=sim.summary, description=f"{sim.summary} ({sim.level} level)")
        sub.add_argument("--check", action="store_true",
                         help="load the simulation and print what it provides, without joining LiveKit")
        add_livekit_arguments(sub)
        sim.runner().add_arguments(sub.add_argument_group(sim.name), sim)
    return root


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("OpenGL").setLevel(logging.WARNING)  # PyOpenGL, imported by MuJoCo's renderer
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] in (["list"], ["fetch-assets"]):
        # Parse these without importing the simulations.
        args = argparse.Namespace(command=argv[0])
        if argv[0] == "fetch-assets":
            fetch = argparse.ArgumentParser(prog="python -m sims fetch-assets")
            fetch.add_argument("sims", nargs="*", metavar="SIM")
            fetch.add_argument("--force", action="store_true")
            args = fetch.parse_args(argv[1:], namespace=args)
    else:
        args = parser(argv[1] if len(argv) > 1 and argv[0] == "run" else None).parse_args(argv)

    if args.command == "list":
        width = max(map(len, SIMULATIONS))
        for sim in SIMULATIONS.values():
            ready = all(assets.SOURCES[name].installed() for name in sim.assets)
            print(f"{sim.name:<{width}}  {sim.level:<11}  {sim.summary}{'' if ready else '  [run fetch-assets]'}")
        return 0
    if args.command == "fetch-assets":
        unknown = [name for name in args.sims if name not in SIMULATIONS]
        if unknown:
            print(f"unknown simulation: {', '.join(unknown)}; see `python -m sims list`", file=sys.stderr)
            return 2
        wanted = args.sims or list(SIMULATIONS)
        for name in dict.fromkeys(a for sim in wanted for a in SIMULATIONS[sim].assets):
            assets.fetch(assets.SOURCES[name], force=args.force)
        return 0

    sim = SIMULATIONS[args.sim]
    missing = [name for name in sim.assets if not assets.SOURCES[name].installed()]
    if missing:
        print(f"{sim.name} needs its models first: python -m sims fetch-assets {sim.name}", file=sys.stderr)
        return 2
    return sim.runner().run(args, sim)


if __name__ == "__main__":
    sys.exit(main())
