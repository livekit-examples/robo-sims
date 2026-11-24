"""Fetches the third-party models each simulation needs, at pinned commits.

Each source is cloned at one commit, and only the files a simulation loads
are kept, unmodified, under assets/<name>/ with the source's LICENSE and a
SOURCE file naming the repository and commit.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .paths import ASSETS_DIR


@dataclass(frozen=True)
class AssetSource:
    name: str
    repo: str
    ref: str
    install: Callable[[Path, Path], None]  # (checkout, destination)

    @property
    def path(self) -> Path:
        return ASSETS_DIR / self.name

    def installed(self) -> bool:
        source = self.path / "SOURCE"
        return source.is_file() and source.read_text().split() == [self.repo, self.ref]


def _install_trossen(checkout: Path, destination: Path) -> None:
    shutil.copytree(checkout / "trossen_arm_mujoco" / "assets", destination / "assets")
    shutil.copy2(checkout / "LICENSE", destination / "LICENSE")


def _install_seeed(checkout: Path, destination: Path) -> None:
    # The MJCF files expect their meshes in ../meshes, as in Seeed's install tree.
    package = checkout / "reBotArm_ros2_DM" / "src"
    (destination / "models").mkdir(parents=True)
    (destination / "meshes").mkdir()
    for model in (package / "rebotarm_mujoco" / "models").glob("*.xml"):
        shutil.copy2(model, destination / "models")
    for mesh in (package / "rebotarm_bringup" / "description" / "meshes").iterdir():
        if mesh.suffix.lower() == ".stl":
            shutil.copy2(mesh, destination / "meshes")
    shutil.copy2(checkout / "LICENSE", destination / "LICENSE")


def _install_f1tenth(checkout: Path, destination: Path) -> None:
    # Preserve the community's meter coordinates and 2.20 m width unmodified.
    for name in ("LICENSE", "README.md"):
        shutil.copy2(checkout / name, destination / name)
    shutil.copytree(checkout / "Spielberg", destination / "Spielberg")


SOURCES = {
    source.name: source
    for source in (
        AssetSource("f1tenth_racetracks", "https://github.com/f1tenth/f1tenth_racetracks.git",
                    "b95c4eff766f6367d66b310ea20cd2c9563712c0", _install_f1tenth),
        # Trossen Arm MuJoCo, BSD-3-Clause. The workbench scene exists only on
        # its workbench_assets branch.
        AssetSource("trossen_arm_mujoco", "https://github.com/TrossenRobotics/trossen_arm_mujoco.git",
                    "a208dad8617848a2cea26a686a2d5d3185d6b667", _install_trossen),
        # Seeed reBot Arm B601-DM digital twin (Yang-Ci/ReBot_Arm_DigitalTwin_DM), Apache-2.0.
        AssetSource("rebot_b601_dm", "https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM.git",
                    "7456d8bc3cc6d9c470680988543f90bc8f3b03b0", _install_seeed),
    )
}


def fetch(source: AssetSource, force: bool = False) -> None:
    if source.installed() and not force:
        print(f"{source.name}: already at {source.ref[:7]}")
        return
    with tempfile.TemporaryDirectory() as temporary:
        checkout = Path(temporary) / "checkout"
        staged = Path(temporary) / "staged"
        print(f"{source.name}: fetching {source.repo} @ {source.ref[:7]}")
        git = ["git", "-c", "advice.detachedHead=false", "-C", str(checkout)]
        checkout.mkdir()
        subprocess.run([*git, "init", "--quiet"], check=True)
        subprocess.run([*git, "fetch", "--quiet", "--depth", "1", source.repo, source.ref], check=True)
        subprocess.run([*git, "checkout", "--quiet", "FETCH_HEAD"], check=True)
        staged.mkdir()
        source.install(checkout, staged)
        (staged / "SOURCE").write_text(f"{source.repo} {source.ref}\n")
        if source.path.exists():
            shutil.rmtree(source.path)
        source.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staged, source.path)
    print(f"{source.name}: installed in {source.path}")
