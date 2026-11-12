"""Simulation-owned construction for the generalized demo gateway."""
from ..assets import SOURCES
from ..contract.config import CONFIGS, Configuration
from ..contract.gateway import ContractGateway
from ..contract.mujoco_backend import MujocoBackend
from .simulation import TrossenSimulation


def create(spec, url: str, token: str, operator_prefix: str, *, fps: int = 12):
    if fps < 1:
        raise ValueError('fps must be a positive integer')
    config = Configuration.load(CONFIGS / f'{spec.name}.bindings.json', fps)
    assets = SOURCES['trossen_arm_mujoco'].path / 'assets'
    # Preserve camera detail when the workspace enlarges the stereo video.
    scene = TrossenSimulation(assets, spec.options['scene'], width=640, height=480,
                             scene_width=960, scene_height=720,
                             model_path=assets / config.bindings['model'], renderer='classic')
    try:
        scene.model.vis.quality.offsamples = 2
        scene.model.vis.quality.shadowsize = 1024
        backend = MujocoBackend(scene, config)
        policy = backend.policy()
        gateway = ContractGateway(backend, config, url, token, fps,
                                  operator_prefix=operator_prefix)
        return gateway, policy
    except BaseException:
        scene.close()
        raise
