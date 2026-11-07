"""Simulation-owned construction for the generalized demo gateway."""
from ..assets import SOURCES
from ..contract.config import CONFIGS, Configuration
from ..contract.gateway import ContractGateway
from ..contract.mujoco_backend import MujocoBackend
from .simulation import TrossenSimulation


def create(spec, url: str, token: str, operator_prefix: str):
    config = Configuration.load(CONFIGS / f'{spec.name}.bindings.json', 12)
    assets = SOURCES['trossen_arm_mujoco'].path / 'assets'
    scene = TrossenSimulation(assets, spec.options['scene'], 160, 120, 480, 360,
                             model_path=assets / config.bindings['model'])
    try:
        scene.model.vis.quality.offsamples = 2
        scene.model.vis.quality.shadowsize = 1024
        backend = MujocoBackend(scene, config)
        policy = {'profile': 'joints.trajectory.v1', 'max_feedback_age_ms': 1000,
                  'interfaces': {key: {name: {'min': float(j.limits[0]), 'max': float(j.limits[1]),
                                            'max_velocity': float(j.speed)}
                                       for name, j in joints.items()}
                                 for key, joints in backend.joints.items()}}
        gateway = ContractGateway(backend, config, url, token, 12,
                                  operator_prefix=operator_prefix)
        return gateway, policy
    except BaseException:
        scene.close()
        raise
