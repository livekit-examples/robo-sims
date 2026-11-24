"""Fail image builds if any advertised scene lacks model assets."""
from ..registry import SIMULATIONS
from ..assets import SOURCES
from .rules import SUPPORTED


def catalog():
    result = []
    for name in SUPPORTED:
        spec = SIMULATIONS[name]
        available = bool(spec.managed_module) and all(SOURCES[asset].installed() for asset in spec.assets)
        result.append(dict(id=name, summary=spec.summary, icon=spec.icon, available=available,
                           reason=None if available else 'The deployed image is missing required scene assets.'))
    return result


if __name__ == '__main__':
    missing = [item['id'] for item in catalog() if not item['available']]
    if missing:
        raise SystemExit(f'Image is missing required scenes/assets: {missing}')
