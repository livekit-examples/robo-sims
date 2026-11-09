"""Construct an isolated managed driving session using the same policy as CLI."""
from .backend import Backend
from .gateway import DriveGateway


def create(spec, url, token, operator_prefix, *, fps=60):
    backend = Backend(fps=fps, width=640, height=360)
    try:
        gateway = DriveGateway(backend, url, token, fps, operator_prefix=operator_prefix)
        return gateway, backend.policy()
    except BaseException:
        backend.close()
        raise
