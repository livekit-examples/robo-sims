"""Construct an isolated managed vacuum session using the same policy as the CLI."""
from .backend import Backend
from .gateway import VacuumGateway


def create(spec, url, token, operator_prefix, *, fps=30):
    # Managed trials have no Start run button, so the first driving command starts the timed run.
    backend = Backend(fps=fps, width=640, height=360, auto_start=True)
    try:
        gateway = VacuumGateway(backend, url, token, fps, operator_prefix=operator_prefix)
        return gateway, backend.policy()
    except BaseException:
        backend.close()
        raise
