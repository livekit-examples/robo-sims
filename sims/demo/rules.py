"""Session rules shared by local and hosted services. No renderer imports."""
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import math
import re
import time

from livekit import api
from ..livekit_auth import scientist_names

ACTIVE = ('starting', 'ready', 'stopping')
SUPPORTED = ('trossen-wxai', 'trossen-wxai-whack', 'trossen-workbench', 'trossen-workbench-whack')
KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{16,100}$')


def now():
    return int(time.time() * 1000)


def bounded(value, name, low, high, integer=False):
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high or (integer and not number.is_integer()):
        raise ValueError(f'{name} must be {"an integer " if integer else ""}between {low} and {high}.')
    return int(number) if integer else number


@dataclass(frozen=True)
class SessionConfig:
    duration: float = 60
    fps: int = 30
    startup: float = 45
    cleanup: float = 5

    def __post_init__(self):
        bounded(self.duration, 'duration', 1, 3600)
        bounded(self.fps, 'fps', 1, 60, True)
        bounded(self.startup, 'startup', 5, 300)
        bounded(self.cleanup, 'cleanup', 1, 30)

    @classmethod
    def from_env(cls, env):
        return cls(bounded(env.get('SIM_SESSION_DURATION_SECONDS', 60), 'duration', 1, 3600),
                   bounded(env.get('SIM_SESSION_FPS', 30), 'fps', 1, 60, True),
                   bounded(env.get('SIM_SESSION_STARTUP_SECONDS', 45), 'startup', 5, 300),
                   bounded(env.get('SIM_SESSION_CLEANUP_SECONDS', 5), 'cleanup', 1, 30))

    def fields(self):
        return dict(duration_seconds=self.duration, fps=self.fps, startup_seconds=self.startup,
                    cleanup_seconds=self.cleanup)


def room_candidates(identifier, limit=128):
    names = scientist_names()
    for retry in range(limit):
        digest = hashlib.sha256(f'{identifier}:{retry}'.encode()).digest()
        yield f'{names[int.from_bytes(digest[1:5], "big") % len(names)]}-{digest[0]:02x}'


def public(run, runs=None, capacity=20):
    result = {k: run.get(k) for k in ('id', 'simulation_id', 'state', 'created_at', 'ready_at',
                                   'expires_at', 'reason', 'exit_code', 'duration_seconds', 'fps', 'queued_at', 'queue_expires_at')}
    if runs is not None:
        from .queue import queue_info
        result.update(queue_info(run, runs, capacity, now()))
    return result


def token(run, identity, role, ttl, api_key, api_secret):
    return (api.AccessToken(api_key, api_secret).with_identity(identity)
            .with_attributes({'lk.portal.role': role, 'lk.portal.version': '3'})
            .with_ttl(timedelta(seconds=ttl))
            .with_grants(api.VideoGrants(room_join=True, room=run['room_name'], can_subscribe=True,
                can_publish=role != 'observer', can_publish_data=True, can_update_own_metadata=True)).to_jwt())


def worker_ttl(run):
    return math.ceil(run['startup_seconds'] + run['duration_seconds'] + run['cleanup_seconds'] + 30)


def accept_ready(run, event):
    """Only the first valid ready report sets the immutable deadline."""
    if run['state'] != 'starting':
        return False
    ready = event['ready_at']
    expiry = event['expires_at']
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (ready, expiry)):
        raise ValueError('Invalid readiness timestamps.')
    if ready < run['created_at'] - 2000 or ready > now() + 2000:
        raise ValueError('Invalid readiness time.')
    if abs(expiry - ready - run['duration_seconds'] * 1000) > 2:
        raise ValueError('Invalid session duration.')
    run.update(state='ready', ready_at=ready, expires_at=expiry, policy=event['policy'])
    return True
