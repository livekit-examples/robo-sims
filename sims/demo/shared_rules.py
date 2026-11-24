"""Wire fields and signed per-tab proofs for the shared simulation room."""
import base64
from dataclasses import dataclass
import hashlib
import hmac
import json

from .rules import bounded, now

KIND = 'vla_demo.kind'
QUEUED = 'vla_demo.queue_ts'
QUEUE_ROBOT = 'vla_demo.queue_robot'
UNTIL = 'vla_demo.control_until'
CONTROL_ROBOT = 'vla_demo.control_robot'
GENERATION = 'teleop.sim.generation'
ACTIVITY = 'teleop.sim.activity'
INSTANCE = 'teleop.sim.instance'
SIMULATION = 'teleop.sim.type'


class Problem(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


@dataclass(frozen=True)
class SharedConfig:
    duration: int = 21600
    fps: int = 30
    startup: float = 45
    cleanup: float = 5
    idle: int = 120
    turn: int = 60

    @classmethod
    def from_env(cls, env):
        return cls(
            bounded(env.get('SIM_MAX_RUNTIME_SECONDS', 21600), 'runtime', 10, 21600, True),
            bounded(env.get('SIM_SESSION_FPS', 30), 'fps', 1, 60, True),
            bounded(env.get('SIM_SESSION_STARTUP_SECONDS', 45), 'startup', 5, 300),
            bounded(env.get('SIM_SESSION_CLEANUP_SECONDS', 5), 'cleanup', 1, 30),
            bounded(env.get('SIM_IDLE_TIMEOUT_SECONDS', 120), 'idle timeout', 5, 3600, True),
            bounded(env.get('SIM_CONTROL_TURN_SECONDS', 60), 'turn', 5, 3600, True))

    def fields(self):
        return dict(duration_seconds=self.duration, fps=self.fps, startup_seconds=self.startup,
                    cleanup_seconds=self.cleanup, idle_seconds=self.idle, turn_seconds=self.turn,
                    shared=True)


def sign(key, **claims):
    payload = base64.urlsafe_b64encode(json.dumps(claims, sort_keys=True,
        separators=(',', ':')).encode()).decode().rstrip('=')
    signature = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return payload + '.' + signature


def verify(key, proof, **expected):
    try:
        payload, signature = proof.split('.')
        actual = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, actual):
            raise ValueError()
        claims = json.loads(base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4)))
        if claims['until'] <= now() or any(claims.get(k) != v for k, v in expected.items()):
            raise ValueError()
        return claims
    except (ValueError, KeyError, TypeError, AttributeError):
        raise Problem(403, 'Invalid or expired simulation proof.') from None


def activity(participant):
    try:
        value = json.loads(participant.attributes.get(ACTIVITY, '{}'))
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def timestamp(participant, field):
    try:
        return int(participant.attributes.get(field, '0') or 0)
    except (ValueError, TypeError):
        return 0


def engaged(participant, run, at):
    item = activity(participant).get(run['id'], {})
    return (item.get('generation') == run['id'] and
            isinstance(item.get('at'), int) and item['at'] + run['idle_seconds'] * 1000 > at)
