"""The existing operator's v1 data frames, shared by every simulated arm."""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass


KINDS = {
    b"TLJC": "joint",
    b"TLVC": "velocity",
    b"TLTC": "pose",
    b"TLPC": "position",
    b"TLHC": "home",
    b"TLFC": "clear_faults",
}


@dataclass(frozen=True)
class Command:
    kind: str
    target: str
    session: int
    sequence: int
    active: bool = False
    joint_kind: int = 0
    joint_index: int = 0
    values: tuple[float, ...] = ()


# The position track also carries home and clear-faults frames, as on the robot.
TRACK_KINDS = {"position": ("position", "home", "clear_faults")}

# The robot parser clamps Cartesian velocity and position commands to these
# v1 limits: linear in m/s or m, angular in rad/s or rad.
MAX_LINEAR = 0.10
MAX_ANGULAR = 0.50


def clamp_motion(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(max(-limit, min(limit, value)) for limit, value in
                 zip((MAX_LINEAR,) * 3 + (MAX_ANGULAR,) * 3, values))


def parse_command(payload: bytes, expected_target: str, expected_kind: str) -> Command:
    if len(payload) < 32:
        raise ValueError("short command frame")
    magic, version, flags, target_size, joint_index, session, sequence = struct.unpack_from(
        "<4sBBBBII", payload
    )
    kind = KINDS.get(magic)
    if kind not in TRACK_KINDS.get(expected_kind, (expected_kind,)) or version != 1 \
            or not 0 < target_size <= 255:
        raise ValueError("unexpected command kind, version, or target length")
    header_size = 32 if kind == "joint" else 48
    if len(payload) != header_size + target_size:
        raise ValueError("incorrect command frame size")
    target = payload[header_size:].decode("utf-8")
    if target != expected_target:
        raise ValueError("command targets a different robot")
    if kind == "joint":
        if flags not in (1, 2, 3):
            raise ValueError("unknown joint command")
        values = (struct.unpack_from("<d", payload, 24)[0],)
        if not math.isfinite(values[0]):
            raise ValueError("non-finite joint command")
        # A joint velocity is a signed fraction of the joint's safe speed.
        if (flags == 2 and not -1.0 <= values[0] <= 1.0) or (flags == 3 and values[0] != 0.0):
            raise ValueError("invalid joint velocity or home value")
        return Command(kind, target, session, sequence, joint_kind=flags,
                       joint_index=joint_index, values=values)
    if flags not in (0, 1) or joint_index != 0:
        raise ValueError("invalid motion flags")
    values = struct.unpack_from("<6f", payload, 24)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite command")
    if kind == "position" and flags:
        kind = "home"  # Bit 0 of a position frame also means home.
    if kind in ("velocity", "position"):
        values = clamp_motion(values)
    return Command(kind, target, session, sequence, active=bool(flags), values=values)


def encode_joint_state(positions: list[float], limits: list[tuple[float, float]], sequence: int) -> bytes:
    if not 1 < len(positions) <= 255 or len(limits) != len(positions):
        raise ValueError("invalid joint layout")
    timestamp = int(time.time() * 1000) & 0xFFFFFFFF
    frame = bytearray(struct.pack("<4sBBBBII", b"TLJS", 1, 1, len(positions), 0,
                                  sequence & 0xFFFFFFFF, timestamp))
    for position, (minimum, maximum) in zip(positions, limits):
        frame.extend(struct.pack("<ddd", position, minimum, maximum))
    return bytes(frame)
