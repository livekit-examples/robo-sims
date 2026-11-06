"""Emulated Damiao motor firmware (DM-J4310, DM-J4340P).

Each motor answers the same CAN frames as the real firmware: enable/disable,
set zero, MIT, POS_VEL, VEL and force-position commands, register reads and
writes, and feedback requests. Every command is answered with a feedback frame.

The motor does not integrate physics. Each physics step it hands the simulator
a torque law that is linear in the motor's own position and velocity,

    tau = offset + stiffness * position + damping * velocity,

so that MuJoCo can integrate the damping term implicitly. That keeps stiff
gains stable even when a motor drives a light finger through a fine rack.

Protocol details mirror motorbridge's motor_vendors/damiao (protocol.rs,
registers.rs, motor.rs). The POS_VEL, VEL and force-position loops only
approximate the real firmware: the gains Seeed writes to registers 25-28 are in
undocumented firmware units, so they are stored but not used.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .dm_serial import CanFrame

MODE_MIT = 1
MODE_POS_VEL = 2
MODE_VEL = 3
MODE_FORCE_POS = 4

STATUS_DISABLED = 0x0
STATUS_ENABLED = 0x1

KP_MAX = 500.0
KD_MAX = 5.0

_SPECIAL_PREFIX = b"\xff" * 7
_ENABLE, _DISABLE, _SET_ZERO, _CLEAR_ERROR = 0xFC, 0xFD, 0xFE, 0xFB
_REGISTER_READ, _REGISTER_WRITE, _STORE, _FEEDBACK_REQUEST = 0x33, 0x55, 0xAA, 0xCC
_BROADCAST_ID = 0x7FF

# Registers holding unsigned integers; all the others hold floats.
_UINT_REGISTERS = frozenset({7, 8, 9, 10, 13, 14, 15, 16, 35, 36})
_READ_ONLY_REGISTERS = frozenset(
    {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 36, 50, 51, 52, 53, 54, 55, 56, 59, 60, 61, 62, 63, 64, 65, 80, 81}
)
_KNOWN_REGISTERS = frozenset(range(0, 37)) | frozenset({50, 51, 52, 53, 54, 55, 56, 59, 60, 61, 62, 63, 64, 65, 80, 81})


@dataclass(frozen=True)
class MotorModel:
    name: str
    pmax: float  # rad, range of the position field in MIT frames and feedback
    vmax: float  # rad/s
    tmax: float  # N*m, also the torque limit
    # Internal speed loop of the emulated POS_VEL/VEL modes.
    speed_gain: float  # N*m per rad/s
    speed_integral: float  # 1/s
    position_gain: float  # 1/s, position error to speed command


MODELS = {
    # Limits match motorbridge's model table. Loop gains were tuned in the sim
    # (settle without oscillation across the workspace), not measured.
    "4310": MotorModel("4310", 12.5, 30.0, 10.0, speed_gain=1.0, speed_integral=3.0, position_gain=15.0),
    "4340P": MotorModel("4340P", 12.5, 10.0, 28.0, speed_gain=6.0, speed_integral=3.0, position_gain=15.0),
}


@dataclass(frozen=True)
class TorqueLaw:
    """tau = offset + stiffness * position + damping * velocity, clamped to +-limit."""

    offset: float = 0.0
    stiffness: float = 0.0
    damping: float = 0.0
    limit: float = 0.0


def float_to_uint(value: float, low: float, high: float, bits: int) -> int:
    clipped = min(max(value, low), high)
    return int((clipped - low) * ((1 << bits) - 1) / (high - low))


def uint_to_float(value: int, low: float, high: float, bits: int) -> float:
    return value * (high - low) / ((1 << bits) - 1) + low


class DamiaoMotor:
    """One emulated motor on the bus."""

    def __init__(self, motor_id: int, feedback_id: int, model: str) -> None:
        self.model = MODELS[model]
        self.enabled = False
        self.zero_offset = 0.0  # Raw position that reads as zero, set by "set zero".
        self.registers: dict[int, float | int] = {rid: 0.0 for rid in _KNOWN_REGISTERS - _UINT_REGISTERS}
        self.registers.update({rid: 0 for rid in _UINT_REGISTERS})
        self.registers.update(
            {
                0: 10.0,  # UV_Value
                2: 80.0,  # OT_Value
                3: 0.98,  # OC_Value
                7: feedback_id,
                8: motor_id,
                10: MODE_MIT,
                14: 5013,  # sw_ver
                16: 14,  # NPP
                20: 10.0,  # Gr
                21: self.model.pmax,
                22: self.model.vmax,
                23: self.model.tmax,
                35: 4,  # can_br: 1 Mbit/s
                60: 24.0,  # VBus
                61: 30.0,  # Tpcb
                62: 30.0,  # Tmtr
            }
        )
        # Latest command for each mode. The firmware keeps one per mode and acts
        # on the one that matches CTRL_MODE.
        self._mit = (0.0, 0.0, 0.0, 0.0, 0.0)  # pos, vel, kp, kd, tau
        self._pos_vel = (0.0, 0.0)  # pos, velocity limit
        self._vel = 0.0
        self._force_pos = (0.0, 0.0, 1.0)  # pos, velocity limit, torque ratio
        self._reference: float | None = None  # Internal trajectory of POS_VEL.
        self._speed_integral = 0.0
        # Measured state, in raw (pre-zero) motor coordinates.
        self.raw_position = 0.0
        self.velocity = 0.0
        self.torque = 0.0

    @property
    def motor_id(self) -> int:
        return int(self.registers[8])

    @property
    def feedback_id(self) -> int:
        return int(self.registers[7])

    @property
    def mode(self) -> int:
        return int(self.registers[10])

    @property
    def position(self) -> float:
        """Position as the motor reports it, relative to its zero."""
        return self.raw_position - self.zero_offset

    # Bus side.

    def accepts(self, frame: CanFrame) -> bool:
        if frame.arbitration_id == _BROADCAST_ID:
            return int.from_bytes(frame.data[0:2], "little") == self.motor_id
        return frame.arbitration_id in (
            self.motor_id,
            0x100 + self.motor_id,
            0x200 + self.motor_id,
            0x300 + self.motor_id,
        )

    def handle(self, frame: CanFrame) -> list[CanFrame]:
        """Applies one command frame and returns the motor's replies."""
        data = frame.data
        arbitration_id = frame.arbitration_id
        if arbitration_id == _BROADCAST_ID:
            return self._handle_broadcast(data)
        if arbitration_id == self.motor_id:
            if data[:7] == _SPECIAL_PREFIX and data[7] in (_ENABLE, _DISABLE, _SET_ZERO, _CLEAR_ERROR):
                self._handle_special(data[7])
            else:
                self._mit = self._decode_mit(data)
        elif arbitration_id == 0x100 + self.motor_id:
            self._pos_vel = struct.unpack_from("<ff", data)
        elif arbitration_id == 0x200 + self.motor_id:
            (self._vel,) = struct.unpack_from("<f", data)
        elif arbitration_id == 0x300 + self.motor_id:
            position, velocity, current = struct.unpack_from("<fHH", data)
            self._force_pos = (position, velocity / 100.0, min(current, 10_000) / 10_000.0)
        return [self.feedback_frame()]

    def _handle_special(self, command: int) -> None:
        if command == _ENABLE:
            if not self.enabled:
                self._restart_loops()
            self.enabled = True
        elif command == _DISABLE:
            self.enabled = False
        elif command == _SET_ZERO and not self.enabled:
            self.zero_offset = self.raw_position
            self._restart_loops()

    def _handle_broadcast(self, data: bytes) -> list[CanFrame]:
        command, rid = data[2], data[3]
        if command == _FEEDBACK_REQUEST:
            return [self.feedback_frame()]
        if command == _STORE:
            return []
        if rid not in _KNOWN_REGISTERS:
            return []
        if command == _REGISTER_WRITE and rid not in _READ_ONLY_REGISTERS:
            if rid in _UINT_REGISTERS:
                (value,) = struct.unpack_from("<I", data, 4)
            else:
                (value,) = struct.unpack_from("<f", data, 4)
            previous_mode = self.mode
            self.registers[rid] = value
            if rid == 10 and value != previous_mode:
                self._restart_loops()
            return [self._register_frame(_REGISTER_WRITE, rid)]
        if command in (_REGISTER_READ, _REGISTER_WRITE):
            return [self._register_frame(command, rid)]
        return []

    def _register_frame(self, marker: int, rid: int) -> CanFrame:
        if rid == 80:
            self.registers[80] = self.position
        value = self.registers[rid]
        payload = struct.pack("<I", int(value)) if rid in _UINT_REGISTERS else struct.pack("<f", float(value))
        header = bytes([self.motor_id & 0xFF, (self.motor_id >> 8) & 0xFF, marker, rid])
        return CanFrame(self.feedback_id, header + payload)

    def _decode_mit(self, data: bytes) -> tuple[float, float, float, float, float]:
        model = self.model
        pos_u = (data[0] << 8) | data[1]
        vel_u = (data[2] << 4) | (data[3] >> 4)
        kp_u = ((data[3] & 0xF) << 8) | data[4]
        kd_u = (data[5] << 4) | (data[6] >> 4)
        tau_u = ((data[6] & 0xF) << 8) | data[7]
        return (
            uint_to_float(pos_u, -model.pmax, model.pmax, 16),
            uint_to_float(vel_u, -model.vmax, model.vmax, 12),
            uint_to_float(kp_u, 0.0, KP_MAX, 12),
            uint_to_float(kd_u, 0.0, KD_MAX, 12),
            uint_to_float(tau_u, -model.tmax, model.tmax, 12),
        )

    def feedback_frame(self) -> CanFrame:
        model = self.model
        pos_u = float_to_uint(self.position, -model.pmax, model.pmax, 16)
        vel_u = float_to_uint(self.velocity, -model.vmax, model.vmax, 12)
        tau_u = float_to_uint(self.torque, -model.tmax, model.tmax, 12)
        status = STATUS_ENABLED if self.enabled else STATUS_DISABLED
        temperature = int(self.registers[61])
        data = bytes(
            [
                (self.motor_id & 0x0F) | (status << 4),
                pos_u >> 8,
                pos_u & 0xFF,
                vel_u >> 4,
                ((vel_u & 0xF) << 4) | (tau_u >> 8),
                tau_u & 0xFF,
                temperature,
                temperature,
            ]
        )
        return CanFrame(self.feedback_id, data)

    # Physics side.

    def _restart_loops(self) -> None:
        self._reference = None
        self._speed_integral = 0.0

    def set_measured(self, raw_position: float, velocity: float, torque: float) -> None:
        self.raw_position = raw_position
        self.velocity = velocity
        self.torque = torque

    def torque_law(self, dt: float) -> TorqueLaw:
        """The torque law for the next physics step, in raw motor coordinates."""
        if not self.enabled:
            self._restart_loops()
            return TorqueLaw()
        model = self.model
        mode = self.mode
        if mode == MODE_MIT:
            position, velocity, kp, kd, tau = self._mit
            # tau = kp * (p - x) + kd * (v - xdot) + t, with x = raw - zero.
            offset = kp * (position + self.zero_offset) + kd * velocity + tau
            return TorqueLaw(offset, -kp, -kd, model.tmax)
        if mode == MODE_VEL:
            return self._speed_loop(self._vel, dt, model.tmax)
        if mode in (MODE_POS_VEL, MODE_FORCE_POS):
            if mode == MODE_POS_VEL:
                target, velocity_limit = self._pos_vel
                limit = model.tmax
            else:
                target, velocity_limit, ratio = self._force_pos
                limit = model.tmax * ratio
            if self._reference is None:
                self._reference = self.position
            step = max(-velocity_limit * dt, min(velocity_limit * dt, target - self._reference))
            self._reference += step
            reference_velocity = step / dt if dt > 0 else 0.0
            # Speed command = feedforward + position_gain * (reference - x).
            gain = model.position_gain
            speed_command_offset = reference_velocity + gain * (self._reference + self.zero_offset)
            law = self._speed_loop(speed_command_offset, dt, limit, position_gain=gain)
            return law
        return TorqueLaw()

    def _speed_loop(self, speed_offset: float, dt: float, limit: float, position_gain: float = 0.0) -> TorqueLaw:
        """PI speed loop. The speed command is speed_offset - position_gain * raw_position."""
        model = self.model
        k = model.speed_gain
        speed_error = speed_offset - position_gain * self.raw_position - self.velocity
        # Conditional integration (anti-windup): hold the integral while the
        # output is saturated and the error would push it further.
        unsaturated = k * speed_error + self._speed_integral
        if abs(unsaturated) < limit or (unsaturated > 0) != (speed_error > 0):
            self._speed_integral = max(
                -limit, min(limit, self._speed_integral + k * model.speed_integral * speed_error * dt)
            )
        return TorqueLaw(k * speed_offset + self._speed_integral, -k * position_gain, -k, limit)
