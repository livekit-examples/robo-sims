"""Byte framing of the Damiao USB2CAN serial bridge (the arm's /dev/ttyACM0).

Host to adapter: 30-byte frames that ask the adapter to forward one CAN frame.
Adapter to host: 16-byte frames carrying one received CAN frame.

Mirrors motorbridge's motor_core/src/dm_serial.rs, which is the host side.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

TX_FRAME_LEN = 30  # host -> adapter
RX_FRAME_LEN = 16  # adapter -> host


@dataclass(frozen=True)
class CanFrame:
    arbitration_id: int
    data: bytes  # always 8 bytes, zero padded
    dlc: int = 8
    is_extended: bool = False


class TxFrameParser:
    """Reassembles host-to-adapter frames from a byte stream."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[CanFrame]:
        self._buffer += chunk
        frames: list[CanFrame] = []
        while True:
            start = self._buffer.find(b"\x55\xaa")
            if start < 0:
                # Keep a trailing 0x55 in case the header straddles two reads.
                del self._buffer[: max(0, len(self._buffer) - 1)]
                return frames
            del self._buffer[:start]
            if len(self._buffer) < TX_FRAME_LEN:
                return frames
            raw = bytes(self._buffer[:TX_FRAME_LEN])
            if raw[2] != 0x1E:
                del self._buffer[:2]  # Not a frame header; resynchronize.
                continue
            del self._buffer[:TX_FRAME_LEN]
            frame = decode_tx(raw)
            if frame is not None:
                frames.append(frame)


def decode_tx(raw: bytes) -> CanFrame | None:
    """Decodes one 30-byte host frame, or returns None for non-data frames."""
    if len(raw) != TX_FRAME_LEN or raw[0:3] != b"\x55\xaa\x1e":
        raise ValueError("not a Damiao serial TX frame")
    if raw[17] != 0:  # frameType: only data frames carry motor commands.
        return None
    dlc = min(raw[18], 8)
    (arbitration_id,) = struct.unpack_from("<I", raw, 13)
    return CanFrame(arbitration_id, bytes(raw[21:29]), dlc, bool(raw[12]))


def encode_tx(frame: CanFrame) -> bytes:
    """Encodes a host frame, as motorbridge does. Used by tests."""
    out = bytearray(TX_FRAME_LEN)
    out[0:4] = b"\x55\xaa\x1e\x03"
    struct.pack_into("<II", out, 4, 1, 10)
    out[12] = int(frame.is_extended)
    struct.pack_into("<I", out, 13, frame.arbitration_id)
    out[18] = frame.dlc
    out[21:29] = frame.data.ljust(8, b"\0")[:8]
    return bytes(out)


def encode_rx(frame: CanFrame) -> bytes:
    """Encodes one adapter-to-host frame."""
    flags = (frame.dlc & 0x3F) | (0x40 if frame.is_extended else 0)
    return (
        bytes([0xAA, 0x11, flags])
        + struct.pack("<I", frame.arbitration_id)
        + frame.data.ljust(8, b"\0")[:8]
        + b"\x55"
    )


def decode_rx(raw: bytes) -> CanFrame:
    """Decodes one adapter-to-host frame. Used by tests."""
    if len(raw) != RX_FRAME_LEN or raw[0] != 0xAA or raw[1] != 0x11 or raw[15] != 0x55:
        raise ValueError("not a Damiao serial RX frame")
    (arbitration_id,) = struct.unpack_from("<I", raw, 3)
    return CanFrame(arbitration_id, bytes(raw[7:15]), raw[2] & 0x3F, bool(raw[2] & 0x40))
