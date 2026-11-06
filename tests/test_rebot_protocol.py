"""Unit tests for the serial framing and the emulated Damiao firmware. No MuJoCo needed.

    python3 -m unittest discover -s tests
"""

import struct
import unittest

from sims.rebot.damiao import (
    MODE_MIT,
    MODE_POS_VEL,
    STATUS_DISABLED,
    STATUS_ENABLED,
    DamiaoMotor,
    float_to_uint,
    uint_to_float,
)
from sims.rebot.dm_serial import CanFrame, TxFrameParser, decode_rx, encode_rx, encode_tx


def mit_frame(motor_id, pos, vel, kp, kd, tau, pmax=12.5, vmax=30.0, tmax=10.0):
    """Encodes an MIT command the way motorbridge's encode_mit_cmd does."""
    p = float_to_uint(pos, -pmax, pmax, 16)
    v = float_to_uint(vel, -vmax, vmax, 12)
    k = float_to_uint(kp, 0.0, 500.0, 12)
    d = float_to_uint(kd, 0.0, 5.0, 12)
    t = float_to_uint(tau, -tmax, tmax, 12)
    data = bytes(
        [p >> 8, p & 0xFF, v >> 4, ((v & 0xF) << 4) | (k >> 8), k & 0xFF, d >> 4, ((d & 0xF) << 4) | (t >> 8), t & 0xFF]
    )
    return CanFrame(motor_id, data)


def decode_feedback(frame, pmax=12.5, vmax=30.0, tmax=10.0):
    d = frame.data
    return {
        "id": d[0] & 0x0F,
        "status": d[0] >> 4,
        "pos": uint_to_float((d[1] << 8) | d[2], -pmax, pmax, 16),
        "vel": uint_to_float((d[3] << 4) | (d[4] >> 4), -vmax, vmax, 12),
        "tau": uint_to_float(((d[4] & 0xF) << 8) | d[5], -tmax, tmax, 12),
    }


ENABLE = bytes([0xFF] * 7 + [0xFC])
DISABLE = bytes([0xFF] * 7 + [0xFD])
SET_ZERO = bytes([0xFF] * 7 + [0xFE])


class SerialFramingTest(unittest.TestCase):
    def test_tx_frames_reassemble_across_reads_and_noise(self):
        frame = CanFrame(0x104, bytes(range(8)))
        stream = b"\x00\x13" + encode_tx(frame) + encode_tx(frame)
        parser = TxFrameParser()
        frames = parser.feed(stream[:7]) + parser.feed(stream[7:40]) + parser.feed(stream[40:])
        self.assertEqual(frames, [frame, frame])

    def test_rx_frame_matches_motorbridge_layout(self):
        raw = encode_rx(CanFrame(0x11, bytes([1, 2, 3, 4, 5, 6, 7, 8])))
        self.assertEqual(len(raw), 16)
        self.assertEqual(raw[:3], bytes([0xAA, 0x11, 0x08]))
        self.assertEqual(raw[3:7], (0x11).to_bytes(4, "little"))
        self.assertEqual(raw[15], 0x55)
        self.assertEqual(decode_rx(raw).data, bytes([1, 2, 3, 4, 5, 6, 7, 8]))


class DamiaoMotorTest(unittest.TestCase):
    def setUp(self):
        self.motor = DamiaoMotor(0x04, 0x14, "4310")

    def test_every_command_gets_feedback_from_the_feedback_id(self):
        replies = self.motor.handle(CanFrame(0x04, ENABLE))
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0].arbitration_id, 0x14)
        self.assertEqual(decode_feedback(replies[0])["status"], STATUS_ENABLED)
        self.assertEqual(decode_feedback(self.motor.handle(CanFrame(0x04, DISABLE))[0])["status"], STATUS_DISABLED)

    def test_feedback_reports_position_velocity_and_torque(self):
        self.motor.set_measured(1.25, -0.5, 2.0)
        feedback = decode_feedback(self.motor.handle(CanFrame(0x7FF, bytes([0x04, 0, 0xCC, 0, 0, 0, 0, 0])))[0])
        self.assertEqual(feedback["id"], 0x04)
        self.assertAlmostEqual(feedback["pos"], 1.25, places=3)
        self.assertAlmostEqual(feedback["vel"], -0.5, delta=0.02)
        self.assertAlmostEqual(feedback["tau"], 2.0, delta=0.01)

    def test_register_write_is_acknowledged_and_read_back(self):
        write = CanFrame(0x7FF, bytes([0x04, 0, 0x55, 10]) + struct.pack("<I", MODE_POS_VEL))
        (ack,) = self.motor.handle(write)
        self.assertEqual(ack.arbitration_id, 0x14)
        self.assertEqual(ack.data, write.data)
        self.assertEqual(self.motor.mode, MODE_POS_VEL)
        (reply,) = self.motor.handle(CanFrame(0x7FF, bytes([0x04, 0, 0x33, 10, 0, 0, 0, 0])))
        self.assertEqual(reply.data, bytes([0x04, 0, 0x33, 10]) + struct.pack("<I", MODE_POS_VEL))

    def test_model_limits_are_readable(self):
        (reply,) = self.motor.handle(CanFrame(0x7FF, bytes([0x04, 0, 0x33, 23, 0, 0, 0, 0])))
        self.assertEqual(struct.unpack("<f", reply.data[4:])[0], 10.0)

    def test_frames_for_other_motors_are_ignored(self):
        self.assertFalse(self.motor.accepts(CanFrame(0x05, ENABLE)))
        self.assertFalse(self.motor.accepts(CanFrame(0x7FF, bytes([0x05, 0, 0xCC, 0, 0, 0, 0, 0]))))
        self.assertTrue(self.motor.accepts(CanFrame(0x104, bytes(8))))

    def test_disabled_motor_applies_no_torque(self):
        self.motor.handle(mit_frame(0x04, 1.0, 0.0, 50.0, 1.0, 0.5))
        law = self.motor.torque_law(0.001)
        self.assertEqual((law.offset, law.stiffness, law.damping, law.limit), (0.0, 0.0, 0.0, 0.0))

    def test_mit_law_is_kp_kd_with_feedforward(self):
        self.assertEqual(self.motor.mode, MODE_MIT)
        self.motor.handle(CanFrame(0x04, ENABLE))
        self.motor.handle(mit_frame(0x04, 1.0, 0.0, 50.0, 1.0, 0.5))
        self.motor.set_measured(0.8, 0.2, 0.0)
        law = self.motor.torque_law(0.001)
        torque = law.offset + law.stiffness * 0.8 + law.damping * 0.2
        self.assertAlmostEqual(torque, 50.0 * 0.2 + 1.0 * -0.2 + 0.5, delta=0.1)
        self.assertEqual(law.limit, 10.0)

    def test_set_zero_rezeroes_reported_position(self):
        self.motor.set_measured(0.7, 0.0, 0.0)
        self.motor.handle(CanFrame(0x04, SET_ZERO))
        self.assertAlmostEqual(self.motor.position, 0.0)
        self.motor.handle(CanFrame(0x04, ENABLE))
        self.motor.handle(CanFrame(0x04, SET_ZERO))  # Ignored while enabled.
        self.motor.set_measured(1.0, 0.0, 0.0)
        self.assertAlmostEqual(self.motor.position, 0.3)

    def test_pos_vel_reference_moves_at_the_velocity_limit(self):
        self.motor.handle(CanFrame(0x7FF, bytes([0x04, 0, 0x55, 10]) + struct.pack("<I", MODE_POS_VEL)))
        self.motor.handle(CanFrame(0x04, ENABLE))
        self.motor.handle(CanFrame(0x104, struct.pack("<ff", 1.0, 2.0)))
        for _ in range(100):
            self.motor.torque_law(0.001)
        self.assertAlmostEqual(self.motor._reference, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
