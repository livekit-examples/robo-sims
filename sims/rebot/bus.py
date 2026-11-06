"""Runs the simulated arm in real time and exposes its Damiao bus.

The bus is reachable two ways, both speaking the USB2CAN serial bridge's byte
protocol, so any client that drives the real arm's /dev/ttyACM0 works:

- a pseudo-terminal, symlinked to a serial-device path, for clients running
  next to the simulator;
- a TCP port, for clients elsewhere. `tools/serial_bridge.py` turns the TCP
  stream back into a local serial device for motorbridge.

Run alone with `python -m sims.rebot.bus`; `python -m sims run rebot-arm`
starts it together with the robot program.

As with a real adapter, replies go back on the connection that sent the
command. Two clients commanding the same motors fight, as they would on
hardware.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import sys
import termios
import time
import tty

from ..assets import SOURCES
from ..paths import RUNTIME_DIR
from .arm import SimulatedArm
from .dm_serial import TxFrameParser, encode_rx

DEFAULT_MODEL = SOURCES["rebot_b601_dm"].path / "models" / "rebotarm_b601_colored.xml"
DEFAULT_LINK = RUNTIME_DIR / "rebot-arm-ttyACM0"
MAX_CATCH_UP_STEPS = 50


def log(message: str) -> None:
    print(f"[rebot-bus] {message}", flush=True)


class Connection:
    """One byte stream to a client: a pty master or a TCP socket."""

    def __init__(self, name: str, fd: int, sock: socket.socket | None = None) -> None:
        self.name = name
        self.fd = fd
        self.sock = sock
        self.parser = TxFrameParser()

    def read(self) -> bytes | None:
        """Returns the bytes available, or None once a TCP client has gone."""
        try:
            chunk = self.sock.recv(4096) if self.sock else os.read(self.fd, 4096)
        except (BlockingIOError, InterruptedError):
            return b""
        except OSError:
            return b"" if self.sock is None else None
        if self.sock is not None and not chunk:
            return None
        return chunk

    def write(self, payload: bytes) -> None:
        try:
            if self.sock:
                self.sock.sendall(payload)
            else:
                os.write(self.fd, payload)
        except (BlockingIOError, InterruptedError):
            pass  # The client is not reading; drop the reply, as a full USB buffer would.
        except OSError:
            pass


def open_pty(link: Path) -> tuple[int, int]:
    master, slave = os.openpty()
    tty.setraw(slave, termios.TCSANOW)
    os.set_blocking(master, False)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.ttyname(slave))
    # Keep the slave open so the master never reports a hangup between clients.
    return master, slave


def write_status(path: Path, arm: SimulatedArm) -> None:
    status = {
        "time": round(arm.data.time, 3),
        "joints": {name: round(value, 5) for name, value in arm.joint_positions().items()},
        "motors": {
            spec_name: {"enabled": motor.enabled, "mode": motor.mode, "position": round(motor.position, 5)}
            for spec_name, motor in zip(arm.joint_positions(), arm.motors)
        },
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status))
    temporary.replace(path)


def serve(args: argparse.Namespace) -> None:
    arm = SimulatedArm(args.model)
    log(f"model {args.model}, {len(arm.motors)} motors, physics step {arm.dt * 1000:.1f} ms")

    selector = selectors.DefaultSelector()
    connections: dict[int, Connection] = {}

    def add(connection: Connection) -> None:
        connections[connection.fd] = connection
        selector.register(connection.fd, selectors.EVENT_READ, connection)

    def drop(connection: Connection) -> None:
        selector.unregister(connection.fd)
        del connections[connection.fd]
        if connection.sock:
            connection.sock.close()
        log(f"{connection.name} disconnected")

    if args.serial_link:
        master, _slave = open_pty(args.serial_link)
        add(Connection(f"serial {args.serial_link}", master))
        log(f"bus on serial device {args.serial_link}")

    listener = None
    if args.tcp_port:
        listener = socket.create_server((args.tcp_host, args.tcp_port), reuse_port=True)
        listener.setblocking(False)
        selector.register(listener.fileno(), selectors.EVENT_READ, None)
        log(f"bus on tcp {args.tcp_host}:{args.tcp_port}")

    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    dt = arm.dt
    start = time.monotonic()
    steps = 0
    next_status = 0.0
    while running:
        due = int((time.monotonic() - start) / dt)
        if due - steps > MAX_CATCH_UP_STEPS:
            # Fell behind (for example, the host was suspended); skip ahead
            # rather than fast-forwarding the arm.
            start += (due - steps - MAX_CATCH_UP_STEPS) * dt
            due = steps + MAX_CATCH_UP_STEPS
        while steps < due:
            arm.step()
            steps += 1

        if args.status_file and arm.data.time >= next_status:
            write_status(args.status_file, arm)
            next_status = arm.data.time + 0.1

        timeout = max(0.0, start + (steps + 1) * dt - time.monotonic())
        for key, _ in selector.select(timeout):
            if key.data is None:
                sock, address = listener.accept()
                sock.setblocking(False)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                add(Connection(f"tcp {address[0]}:{address[1]}", sock.fileno(), sock))
                log(f"tcp client {address[0]}:{address[1]} connected")
                continue
            connection: Connection = key.data
            chunk = connection.read()
            if chunk is None:
                drop(connection)
                continue
            replies = bytearray()
            for frame in connection.parser.feed(chunk):
                for motor in arm.motors:
                    if motor.accepts(frame):
                        for reply in motor.handle(frame):
                            replies += encode_rx(reply)
            if replies:
                connection.write(bytes(replies))
    if args.serial_link and args.serial_link.is_symlink():
        args.serial_link.unlink()  # Clients then see the device is gone, as on unplugging.
    log("stopped")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Seeed MJCF model")
    parser.add_argument(
        "--serial-link", type=Path, default=DEFAULT_LINK,
        help="symlink to create for the bus pty ('' for none; default: %(default)s)"
    )
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=5555, help="TCP port for the bus (0 for none)")
    parser.add_argument("--status-file", type=Path, default=None, help="write joint state JSON here at 10 Hz")
    args = parser.parse_args(argv)
    if args.serial_link == Path(""):
        args.serial_link = None
    try:
        serve(args)
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
