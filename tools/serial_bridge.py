#!/usr/bin/env python3
"""Virtual USB cable: makes the simulator's bus appear as a local serial device.

    python3 serial_bridge.py --connect localhost:5555 --link /tmp/rebot-ttyACM0

creates a pseudo-terminal, symlinks it at --link, and copies bytes between it
and the simulator's TCP bus port. Point the arm driver at the link instead of
/dev/ttyACM0; nothing else changes. Reconnects if the simulator restarts.

Standard library only, so it runs on any machine with Python 3.8+.
"""

import argparse
import os
from pathlib import Path
import selectors
import signal
import socket
import sys
import termios
import time
import tty


def log(message):
    print(f"[serial-bridge] {message}", file=sys.stderr, flush=True)


def open_pty(link):
    master, slave = os.openpty()
    tty.setraw(slave, termios.TCSANOW)
    os.set_blocking(master, False)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.ttyname(slave))
    return master, slave  # Keep the slave open so the link survives client reopens.


def connect(host, port):
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
            log(f"connected to {host}:{port}")
            return sock
        except OSError as error:
            log(f"waiting for {host}:{port} ({error})")
            time.sleep(1.0)


def write_all(fd_or_sock, payload):
    """Best-effort write; drops data if the other side is not reading."""
    try:
        if isinstance(fd_or_sock, socket.socket):
            fd_or_sock.sendall(payload)
        else:
            os.write(fd_or_sock, payload)
    except (BlockingIOError, InterruptedError):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", default="localhost:5555", help="simulator bus HOST:PORT")
    parser.add_argument("--link", type=Path, default=Path("/tmp/rebot-ttyACM0"), help="serial device path to create")
    args = parser.parse_args()
    host, _, port = args.connect.rpartition(":")

    master, _slave = open_pty(args.link)
    log(f"serial device {args.link} -> {os.ttyname(_slave)}")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        while True:
            sock = connect(host or "localhost", int(port))
            selector = selectors.DefaultSelector()
            selector.register(master, selectors.EVENT_READ, "pty")
            selector.register(sock, selectors.EVENT_READ, "tcp")
            connected = True
            while connected:
                for key, _ in selector.select():
                    if key.data == "pty":
                        try:
                            write_all(sock, os.read(master, 4096))
                        except (BlockingIOError, InterruptedError):
                            pass
                    else:
                        try:
                            chunk = sock.recv(4096)
                        except (BlockingIOError, InterruptedError):
                            continue
                        except OSError:
                            chunk = b""
                        if not chunk:
                            log("simulator disconnected")
                            connected = False
                            break
                        write_all(master, chunk)
            selector.close()
            sock.close()
    except KeyboardInterrupt:
        pass
    finally:
        if args.link.is_symlink():
            args.link.unlink()


if __name__ == "__main__":
    main()
