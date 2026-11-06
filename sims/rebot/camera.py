"""A simulated scene camera, served as an MJPEG stream the robot program opens like a webcam.

The real reBot setup has a UVC scene camera that the robot program opens with
OpenCV (`--camera-device scene=/dev/video0`). Here the bus renders the MuJoCo
scene from a fixed viewpoint and serves it over HTTP at

    http://127.0.0.1:<port>/scene.mjpg

which the same OpenCvCamera opens through its FFmpeg backend
(`--camera-device scene=http://127.0.0.1:<port>/scene.mjpg`). No kernel module
or root is needed.

Rendering runs on its own thread with its own MjData, from snapshots of the
physics state, so the 1 kHz physics loop is not slowed by OpenGL. MuJoCo and
Pillow release the GIL while they render and encode.
"""

from __future__ import annotations

import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..mujoco_tools import look_at  # Sets MUJOCO_GL before mujoco is imported.
import mujoco
import numpy as np

CAMERA_NAME = "scene"
STREAM_PATH = "/scene.mjpg"
BOUNDARY = "frame"
JPEG_QUALITY = 85

# Beside and in front of the arm, looking at the table, as the Trossen
# single-arm scene camera does. The arm's base is at the origin and the table
# spans x 0.12..0.72 m, y -0.26..0.26 m.
CAMERA_POSITION = (0.80, -0.62, 0.52)
CAMERA_TARGET = (0.24, 0.0, 0.14)
CAMERA_FOVY = 52.0

# Seeed's arm materials carry no color (their web viewer colors them), so
# they would render white. These approximate the real arm.
MATERIAL_COLORS = {
    "anodized_grey_mat": (0.42, 0.44, 0.46),
    "hardware_black_mat": (0.06, 0.06, 0.07),
    "matte_black_mat": (0.10, 0.10, 0.11),
    "seeed_yellow_mat": (0.95, 0.74, 0.06),
    "silver_trim_mat": (0.72, 0.73, 0.75),
    "gripper_finger_black_mat": (0.10, 0.10, 0.11),
    "gripper_carriage_grey_mat": (0.34, 0.36, 0.37),
    "gripper_rack_metal_mat": (0.70, 0.71, 0.73),
    "gripper_seeed_yellow_mat": (0.95, 0.74, 0.06),
    "gripper_hardware_black_mat": (0.06, 0.06, 0.07),
    "gripper_base_metal_mat": (0.70, 0.71, 0.73),
}


def add_scene_camera(spec: mujoco.MjSpec) -> None:
    """Adds the scene camera and colors the arm. Neither changes the physics."""
    spec.worldbody.add_camera(name=CAMERA_NAME, pos=list(CAMERA_POSITION),
                              quat=look_at(CAMERA_POSITION, CAMERA_TARGET), fovy=CAMERA_FOVY)
    for material in spec.materials:
        if material.name in MATERIAL_COLORS:
            material.rgba = [*MATERIAL_COLORS[material.name], 1.0]


class SceneCamera:
    """Renders the scene camera at a fixed rate and serves the frames as MJPEG."""

    def __init__(self, model: mujoco.MjModel, width: int, height: int, fps: int,
                 host: str, port: int, log) -> None:
        self.model = model
        self.width, self.height, self.fps = width, height, fps
        self.log = log
        self._render_data = mujoco.MjData(model)
        self._qpos = np.zeros(model.nq)
        self._mocap_pos = np.zeros((model.nmocap, 3))
        self._mocap_quat = np.zeros((model.nmocap, 4))
        self._state_lock = threading.Lock()
        self._have_state = False
        self._frame: bytes | None = None
        self._frame_number = 0
        self._frame_ready = threading.Condition()
        self._stopping = threading.Event()
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._threads = [threading.Thread(target=self._render_loop, name="scene-render", daemon=True),
                         threading.Thread(target=self._server.serve_forever, name="scene-http", daemon=True)]

    @property
    def url(self) -> str:
        host = self._server.server_address[0]
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{self.port}{STREAM_PATH}"

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stopping.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._threads[0].join(timeout=2.0)

    def publish_state(self, data: mujoco.MjData) -> None:
        """Called from the physics loop: snapshot what the camera needs to draw."""
        with self._state_lock:
            np.copyto(self._qpos, data.qpos)
            np.copyto(self._mocap_pos, data.mocap_pos)
            np.copyto(self._mocap_quat, data.mocap_quat)
            self._have_state = True

    def _render_loop(self) -> None:
        from PIL import Image  # Only the camera needs Pillow.

        # The OpenGL context belongs to this thread, so create and free it here.
        try:
            renderer = mujoco.Renderer(self.model, self.height, self.width)
        except Exception as error:  # noqa: BLE001 - report any GL failure and keep the bus running
            self.log(f"scene camera disabled: could not create an OpenGL context ({error}). "
                     "Set MUJOCO_GL=osmesa and install libosmesa6 to render without a GPU.")
            return
        period = 1.0 / self.fps
        next_frame = time.monotonic()
        try:
            while not self._stopping.is_set():
                with self._state_lock:
                    if self._have_state:
                        self._render_data.qpos[:] = self._qpos
                        self._render_data.mocap_pos[:] = self._mocap_pos
                        self._render_data.mocap_quat[:] = self._mocap_quat
                # Positions only: no dynamics, so the motor laws the physics
                # loop writes into the model are never read here.
                mujoco.mj_kinematics(self.model, self._render_data)
                mujoco.mj_camlight(self.model, self._render_data)
                renderer.update_scene(self._render_data, camera=CAMERA_NAME)
                pixels = renderer.render()
                encoded = io.BytesIO()
                Image.fromarray(pixels).save(encoded, "JPEG", quality=JPEG_QUALITY)
                with self._frame_ready:
                    self._frame = encoded.getvalue()
                    self._frame_number += 1
                    self._frame_ready.notify_all()
                next_frame = max(next_frame + period, time.monotonic())
                self._stopping.wait(max(0.0, next_frame - time.monotonic()))
        finally:
            renderer.close()

    def _next_frame(self, after: int) -> tuple[int, bytes] | None:
        with self._frame_ready:
            self._frame_ready.wait_for(lambda: self._stopping.is_set() or self._frame_number > after, timeout=5.0)
            if self._stopping.is_set() or self._frame is None or self._frame_number <= after:
                return None
            return self._frame_number, self._frame

    def _handler(self):
        camera = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's naming
                if self.path.split("?")[0] != STREAM_PATH:
                    self.send_error(404, f"the stream is at {STREAM_PATH}")
                    return
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                camera.log(f"scene camera client {self.client_address[0]}:{self.client_address[1]} connected")
                last = 0
                try:
                    while (frame := camera._next_frame(last)) is not None:
                        last, jpeg = frame
                        self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                         f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                camera.log(f"scene camera client {self.client_address[0]}:{self.client_address[1]} disconnected")

            def log_message(self, *_args) -> None:
                pass  # Connections are logged above; skip the per-request lines.

        return Handler
