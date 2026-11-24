"""Two camera views, the floor texture upload and the on-screen display.

`prepare()` runs on the event loop and copies everything a frame needs. `render()` runs on the
single render thread that owns the graphics context, so the texture upload happens there too.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..mujoco_tools import mujoco
from .simulation import DEADLINE_S, ONBOARD, OVERHEAD, Vacuum
from . import room

LOG = logging.getLogger("sims.vacuum")


class Frame:
    def __init__(self, view: "Views", key: str, data, texture, hud):
        self.view, self.key, self.data, self.texture, self.hud = view, key, data, texture, hud

    def render(self) -> np.ndarray:
        return self.view.render_frame(self)


class Views:
    feeds = {OVERHEAD: type("Feed", (), {"name": OVERHEAD})(), ONBOARD: type("Feed", (), {"name": ONBOARD})()}
    batch_rendering = False

    def __init__(self, sim: Vacuum, width=960, height=540):
        self.sim, self.width, self.height = sim, width, height
        self.renderer = None
        self.uploaded = -1
        self.render_data = mujoco.MjData(sim.model)
        self.font = self.big = None

    def render_size(self, _key):
        return self.width, self.height

    def open_renderers(self):
        self.renderer = mujoco.Renderer(self.sim.model, height=self.height, width=self.width)
        scene_flags = self.renderer.scene.flags
        scene_flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        try:
            self.font = ImageFont.load_default(size=20)
            self.big = ImageFont.load_default(size=30)
        except TypeError:  # Pillow without sized default fonts
            self.font = self.big = ImageFont.load_default()

    def prepare(self, key: str) -> Frame:
        sim = self.sim
        mujoco.mj_copyData(self.render_data, sim.model, sim.data)
        mujoco.mj_forward(sim.model, self.render_data)
        texture = (sim.floor.version, sim.floor.image()) if sim.floor.version != self.uploaded else None
        hud = None
        if key == OVERHEAD:
            hud = {"percent": sim.floor.percent, "elapsed": sim.elapsed, "state": sim.state,
                   "result": sim.result, "battery": sim.battery.state()["charge_percent"],
                   "bin_full": sim.bin.full, "bin_percent": sim.bin.state()["fill_percent"],
                   "assist": sim.assist_active, "valid": sim.trial_valid, "drive": sim.drive_enabled}
        return Frame(self, key, self.render_data, texture, hud)

    def upload_floor(self, version, image):
        model = self.sim.model
        tid = self.sim.texture_id
        adr, size = int(model.tex_adr[tid]), room.N * room.N * 3
        # Texture row 0 is the north edge, grid row 0 is the south edge.
        model.tex_data[adr:adr + size] = image[::-1].reshape(-1)
        context = self.renderer._gl_context
        context.make_current()
        mujoco.mjr_uploadTexture(model, self.renderer._mjr_context, tid)
        self.uploaded = version

    def render_frame(self, frame: Frame) -> np.ndarray:
        if frame.texture is not None:
            self.upload_floor(*frame.texture)
        self.renderer.update_scene(frame.data, camera=frame.key)
        pixels = self.renderer.render()
        return self.overlay(pixels, frame.hud) if frame.hud else pixels

    def overlay(self, pixels, hud) -> np.ndarray:
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        lines = [(f"Cleaned {hud['percent']:5.1f}%", self.big, (255, 255, 255)),
                 (f"Time {hud['elapsed']:4.1f} / {DEADLINE_S:.0f} s", self.big, (255, 255, 255)),
                 (f"Battery {hud['battery']:4.0f}%", self.font, (255, 255, 255)),
                 (f"Bin {hud['bin_percent']:4.0f}%", self.font, (255, 255, 255))]
        if hud["bin_full"]:
            lines.append(("BIN FULL: cleaning stopped", self.font, (255, 120, 90)))
        if not hud["drive"]:
            lines.append(("Driving disabled", self.font, (255, 120, 90)))
        if hud["assist"]:
            lines.append(("Driving assist slowing", self.font, (250, 220, 80)))
        status = {"idle": "Ready: start the run", "running": "Running", "complete": "Complete",
                  "timeout": "Time is up"}[hud["state"]]
        if hud["state"] != "idle" and not hud["valid"]:
            status += " (trial invalid)"
        lines.insert(0, (status, self.big, (120, 220, 255)))
        y = 14
        for text, font, color in lines:
            draw.text((16, y), text, font=font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
            y += font.size + 8 if hasattr(font, "size") else 26
        legend = [("Dust", (205, 125, 20)), ("Clean", (196, 172, 138)), ("Brush cannot reach", (104, 124, 156))]
        y = self.height - 30 * len(legend) - 14
        for text, color in legend:
            draw.rectangle((16, y + 3, 34, y + 21), fill=color, outline=(255, 255, 255))
            draw.text((44, y), text, font=self.font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            y += 30
        return np.asarray(image)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
