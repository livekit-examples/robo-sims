"""Batched offscreen cameras using MuJoCo 4's Filament renderer."""

import asyncio
import copy
import ctypes
import logging

import mujoco
import numpy as np

LOG = logging.getLogger("sims.trossen.filament")


def service_macos_events():
    """Drain Cocoa's main queue while Filament renders on its own worker.

    Filament's Cocoa OpenGL swapchain dispatches synchronous work to this
    queue even for offscreen targets. asyncio alone does not drain it.
    """
    core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    core.CFRunLoopGetMain.restype = ctypes.c_void_p
    core.CFRunLoopGetCurrent.restype = ctypes.c_void_p
    if core.CFRunLoopGetMain() != core.CFRunLoopGetCurrent():
        raise RuntimeError("Run Filament with python on the macOS main thread, rather than mjpython")
    mode = ctypes.c_void_p.in_dll(core, "kCFRunLoopDefaultMode")
    step = core.CFRunLoopRunInMode
    step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_bool]
    step.restype = ctypes.c_int
    async def pump():
        while True:
            step(mode, 0.0, True)
            await asyncio.sleep(0.005)

    return pump()


class FilamentFrames:
    """GPU objects live on one render worker; prepare() only copies CPU state."""

    def __init__(self, model, feeds, sizes, fog=False):
        # Register MuJoCo's bundled material/texture resource providers.
        import mujoco.studio.window  # noqa: F401
        from mujoco import mjrf
        from mujoco.rendering.filament import renderer

        self.mjrf, self.api = mjrf, renderer
        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.feeds, self.fog = feeds, fog
        self.context = mjrf.Context(mjrf.ContextConfig())
        info = mujoco.MjrRendererInfo()
        self.context.get_renderer_info(info)
        LOG.info("Renderer: %s, graphics backend: %s", info.renderer, info.backend)
        self.renderer = renderer.Renderer(self.context)
        self.scene = self.renderer.scene("robot")
        self.scene.configure_from_model(self.model)
        self.scene.set_msaa(int(model.vis.quality.offsamples))
        self.resources = mjrf.ModelResources(self.context, self.model)
        self.lights = mjrf.ModelLights(self.scene, self.resources)
        self.renderables = mjrf.ModelRenderables(self.scene, self.resources)
        self.renderables.set_options(mujoco.MjvOption())
        # MuJoCo 4.0 currently permits only one pixel read per render call.
        # Pack all views into one atlas, then expose slices as camera frames.
        self.height = sum(height for _, height in sizes.values())
        self.width = max(width for width, _ in sizes.values())
        self.atlas = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.renderer.target("cameras", self.atlas)
        self.pixels, self.rects = {}, {}
        top = 0
        for key, (width, height) in sizes.items():
            self.pixels[key] = self.atlas[top:top + height, :width]
            self.rects[key] = (0, self.height - top - height, width, height)
            top += height
        self.camera_ids = {key: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, feed.camera)
                           for key, feed in feeds.items()}
        self.keys = tuple(feeds)

    def prepare(self, model, data, keys):
        mujoco.mj_copyData(self.data, self.model, data)
        # Games change colors and lamp emission at runtime. Keep these on the
        # render model so physics never mutates the worker's model or data.
        self.model.geom_rgba[:] = model.geom_rgba
        self.model.mat_rgba[:] = model.mat_rgba
        self.model.mat_emission[:] = model.mat_emission
        self.keys = tuple(keys)
        return self

    def render(self):
        self.lights.update(self.data)
        self.renderables.update(self.data)
        views = []
        for key in self.keys:
            camera_id = self.camera_ids[key]
            fixed = mujoco.MjvCamera()
            fixed.type = mujoco.mjtCamera.mjCAMERA_FIXED
            fixed.fixedcamid = camera_id
            camera = mujoco.mjv_camera2GLCamera(self.model, self.data, fixed)
            stereo = self.feeds[key].stereo
            x, y, width, height = self.rects[key]
            for eye in range(2 if stereo else 1):
                pos = self.data.cam_xpos[camera_id].copy()
                if eye:
                    right = self.data.cam_xmat[camera_id].reshape(3, 3)[:, 0]
                    pos += stereo.baseline * right
                view_camera = self.mjrf.Camera(
                    pos=pos, forward=camera.forward, up=camera.up,
                    frustum_center=camera.frustum_center, frustum_width=camera.frustum_width,
                    frustum_bottom=camera.frustum_bottom, frustum_top=camera.frustum_top,
                    frustum_near=camera.frustum_near, frustum_far=camera.frustum_far,
                    orthographic=camera.orthographic)
                views.append(self.api.View(
                    scene="robot", target="cameras", camera=view_camera,
                    viewport=((x + eye * width / 2) / self.width, y / self.height,
                              width / (2 if stereo else 1) / self.width, height / self.height),
                    reflections=False, haze=self.fog,
                    headlight=bool(self.model.vis.headlight.active),
                    headlight_color=tuple(self.model.vis.headlight.diffuse),
                    headlight_intensity=10_000.0))
        self.renderer.render(views)
        return {key: self.pixels[key] for key in self.keys}

    def close(self):
        # Drop dependent GPU objects before their context, on the render worker.
        self.renderables = None
        self.lights = None
        self.resources = None
        self.scene = None
        self.renderer = None
        self.context = None
