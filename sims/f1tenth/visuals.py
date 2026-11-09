"""Procedural textures and merged meshes for the racing scene.

Everything is generated once at startup, in memory, with no downloaded art.
Static scenery is merged into a handful of textured meshes so the classic
renderer issues a few dozen draw calls per frame instead of ~900 boxes.
"""
from functools import lru_cache
import io
import math

import numpy as np

CYAN = (0.12, 0.84, 0.98)  # LiveKit accent


def _png(rgb):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _noise(shape, scale, rng):
    """Tileable value noise: random lattice upsampled with wraparound."""
    h, w = shape
    lattice = rng.random((max(1, h // scale), max(1, w // scale)))
    small = np.fft.rfft2(lattice)
    out = np.fft.irfft2(small, s=lattice.shape)
    # Bilinear upsample with wrap so the texture tiles seamlessly.
    ys = np.arange(h) * out.shape[0] / h
    xs = np.arange(w) * out.shape[1] / w
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    y1, x1 = (y0 + 1) % out.shape[0], (x0 + 1) % out.shape[1]
    top = out[y0][:, x0] * (1 - fx) + out[y0][:, x1] * fx
    bottom = out[y1][:, x0] * (1 - fx) + out[y1][:, x1] * fx
    value = top * (1 - fy) + bottom * fy
    return (value - value.min()) / (np.ptp(value) + 1e-9)


@lru_cache(maxsize=1)
def textures():
    """PNG bytes keyed by MuJoCo asset filename."""
    rng = np.random.default_rng(7)
    out = {}

    # Asphalt across the full road width (u) and 2.2 m along the road (v).
    size = 512
    grain = (0.12 * rng.random((size, size)) + 0.5 * _noise((size, size), 8, rng)
             + 0.38 * _noise((size, size), 64, rng))
    u = (np.arange(size) + 0.5) / size
    base = 52 + 22 * grain
    # Rubbered-in racing groove and slightly lighter, dusty edges.
    base -= 9 * np.exp(-((u - 0.5) / 0.2) ** 2)[None, :]
    base += 8 * (np.abs(u - 0.5) > 0.38)[None, :]
    road = np.repeat(base[:, :, None], 3, axis=2) * np.array([1.0, 1.0, 1.04])
    for lo, hi in ((0.018, 0.045), (0.955, 0.982)):
        mask = (u >= lo) & (u <= hi)
        road[:, mask] = 222 + 18 * grain[:, mask, None]
    out["asphalt.png"] = _png(road)

    # Red/white curb stripes; u across the curb, v along it (one red + one white).
    curb = np.zeros((128, 32, 3))
    curb[:64] = (205, 28, 30)
    curb[64:] = (238, 238, 232)
    shade = 0.82 + 0.18 * np.sin(np.linspace(0, math.pi, 32))
    curb *= shade[None, :, None]
    curb *= (0.94 + 0.06 * rng.random((128, 32)))[:, :, None]
    out["curb.png"] = _png(curb)

    # Mown grass: two-tone bands are produced by the plane's texture repeat.
    g = 256
    blades = 0.15 * rng.random((g, g)) + 0.85 * _noise((g, g), 16, rng)
    band = (np.arange(g) < g // 2)[None, :]
    grass = np.stack([46 + 18 * blades, 98 + 30 * blades, 40 + 14 * blades], axis=2)
    grass *= np.where(band, 1.0, 0.86)[:, :, None]
    out["grass.png"] = _png(grass)

    # Trackside hoardings: u along the wall (8 m per tile), v up the wall.
    out["hoarding.png"] = _png(_hoarding(rng))

    # Grandstand crowd: dense, colorful specks over dark seating.
    c = 256
    crowd = np.full((c, c, 3), 40.0)
    palette = np.array([[230, 230, 230], [31, 213, 249], [220, 60, 50], [250, 200, 40],
                        [60, 90, 200], [30, 30, 30], [200, 120, 80], [240, 160, 170]])
    spots = rng.random((c // 4, c // 4)) < 0.75
    colors = palette[rng.integers(0, len(palette), spots.shape)]
    tile = np.where(spots[:, :, None], colors, 40)
    crowd = np.kron(tile, np.ones((4, 4, 1)))
    crowd[3::16] *= 0.55  # seat rows
    out["crowd.png"] = _png(crowd)

    # Pit building: light panels with a dark glazing band.
    b = np.full((128, 128, 3), 205.0)
    b[40:88] = (38, 52, 64)
    b[40:88, ::16] = 150
    b[:6] = b[-6:] = 120
    out["building.png"] = _png(b * (0.95 + 0.05 * rng.random((128, 128)))[:, :, None])

    return out


def _hoarding(rng):
    from PIL import Image, ImageDraw, ImageFont
    w, h = 1024, 128
    image = Image.new("RGB", (w, h), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=74)
    except TypeError:  # Pillow < 10.1 has only the tiny bitmap font.
        font = ImageFont.load_default()
    panels = [((10, 10, 10), "LIVEKIT", tuple(int(255 * c) for c in CYAN)),
              ((240, 240, 240), "REALTIME", (20, 20, 20)),
              (tuple(int(255 * c) for c in CYAN), "TELEOP", (10, 10, 10)),
              ((28, 28, 30), "1/10 SCALE", (240, 240, 240))]
    width = w // len(panels)
    for i, (background, text, ink) in enumerate(panels):
        x0 = i * width
        draw.rectangle((x0, 0, x0 + width - 1, h - 1), fill=background)
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        scale = min(1.0, (width - 24) / max(tw, 1))
        if scale < 1:
            label = Image.new("RGB", (tw + 4, th + 4), background)
            ImageDraw.Draw(label).text((2 - box[0], 2 - box[1]), text, font=font, fill=ink)
            label = label.resize((max(1, int(label.width * scale)), max(1, int(label.height * scale))))
            image.paste(label, (x0 + (width - label.width) // 2, (h - label.height) // 2))
        else:
            draw.text((x0 + (width - tw) // 2 - box[0], (h - th) // 2 - box[1]), text, font=font, fill=ink)
    pixels = np.asarray(image, dtype=float)
    pixels[:8] = pixels[-8:] = 70  # top/bottom frame of the barrier
    # MuJoCo samples v from the bottom row up; flip so text reads upright.
    return np.flipud(pixels)


class MeshBuilder:
    """Accumulates one textured triangle mesh."""

    def __init__(self):
        self.vertices, self.uvs, self.faces = [], [], []

    def __bool__(self):
        return bool(self.faces)

    def quad(self, a, b, c, d, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
        """Counter-clockwise quad (seen from the visible side)."""
        base = len(self.vertices)
        self.vertices.extend([a, b, c, d])
        self.uvs.extend(uv)
        self.faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    def tri(self, a, b, c):
        base = len(self.vertices)
        self.vertices.extend([a, b, c])
        self.uvs.extend([(0, 0), (1, 0), (0.5, 1)])
        self.faces.append([base, base + 1, base + 2])

    def strip(self, left, right, uv_left, uv_right):
        """Upward-facing ribbon between two (n, 3) polylines; consecutive rings form quads."""
        base = len(self.vertices)
        n = len(left)
        for i in range(n):
            self.vertices.extend([left[i], right[i]])
            self.uvs.extend([uv_left[i], uv_right[i]])
        for i in range(n - 1):
            a, b = base + 2 * i, base + 2 * i + 2
            self.faces.extend([[a, a + 1, b + 1], [a, b + 1, b]])

    def box(self, center, yaw, half, tile=1.0, bottom=False):
        """Axis-aligned-in-yaw box with uv in meters / tile on every face."""
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, z = half

        def p(i, j, k):
            return (center[0] + c * i * x - s * j * y, center[1] + s * i * x + c * j * y, center[2] + k * z)

        def uv(du, dv):
            return ((0, 0), (du / tile, 0), (du / tile, dv / tile), (0, dv / tile))

        self.quad(p(-1, -1, 1), p(1, -1, 1), p(1, 1, 1), p(-1, 1, 1), uv(2 * x, 2 * y))
        self.quad(p(-1, -1, -1), p(1, -1, -1), p(1, -1, 1), p(-1, -1, 1), uv(2 * x, 2 * z))
        self.quad(p(1, 1, -1), p(-1, 1, -1), p(-1, 1, 1), p(1, 1, 1), uv(2 * x, 2 * z))
        self.quad(p(1, -1, -1), p(1, 1, -1), p(1, 1, 1), p(1, -1, 1), uv(2 * y, 2 * z))
        self.quad(p(-1, 1, -1), p(-1, -1, -1), p(-1, -1, 1), p(-1, 1, 1), uv(2 * y, 2 * z))
        if bottom:
            self.quad(p(-1, 1, -1), p(1, 1, -1), p(1, -1, -1), p(-1, -1, -1), uv(2 * x, 2 * y))

    def cone(self, base, radius, height, sides=6, phase=0.0):
        apex = (base[0], base[1], base[2] + height)
        ring = [(base[0] + radius * math.cos(phase + 2 * math.pi * k / sides),
                 base[1] + radius * math.sin(phase + 2 * math.pi * k / sides), base[2]) for k in range(sides)]
        for k in range(sides):
            self.tri(ring[k], ring[(k + 1) % sides], apex)

    def attributes(self):
        from .scene import numbers
        return {"vertex": numbers(np.asarray(self.vertices, dtype=float).ravel()),
                "face": " ".join(str(int(i)) for i in np.asarray(self.faces).ravel()),
                "texcoord": numbers(np.asarray(self.uvs, dtype=float).ravel())}


def loft(sections, samples=14, y_offset=0.0):
    """Closed superellipse loft along body X: sections are (x, half_width, z_bottom, z_top, exponent)."""
    builder = MeshBuilder()
    rings = []
    for x, hw, zb, zt, e in sections:
        zc, hz = (zb + zt) / 2, (zt - zb) / 2
        ring = []
        for k in range(samples):
            t = 2 * math.pi * k / samples
            c, s = math.cos(t), math.sin(t)
            ring.append((x, y_offset + hw * math.copysign(abs(c) ** (2 / e), c),
                         zc + hz * math.copysign(abs(s) ** (2 / e), s)))
        rings.append(ring)
    # Sections run front (+x) to back; quads face outward.
    for r in range(len(rings) - 1):
        a, b = rings[r], rings[r + 1]
        for k in range(samples):
            m = (k + 1) % samples
            builder.quad(a[k], b[k], b[m], a[m])
    for ring, front in ((rings[0], True), (rings[-1], False)):
        center = tuple(np.mean(ring, axis=0))
        for k in range(samples):
            m = (k + 1) % samples
            builder.tri(center, ring[k], ring[m]) if front else builder.tri(center, ring[m], ring[k])
    return builder
