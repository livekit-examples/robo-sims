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


# Hoarding atlas: the advertising strip fills the top half (v 0.5-1) and the
# trackside signs live in 128 px cells along the bottom half (v 0-0.5).
ATLAS_W, ATLAS_H, CELL = 1024, 256, 128
WALL_V = (0.502, 0.998)  # wall face, bottom to top
WALL_TOP_V = (0.505, 0.51)  # dark frame row reused for the wall cap
SIGNS = {"3": 0, "2": 1, "1": 2, "apex_left": 3, "apex_right": 4, "back": 5}


def atlas_cell(name, inset=10):
    """(u0, v0, u1, v1) of one sign cell, inset past its gutter to avoid mip bleed."""
    k = SIGNS[name]
    return ((k * CELL + inset) / ATLAS_W, inset / ATLAS_H,
            ((k + 1) * CELL - inset) / ATLAS_W, (CELL - inset) / ATLAS_H)


def _stamp(canvas, mask, color, strength=1.0):
    color = np.asarray(color, dtype=float)
    a = np.clip(mask, 0, 1)[..., None] * strength
    canvas[:] = canvas * (1 - a) + color * a


@lru_cache(maxsize=1)
def textures():
    """PNG bytes keyed by MuJoCo asset filename."""
    rng = np.random.default_rng(7)
    out = {}

    # Asphalt across the full road width (u) and 2.2 m along the road (v). The
    # road mesh maps u = 0.5 onto the racing line, so the rubbered groove below
    # follows it; the outer 6% of u on each side is the edge zone under the white line.
    size = 512
    grain = (0.12 * rng.random((size, size)) + 0.5 * _noise((size, size), 8, rng)
             + 0.38 * _noise((size, size), 64, rng))
    u = (np.arange(size) + 0.5) / size
    base = 50 + 22 * grain
    # Aggregate: bright stones and dark voids at pixel scale, some two pixels wide.
    stones = rng.random((size, size))
    base += 26 * (stones < 0.03) * rng.random((size, size))
    base += 14 * (np.roll(stones, 1, axis=1) < 0.012)
    base -= 18 * (stones > 0.975)
    # Polished, rubbered racing groove with two faint tire lanes in it.
    polish = np.exp(-((u - 0.5) / 0.09) ** 2)[None, :]
    lanes = sum(np.exp(-((u - c) / 0.018) ** 2) for c in (0.44, 0.56))[None, :]
    streaks = 0.6 + 0.4 * _noise((size, 64), 4, rng)[:, np.minimum((u * 64).astype(int), 63)]
    base = base * (1 - 0.35 * polish) + 50 * 0.35 * polish  # less grain where polished
    base -= (11 * polish + 5 * lanes) * streaks
    # Dusty, lighter margins outside the driven width.
    base += 7 * np.clip((np.abs(u - 0.5) - 0.40) / 0.06, 0, 1)[None, :]
    road = np.repeat(base[:, :, None], 3, axis=2) * np.array([1.0, 1.0, 1.04])
    # The white edge lines are separate geometry (scene "edge_line"): painted into
    # this texture they mip-mapped into grey blur at grazing distances.
    out["asphalt.png"] = _png(road)

    # Corner rubber: a translucent overlay laid along the racing line through
    # bends. Mid-gray edges vanish into the asphalt; the core and the two
    # tire tracks read as a dark, rubbered-in line. u across (0.5 m), v 2 m along.
    rw, rh = 64, 256
    ru = (np.arange(rw) + 0.5) / rw
    core = np.exp(-((ru - 0.5) / 0.23) ** 2)
    tracks = sum(np.exp(-((ru - c) / 0.06) ** 2) for c in (0.24, 0.76))
    along = _noise((rh, rw), 4, rng)
    fibres = 0.75 + 0.25 * np.repeat(rng.random((1, rw)), rh, axis=0)
    rubber = 62 - (32 * core + 12 * tracks)[None, :] * fibres * (0.8 + 0.2 * along)
    out["rubber.png"] = _png(np.repeat(rubber[:, :, None], 3, axis=2))

    # Red/white curb stripes; u across (0 road side, 1 raised outer edge),
    # v along: four red/white pairs per tile so scuffs do not repeat every stripe.
    ch, cw = 512, 64
    curb = np.zeros((ch, cw, 3))
    stripe = (np.arange(ch) // 64) % 2 == 0
    curb[stripe] = (205, 28, 30)
    curb[~stripe] = (238, 238, 232)
    cu = (np.arange(cw) + 0.5) / cw
    curb *= (0.80 + 0.20 * np.sin(np.pi * np.clip(cu * 1.15, 0, 1)))[None, :, None]
    curb *= (0.93 + 0.07 * rng.random((ch, cw)))[:, :, None]
    # Tire scuffs: dark, slightly diagonal rubber smears, mostly on the road side.
    rows, cols = np.mgrid[0:ch, 0:cw]
    scuff = np.zeros((ch, cw))
    for _ in range(46):
        v0, length = rng.uniform(0, ch), rng.uniform(25, 150)
        c0, drift = rng.beta(1.3, 3.2) * cw, rng.uniform(-0.06, 0.12)
        width, depth = rng.uniform(1.2, 4.5), rng.uniform(0.25, 0.65)
        t = ((rows - v0) % ch) / length
        inside = (t >= 0) & (t <= 1)
        centre = c0 + drift * t * length
        fade = np.sin(np.pi * np.clip(t, 0, 1)) ** 0.6
        scuff = np.maximum(scuff, inside * fade * depth * np.exp(-((cols - centre) / width) ** 2))
    grime = 0.22 * _noise((ch, cw), 8, rng) * np.clip(1.2 - cu * 1.6, 0, 1)[None, :]
    curb *= (1 - np.clip(scuff + grime, 0, 0.8))[:, :, None]
    curb += (np.clip(scuff + grime, 0, 0.8) * 12)[:, :, None]  # rubber is near-black, not pure black
    out["curb.png"] = _png(curb)

    # Mown grass: two-tone bands are produced by the plane's texture repeat.
    g = 256
    blades = 0.15 * rng.random((g, g)) + 0.85 * _noise((g, g), 16, rng)
    band = (np.arange(g) < g // 2)[None, :]
    grass = np.stack([46 + 18 * blades, 98 + 30 * blades, 40 + 14 * blades], axis=2)
    grass *= np.where(band, 1.0, 0.86)[:, :, None]
    out["grass.png"] = _png(grass)

    # Trackside hoardings and braking/apex signs share one atlas, so the signs
    # merge into the barrier mesh instead of adding geoms.
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

    # 2x2 twill carbon weave; each tow is shaded like a small cylinder so the
    # weave catches the specular highlight at an angle.
    cell, n = 8, 8
    y, x = np.mgrid[0:cell * n, 0:cell * n]
    i, j = y // cell, x // cell
    horizontal = ((i + j) % 4) < 2
    yin, xin = (y % cell + 0.5) / cell, (x % cell + 0.5) / cell
    tow = np.where(horizontal, np.sin(np.pi * yin), np.sin(np.pi * xin)) ** 0.7
    sheen = np.where(horizontal, 1.0, 0.72)
    weave = 12 + 34 * tow * sheen
    out["carbon.png"] = _png(np.stack([weave, weave * 1.03, weave * 1.12], axis=2))

    out["livery.png"] = _png(_livery(text=False))
    out["podlivery.png"] = _png(_livery(text=True))
    return out


def _livery(text):
    """Gloss black with a sharp shoulder highlight and a cyan swoosh.

    Loft uv: u runs nose (1) to tail (0) along body x; v = angle / 2pi around
    the section, with v 0 / 0.5 the +y / -y waist and v 0.25 the top. Array row
    r samples v = r / height.
    """
    w, h = 256, 128
    u = (np.arange(w) + 0.5) / w
    t = (np.arange(h) + 0.5) / h * 2 * np.pi
    height = np.sin(t)[:, None]  # +1 top, -1 bottom
    # Painted-on environment: bright sky above the shoulder, dark ground below.
    shade = 0.55 + 0.75 * np.clip(height, 0, 1) ** 2 - 0.25 * np.clip(-height, 0, 1)
    shoulder = sum(np.exp(-((t - c) / 0.11) ** 2) for c in (0.95, np.pi - 0.95))[:, None]
    rgb = np.empty((h, w, 3))
    rgb[:] = (17, 20, 26)
    rgb *= shade[..., None]
    rgb += (60 * shoulder)[..., None] * np.array([0.8, 0.95, 1.0])
    if text:
        _side_text(rgb, "LIVEKIT")
        return rgb
    # Cyan swoosh rising toward the tail, with a white pinstripe beneath.
    rise = 0.12 + 0.22 * np.clip((0.85 - u) / 0.7, 0, 1) ** 1.5
    ends = np.clip((u - 0.06) / 0.08, 0, 1) * np.clip((0.97 - u) / 0.08, 0, 1)
    for side in (1, -1):
        centre = rise if side > 0 else np.pi - rise
        d = side * (t[:, None] - centre[None, :])  # positive = above the swoosh
        # Edges are anti-aliased in the texture so the stripe stays crisp, not stair-stepped.
        _stamp(rgb, np.clip((0.075 - np.abs(d)) / 0.05 + 0.5, 0, 1) * ends, (31, 213, 249))
        _stamp(rgb, np.clip((0.022 - np.abs(d + 0.12)) / 0.04 + 0.5, 0, 1) * ends, (235, 240, 245), 0.9)
    return rgb


def _side_text(rgb, word, aspect=1.3):
    """Wordmark centred on both pod flanks, upright and readable from either side.

    aspect corrects the loft uv stretch: (pod length / texture width) / (girth / height).
    """
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=60)
    except TypeError:
        font = ImageFont.load_default()
    box = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), word, font=font)
    label = Image.new("L", (box[2] - box[0] + 8, box[3] - box[1] + 8), 0)
    ImageDraw.Draw(label).text((4 - box[0], 4 - box[1]), word, font=font, fill=255)
    h, w = rgb.shape[:2]
    tw = int(w * 0.42)
    th = max(8, int(label.height * tw / label.width / aspect))
    mask = np.asarray(label.resize((tw, th), Image.LANCZOS), dtype=float) / 255
    layer = np.zeros((h, w))
    u0, gap = int(w * 0.30), 3
    # -y flank (v 0.5): rows run downward on the flank, u toward the nose = viewer's right.
    # Both wordmarks sit just above the waist, clear of the pod's cyan stripe.
    layer[h // 2 - gap - th:h // 2 - gap, u0:u0 + tw] = mask
    # +y flank (v 0): rows run upward and the nose is on the viewer's left.
    plus = np.zeros((h, w))
    plus[h // 2 + gap:h // 2 + gap + th, u0:u0 + tw] = mask[::-1, ::-1]
    layer = np.maximum(layer, np.roll(plus, -h // 2, axis=0))
    _stamp(rgb, layer, (31, 213, 249))


def _hoarding(rng):
    from PIL import Image, ImageDraw, ImageFont
    w, h = ATLAS_W, CELL
    image = Image.new("RGB", (w, h), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=74)
        big = ImageFont.load_default(size=104)
    except TypeError:  # Pillow < 10.1 has only the tiny bitmap font.
        font = big = ImageFont.load_default()
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
    strip = np.asarray(image, dtype=float).copy()
    strip[:8] = strip[-8:] = 70  # top/bottom frame of the barrier

    # Sign cells: 3-2-1 braking boards, apex chevrons, plain back.
    signs = Image.new("RGB", (w, CELL), (90, 90, 92))
    draw = ImageDraw.Draw(signs)
    cyan = tuple(int(255 * c) for c in CYAN)
    for label in ("3", "2", "1"):
        x0 = SIGNS[label] * CELL
        draw.rectangle((x0 + 6, 6, x0 + CELL - 7, CELL - 7), fill=(12, 12, 14))
        draw.rectangle((x0 + 13, 13, x0 + CELL - 14, CELL - 14), fill=(245, 245, 245))
        draw.rectangle((x0 + 13, 13, x0 + CELL - 14, 28), fill=cyan)
        box = draw.textbbox((0, 0), label, font=big)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.text((x0 + (CELL - tw) // 2 - box[0], 30 + (CELL - 44 - th) // 2 - box[1]), label, font=big,
                  fill=(12, 12, 14))
    for name, sign in (("apex_left", -1), ("apex_right", 1)):
        x0 = SIGNS[name] * CELL
        draw.rectangle((x0 + 6, 6, x0 + CELL - 7, CELL - 7), fill=(12, 12, 14))
        for k in range(3):
            cx = x0 + CELL // 2 + sign * (k - 1) * 30
            tip, back = cx + sign * 16, cx - sign * 16
            draw.polygon([(back, 22), (tip, CELL // 2), (back, CELL - 22), (back + sign * 12, CELL - 22),
                          (tip + sign * 12, CELL // 2), (back + sign * 12, 22)], fill=cyan if k != 1 else (245, 245, 245))
    x0 = SIGNS["back"] * CELL
    draw.rectangle((x0 + 6, 6, x0 + CELL - 7, CELL - 7), fill=(120, 122, 126))
    atlas = np.vstack([strip, np.asarray(signs, dtype=float)])
    # MuJoCo samples v from the bottom row up; flip so text reads upright.
    return np.flipud(atlas)


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

    def tri(self, a, b, c, uv=((0, 0), (1, 0), (0.5, 1))):
        base = len(self.vertices)
        self.vertices.extend([a, b, c])
        self.uvs.extend(uv)
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

    def tube(self, start, end, radius, sides=5):
        """Open prism between two points (wishbones, struts); uv wraps once around."""
        start, end = np.asarray(start, float), np.asarray(end, float)
        axis = end - start
        axis /= np.linalg.norm(axis)
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(axis, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        ring = [radius * (math.cos(2 * math.pi * k / sides) * e1 + math.sin(2 * math.pi * k / sides) * e2)
                for k in range(sides)]
        for k in range(sides):
            m = (k + 1) % sides
            self.quad(tuple(start + ring[k]), tuple(start + ring[m]), tuple(end + ring[m]), tuple(end + ring[k]),
                      ((k / sides, 0), ((k + 1) / sides, 0), ((k + 1) / sides, 1), (k / sides, 1)))

    def disc(self, center, rx, ry, sides=24):
        """Upward-facing ellipse fan."""
        cx, cy, cz = center
        ring = [(cx + rx * math.cos(2 * math.pi * k / sides), cy + ry * math.sin(2 * math.pi * k / sides), cz)
                for k in range(sides)]
        for k in range(sides):
            self.tri((cx, cy, cz), ring[k], ring[(k + 1) % sides], uv=((0.5, 0.5),) * 3)

    def panel(self, center, right, up, half_w, half_h, depth, front_uv, back_uv):
        """Thin two-sided sign: front quad textured from the atlas, back and edges plain.

        right/up are unit vectors in the sign plane; the front faces right x up.
        """
        c, r, u = (np.asarray(v, float) for v in (center, right, up))
        n = np.cross(r, u) * depth
        u0, v0, u1, v1 = front_uv
        b0, c0, b1, c1 = back_uv
        corner = lambda i, j, k: tuple(c + i * half_w * r + j * half_h * u + k * n)  # noqa: E731
        self.quad(corner(-1, -1, 1), corner(1, -1, 1), corner(1, 1, 1), corner(-1, 1, 1),
                  ((u0, v0), (u1, v0), (u1, v1), (u0, v1)))
        plain = ((b0, c0), (b1, c0), (b1, c1), (b0, c1))
        self.quad(corner(1, -1, -1), corner(-1, -1, -1), corner(-1, 1, -1), corner(1, 1, -1), plain)
        self.quad(corner(1, -1, 1), corner(1, -1, -1), corner(1, 1, -1), corner(1, 1, 1), plain)
        self.quad(corner(-1, -1, -1), corner(-1, -1, 1), corner(-1, 1, 1), corner(-1, 1, -1), plain)
        self.quad(corner(-1, 1, 1), corner(1, 1, 1), corner(1, 1, -1), corner(-1, 1, -1), plain)

    def attributes(self):
        from .scene import numbers
        return {"vertex": numbers(np.asarray(self.vertices, dtype=float).ravel()),
                "face": " ".join(str(int(i)) for i in np.asarray(self.faces).ravel()),
                "texcoord": numbers(np.asarray(self.uvs, dtype=float).ravel())}


def spokes(radius, half_width, count=5, hub=0.2):
    """Five-spoke wheel insert spanning the rim width, axle along y (wheel body frame)."""
    builder = MeshBuilder()
    inner, outer, width = radius * hub, radius * 0.97, radius * 0.17
    for k in range(count):
        a = 2 * math.pi * k / count
        d = np.array([math.cos(a), 0, math.sin(a)])
        side = np.array([-math.sin(a), 0, math.cos(a)])
        for sy in (1, -1):  # spoke faces on both rim faces, plus edges
            y = np.array([0, sy * half_width, 0])
            p = [inner * d - side * width, outer * d - side * width * 0.7,
                 outer * d + side * width * 0.7, inner * d + side * width]
            quad = [tuple(q + y) for q in p]
            builder.quad(*(quad if sy < 0 else quad[::-1]))
    hub_ring = [(inner * 1.25 * math.cos(2 * math.pi * k / 6), 0, inner * 1.25 * math.sin(2 * math.pi * k / 6))
                for k in range(6)]
    for sy in (1, -1):
        y = sy * (half_width + 0.0006)
        for k in range(6):
            a, b = hub_ring[k], hub_ring[(k + 1) % 6]
            tri = ((0, y, 0), (a[0], y, a[2]), (b[0], y, b[2]))
            builder.tri(*(tri if sy < 0 else (tri[0], tri[2], tri[1])))
    return builder


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
    # uv: u = 0 at the tail to 1 at the nose, v = section angle / 2pi (see _livery).
    xs = [section[0] for section in sections]
    us = [(x - min(xs)) / (max(xs) - min(xs)) for x in xs]
    # Sections run front (+x) to back; quads face outward.
    for r in range(len(rings) - 1):
        a, b = rings[r], rings[r + 1]
        for k in range(samples):
            m = (k + 1) % samples
            v0, v1 = k / samples, (k + 1) / samples
            builder.quad(a[k], b[k], b[m], a[m], ((us[r], v0), (us[r + 1], v0), (us[r + 1], v1), (us[r], v1)))
    for ring, front, u in ((rings[0], True, us[0]), (rings[-1], False, us[-1])):
        center = tuple(np.mean(ring, axis=0))
        cap = ((u, 0.75),) * 3  # underside colour: no stripes on the end caps
        for k in range(samples):
            m = (k + 1) % samples
            builder.tri(center, ring[k], ring[m], cap) if front else builder.tri(center, ring[m], ring[k], cap)
    return builder
