"""Generated formula-style visuals; vehicle dynamics have one separate owner.

Static scenery is merged into a few textured meshes (road, curbs, barriers,
stands, trees, hills), so a frame costs a few dozen draw calls. Road, curbs and
barriers are built from the same centerline and widths the physics uses.
"""
from dataclasses import dataclass
import logging
import math
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from ..mujoco_tools import mujoco
from .visuals import CYAN, WALL_TOP_V, WALL_V, MeshBuilder, atlas_cell, loft, spokes, textures

LOG = logging.getLogger("sims.f1tenth.scene")

# Heights in meters above the grass plane. The road surface sits slightly above
# the grass so depth precision stays clean at distance; tires touch ROAD_Z.
GRASS_Z, ROAD_Z, BODY_Z = -0.01, 0.004, 0.065

# Body-frame (position, look-at target) for the single streamed camera.
CAMERAS = {
    # Broadcast-style "T-cam" above the airbox: helmet, halo, nose and front tires in view.
    "tcam": ((-0.16, 0.0, 0.118), (1.4, 0.0, -0.03)),
    # Low cockpit view close to the driver's eye line.
    "cockpit": ((-0.06, 0.0, 0.07), (1.4, 0.0, 0.0)),
    # Rigid third-person chase view (no smoothing, so camera_info stays exact).
    "chase": ((-0.78, 0.0, 0.25), (0.6, 0.0, 0.03)),
}

# Indicator and wheel colors (linear RGB) for per-frame geom_rgba updates.
BRAKE_RED, AMBER = (1.0, 0.06, 0.04), (1.0, 0.55, 0.08)
SPOKE, RIM = (0.68, 0.7, 0.74), (0.11, 0.12, 0.13)
SPOKE_RGB, RIM_RGB = np.array(SPOKE), np.array(RIM)
STEERING_LOCK = 1.4  # steering-wheel radians at full road-wheel lock
MAX_SPIN = 24.0  # rad/s of visible spoke rotation; ~0.4 rad per 60 fps frame


def _rgba(color, level):
    return (*(level * np.asarray(color, dtype=float)), 1.0)


def _rgb(color, level):
    return numbers(_rgba(color, level))


# shadow: shadow-map size (0 disables); msaa: offscreen samples; scenery: trees/hills/stands.
# "auto" picks "high" on a GPU and "medium" on a software rasterizer (llvmpipe/OSMesa),
# where a shadow pass alone would more than double the frame time.
GRAPHICS = {
    "low": {"shadow": 0, "msaa": 0, "scenery": False},
    "medium": {"shadow": 0, "msaa": 0, "scenery": True},
    "high": {"shadow": 4096, "msaa": 4, "scenery": True},
}


@dataclass(frozen=True)
class Feed:
    name: str
    camera: str
    stereo: object = None


def numbers(values):
    return " ".join(f"{v:.7g}" for v in values)


def _xyaxes(position, target):
    forward = np.subtract(target, position).astype(float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return numbers([*right, *np.cross(right, forward)])


class Scene:
    def __init__(self, car, width=960, height=540, renderer="classic", camera="tcam", graphics="auto"):
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {', '.join(CAMERAS)}")
        if graphics != "auto" and graphics not in GRAPHICS:
            raise ValueError(f"graphics must be auto or one of {', '.join(GRAPHICS)}")
        self.car, self.width, self.height, self.renderer_kind = car, width, height, renderer
        # Auto builds the full model; open_renderers() turns off what a CPU cannot afford.
        self.camera, self.graphics = camera, graphics
        self.quality = GRAPHICS["high" if graphics == "auto" else graphics]
        self.feeds = {"onboard": Feed("onboard", "onboard")}
        self.frames = self.renderer = None
        self.model = mujoco.MjModel.from_xml_string(self._xml(), assets=textures())
        self.data = mujoco.MjData(self.model)
        self.render_data = mujoco.MjData(self.model)
        self._bind_runtime()
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "onboard")
        self.update()

    # ------------------------------------------------------------------ model
    def _xml(self):
        car, track, quality = self.car, self.car.track, self.quality
        lo, hi = track.points.min(axis=0), track.points.max(axis=0)
        center = (lo + hi) / 2
        root = Element("mujoco", model="formula_rc_spielberg")
        SubElement(root, "compiler", angle="radian")
        # A fixed extent keeps clip planes, fog and the shadow box in meters:
        # near 3 cm, far 300 m, fog 35-230 m, shadows within ~2.5 m of the car.
        SubElement(root, "statistic", extent="100", center=numbers([*center, 0]))
        visual = SubElement(root, "visual")
        SubElement(visual, "global", offwidth=str(self.width), offheight=str(self.height))
        SubElement(visual, "map", znear="0.0003", zfar="3", fogstart="0.35", fogend="2.3",
                   haze="0.35", shadowclip="0.08")
        SubElement(visual, "quality", shadowsize=str(max(quality["shadow"], 1)), offsamples=str(quality["msaa"]))
        SubElement(visual, "headlight", ambient="0.36 0.37 0.40", diffuse="0.22 0.22 0.22", specular="0.05 0.05 0.05")
        SubElement(visual, "rgba", fog="0.74 0.81 0.88 1", haze="0.82 0.87 0.92 1")

        defaults = SubElement(root, "default")
        SubElement(defaults, "geom", contype="0", conaffinity="0")
        SubElement(defaults, "mesh", inertia="shell")

        assets = SubElement(root, "asset")
        SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient",
                   rgb1="0.30 0.50 0.78", rgb2="0.86 0.90 0.94", width="256", height="256")
        for name in ("asphalt", "curb", "grass", "hoarding", "crowd", "building", "rubber", "livery",
                     "podlivery"):
            SubElement(assets, "texture", name=name, type="2d", file=f"{name}.png")
        SubElement(assets, "texture", name="checker", type="2d", builtin="checker",
                   rgb1="0.96 0.96 0.96", rgb2="0.04 0.04 0.04", width="64", height="64")
        SubElement(assets, "texture", name="weave", type="2d", file="carbon.png")
        materials = {
            "road": dict(texture="asphalt", specular="0.12", shininess="0.15"),
            "curb": dict(texture="curb", specular="0.25", shininess="0.3"),
            "grass": dict(texture="grass", texrepeat="0.2 0.2", texuniform="true", specular="0", shininess="0"),
            "hoarding": dict(texture="hoarding", specular="0.2", shininess="0.4"),
            "crowd": dict(texture="crowd", specular="0", shininess="0"),
            "building": dict(texture="building", specular="0.3", shininess="0.5"),
            "checker": dict(texture="checker", specular="0.1"),
            # Untextured so distant lines stay crisp (no mipmap blur at grazing angles).
            "edge_line": dict(rgba="0.86 0.86 0.84 1", specular="0.15", shininess="0.3"),
            "rubber_line": dict(texture="rubber", rgba="1 1 1 0.45", specular="0.05", shininess="0.2"),
            # Stacked layers in one mesh: each adds ~10% darkness toward the center.
            "shadow": dict(rgba="0 0 0 0.1", specular="0", shininess="0"),
            "steel": dict(rgba="0.62 0.64 0.67 1", specular="0.5", shininess="0.6"),
            "foliage": dict(rgba="0.13 0.27 0.12 1", specular="0", shininess="0"),
            "hills": dict(rgba="0.26 0.38 0.27 1", specular="0"),
            "mountains": dict(rgba="0.38 0.46 0.55 1", specular="0"),
            "lamp": dict(rgba="0.9 0.05 0.05 1", emission="0.6"),
            # Car livery: gloss black with LiveKit cyan accents. The loft textures carry a
            # painted-on sky/ground gradient and shoulder highlight for contrast without
            # reflections; materials keep a tight, bright specular.
            "livery": dict(texture="livery", specular="1", shininess="0.97"),
            "podlivery": dict(texture="podlivery", specular="1", shininess="0.97"),
            "paint": dict(rgba="0.06 0.07 0.085 1", specular="1", shininess="0.95"),
            "accent": dict(rgba=f"{CYAN[0]} {CYAN[1]} {CYAN[2]} 1", specular="0.9", shininess="0.92", emission="0.2"),
            "carbon": dict(texture="weave", texrepeat="140 140", texuniform="true", specular="0.75", shininess="0.85"),
            "rubber": dict(rgba="0.035 0.035 0.038 1", specular="0.06", shininess="0.15"),
            "rim": dict(rgba="0.11 0.12 0.13 1", specular="0.8", shininess="0.9"),
            "spoke": dict(rgba="0.68 0.7 0.74 1", specular="0.9", shininess="0.92"),
            # Self-lit indicators; per-frame geom_rgba sets their color and intensity.
            "indicator": dict(rgba="1 1 1 1", emission="1", specular="0.3", shininess="0.6"),
            "white": dict(rgba="0.93 0.93 0.93 1", specular="0.6", shininess="0.8"),
            "visor": dict(rgba="0.02 0.03 0.05 1", specular="1", shininess="1"),
        }
        for name, attributes in materials.items():
            SubElement(assets, "material", name=name, **attributes)

        world = SubElement(root, "worldbody")
        for name, builder in self._scenery(track, center).items():
            SubElement(assets, "mesh", name=name, **builder.attributes())
            SubElement(world, "geom", type="mesh", mesh=name, material=name)
        span = float(np.max(hi - lo)) / 2 + 80
        SubElement(world, "geom", type="plane", size=f"{span} {span} 8", pos=numbers([*center, GRASS_Z]), material="grass")
        self._gantry(world, track)
        self._car(world, assets, car)
        x, y, heading = track.spawn()
        position = (x - 1.4 * math.cos(heading), y - 1.4 * math.sin(heading), 0.9)
        # A fixed camera for exported previews; only onboard is streamed.
        SubElement(world, "camera", name="preview", pos=numbers(position),
                   xyaxes=_xyaxes(position, (x, y, 0.06)), fovy="50")
        return tostring(root, encoding="unicode")

    def _scenery(self, track, center):
        n = len(track.points)
        normals = track.normals + np.roll(track.normals, 1, axis=0)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        ring = np.r_[np.arange(n), 0]  # closes the loop with a fresh uv seam
        arc = track.arc  # n + 1 entries, arc[n] = full length
        points, ring_normals = track.points[ring], normals[ring]
        left_w, right_w = track.rows[ring, 3], track.rows[ring, 2]

        def lift(xy, z):
            return np.column_stack((xy, np.full(len(xy), z)))

        meshes = {}
        curvature = _curvature(track)
        racing_line = _racing_line(track, curvature)
        line = racing_line[ring]
        # Five columns across the road: fixed painted edge zones (u 0-0.06 and
        # 0.94-1) and the texture's rubbered groove (u 0.5) on the racing line.
        width = left_w + right_w
        lateral = [left_w, left_w - 0.06 * width, line, -right_w + 0.06 * width, -right_w]
        columns = [points + ring_normals * offset[:, None] for offset in lateral]
        road = MeshBuilder()
        v = arc / 2.2
        for (a, ua), (b, ub) in zip(zip(columns, (0, 0.06, 0.5, 0.94, 1)), zip(columns[1:], (0.06, 0.5, 0.94, 1))):
            road.strip(lift(a, ROAD_Z), lift(b, ROAD_Z), np.column_stack((np.full(n + 1, ua), v)),
                       np.column_stack((np.full(n + 1, ub), v)))
        meshes["road"] = road

        # White edge lines as their own flat strips (u 0.018-0.045 of the width in
        # from each edge). Geometry keeps them sharp at any distance.
        lines = MeshBuilder()
        flat = np.zeros((n + 1, 2))
        for outer_edge, inner_edge in ((left_w - 0.018 * width, left_w - 0.045 * width),
                                       (-right_w + 0.045 * width, -right_w + 0.018 * width)):
            a = _offset_line(points + ring_normals * outer_edge[:, None])
            b = _offset_line(points + ring_normals * inner_edge[:, None])
            lines.strip(lift(a, ROAD_Z + 0.0005), lift(b, ROAD_Z + 0.0005), flat, flat)
        meshes["edge_line"] = lines

        # Extra rubber laid down through the bends: a translucent strip on the
        # racing line that fades in on the brakes and out on the exit.
        corner = _smooth((np.abs(curvature) > 0.05).astype(float), 2.5 / track.lengths.mean())
        rubber = MeshBuilder()
        for run in _runs(corner > 0.04):
            idx = np.r_[run, (run[-1] + 1) % n]
            half = 0.25 * np.clip(corner[idx] * 1.6, 0, 1)
            centre = track.points[idx] + normals[idx] * racing_line[idx][:, None]
            along = np.unwrap(arc[idx], period=track.length) / 2.0
            rubber.strip(lift(centre + normals[idx] * half[:, None], ROAD_Z + 0.0004),
                         lift(centre - normals[idx] * half[:, None], ROAD_Z + 0.0004),
                         np.column_stack((np.zeros(len(idx)), along)), np.column_stack((np.ones(len(idx)), along)))
        if rubber:
            meshes["rubber_line"] = rubber

        # Curbs only where the circuit bends, like a real track.
        bend = np.abs(curvature) > 0.05
        mask = np.convolve(np.r_[bend[-8:], bend, bend[:8]], np.ones(17), "valid") > 0
        curbs = MeshBuilder()
        for run in _runs(mask):
            idx = np.r_[run, (run[-1] + 1) % n]
            p, q = track.points[idx], normals[idx]
            along = np.unwrap(arc[idx], period=track.length) / 1.28  # four stripe pairs per tile
            for sign, widths in ((1, track.rows[idx, 3]), (-1, track.rows[idx, 2])):
                inner = p + sign * q * widths[:, None]
                outer = p + sign * q * (widths[:, None] + 0.14)
                a, b = lift(outer, ROAD_Z + 0.004), lift(inner, ROAD_Z + 0.0005)
                uv_a = np.column_stack((np.ones(len(idx)), along))
                uv_b = np.column_stack((np.zeros(len(idx)), along))
                if sign > 0:
                    curbs.strip(a, b, uv_a, uv_b)
                else:
                    curbs.strip(b, a, uv_b, uv_a)
        meshes["curb"] = curbs

        # Barriers at the physics limit (half width + runoff), hoardings facing the road.
        walls = MeshBuilder()
        offset = track.barrier_runoff
        for sign, widths in ((1, left_w), (-1, right_w)):
            # Trim fold-backs on hairpin insides so the wall follows the true
            # physics boundary instead of cutting across the road.
            inner = _offset_line(points + sign * ring_normals * (widths + offset)[:, None])
            outer = _offset_line(points + sign * ring_normals * (widths + offset + 0.05)[:, None])
            u = sign * arc / 8.0
            for i in range(n):
                j = i + 1
                if np.linalg.norm(inner[j] - inner[i]) < 1e-6 and np.linalg.norm(outer[j] - outer[i]) < 1e-6:
                    continue  # collapsed into the fold-back corner
                bi, bj = (*inner[i], GRASS_Z), (*inner[j], GRASS_Z)
                ti, tj = (*inner[i], 0.24), (*inner[j], 0.24)
                oi, oj = (*outer[i], 0.24), (*outer[j], 0.24)
                gi, gj = (*outer[i], GRASS_Z), (*outer[j], GRASS_Z)
                face = ((u[i], WALL_V[0]), (u[j], WALL_V[0]), (u[j], WALL_V[1]), (u[i], WALL_V[1]))
                top = ((u[i], WALL_TOP_V[0]), (u[j], WALL_TOP_V[0]), (u[j], WALL_TOP_V[1]), (u[i], WALL_TOP_V[1]))
                if sign > 0:
                    walls.quad(bi, bj, tj, ti, face)
                    walls.quad(ti, tj, oj, oi, top)
                    walls.quad(gj, gi, oi, oj, (face[1], face[0], face[3], face[2]))
                else:
                    walls.quad(bj, bi, ti, tj, (face[1], face[0], face[3], face[2]))
                    walls.quad(tj, ti, oi, oj, top)
                    walls.quad(gi, gj, oj, oi, face)
        self._signs(walls, track, curvature, normals)
        meshes["hoarding"] = walls

        # Checkered start/finish line across the road.
        start = MeshBuilder()
        p, t, q = track.points[0], track.tangents[0], normals[0]
        corners = [p + q * track.rows[0, 3] - t * 0.09, p - q * track.rows[0, 2] - t * 0.09,
                   p - q * track.rows[0, 2] + t * 0.09, p + q * track.rows[0, 3] + t * 0.09]
        z = ROAD_Z + 0.0006
        start.quad(*[(*c, z) for c in corners], uv=((0, 0), (6, 0), (6, 1), (0, 1)))
        meshes["checker"] = start
        if self.quality["scenery"]:
            meshes.update(self._surroundings(track, center, normals))
        return meshes

    def _signs(self, walls, track, curvature, normals):
        """3-2-1 braking boards on the outside before each real corner and an apex
        chevron on the inside, merged into the barrier mesh (no extra geoms)."""
        n, arc = len(track.points), track.arc
        z_axis = np.array([0.0, 0.0, 1.0])
        back = atlas_cell("back")

        def frame(s, side, toward_road):
            s %= track.length
            i = int(np.searchsorted(arc, s, side="right") - 1) % n
            t, q = track.tangents[i], normals[i]
            width = track.rows[i, 3 if side > 0 else 2]
            point = track.points[i] + t * (s - arc[i]) + side * q * (width + track.barrier_runoff + 0.025)
            facing = -t * math.cos(toward_road) - side * q * math.sin(toward_road)
            facing3 = np.array([*facing, 0.0])
            return point, np.cross(z_axis, facing3), facing3

        def place(s, side, cell, half_w, half_h, toward_road):
            point, right, _ = frame(s, side, toward_road)
            # Skip spots where the barrier line folds back toward the road (tight hairpin insides).
            nearest = track.project(point)
            if abs(nearest.lateral) - nearest.half_width < track.barrier_runoff * 0.9:
                return
            walls.panel((*point, 0.245 + half_h), right, z_axis, half_w, half_h, 0.006, atlas_cell(cell), back)

        for entry, apex, direction, straight in _corners(track, curvature):
            outside = -direction  # left turn (positive curvature) -> boards on the right
            for label, distance in (("3", 7.5), ("2", 5.0), ("1", 2.5)):
                if straight >= distance + 1.0:
                    place(arc[entry] - distance, outside, label, 0.12, 0.12, math.radians(18))
            place(arc[apex] + 0.4, direction, "apex_left" if direction > 0 else "apex_right", 0.11, 0.08,
                  math.radians(35))

    def _surroundings(self, track, center, normals):
        rng = np.random.default_rng(3)
        out = {}
        clearance = track.barrier_runoff + 0.4

        def clear(xy, margin):
            projection = track.project(xy)
            return abs(projection.lateral) > projection.half_width + margin

        # Grandstand on one side of the start straight, pit building on the other.
        stands, building = MeshBuilder(), MeshBuilder()
        reserved = []
        p0, t0 = track.points[0], track.tangents[0]
        yaw = math.atan2(t0[1], t0[0])
        sides = []
        for sign in (1, -1):
            width = track.rows[0, 3 if sign > 0 else 2]
            footprint = [p0 + sign * normals[0] * (width + clearance + d) + t0 * s
                         for d in (0.0, 1.0) for s in (-12.0, 0.0, 12.0)]
            if all(clear(xy, clearance - 0.05) for xy in footprint):
                sides.append(sign)
        for k, sign in enumerate(sides[:2]):
            base = p0 + sign * normals[0] * (track.rows[0, 3 if sign > 0 else 2] + clearance)
            if k == 0:
                for step in range(7):
                    d, h = 0.06 + step * 0.12, 0.05 + step * 0.08
                    c = base + sign * normals[0] * d
                    stands.box((*c, GRASS_Z + h / 2), yaw, (12.0, 0.06, h / 2), tile=0.8)
                back = base + sign * normals[0] * 0.95
                stands.box((*back, GRASS_Z + 0.45), yaw, (12.0, 0.04, 0.45), tile=0.8)
                roof = base + sign * normals[0] * 0.5
                building.box((*roof, 0.93), yaw, (12.2, 0.55, 0.02), tile=1.0)
                for s in (-11.5, -4, 4, 11.5):
                    building.box((*(back + t0 * s), 0.45), yaw, (0.03, 0.03, 0.47), tile=1.0)
            else:
                c = base + sign * normals[0] * 0.6
                building.box((*c, GRASS_Z + 0.32), yaw, (14.0, 0.6, 0.33), tile=0.7)
            reserved.append(base + sign * normals[0] * 0.6)
        if stands:
            out["crowd"] = stands
        if building:
            out["building"] = building

        # Trees lining the circuit, merged into one low-poly mesh.
        trees = MeshBuilder()
        lo, hi = track.points.min(axis=0) - 30, track.points.max(axis=0) + 30
        placed = 0
        for xy in rng.uniform(lo, hi, size=(5000, 2)):
            if placed >= 320:
                break
            projection = track.project(xy)
            distance = abs(projection.lateral) - projection.half_width
            if not clearance + 1.2 < distance < 22 or any(np.linalg.norm(xy - r) < 16 for r in reserved):
                continue
            height = rng.uniform(0.9, 1.9)
            trees.cone((*xy, GRASS_Z + 0.12 * height), rng.uniform(0.3, 0.5) * height / 1.4,
                       height, sides=6, phase=rng.uniform(0, math.pi))
            placed += 1
        out["foliage"] = trees

        # Two rings of hills on the horizon; fog turns the far ring blue-gray.
        radius0 = float(np.linalg.norm(track.points.max(axis=0) - track.points.min(axis=0))) / 2
        for name, radius, base_h, amp in (("hills", radius0 + 45, 7.0, 6.0),
                                          ("mountains", radius0 + 120, 22.0, 16.0)):
            ring = MeshBuilder()
            segments = 120
            phases = rng.uniform(0, 2 * math.pi, 3)
            for k in range(segments):
                pts = []
                for theta in (2 * math.pi * k / segments, 2 * math.pi * (k + 1) / segments):
                    h = base_h + amp * (0.5 * math.sin(3 * theta + phases[0]) + 0.3 * math.sin(7 * theta + phases[1])
                                        + 0.2 * math.sin(13 * theta + phases[2]))
                    xy = center + radius * np.array([math.cos(theta), math.sin(theta)])
                    pts.append(((*xy, GRASS_Z), (*xy, max(h, 1.0))))
                (b0, top0), (b1, top1) = pts
                ring.quad(b0, top0, top1, b1)  # faces the circuit
            out[name] = ring
        return out

    def _gantry(self, world, track):
        p, t, q = track.points[0], track.tangents[0], track.normals[0]
        yaw = math.atan2(t[1], t[0])
        reach = track.rows[0, 3] + track.barrier_runoff + 0.1
        for sign in (1, -1):
            post = p + sign * q * reach + t * 0.4
            SubElement(world, "geom", type="box", pos=numbers([*post, 0.42]), size="0.035 0.035 0.43",
                       euler=f"0 0 {yaw}", material="steel")
        beam = p + t * 0.4
        SubElement(world, "geom", type="box", pos=numbers([*beam, 0.82]), size=f"0.04 {reach} 0.035",
                   euler=f"0 0 {yaw}", material="steel")
        SubElement(world, "geom", type="box", pos=numbers([*(beam - t * 0.045), 0.74]), size="0.012 0.32 0.05",
                   euler=f"0 0 {yaw}", material="carbon")
        for k in range(5):
            lamp = beam - t * 0.06 + q * (k - 2) * 0.12
            SubElement(world, "geom", type="sphere", pos=numbers([*lamp, 0.74]), size="0.025", material="lamp")

    def _car(self, world, assets, car):
        body = SubElement(world, "body", name="base_link", mocap="true", pos=f"0 0 {BODY_Z}")
        # Sun follows the car in position only, so its shadow box stays tight around the car.
        SubElement(body, "light", name="sun", mode="track", pos="-6 -4 15", dir="0.35 0.25 -0.9", directional="true",
                   castshadow=str(bool(self.quality["shadow"])).lower(),
                   diffuse="0.78 0.75 0.68", specular="0.35 0.35 0.33", ambient="0.08 0.08 0.1")

        def geom(material, **attributes):
            SubElement(body, "geom", material=material, **attributes)

        def mesh(name, builder, material):
            SubElement(assets, "mesh", name=name, **builder.attributes())
            geom(material, type="mesh", mesh=name)

        # Soft contact shadow without a shadow pass: one mesh of stacked translucent
        # ellipses (largest lowest) plus darker patches under each tire.
        ground = ROAD_Z - BODY_Z
        shadow = MeshBuilder()
        for k in range(6):
            f = 1 - k / 7
            shadow.disc((-0.01, 0, ground + 0.0006 + 0.00012 * k), 0.36 * f + 0.06, 0.2 * f + 0.03)
        for x, y in car.contacts:
            for k in range(2):
                shadow.disc((x, y, ground + 0.0014 + 0.00012 * k), 0.034 - 0.01 * k, 0.03 - 0.008 * k, sides=12)
        mesh("contact_shadow", shadow, "shadow")
        mesh("monocoque", loft([
            (0.292, 0.010, -0.040, -0.027, 2.2), (0.25, 0.017, -0.042, -0.019, 2.4),
            (0.17, 0.027, -0.043, -0.004, 2.6), (0.08, 0.037, -0.044, 0.014, 2.8),
            (0.00, 0.045, -0.045, 0.024, 3.0), (-0.09, 0.050, -0.045, 0.026, 3.0),
            (-0.125, 0.050, -0.045, 0.05, 3.0), (-0.165, 0.044, -0.044, 0.066, 2.8),
            (-0.215, 0.032, -0.040, 0.040, 2.6), (-0.262, 0.016, -0.034, 0.010, 2.4)]), "livery")
        for sign, name in ((1, "pod_left"), (-1, "pod_right")):
            mesh(name, loft([(0.055, 0.010, -0.040, -0.014, 2.5), (0.03, 0.028, -0.043, 0.004, 3),
                             (-0.07, 0.031, -0.043, 0.006, 3), (-0.165, 0.015, -0.040, -0.014, 2.5)],
                            y_offset=sign * 0.072), "podlivery")
            # Cyan stripe along the pod and a mirror on a stalk.
            geom("accent", type="box", pos=f"-0.04 {sign * 0.1035} -0.02", size="0.07 0.0015 0.006")
            geom("carbon", type="capsule", fromto=f"0.0 {sign * 0.045} 0.018 0.005 {sign * 0.075} 0.034", size="0.002")
            geom("paint", type="box", pos=f"0.005 {sign * 0.08} 0.036", size="0.005 0.012 0.007")
        geom("carbon", type="box", pos="-0.05 0 -0.047", size="0.16 0.108 0.0018")  # floor
        geom("carbon", type="ellipsoid", pos="-0.05 0 0.024", size="0.048 0.033 0.006")  # cockpit opening
        geom("white", type="ellipsoid", pos="-0.062 0 0.037", size="0.021 0.0175 0.019")  # helmet
        geom("visor", type="ellipsoid", pos="-0.05 0 0.043", size="0.012 0.019 0.0055")
        geom("accent", type="box", pos="0.27 0 -0.026", size="0.02 0.006 0.0015")  # nose stripe
        geom("paint", type="box", pos="-0.215 0 0.05", size="0.05 0.0018 0.022")  # engine-cover fin
        geom("accent", type="box", pos="-0.2 0 0.068", size="0.03 0.002 0.003")
        # ERS deployment light strip along the nose spine, in view of tcam and cockpit.
        SubElement(body, "geom", name="ers_led", type="capsule", material="indicator", rgba=_rgb(CYAN, 0.3),
                   fromto="0.1 0 0.0096 0.036 0 0.0191", size="0.0017")
        # Halo.
        geom("carbon", type="capsule", fromto="0.02 0 0.022 -0.002 0 0.05", size="0.0045")
        self._steering_wheel(body, assets, car)
        for sign in (1, -1):
            geom("carbon", type="capsule", fromto=f"-0.002 0 0.05 -0.04 {sign * 0.03} 0.05", size="0.0045")
            geom("carbon", type="capsule", fromto=f"-0.04 {sign * 0.03} 0.05 -0.09 {sign * 0.04} 0.03", size="0.0045")
        # Front wing: main plane, two flaps, endplates.
        geom("carbon", type="box", pos="0.258 0 -0.047", size="0.026 0.148 0.0022")
        geom("accent", type="box", pos="0.243 0 -0.041", size="0.013 0.14 0.0018", euler="0 -0.35 0")
        geom("paint", type="box", pos="0.233 0 -0.035", size="0.009 0.13 0.0016", euler="0 -0.6 0")
        geom("paint", type="box", pos="0.252 0.148 -0.04", size="0.03 0.002 0.012")
        geom("paint", type="box", pos="0.252 -0.148 -0.04", size="0.03 0.002 0.012")
        # Rear wing: pylon, main plane, cyan flap, endplates, rain light.
        geom("carbon", type="box", pos="-0.255 0 0.03", size="0.01 0.003 0.032")
        geom("paint", type="box", pos="-0.265 0 0.066", size="0.022 0.124 0.0025", euler="0 0.12 0")
        geom("accent", type="box", pos="-0.262 0 0.081", size="0.012 0.124 0.002", euler="0 0.45 0")
        geom("paint", type="box", pos="-0.262 0.126 0.062", size="0.03 0.002 0.03")
        geom("paint", type="box", pos="-0.262 -0.126 0.062", size="0.03 0.002 0.03")
        SubElement(body, "geom", name="brake_light", type="box", material="indicator", rgba=_rgb(BRAKE_RED, 0.22),
                   pos="-0.287 0 0.0", size="0.003 0.012 0.0045")
        # Suspension wishbones to each hub, merged into one carbon mesh.
        radius = car.tuning["tire_radius"]
        hub_z = ROAD_Z + radius - BODY_Z
        wishbones = MeshBuilder()
        for x, y in car.contacts:
            inner = 0.03 * math.copysign(1, y)
            for dx, z, dz in ((0.03, 0.002, 0.012), (-0.03, -0.03, -0.008)):
                wishbones.tube((x + dx, inner, z), (x, y * 0.82, hub_z + dz), 0.0022)
        mesh("wishbones", wishbones, "carbon")
        # Wheels: slick tire, cyan sidewall band and dark rim (axisymmetric, steer
        # only) around a bright five-spoke insert that spins with the wheel speed.
        axle = "0.7071068 0.7071068 0 0"
        SubElement(assets, "mesh", name="spokes", **spokes(radius * 0.72, 0.0261).attributes())
        for i, (x, y) in enumerate(car.contacts):
            wheel = SubElement(body, "body", pos=numbers([x, y, hub_z]))
            if i < 2:
                SubElement(wheel, "joint", name=f"steer{i}", type="hinge", axis="0 0 1")
            for r, half, mass, material in ((radius, 0.025, "0.1", "rubber"), (radius * 0.86, 0.0254, "0.001", "accent"),
                                            (radius * 0.72, 0.0258, "0.01", "rim")):
                SubElement(wheel, "geom", type="cylinder", size=f"{r} {half}", quat=axle, mass=mass, material=material)
            spin = SubElement(wheel, "body", name=f"hub{i}")
            SubElement(spin, "joint", name=f"spin{i}", type="hinge", axis="0 1 0")
            SubElement(spin, "geom", name=f"spokes{i}", type="mesh", mesh="spokes", material="spoke",
                       rgba=_rgb(SPOKE, 1.0), mass="0.005")
        position, target = CAMERAS[self.camera]
        SubElement(body, "camera", name="onboard", pos=numbers(position), xyaxes=_xyaxes(position, target),
                   fovy=str(math.degrees(2 * math.atan(self.height / self.width))))

    def _steering_wheel(self, body, assets, car):
        """Rectangular wheel on a raked column, visible low in tcam and cockpit views."""
        column = SubElement(body, "body", name="steering", pos="-0.004 0 0.031", euler="0 -0.5 0")
        SubElement(column, "joint", name="steering_wheel", type="hinge", axis="1 0 0")
        wheel = MeshBuilder()
        wheel.box((0, 0, 0), 0, (0.0025, 0.014, 0.007), tile=0.02, bottom=True)
        for sign in (1, -1):
            wheel.box((0, sign * 0.0175, -0.0005), 0, (0.0035, 0.0038, 0.0095), tile=0.02, bottom=True)
        SubElement(assets, "mesh", name="steering_wheel", **wheel.attributes())
        SubElement(column, "geom", type="mesh", mesh="steering_wheel", material="paint", mass="0.005")
        # Shift/rev light bar across the top of the wheel, facing the driver.
        SubElement(column, "geom", name="rev_led", type="box", material="indicator", rgba=_rgb(CYAN, 0.2),
                   pos="-0.0023 0 0.0048", size="0.0008 0.012 0.0016", mass="0.0001")

    # ---------------------------------------------------------------- runtime
    def _bind_runtime(self):
        model = self.model

        def ids(kind, *names):
            return [mujoco.mj_name2id(model, kind, name) for name in names]

        self.steer_ids = ids(mujoco.mjtObj.mjOBJ_JOINT, "steer0", "steer1")
        self.spin_adr = [model.jnt_qposadr[j] for j in ids(mujoco.mjtObj.mjOBJ_JOINT, *(f"spin{i}" for i in range(4)))]
        self.steering_adr = model.jnt_qposadr[ids(mujoco.mjtObj.mjOBJ_JOINT, "steering_wheel")[0]]
        self.spoke_geoms = ids(mujoco.mjtObj.mjOBJ_GEOM, *(f"spokes{i}" for i in range(4)))
        self.brake_geom, self.ers_geom, self.rev_geom = ids(mujoco.mjtObj.mjOBJ_GEOM, "brake_light", "ers_led", "rev_led")
        self.spin = np.zeros(4)
        self.last_time = self.car.time

    def update(self):
        car = self.car
        data, model = self.data, self.model
        data.mocap_pos[0] = [car.x, car.y, BODY_Z]
        data.mocap_quat[0] = [math.cos(car.yaw / 2), 0, 0, math.sin(car.yaw / 2)]
        angles = car.wheel_angles()
        for i, joint in enumerate(self.steer_ids):
            data.qpos[model.jnt_qposadr[joint]] = angles[i]
        data.qpos[self.steering_adr] = -car.steering / car.p["max_steering"] * STEERING_LOCK
        # Spokes: rotation is capped below the 60 fps wagon-wheel limit and the
        # insert fades toward the rim color as it would motion-blur.
        dt = min(max(car.time - self.last_time, 0.0), 0.1)
        self.last_time = car.time
        omega = np.asarray(car.wheels, dtype=float)
        self.spin = np.remainder(self.spin + np.clip(omega, -MAX_SPIN, MAX_SPIN) * dt, 2 * math.pi)
        data.qpos[self.spin_adr] = self.spin
        blur = np.clip((np.abs(omega) - 12) / 40, 0, 0.6)[:, None]
        model.geom_rgba[self.spoke_geoms, :3] = SPOKE_RGB * (1 - blur) + RIM_RGB * blur
        # Brake light: dim tail light, full on under deceleration.
        moving = abs(car.vx) > 0.05
        decel = -car.ax * math.copysign(1, car.vx) if moving else 0.0
        braking = float(np.clip(decel / 5.0, 0, 1)) if car.target.speed * car.vx <= car.vx * car.vx else 0.0
        if not moving and car.target.speed == 0:
            braking = max(braking, 0.35)
        model.geom_rgba[self.brake_geom] = _rgba(BRAKE_RED, 0.22 + 0.78 * braking)
        # ERS strip: brighter with drive deployment; amber when the pack runs low.
        deploy = float(np.clip(car.ax * math.copysign(1, car.vx) / (0.5 * car.p["max_acceleration"]), 0, 1)) \
            if moving and braking == 0 else 0.0
        charge = car.battery.remaining_wh / car.battery.capacity_wh
        model.geom_rgba[self.ers_geom] = _rgba(CYAN if charge > 0.2 else AMBER, 0.28 + 0.72 * deploy)
        # Rev lights scale with speed and flash near the top of the gearing.
        revs = float(np.clip(abs(car.vx) / car.p["speed_switch"], 0, 1))
        flash = revs > 0.95 and int(car.time * 12) % 2
        model.geom_rgba[self.rev_geom] = _rgba(BRAKE_RED if flash else CYAN, 0.2 + 0.8 * revs)
        mujoco.mj_forward(model, data)

    def render_size(self, _key):
        return self.width, self.height

    def open_renderers(self):
        if self.renderer_kind == "filament":
            from ..trossen.filament import FilamentFrames
            self.frames = FilamentFrames(self.model, self.feeds, {"onboard": (self.width, self.height)}, fog=True)
            return
        quality = self.quality
        if self.graphics == "auto":
            self.model.vis.quality.offsamples = 0
            self.renderer = self._renderer()
            if _software_rasterizer():
                quality = GRAPHICS["medium"]
            else:
                self.renderer.close()
                self.model.vis.quality.offsamples = quality["msaa"]
                self.renderer = self._renderer()
            LOG.info("Graphics auto-selected %s", "medium" if quality is GRAPHICS["medium"] else "high")
        else:
            self.renderer = self._renderer()
        flags = self.renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_FOG] = True
        flags[mujoco.mjtRndFlag.mjRND_HAZE] = True
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(quality["shadow"])

    def _renderer(self):
        return mujoco.Renderer(self.model, height=self.height, width=self.width, max_geom=2000)

    def prepare(self, _key):
        self.update()
        mujoco.mj_copyData(self.render_data, self.model, self.data)
        return self

    def render(self):
        self.renderer.update_scene(self.render_data, camera="onboard")
        return self.renderer.render()

    def prepare_batch(self, keys):
        self.update()
        return self.frames.prepare(self.model, self.data, keys)

    @property
    def batch_rendering(self):
        return self.renderer_kind == "filament"

    def camera_info(self):
        # Optical frame: right = MuJoCo camera x, down = -camera y, forward = -camera z.
        f = self.width / 2  # 90 degree horizontal field of view.
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.model.cam_quat[self.camera_id])
        optical = rotation.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, optical.ravel())
        if quat[0] < 0:
            quat = -quat
        w, x, y, z = (round(float(v), 9) for v in quat)
        return {"front": {"track_name": "onboard", "width": self.width, "height": self.height,
                "views": {"mono": {"frame_id": "front_optical", "image_region": {
                    "x": 0, "y": 0, "width": self.width, "height": self.height},
                    "intrinsics": {"fx": f, "fy": f, "cx": self.width / 2, "cy": self.height / 2,
                                   "distortion_model": "none", "distortion_coefficients": []},
                    "extrinsics": {"parent_frame_id": "base_link",
                                   "translation": [round(float(v), 9) for v in self.model.cam_pos[self.camera_id]],
                                   "rotation_xyzw": [x, y, z, w]}}}}}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.frames is not None:
            self.frames.close()
            self.frames = None


def _software_rasterizer():
    """True for CPU OpenGL (llvmpipe, softpipe, OSMesa), or when the renderer cannot be identified."""
    try:
        from OpenGL import GL
        name = GL.glGetString(GL.GL_RENDERER).decode().lower()
    except Exception:  # noqa: BLE001 - be conservative: assume the expensive path is unaffordable.
        return True
    return any(word in name for word in ("llvmpipe", "softpipe", "swrast", "software", "osmesa", "offscreen"))


def _smooth(values, sigma):
    """Circular Gaussian smoothing over track samples (sigma in samples)."""
    radius = max(1, int(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.r_[values[-radius:], values, values[:radius]], kernel, "valid")


def _curvature(track):
    """Signed curvature per centerline sample (1/m, positive = left), lightly smoothed."""
    heading = np.arctan2(track.tangents[:, 1], track.tangents[:, 0])
    bend = np.angle(np.exp(1j * (heading - np.roll(heading, 1)))) / track.lengths
    return np.convolve(np.r_[bend[-6:], bend, bend[:6]], np.ones(13) / 13, "valid")


def _racing_line(track, curvature):
    """Approximate outside-inside-outside line as a lateral offset per sample (+ = left).

    A narrow curvature kernel pulls toward the apex; a wide one pushes wide on
    entry and exit. Visual only: it steers the rubber texture, never the car.
    """
    step = track.lengths.mean()
    offset = 5.0 * _smooth(curvature, 1.8 / step) - 6.0 * _smooth(curvature, 6.0 / step)
    offset = _smooth(offset, 1.2 / step)
    limit = np.minimum(track.rows[:, 2], track.rows[:, 3]) - 0.06 * track.rows[:, 2:].sum(axis=1) - 0.22
    return limit * np.tanh(offset / limit)


def _corners(track, curvature, peak=0.12, edge=0.05):
    """(entry index, apex index, direction, straight meters before entry) per real corner."""
    n = len(curvature)
    out = []
    for run in _runs(np.abs(curvature) > peak):
        apex = int(run[np.argmax(np.abs(curvature[run]))])
        entry = apex
        while abs(curvature[(entry - 1) % n]) > edge and (apex - entry) < n // 4:
            entry -= 1
        straight, k = 0.0, entry - 1
        while abs(curvature[k % n]) <= edge and straight < 30:
            straight += track.lengths[k % n]
            k -= 1
        out.append((entry % n, apex, 1 if curvature[apex] > 0 else -1, straight))
    return out


def _runs(mask):
    """Index arrays of consecutive True entries in a circular boolean mask."""
    n = len(mask)
    if mask.all():
        return [np.arange(n)]
    start = int(np.argmin(mask))  # begin at a False so no run wraps the seam
    runs, current = [], []
    for i in (np.arange(n) + start) % n:
        if mask[i]:
            current.append(i)
        elif current:
            runs.append(np.array(current))
            current = []
    if current:
        runs.append(np.array(current))
    return runs


def _offset_line(line, window=80):
    """Remove the loops an offset polyline makes where a bend is tighter than the
    offset: every vertex inside a self-intersection collapses onto the crossing
    point, so the line keeps its vertex count but traces the outer envelope."""
    line = np.array(line, dtype=float)
    count = len(line)
    i = 0
    while i < count - 3:
        a, b = line[i], line[i + 1]
        crossing = None
        for j in range(min(count - 2, i + window), i + 1, -1):  # farthest crossing removes the whole loop
            c, d = line[j], line[j + 1]
            r, q = b - a, d - c
            denom = r[0] * q[1] - r[1] * q[0]
            if abs(denom) < 1e-12:
                continue
            w = c - a
            t = (w[0] * q[1] - w[1] * q[0]) / denom
            v = (w[0] * r[1] - w[1] * r[0]) / denom
            if 0 <= t <= 1 and 0 <= v <= 1:
                crossing = (j, a + t * r)
                break
        if crossing is None:
            i += 1
            continue
        j, point = crossing
        line[i + 1:j + 1] = point
        i = j
    return line
