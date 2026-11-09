"""Whack-a-mole games for Trossen arms (`trossen-wxai-whack`, `trossen-workbench-whack`).

A black table with holes stands in a Tron-style world. Each hole pops up a
violet, white or blue ball at random. A light bar behind the table lights one
color. The arms pick up wooden mallets (a mallet snaps into a gripper when the
gripper comes close to its handle) and score a point for each ball a mallet
hits that matches the lit color. Hitting a ball of another color costs a
point. The score never goes below 0. Hits by any mallet count.

A `WhackLayout` describes one table: its size, holes, mallets and the arms that
can hold them. `build_scene` adds it to a scene before the model compiles. A
`WhackAMole` instance then runs the game on every physics step through
`update`. It only reads contacts and moves mocap bodies, materials and weld
constraints, so it works with any controller that steps the model.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

import mujoco
import numpy as np


LOG = logging.getLogger("sims.trossen.whack")

# Ball and lamp colors, from the LiveKit code-sample palette: violet #C34BCA,
# white #FFFFFF and blue #658AF7.
COLORS = {"violet": (0.765, 0.294, 0.792), "white": (1.0, 1.0, 1.0), "blue": (0.396, 0.541, 0.969)}
COLOR_NAMES = tuple(COLORS)

HOLE_RADIUS = 0.055
BALL_RADIUS = 0.04
SLAB_THICKNESS = 0.10     # the balls hide inside the tabletop slab when they are down

# Mallet, in its body frame: origin at the grip point, handle along +x, head
# axis along +z. Held, the body frame matches the gripper's ee_site frame.
HANDLE_RADIUS = 0.011
HANDLE_LENGTH = 0.20
GRIP_FROM_END = 0.03      # the grip point sits this far from the handle's end
HEAD_RADIUS = 0.038
HEAD_HALF_LENGTH = 0.06
WOOD_DENSITY = 650.0
GRAB_DISTANCE = 0.06      # m between an ee_site and a grip point to snap

# Light bar.
LIGHT_SPACING = 0.13
LIGHT_RADIUS = 0.04

# Collision bits. Arm, workbench and floor geoms use 1, and the Workbench
# finger pads use 2. The mallets touch the table, the floor and the balls, but
# not the arms that hold them. The balls pass through the table, so they can
# sink into their holes.
HAMMER_BITS = (8, 8)        # contype, conaffinity
TABLE_BITS = (1, 9)
BALL_BITS = (16, 8)
FLOOR_CONAFFINITY = 9

# Timing, in simulated seconds. The trajectory profile caps joints at
# 0.5 rad/s, so balls stay up long enough to reach.
RISE_TIME = 0.18
FALL_TIME = 0.18
HIT_FALL_TIME = 0.08
HIT_FLASH_TIME = 0.25
UP_TIME = (2.5, 4.5)
DOWN_TIME = (0.8, 2.8)
TARGET_TIME = 12.0        # the lit color also changes after this long
TARGET_COLOR_CHANCE = 0.5  # chance that a popping ball takes the lit color
GAME_START_DELAY = 1.0


@dataclass(frozen=True)
class Gripper:
    """An arm that can hold a mallet."""
    hand_body: str              # the body that carries the ee_site
    ee_site: str
    actuator: str               # gripper position actuator
    joints: tuple[str, ...]     # the carriage joints it drives
    hold_opening: float         # m per finger, so the fingers rest around the handle


@dataclass(frozen=True)
class WhackLayout:
    table_center: tuple[float, float]
    table_half: tuple[float, float]
    table_top: float
    floor_z: float
    holes: tuple[tuple[float, float], ...]
    hammers: tuple[tuple[float, float, float], ...]   # grip x, y and handle yaw (rad)
    grippers: tuple[Gripper, ...]
    light_bar_x: float
    light_height: float = 0.30  # above the tabletop

    @property
    def ball_down_z(self) -> float:
        return self.table_top - BALL_RADIUS - 0.004   # top just below the tabletop

    @property
    def ball_up_z(self) -> float:
        return self.table_top + 0.6 * BALL_RADIUS     # sits up out of its hole


def _arc(distance: float, angles: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    return tuple((distance * math.cos(math.radians(a)), distance * math.sin(math.radians(a))) for a in angles)


# One WidowX AI arm at the origin on the floor, with the table in front of it.
SINGLE_WXAI = WhackLayout(
    table_center=(0.40, 0.0), table_half=(0.26, 0.36), table_top=0.10, floor_z=0.0,
    holes=_arc(0.42, (-40.0, 0.0, 40.0)),
    hammers=((0.22, -0.16, math.pi / 2),),
    grippers=(Gripper("link_6", "ee_site", "left_gripper",
                      ("left_carriage_joint", "right_carriage_joint"), 0.013),),
    light_bar_x=0.74)

# The Workbench, two WidowX Pro arms 0.55 m apart facing +x, stands on a table
# about twice the size of the single-arm one. Five holes in a W, a mallet in
# front of each arm with its handle pointing out to the side.
WORKBENCH = WhackLayout(
    table_center=(0.30, 0.0), table_half=(0.50, 0.72), table_top=0.0, floor_z=-0.75,
    holes=((0.47, -0.36), (0.47, 0.0), (0.47, 0.36), (0.64, -0.18), (0.64, 0.18)),
    hammers=((0.22, 0.52, math.pi / 2), (0.22, -0.52, -math.pi / 2)),
    grippers=tuple(Gripper(f"{arm}_link_6", f"{arm}_ee_site", f"{arm}_gripper",
                           (f"{arm}_left_carriage_joint", f"{arm}_right_carriage_joint"), 0.0115)
                   for arm in ("left", "right")),
    light_bar_x=0.90)

LAYOUTS = {"whack_wxai": SINGLE_WXAI, "workbench_whack": WORKBENCH}


def _rgba(color: str, scale: float = 1.0, alpha: float = 1.0) -> list[float]:
    return [c * scale for c in COLORS[color]] + [alpha]


def _lying_quat(yaw: float) -> np.ndarray:
    """A mallet lying on its side, handle along `yaw`, head axis horizontal."""
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, 0.0, -s], [s, 0.0, c], [0.0, -1.0, 0.0]])  # columns: body x, y, z
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return quat


# -- Tron world -----------------------------------------------------------------

# LiveKit's signature cyan, #1FD5F9, and its muted cyan, #15889F.
NEON = (0.122, 0.835, 0.976)
NEON_DIM = (0.082, 0.533, 0.624)
GRID_SPACING = 0.25
GRID_HALF_EXTENT = 8.0
GRID_LINE_HALF_WIDTH = 0.0045


def add_tron_environment(spec: mujoco.MjSpec, floor_z: float = 0.0, table_top: float = 0.1) -> None:
    """A black world with a glowing cyan floor grid and a neon skyline, fading into fog.

    Replaces any skybox, lights and floor the source scene brings.
    """
    world = spec.worldbody
    for texture in list(spec.textures):
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(texture)
    for light in list(spec.lights):
        spec.delete(light)
    # Fog distances are multiples of the model extent. Pin it, so the skyline
    # geoms do not stretch the fog, near plane or shadows.
    spec.stat.extent = 1.3
    spec.visual.map.fogstart = 1.0
    spec.visual.map.fogend = 4.6
    spec.visual.rgba.fog = [0.01, 0.03, 0.04, 1.0]
    spec.visual.rgba.haze = [n * 0.35 for n in NEON] + [1.0]
    spec.visual.headlight.ambient = [0.08, 0.1, 0.12]
    spec.visual.headlight.diffuse = [0.35, 0.38, 0.4]

    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.01, 0.05, 0.07], rgb2=[0.027, 0.027, 0.027], width=512, height=3072)
    spec.add_material(name="tron_floor", rgba=[0.005, 0.01, 0.015, 1], specular=0.6, shininess=0.8)
    spec.add_material(name="tron_panel", rgba=[0.02, 0.025, 0.03, 1], specular=0.7, shininess=0.9)
    spec.add_material(name="tron_building", rgba=[0.0, 0.012, 0.02, 1], specular=0.2)
    spec.add_material(name="neon_cyan", rgba=[*NEON, 1], emission=1.0)
    spec.add_material(name="neon_cyan_dim", rgba=[*NEON_DIM, 1], emission=0.8)
    spec.add_material(name="neon_white", rgba=[0.85, 0.97, 1.0, 1], emission=1.0)
    spec.add_material(name="neon_beam", rgba=[*NEON, 0.22], emission=1.0)

    floor = spec.geom("floor")
    if floor is None:
        floor = world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    floor.pos = [0, 0, floor_z]
    floor.material = "tron_floor"
    # The grid lines are thin glowing strips, so they stay crisp at any distance.
    count = int(GRID_HALF_EXTENT / GRID_SPACING)
    for index in range(-count, count + 1):
        offset = index * GRID_SPACING
        material = "neon_cyan" if index % 4 == 0 else "neon_cyan_dim"
        for size, pos in (([GRID_HALF_EXTENT, GRID_LINE_HALF_WIDTH, 0.0004], [0, offset, floor_z + 0.0004]),
                          ([GRID_LINE_HALF_WIDTH, GRID_HALF_EXTENT, 0.0004], [offset, 0, floor_z + 0.0004])):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=material,
                           contype=0, conaffinity=0, group=1)

    _skyline(world, floor_z)

    spec.worldbody.add_light(name="scene_light", pos=[0.25, -0.35, 1.2], dir=[-0.2, 0.25, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.75, 0.8, 0.85], ambient=[0.12, 0.14, 0.16])
    # A cyan rim light from behind the table outlines the black arm against the dark.
    spec.worldbody.add_light(name="rim_light", pos=[1.2, 0.8, table_top + 0.8], dir=[-1.0, -0.6, -0.5],
                             type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=50, exponent=2,
                             diffuse=[n * 0.55 for n in NEON], castshadow=False)


def _plate(world, center, yaw, half, material, alpha=None):
    """A box at `center`, turned `yaw` radians about z, with local half sizes `half`."""
    quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), quat=quat, size=list(half),
                   material=material, contype=0, conaffinity=0, group=1)


def _skyline(world, floor_z: float) -> None:
    """Neon-outlined towers in a ring around the arena, plus pillars, bars and a light beam."""
    def plate(center, *args):
        _plate(world, (center[0], center[1], center[2] + floor_z), *args)

    rng = random.Random(1982)
    edge = 0.012
    angle = 0.0
    while angle < 2 * math.pi:
        width = rng.uniform(0.5, 1.4)
        distance = rng.uniform(3.8, 5.5)
        height = rng.uniform(0.6, 2.2)
        yaw = angle
        radial = np.array([math.cos(yaw), math.sin(yaw)])
        tangent = np.array([-radial[1], radial[0]])
        base = radial * distance
        # Black tower body, then glowing outline on the face toward the arena.
        plate((*base, height / 2), yaw, (0.25, width / 2, height / 2), "tron_building")
        face = base - radial * 0.26
        for side in (-1, 1):
            plate((*(face + tangent * side * width / 2), height / 2), yaw,
                   (edge, edge, height / 2), "neon_cyan")
        plate((*face, height), yaw, (edge, width / 2, edge), "neon_cyan")
        # Circuit-like ledges: partial horizontal lines and a stepped inner outline.
        for _ in range(rng.randint(1, 3)):
            z = rng.uniform(0.25, height - 0.15)
            span = rng.uniform(0.25, 0.5) * width
            shift = rng.uniform(-1, 1) * (width / 2 - span)
            plate((*(face + tangent * shift), z), yaw, (edge, span, edge * 0.8), "neon_cyan_dim")
        if rng.random() < 0.5:
            inset = width * 0.3
            step = height * rng.uniform(0.5, 0.8)
            plate((*(face + tangent * inset), step / 2), yaw, (edge, edge, step / 2), "neon_cyan_dim")
            plate((*(face + tangent * (inset + width / 4) / 1.0), step), yaw,
                   (edge, (width / 2 - inset) / 2, edge), "neon_cyan_dim")
        angle += (width + rng.uniform(0.3, 0.9)) / distance

    # Tall glowing pillars between the towers.
    for _ in range(12):
        yaw = rng.uniform(0, 2 * math.pi)
        distance = rng.uniform(3.4, 5.2)
        height = rng.uniform(1.5, 3.2)
        material = "neon_white" if rng.random() < 0.25 else "neon_cyan"
        plate((distance * math.cos(yaw), distance * math.sin(yaw), height / 2), yaw,
               (0.01, 0.01, height / 2), material)
    # Floating horizontal light bars.
    for _ in range(5):
        yaw = rng.uniform(0, 2 * math.pi)
        distance = rng.uniform(3.0, 4.2)
        plate((distance * math.cos(yaw), distance * math.sin(yaw), rng.uniform(1.6, 2.8)), yaw,
               (0.02, rng.uniform(0.8, 1.6), 0.03), "neon_cyan_dim")
    # A translucent light beam slanting down across the sky.
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.0, -0.5, 0.9]), "xyz")
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[2.8, 2.2, floor_z + 2.4], quat=quat.tolist(),
                   size=[2.6, 0.08, 0.015], material="neon_beam", contype=0, conaffinity=0, group=1)


def _table(world, layout: WhackLayout) -> None:
    """A black slab with glowing edges, on legs when it stands above the floor."""
    (cx, cy), (hx, hy) = layout.table_center, layout.table_half
    top = layout.table_top
    thickness = min(SLAB_THICKNESS, top - layout.floor_z)
    bottom = top - thickness
    world.add_geom(name="whack_table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[cx, cy, top - thickness / 2], size=[hx, hy, thickness / 2],
                   material="tron_panel", contype=TABLE_BITS[0], conaffinity=TABLE_BITS[1])
    leg_half = (bottom - layout.floor_z) / 2
    if leg_half > 0:
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = cx + sx * (hx - 0.05), cy + sy * (hy - 0.05)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, layout.floor_z + leg_half],
                               size=[0.03, 0.03, leg_half], material="tron_panel", contype=0, conaffinity=0)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x - sx * 0.031, y, layout.floor_z + leg_half],
                               size=[0.0015, 0.004, leg_half], material="neon_cyan_dim", contype=0, conaffinity=0)
    w = 0.003
    for sign in (-1, 1):
        for z in (top - w, bottom + w):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx + sign * hx, cy, z], size=[w, hy, w],
                           material="neon_cyan", contype=0, conaffinity=0)
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy + sign * hy, z], size=[hx, w, w],
                           material="neon_cyan", contype=0, conaffinity=0)
        for other in (-1, 1):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx + sign * hx, cy + other * hy, top - thickness / 2],
                           size=[w, w, thickness / 2], material="neon_cyan", contype=0, conaffinity=0)


def _hammer(spec: mujoco.MjSpec, index: int, grip_x: float, grip_y: float, yaw: float,
            layout: WhackLayout) -> np.ndarray:
    """Adds mallet `index` lying on the table. Returns its free-joint pose for keyframes."""
    head_x = HANDLE_LENGTH - GRIP_FROM_END
    quat = _lying_quat(yaw)
    pos = [grip_x, grip_y, layout.table_top + HEAD_RADIUS + 0.002]
    hammer = spec.worldbody.add_body(name=f"hammer_{index}", pos=pos, quat=quat.tolist())
    hammer.add_freejoint().name = f"hammer_{index}_free"
    common = dict(contype=HAMMER_BITS[0], conaffinity=HAMMER_BITS[1], density=WOOD_DENSITY,
                  friction=[0.9, 0.005, 0.0001])
    hammer.add_geom(name=f"hammer_{index}_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=[-GRIP_FROM_END + HANDLE_RADIUS, 0, 0, head_x, 0, 0],
                    size=[HANDLE_RADIUS, 0, 0], material="mallet_handle_wood", **common)
    hammer.add_geom(name=f"hammer_{index}_head", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[head_x, 0, 0], size=[HEAD_RADIUS, HEAD_HALF_LENGTH, 0],
                    material="mallet_head_wood", **common)
    for sign in (-1, 1):  # dark bands near each striking face
        hammer.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        pos=[head_x, 0, sign * (HEAD_HALF_LENGTH - 0.012)],
                        size=[HEAD_RADIUS + 0.0008, 0.004, 0], rgba=[0.3, 0.18, 0.08, 1],
                        contype=0, conaffinity=0, mass=0)
    hammer.add_site(name=f"hammer_{index}_grip", pos=[0, 0, 0], size=[0.004, 0, 0])
    # One weld per gripper; the game turns on the one that picks the mallet up.
    for arm, gripper in enumerate(layout.grippers):
        weld = spec.add_equality(name=f"hammer_{index}_grab_{arm}", type=mujoco.mjtEq.mjEQ_WELD,
                                 objtype=mujoco.mjtObj.mjOBJ_BODY, name1=gripper.hand_body,
                                 name2=f"hammer_{index}")
        weld.active = False
        weld.solref = [0.004, 1.0]
    return np.array([*pos, *quat])


def _light_bar(spec: mujoco.MjSpec, layout: WhackLayout) -> None:
    """A light bar on two posts behind the table. Each lamp has its own material so that the game can light it."""
    world = spec.worldbody
    x, z = layout.light_bar_x, layout.table_top + layout.light_height
    base = layout.floor_z
    post_half = (z - base) / 2
    bar_half_y = LIGHT_SPACING + LIGHT_RADIUS + 0.04
    for sign in (-1, 1):
        y = layout.table_center[1] + sign * (bar_half_y - 0.02)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x + 0.03, y, base + post_half],
                       size=[0.015, 0.015, post_half], material="tron_panel", contype=0, conaffinity=0)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x + 0.014, y, base + post_half],
                       size=[0.0015, 0.004, post_half], material="neon_cyan", contype=0, conaffinity=0)
    world.add_geom(name="light_bar", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x + 0.03, layout.table_center[1], z],
                   size=[0.03, bar_half_y, LIGHT_RADIUS + 0.025], material="tron_panel",
                   contype=0, conaffinity=0)
    bar_half_z = LIGHT_RADIUS + 0.025
    for z_sign in (-1, 1):  # glowing trim along the top and bottom of the bar
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[x - 0.001, layout.table_center[1], z + z_sign * (bar_half_z - 0.006)],
                       size=[0.0015, bar_half_y - 0.006, 0.0025], material="neon_cyan", contype=0, conaffinity=0)
    for index, color in enumerate(COLOR_NAMES):
        spec.add_material(name=f"lamp_{color}", rgba=_rgba(color, 0.25), emission=0.0, specular=0.6)
        world.add_geom(name=f"lamp_{color}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       pos=[x, layout.table_center[1] + (1 - index) * LIGHT_SPACING, z],
                       size=[LIGHT_RADIUS, 0, 0], material=f"lamp_{color}", contype=0, conaffinity=0)


def build_scene(spec: mujoco.MjSpec, layout: WhackLayout) -> None:
    """Adds the table, holes, balls, mallets and light bar. Call after add_tron_environment."""
    world = spec.worldbody
    spec.geom("floor").conaffinity = FLOOR_CONAFFINITY  # catches a dropped mallet
    _table(world, layout)
    # Holes are dark discs on the top; the balls rise through the slab, which
    # hides them when they are down.
    for index, (x, y) in enumerate(layout.holes):
        world.add_geom(name=f"hole_{index}_rim", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=[x, y, layout.table_top + 0.0006], size=[HOLE_RADIUS + 0.006, 0.0006],
                       material="neon_cyan", contype=0, conaffinity=0)
        world.add_geom(name=f"hole_{index}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=[x, y, layout.table_top + 0.0013], size=[HOLE_RADIUS, 0.0007],
                       rgba=[0.03, 0.025, 0.02, 1], contype=0, conaffinity=0)
        ball = world.add_body(name=f"mole_{index}", pos=[x, y, layout.ball_down_z], mocap=True)
        ball.add_geom(name=f"mole_{index}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[BALL_RADIUS, 0, 0], rgba=_rgba("violet"),
                      contype=BALL_BITS[0], conaffinity=BALL_BITS[1])

    spec.add_material(name="mallet_head_wood", rgba=[0.62, 0.4, 0.2, 1], specular=0.2, shininess=0.3)
    spec.add_material(name="mallet_handle_wood", rgba=[0.85, 0.68, 0.45, 1], specular=0.2, shininess=0.3)
    poses = [_hammer(spec, index, *hammer, layout) for index, hammer in enumerate(layout.hammers)]
    _light_bar(spec, layout)
    # Add the mallets to the keyframes. WhackAMole opens the grippers in them.
    for key in spec.keys:
        key.qpos = np.concatenate((np.asarray(key.qpos, dtype=float), *poses))


@dataclass
class Mole:
    body: int
    mocap: int
    geom: int
    color: str = "violet"
    phase: str = "down"   # down, rising, up, falling
    phase_start: float = 0.0
    phase_end: float = 0.0
    z_from: float = 0.0
    hit: bool = False
    down_until: float = math.inf


@dataclass
class Hammer:
    body: int
    qpos: int
    dof: int
    grip_site: int
    head_geom: int
    welds: list[int]             # one per gripper
    held_by: int | None = None   # gripper index


@dataclass
class Hand:
    name: str
    ee_site: int
    body: int
    holding: int | None = None   # hammer index


class WhackAMole:
    """Runs the game on the compiled model. Construct it before resetting to a keyframe."""

    def __init__(self, model: mujoco.MjModel, layout: WhackLayout, seed: int | None = None):
        self.model = model
        self.layout = layout
        self.random = random.Random(seed)
        name = lambda kind, n: mujoco.mj_name2id(model, kind, n)
        self.hands = [Hand(g.ee_site.removesuffix("ee_site").rstrip("_") or "arm",
                           name(mujoco.mjtObj.mjOBJ_SITE, g.ee_site), name(mujoco.mjtObj.mjOBJ_BODY, g.hand_body))
                      for g in layout.grippers]
        self.hammers = []
        for index in range(len(layout.hammers)):
            joint = name(mujoco.mjtObj.mjOBJ_JOINT, f"hammer_{index}_free")
            self.hammers.append(Hammer(
                name(mujoco.mjtObj.mjOBJ_BODY, f"hammer_{index}"), int(model.jnt_qposadr[joint]),
                int(model.jnt_dofadr[joint]), name(mujoco.mjtObj.mjOBJ_SITE, f"hammer_{index}_grip"),
                name(mujoco.mjtObj.mjOBJ_GEOM, f"hammer_{index}_head"),
                [name(mujoco.mjtObj.mjOBJ_EQUALITY, f"hammer_{index}_grab_{arm}")
                 for arm in range(len(layout.grippers))]))
        self.head_geoms = {h.head_geom for h in self.hammers}
        self.lamps = {c: name(mujoco.mjtObj.mjOBJ_MATERIAL, f"lamp_{c}") for c in COLOR_NAMES}
        self.moles = []
        for index in range(len(layout.holes)):
            body = name(mujoco.mjtObj.mjOBJ_BODY, f"mole_{index}")
            self.moles.append(Mole(body, int(model.body_mocapid[body]),
                                   name(mujoco.mjtObj.mjOBJ_GEOM, f"mole_{index}_geom")))
        self.geom_to_mole = {m.geom: m for m in self.moles}
        self._open_grippers()
        self.reset()

    def _open_grippers(self) -> None:
        """Opens each gripper in the keyframes to rest just around a mallet handle."""
        for gripper in self.layout.grippers:
            actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper.actuator)
            self.model.key_ctrl[:, actuator] = gripper.hold_opening
            for joint in gripper.joints:
                address = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
                self.model.key_qpos[:, address] = gripper.hold_opening

    # -- state ---------------------------------------------------------------

    @property
    def held(self) -> bool:
        return any(h.held_by is not None for h in self.hammers)

    def reset(self, data: mujoco.MjData | None = None) -> None:
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.started = False
        self.target = self.random.choice(COLOR_NAMES)
        self.target_until = 0.0
        for hammer in self.hammers:
            hammer.held_by = None
            self.model.body_gravcomp[hammer.body] = 0.0
            if data is not None:
                data.eq_active[hammer.welds] = 0
        for hand in self.hands:
            hand.holding = None
        for mole in self.moles:
            mole.phase, mole.hit = "down", False
        self._light(self.target)

    def state(self) -> dict:
        """A snapshot for a future score display (data track, attributes or overlay)."""
        return {"score": self.score, "hits": self.hits, "misses": self.misses,
                "target": self.target, "holding_hammer": self.held,
                "hammers": [None if h.held_by is None else self.hands[h.held_by].name for h in self.hammers],
                "balls": [m.color if m.phase in ("rising", "up") and not m.hit else None
                          for m in self.moles]}

    # -- per step ------------------------------------------------------------

    def update(self, data: mujoco.MjData) -> None:
        now = float(data.time)
        if not self.started:
            if now < GAME_START_DELAY:
                return
            self.started = True
            self.target_until = now + TARGET_TIME
            for mole in self.moles:
                self._go_down(mole, now, now + self.random.uniform(*DOWN_TIME), instant=True)
        self._try_grab(data)
        if now >= self.target_until:
            self._new_target(now)
        self._check_hits(data, now)
        for mole in self.moles:
            self._advance(mole, data, now)

    def _try_grab(self, data: mujoco.MjData) -> None:
        for arm, hand in enumerate(self.hands):
            if hand.holding is not None:
                continue
            for index, hammer in enumerate(self.hammers):
                if hammer.held_by is None and np.linalg.norm(
                        data.site_xpos[hand.ee_site] - data.site_xpos[hammer.grip_site]) <= GRAB_DISTANCE:
                    self._grab(data, arm, hand, index, hammer)
                    break

    def _grab(self, data: mujoco.MjData, arm: int, hand: Hand, index: int, hammer: Hammer) -> None:
        # Snap the mallet's grip frame onto the ee_site, then weld it to the wrist.
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[hand.ee_site])
        data.qpos[hammer.qpos:hammer.qpos + 3] = data.site_xpos[hand.ee_site]
        data.qpos[hammer.qpos + 3:hammer.qpos + 7] = site_quat
        data.qvel[hammer.dof:hammer.dof + 6] = 0.0
        mujoco.mj_kinematics(self.model, data)
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, data.xquat[hand.body])
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, data.xpos[hammer.body] - data.xpos[hand.body], inverse)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inverse, data.xquat[hammer.body])
        weld = hammer.welds[arm]
        # Weld data: anchor (3), relative pose of body2 in body1 (7), torque scale (1).
        self.model.eq_data[weld, 0:3] = 0.0
        self.model.eq_data[weld, 3:6] = rel_pos
        self.model.eq_data[weld, 6:10] = rel_quat
        self.model.eq_data[weld, 10] = 1.0
        data.eq_active[weld] = 1
        # The arm's servos only compensate the arm's own weight.
        self.model.body_gravcomp[hammer.body] = 1.0
        hammer.held_by, hand.holding = arm, index
        LOG.info("Mallet %d picked up by %s", index, hand.name)

    def _new_target(self, now: float) -> None:
        self.target = self.random.choice([c for c in COLOR_NAMES if c != self.target])
        self.target_until = now + TARGET_TIME
        self._light(self.target)

    def _light(self, color: str) -> None:
        for name, material in self.lamps.items():
            on = name == color
            self.model.mat_rgba[material] = _rgba(name, 1.0 if on else 0.4)
            self.model.mat_emission[material] = 0.9 if on else 0.0

    def _check_hits(self, data: mujoco.MjData, now: float) -> None:
        middle = (self.layout.ball_down_z + self.layout.ball_up_z) / 2
        for contact in data.contact[:data.ncon]:
            if contact.geom1 in self.head_geoms:
                other = contact.geom2
            elif contact.geom2 in self.head_geoms:
                other = contact.geom1
            else:
                continue
            mole = self.geom_to_mole.get(other)
            if mole is None or mole.hit or mole.phase not in ("rising", "up"):
                continue
            if data.mocap_pos[mole.mocap, 2] < middle:
                continue  # barely out of its hole
            self._whack(mole, data, now)

    def _whack(self, mole: Mole, data: mujoco.MjData, now: float) -> None:
        mole.hit = True
        if mole.color == self.target:
            self.score += 1
            self.hits += 1
            LOG.info("Hit %s: +1, score %d", mole.color, self.score)
            self._new_target(now)
        else:
            self.score = max(0, self.score - 1)
            self.misses += 1
            LOG.info("Hit %s while %s was lit: -1, score %d", mole.color, self.target, self.score)
        self.model.geom_rgba[mole.geom] = [*NEON, 1.0]
        self._go_down(mole, now, now + HIT_FLASH_TIME + self.random.uniform(*DOWN_TIME),
                      fall_time=HIT_FALL_TIME, z_from=float(data.mocap_pos[mole.mocap, 2]))

    def _go_down(self, mole: Mole, now: float, until: float, instant: bool = False,
                 fall_time: float = FALL_TIME, z_from: float | None = None) -> None:
        mole.phase = "down" if instant else "falling"
        mole.phase_start = now
        mole.z_from = self.layout.ball_up_z if z_from is None else z_from
        mole.phase_end = now if instant else now + fall_time
        mole.down_until = until

    def _advance(self, mole: Mole, data: mujoco.MjData, now: float) -> None:
        down, up = self.layout.ball_down_z, self.layout.ball_up_z

        def ease(duration: float) -> float:
            return min(1.0, (now - mole.phase_start) / max(duration, 1e-6))
        z = down
        if mole.phase == "falling":
            fraction = ease(mole.phase_end - mole.phase_start)
            z = mole.z_from + (down - mole.z_from) * fraction
            if fraction >= 1.0:
                mole.phase = "down"
        if mole.phase == "down":
            z = down
            if now >= mole.down_until:
                self._pop(mole, now)
        if mole.phase == "rising":
            fraction = ease(RISE_TIME)
            z = down + (up - down) * fraction
            if fraction >= 1.0:
                mole.phase = "up"
                mole.phase_start = now
                mole.phase_end = now + self.random.uniform(*UP_TIME)
        if mole.phase == "up":
            z = up
            if now >= mole.phase_end:
                self._go_down(mole, now, now + self.random.uniform(*DOWN_TIME))
        data.mocap_pos[mole.mocap, 2] = z

    def _pop(self, mole: Mole, now: float) -> None:
        showing = [m.color for m in self.moles if m is not mole and m.phase in ("rising", "up")]
        if self.random.random() < TARGET_COLOR_CHANCE:
            color = self.target
        else:
            color = self.random.choice([c for c in COLOR_NAMES if c != self.target])
        if color in showing and len(set(showing)) < len(COLOR_NAMES):
            # Prefer a color that is not already up, so the holes look varied.
            color = self.random.choice([c for c in COLOR_NAMES if c not in showing])
        mole.color = color
        mole.hit = False
        self.model.geom_rgba[mole.geom] = _rgba(color)
        mole.phase = "rising"
        mole.phase_start = now
        mole.down_until = math.inf
