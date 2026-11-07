"""Whack-a-mole game for the single WidowX AI arm (`trossen-wxai-whack`).

The table has three holes. Each hole pops up a violet, white or blue ball at
random. A light bar behind the table lights one color. The arm picks up a
wooden mallet (it snaps into the gripper when the gripper comes close to the
handle) and scores a point for each ball it hits that matches the lit color.
Hitting a ball of another color costs a point. The score never goes below 0.

The scene is built with `build_scene` before the model compiles. A
`WhackAMole` instance then runs the game on every physics step through
`update`. It only reads contacts and moves mocap bodies, materials and one
weld constraint, so it works with any controller that steps the model.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

import mujoco
import numpy as np


LOG = logging.getLogger("sims.trossen.whack")

# Ball and lamp colors, from the LiveKit code-sample palette: violet #C34BCA,
# white #FFFFFF and blue #658AF7.
COLORS = {"violet": (0.765, 0.294, 0.792), "white": (1.0, 1.0, 1.0), "blue": (0.396, 0.541, 0.969)}
COLOR_NAMES = tuple(COLORS)

# Table: a thick black slab with glowing edges in front of the arm. The balls hide inside it.
TABLE_TOP = 0.10
TABLE_CENTER = (0.40, 0.0)
TABLE_HALF = (0.26, 0.36)
HOLE_RADIUS = 0.055
HOLE_DISTANCE = 0.42      # m from the arm base
HOLE_ANGLES = (-40.0, 0.0, 40.0)  # degrees around the arm base
BALL_RADIUS = 0.04
BALL_DOWN_Z = TABLE_TOP - BALL_RADIUS - 0.004   # top just below the tabletop
BALL_UP_Z = TABLE_TOP + 0.6 * BALL_RADIUS      # sits up out of its hole

# Mallet, in its body frame: origin at the grip point, handle along +x, head
# axis along +z. Held, the body frame matches the gripper's ee_site frame.
HANDLE_RADIUS = 0.011
HANDLE_LENGTH = 0.20
GRIP_FROM_END = 0.03      # the grip point sits this far from the handle's end
HEAD_RADIUS = 0.038
HEAD_HALF_LENGTH = 0.06
WOOD_DENSITY = 650.0
HAMMER_START = (0.22, -0.16)  # grip point on the table, handle toward +y
GRAB_DISTANCE = 0.06      # m between ee_site and the grip point to snap
GRIPPER_HOLD_OPENING = 0.013  # m per finger; fingers rest just around the handle

# Light bar behind the table.
LIGHT_BAR_X = 0.74
LIGHT_Z = 0.40
LIGHT_SPACING = 0.13
LIGHT_RADIUS = 0.04

# Collision bits. Arm and floor geoms use 1. The mallet touches the table, the
# floor and the balls, but not the arm that holds it. The balls pass through
# the table, so they can sink into their holes.
HAMMER_BITS = (2, 2)        # contype, conaffinity
TABLE_BITS = (1, 3)
BALL_BITS = (4, 2)

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


def _rgba(color: str, scale: float = 1.0, alpha: float = 1.0) -> list[float]:
    return [c * scale for c in COLORS[color]] + [alpha]


def hole_positions() -> list[tuple[float, float]]:
    return [(HOLE_DISTANCE * math.cos(math.radians(a)), HOLE_DISTANCE * math.sin(math.radians(a)))
            for a in HOLE_ANGLES]


# -- Tron world -----------------------------------------------------------------

# LiveKit's signature cyan, #1FD5F9, and its muted cyan, #15889F.
NEON = (0.122, 0.835, 0.976)
NEON_DIM = (0.082, 0.533, 0.624)
GRID_SPACING = 0.25
GRID_HALF_EXTENT = 8.0
GRID_LINE_HALF_WIDTH = 0.0045


def add_tron_environment(spec: mujoco.MjSpec) -> None:
    """A black world with a glowing cyan floor grid and a neon skyline, fading into fog."""
    world = spec.worldbody
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

    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, 0],
                   size=[0, 0, 0.05], material="tron_floor")
    # The grid lines are thin glowing strips, so they stay crisp at any distance.
    count = int(GRID_HALF_EXTENT / GRID_SPACING)
    for index in range(-count, count + 1):
        offset = index * GRID_SPACING
        material = "neon_cyan" if index % 4 == 0 else "neon_cyan_dim"
        for size, pos in (([GRID_HALF_EXTENT, GRID_LINE_HALF_WIDTH, 0.0004], [0, offset, 0.0004]),
                          ([GRID_LINE_HALF_WIDTH, GRID_HALF_EXTENT, 0.0004], [offset, 0, 0.0004])):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=material,
                           contype=0, conaffinity=0, group=1)

    _skyline(world)

    spec.worldbody.add_light(name="scene_light", pos=[0.25, -0.35, 1.2], dir=[-0.2, 0.25, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.75, 0.8, 0.85], ambient=[0.12, 0.14, 0.16])
    # A cyan rim light from behind the table outlines the black arm against the dark.
    spec.worldbody.add_light(name="rim_light", pos=[1.2, 0.8, 0.9], dir=[-1.0, -0.6, -0.5],
                             type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=50, exponent=2,
                             diffuse=[n * 0.55 for n in NEON], castshadow=False)


def _plate(world, center, yaw, half, material, alpha=None):
    """A box at `center`, turned `yaw` radians about z, with local half sizes `half`."""
    quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), quat=quat, size=list(half),
                   material=material, contype=0, conaffinity=0, group=1)


def _skyline(world) -> None:
    """Neon-outlined towers in a ring around the arena, plus pillars, bars and a light beam."""
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
        _plate(world, (*base, height / 2), yaw, (0.25, width / 2, height / 2), "tron_building")
        face = base - radial * 0.26
        for side in (-1, 1):
            _plate(world, (*(face + tangent * side * width / 2), height / 2), yaw,
                   (edge, edge, height / 2), "neon_cyan")
        _plate(world, (*face, height), yaw, (edge, width / 2, edge), "neon_cyan")
        # Circuit-like ledges: partial horizontal lines and a stepped inner outline.
        for _ in range(rng.randint(1, 3)):
            z = rng.uniform(0.25, height - 0.15)
            span = rng.uniform(0.25, 0.5) * width
            shift = rng.uniform(-1, 1) * (width / 2 - span)
            _plate(world, (*(face + tangent * shift), z), yaw, (edge, span, edge * 0.8), "neon_cyan_dim")
        if rng.random() < 0.5:
            inset = width * 0.3
            step = height * rng.uniform(0.5, 0.8)
            _plate(world, (*(face + tangent * inset), step / 2), yaw, (edge, edge, step / 2), "neon_cyan_dim")
            _plate(world, (*(face + tangent * (inset + width / 4) / 1.0), step), yaw,
                   (edge, (width / 2 - inset) / 2, edge), "neon_cyan_dim")
        angle += (width + rng.uniform(0.3, 0.9)) / distance

    # Tall glowing pillars between the towers.
    for _ in range(12):
        yaw = rng.uniform(0, 2 * math.pi)
        distance = rng.uniform(3.4, 5.2)
        height = rng.uniform(1.5, 3.2)
        material = "neon_white" if rng.random() < 0.25 else "neon_cyan"
        _plate(world, (distance * math.cos(yaw), distance * math.sin(yaw), height / 2), yaw,
               (0.01, 0.01, height / 2), material)
    # Floating horizontal light bars.
    for _ in range(5):
        yaw = rng.uniform(0, 2 * math.pi)
        distance = rng.uniform(3.0, 4.2)
        _plate(world, (distance * math.cos(yaw), distance * math.sin(yaw), rng.uniform(1.6, 2.8)), yaw,
               (0.02, rng.uniform(0.8, 1.6), 0.03), "neon_cyan_dim")
    # A translucent light beam slanting down across the sky.
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.0, -0.5, 0.9]), "xyz")
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[2.8, 2.2, 2.4], quat=quat.tolist(),
                   size=[2.6, 0.08, 0.015], material="neon_beam", contype=0, conaffinity=0, group=1)


def _table_trim(world) -> None:
    """Glowing strips along the table's top edges and corners."""
    (cx, cy), (hx, hy) = TABLE_CENTER, TABLE_HALF
    w = 0.003
    for sign in (-1, 1):
        for z in (TABLE_TOP - w, w):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx + sign * hx, cy, z], size=[w, hy, w],
                           material="neon_cyan", contype=0, conaffinity=0)
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy + sign * hy, z], size=[hx, w, w],
                           material="neon_cyan", contype=0, conaffinity=0)
        for other in (-1, 1):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx + sign * hx, cy + other * hy, TABLE_TOP / 2],
                           size=[w, w, TABLE_TOP / 2], material="neon_cyan", contype=0, conaffinity=0)


def build_scene(spec: mujoco.MjSpec) -> None:
    """Adds the table, holes, balls, mallet and light bar. Call after add_tron_environment."""
    world = spec.worldbody
    # The floor catches a dropped mallet.
    spec.geom("floor").conaffinity = 3

    # Table slab and legs. Holes are dark discs on the top; the balls rise
    # through the slab, which hides them when they are down.
    world.add_geom(name="whack_table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[*TABLE_CENTER, TABLE_TOP / 2], size=[*TABLE_HALF, TABLE_TOP / 2],
                   material="tron_panel", contype=TABLE_BITS[0], conaffinity=TABLE_BITS[1])
    _table_trim(world)
    for index, (x, y) in enumerate(hole_positions()):
        world.add_geom(name=f"hole_{index}_rim", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=[x, y, TABLE_TOP + 0.0006], size=[HOLE_RADIUS + 0.006, 0.0006],
                       material="neon_cyan", contype=0, conaffinity=0)
        world.add_geom(name=f"hole_{index}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=[x, y, TABLE_TOP + 0.0013], size=[HOLE_RADIUS, 0.0007],
                       rgba=[0.03, 0.025, 0.02, 1], contype=0, conaffinity=0)
        ball = world.add_body(name=f"mole_{index}", pos=[x, y, BALL_DOWN_Z], mocap=True)
        ball.add_geom(name=f"mole_{index}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[BALL_RADIUS, 0, 0], rgba=_rgba("violet"),
                       contype=BALL_BITS[0], conaffinity=BALL_BITS[1])

    # Wooden mallet lying on the table, handle along +y, head on its side.
    spec.add_material(name="mallet_head_wood", rgba=[0.62, 0.4, 0.2, 1], specular=0.2, shininess=0.3)
    spec.add_material(name="mallet_handle_wood", rgba=[0.85, 0.68, 0.45, 1], specular=0.2, shininess=0.3)
    head_x = HANDLE_LENGTH - GRIP_FROM_END
    # Body x -> world +y, body z -> world -x: the head lies on its side.
    lying = np.zeros(4)
    mujoco.mju_mat2Quat(lying, np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float).ravel())
    hammer = world.add_body(name="hammer", pos=[*HAMMER_START, TABLE_TOP + HEAD_RADIUS + 0.002],
                            quat=lying.tolist())
    hammer.add_freejoint().name = "hammer_free"
    common = dict(contype=HAMMER_BITS[0], conaffinity=HAMMER_BITS[1], density=WOOD_DENSITY,
                  friction=[0.9, 0.005, 0.0001])
    hammer.add_geom(name="hammer_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=[-GRIP_FROM_END + HANDLE_RADIUS, 0, 0, head_x, 0, 0],
                    size=[HANDLE_RADIUS, 0, 0], material="mallet_handle_wood", **common)
    hammer.add_geom(name="hammer_head", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[head_x, 0, 0], size=[HEAD_RADIUS, HEAD_HALF_LENGTH, 0],
                    material="mallet_head_wood", **common)
    for sign in (-1, 1):  # dark bands near each striking face
        hammer.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        pos=[head_x, 0, sign * (HEAD_HALF_LENGTH - 0.012)],
                        size=[HEAD_RADIUS + 0.0008, 0.004, 0], rgba=[0.3, 0.18, 0.08, 1],
                        contype=0, conaffinity=0, mass=0)
    hammer.add_site(name="hammer_grip", pos=[0, 0, 0], size=[0.004, 0, 0])
    weld = spec.add_equality(name="hammer_grab", type=mujoco.mjtEq.mjEQ_WELD,
                             objtype=mujoco.mjtObj.mjOBJ_BODY, name1="link_6", name2="hammer")
    weld.active = False
    weld.solref = [0.004, 1.0]

    # Light bar on two posts behind the table. Each lamp has its own material
    # so that the game can light it.
    bar_half_y = LIGHT_SPACING + LIGHT_RADIUS + 0.04
    for sign in (-1, 1):
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[LIGHT_BAR_X + 0.03, sign * (bar_half_y - 0.02), LIGHT_Z / 2],
                       size=[0.015, 0.015, LIGHT_Z / 2], material="tron_panel", contype=0, conaffinity=0)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[LIGHT_BAR_X + 0.014, sign * (bar_half_y - 0.02), LIGHT_Z / 2],
                       size=[0.0015, 0.004, LIGHT_Z / 2], material="neon_cyan", contype=0, conaffinity=0)
    world.add_geom(name="light_bar", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[LIGHT_BAR_X + 0.03, 0, LIGHT_Z],
                   size=[0.03, bar_half_y, LIGHT_RADIUS + 0.025], material="tron_panel",
                   contype=0, conaffinity=0)
    bar_half_z = LIGHT_RADIUS + 0.025
    for z_sign in (-1, 1):  # glowing trim along the top and bottom of the bar
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[LIGHT_BAR_X - 0.001, 0, LIGHT_Z + z_sign * (bar_half_z - 0.006)],
                       size=[0.0015, bar_half_y - 0.006, 0.0025], material="neon_cyan", contype=0, conaffinity=0)
    for index, color in enumerate(COLOR_NAMES):
        spec.add_material(name=f"lamp_{color}", rgba=_rgba(color, 0.25), emission=0.0, specular=0.6)
        world.add_geom(name=f"lamp_{color}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       pos=[LIGHT_BAR_X, (1 - index) * LIGHT_SPACING, LIGHT_Z], size=[LIGHT_RADIUS, 0, 0],
                       material=f"lamp_{color}", contype=0, conaffinity=0)

    # Open the gripper just around the handle, and add the mallet to the keyframes.
    start = np.array([*HAMMER_START, TABLE_TOP + HEAD_RADIUS + 0.002, *lying])
    for key in spec.keys:
        qpos = np.asarray(key.qpos, dtype=float)
        qpos[6:8] = GRIPPER_HOLD_OPENING  # the two carriage joints
        key.qpos = np.concatenate((qpos, start))
        ctrl = np.asarray(key.ctrl, dtype=float)
        if ctrl.size >= 7:
            ctrl[6] = GRIPPER_HOLD_OPENING
            key.ctrl = ctrl


@dataclass
class Mole:
    body: int
    mocap: int
    geom: int
    color: str = "violet"
    phase: str = "down"   # down, rising, up, falling
    phase_start: float = 0.0
    phase_end: float = 0.0
    z_from: float = BALL_DOWN_Z
    hit: bool = False
    down_until: float = math.inf


class WhackAMole:
    """Runs the game on the compiled model."""

    def __init__(self, model: mujoco.MjModel, site_name: str = "ee_site", seed: int | None = None):
        self.model = model
        self.random = random.Random(seed)
        name = lambda kind, n: mujoco.mj_name2id(model, kind, n)
        self.ee_site = name(mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.grip_site = name(mujoco.mjtObj.mjOBJ_SITE, "hammer_grip")
        self.hammer_body = name(mujoco.mjtObj.mjOBJ_BODY, "hammer")
        self.hand_body = name(mujoco.mjtObj.mjOBJ_BODY, "link_6")
        self.head_geom = name(mujoco.mjtObj.mjOBJ_GEOM, "hammer_head")
        self.weld = name(mujoco.mjtObj.mjOBJ_EQUALITY, "hammer_grab")
        joint = name(mujoco.mjtObj.mjOBJ_JOINT, "hammer_free")
        self.hammer_qpos = int(model.jnt_qposadr[joint])
        self.hammer_dof = int(model.jnt_dofadr[joint])
        self.lamps = {c: name(mujoco.mjtObj.mjOBJ_MATERIAL, f"lamp_{c}") for c in COLOR_NAMES}
        self.moles = []
        for index in range(len(HOLE_ANGLES)):
            body = name(mujoco.mjtObj.mjOBJ_BODY, f"mole_{index}")
            self.moles.append(Mole(body, int(model.body_mocapid[body]),
                                   name(mujoco.mjtObj.mjOBJ_GEOM, f"mole_{index}_geom")))
        self.geom_to_mole = {m.geom: m for m in self.moles}
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self, data: mujoco.MjData | None = None) -> None:
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.held = False
        self.started = False
        self.target = self.random.choice(COLOR_NAMES)
        self.target_until = 0.0
        if data is not None:
            data.eq_active[self.weld] = 0
        self.model.body_gravcomp[self.hammer_body] = 0.0
        for mole in self.moles:
            mole.phase, mole.hit = "down", False
        self._light(self.target)

    def state(self) -> dict:
        """A snapshot for a future score display (data track, attributes or overlay)."""
        return {"score": self.score, "hits": self.hits, "misses": self.misses,
                "target": self.target, "holding_hammer": self.held,
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
        if not self.held:
            self._try_grab(data)
        if now >= self.target_until:
            self._new_target(now)
        self._check_hits(data, now)
        for mole in self.moles:
            self._advance(mole, data, now)

    def _try_grab(self, data: mujoco.MjData) -> None:
        if np.linalg.norm(data.site_xpos[self.ee_site] - data.site_xpos[self.grip_site]) > GRAB_DISTANCE:
            return
        # Snap the mallet's grip frame onto ee_site, then weld it to the wrist.
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[self.ee_site])
        data.qpos[self.hammer_qpos:self.hammer_qpos + 3] = data.site_xpos[self.ee_site]
        data.qpos[self.hammer_qpos + 3:self.hammer_qpos + 7] = site_quat
        data.qvel[self.hammer_dof:self.hammer_dof + 6] = 0.0
        mujoco.mj_kinematics(self.model, data)
        hand_pos = data.xpos[self.hand_body]
        hand_quat = data.xquat[self.hand_body]
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, hand_quat)
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, data.xpos[self.hammer_body] - hand_pos, inverse)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inverse, data.xquat[self.hammer_body])
        # Weld data: anchor (3), relative pose of body2 in body1 (7), torque scale (1).
        self.model.eq_data[self.weld, 0:3] = 0.0
        self.model.eq_data[self.weld, 3:6] = rel_pos
        self.model.eq_data[self.weld, 6:10] = rel_quat
        self.model.eq_data[self.weld, 10] = 1.0
        data.eq_active[self.weld] = 1
        # The arm's servos only compensate the arm's own weight.
        self.model.body_gravcomp[self.hammer_body] = 1.0
        self.held = True
        LOG.info("Mallet picked up")

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
        for contact in data.contact[:data.ncon]:
            pair = (contact.geom1, contact.geom2)
            if self.head_geom not in pair:
                continue
            mole = self.geom_to_mole.get(pair[0] if pair[1] == self.head_geom else pair[1])
            if mole is None or mole.hit or mole.phase not in ("rising", "up"):
                continue
            if data.mocap_pos[mole.mocap, 2] < (BALL_DOWN_Z + BALL_UP_Z) / 2:
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
                 fall_time: float = FALL_TIME, z_from: float = BALL_UP_Z) -> None:
        mole.phase = "down" if instant else "falling"
        mole.phase_start = now
        mole.z_from = z_from
        mole.phase_end = now if instant else now + fall_time
        mole.down_until = until

    def _advance(self, mole: Mole, data: mujoco.MjData, now: float) -> None:
        def ease(duration: float) -> float:
            return min(1.0, (now - mole.phase_start) / max(duration, 1e-6))
        z = BALL_DOWN_Z
        if mole.phase == "falling":
            fraction = ease(mole.phase_end - mole.phase_start)
            z = mole.z_from + (BALL_DOWN_Z - mole.z_from) * fraction
            if fraction >= 1.0:
                mole.phase = "down"
        if mole.phase == "down":
            z = BALL_DOWN_Z
            if now >= mole.down_until:
                self._pop(mole, now)
        if mole.phase == "rising":
            fraction = ease(RISE_TIME)
            z = BALL_DOWN_Z + (BALL_UP_Z - BALL_DOWN_Z) * fraction
            if fraction >= 1.0:
                mole.phase = "up"
                mole.phase_start = now
                mole.phase_end = now + self.random.uniform(*UP_TIME)
        if mole.phase == "up":
            z = BALL_UP_Z
            if now >= mole.phase_end:
                self._go_down(mole, now, now + self.random.uniform(*DOWN_TIME))
        data.mocap_pos[mole.mocap, 2] = z

    def _pop(self, mole: Mole, now: float) -> None:
        showing = {m.color for m in self.moles if m is not mole and m.phase in ("rising", "up")}
        if self.random.random() < TARGET_COLOR_CHANCE:
            color = self.target
        else:
            color = self.random.choice([c for c in COLOR_NAMES if c != self.target])
        if color in showing and len(showing) < len(COLOR_NAMES):
            # Prefer a color that is not already up, so the holes look varied.
            color = self.random.choice([c for c in COLOR_NAMES if c not in showing])
        mole.color = color
        mole.hit = False
        self.model.geom_rgba[mole.geom] = _rgba(color)
        mole.phase = "rising"
        mole.phase_start = now
        mole.down_until = math.inf
