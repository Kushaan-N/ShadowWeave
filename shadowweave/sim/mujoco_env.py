"""MuJoCo 3.x single-room environment with debris spawning.

Room: 6m × 6m, first-person camera at cfg.sim.camera_height, 90° FOV.
Difficulty tiers: static | moving | debris.
Exports: RGB, metric depth, ground-truth EGOCENTRIC BEV occupancy, collision flag.

Key changes from the first version:
  * Depth is normalised by a FIXED range (cfg.shadow.max_range_m), not the per-frame
    max. Per-frame normalisation made the depth scale drift between frames, so
    uncertainty values were not comparable across time.
  * Ground truth is an egocentric BEV grid that moves with the agent, replacing the
    world-frame grid that was identical no matter where the agent stood.
  * Occupancy respects the z axis, so falling debris actually enters the grid as it
    descends. The old version projected every geom regardless of height, so debris
    was "present" from spawn and the target never changed.
  * Collision is a real geometric test against obstacle footprints. The old flag was
    ``occupancy.max() > 0.5``, and since walls were always marked 1.0 it was True on
    every single step.
  * The renderer is closed before being replaced on reset (it used to leak one
    renderer and its GL context per episode).

MuJoCo 3.x may not be installed — import is guarded with a graceful message.
Install: pip install mujoco
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig

from ..utils import setup_headless_rendering

setup_headless_rendering()

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False


_ROOM_XML_TEMPLATE = textwrap.dedent("""
<mujoco model="shadowweave_room">
  <option timestep="{timestep}" gravity="0 0 -9.81"/>
  <visual>
    <map fogstart="3" fogend="10" znear="0.02" zfar="{zfar}"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="{half_x} {half_y} 0.1" material="floor_mat"/>
    <geom name="wall_n" type="box" pos="0 {half_y} 1.5"  size="{half_x} 0.1 1.5" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall_s" type="box" pos="0 -{half_y} 1.5" size="{half_x} 0.1 1.5" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall_e" type="box" pos="{half_x} 0 1.5"  size="0.1 {half_y} 1.5" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall_w" type="box" pos="-{half_x} 0 1.5" size="0.1 {half_y} 1.5" rgba="0.5 0.5 0.5 1"/>
    <!-- agent body with freejoint — position driven by RL actions (kinematic only) -->
    <body name="agent_body" pos="0 0 {camera_height}">
      <freejoint name="agent_joint"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <!-- xyaxes: x=right(+X), y=up(+Z) — camera looks in +Y when yaw=0 -->
      <camera name="fp_cam" fovy="{fov}" xyaxes="1 0 0 0 0 1"/>
    </body>
    {objects_xml}
  </worldbody>
</mujoco>
""")

# Bodies whose names start with these prefixes are navigation obstacles.
_OBSTACLE_PREFIXES = ("obs_", "mover_", "debris_", "wall_")

_N_MOVERS = 3
_N_DEBRIS = 6
# Upper bound on scripted-body indices to scan for; counts scale up with room area
# (max ~1.8x baseline), so this cap is comfortably above any generated count.
_MAX_SCRIPTED = 16
# Returned in place of a real frame when RGB rendering is switched off.
_EMPTY_RGB = np.zeros((1, 1, 3), dtype=np.uint8)
_EMPTY_RGB.flags.writeable = False  # shared sentinel; never mutate in place

# Height band an obstacle must overlap to block the agent.
_BODY_BAND_LO = 0.05
_BODY_BAND_MARGIN = 0.4
# Debris waits here until its scheduled release. Well above the body band, so a held
# piece contributes nothing to occupancy until it actually starts falling.
_DEBRIS_HOLD_Z = 6.0


# Obstacle counts scale with room area (relative to the 6x6=36 m^2 baseline) so the target's
# positive fraction — which pos_weight is tuned to — stays roughly constant as the room grows.
def _area_scale(half_x: float, half_y: float) -> float:
    return (4.0 * half_x * half_y) / 36.0


# Legacy fixed-room spawn margins, per obstacle type — the bounds the published Aug 18
# numbers were generated with (±2.2 static / ±2.0 movers / ±2.2 blockers / ±2.5 debris
# in the 6x6 room). The fixed-room path passes margin=None and each builder falls back
# to these, so randomize_room=false reproduces the legacy scenes EXACTLY; a uniform
# margin here silently changed the scenes and broke reproduction of the Aug 18 rollout
# metrics (collision 0.37 -> 0.76).
_LEGACY_MARGINS = {"static": 0.8, "mover": 1.0, "blocker": 0.8, "debris": 0.5}


def _xy(rng, half_x, half_y, margin):
    """A random position at least `margin` inside each wall (per-axis). One size-2 draw, to
    match the legacy rng.uniform(-b, b, size=2) consumption so the fixed-room path stays
    stream-identical."""
    hx, hy = max(half_x - margin, 0.1), max(half_y - margin, 0.1)
    xy = rng.uniform([-hx, -hy], [hx, hy])
    return float(xy[0]), float(xy[1])


def _static_objects_xml(rng, half_x, half_y, margin=None) -> str:
    xml = ""
    sc = _area_scale(half_x, half_y)
    lo = max(2, round(3 * sc)); hi = max(lo + 1, round(6 * sc))
    n = rng.integers(lo, hi)
    for i in range(int(n)):
        x, y = _xy(rng, half_x, half_y, _LEGACY_MARGINS["static"] if margin is None else margin)
        s = rng.uniform(0.2, 0.45)
        hz = rng.uniform(0.3, 1.1)
        xml += (
            f'<geom name="obs_{i}" type="box" pos="{x:.2f} {y:.2f} {hz:.2f}" '
            f'size="{s:.2f} {s:.2f} {hz:.2f}" rgba="0.8 0.4 0.1 1"/>\n    '
        )
    return xml


def _moving_objects_xml(rng, half_x, half_y, margin=None) -> str:
    xml = ""
    sc = _area_scale(half_x, half_y)
    for i in range(max(1, round(_N_MOVERS * sc))):
        x, y = _xy(rng, half_x, half_y, _LEGACY_MARGINS["mover"] if margin is None else margin)
        xml += f"""
    <body name="mover_{i}" pos="{x:.2f} {y:.2f} 0.5">
      <freejoint name="mover_joint_{i}"/>
      <geom type="cylinder" size="0.22 0.5" rgba="0.2 0.6 0.9 1" mass="2"/>
    </body>"""
    # A couple of static blockers so "moving" is not trivially empty.
    for i in range(max(1, round(2 * sc))):
        x, y = _xy(rng, half_x, half_y, _LEGACY_MARGINS["blocker"] if margin is None else margin)
        xml += (
            f'\n    <geom name="obs_{i}" type="box" pos="{x:.2f} {y:.2f} 0.5" '
            f'size="0.3 0.3 0.5" rgba="0.8 0.4 0.1 1"/>'
        )
    return xml


def _debris_objects_xml(rng, half_x, half_y, margin=None) -> str:
    xml = ""
    sc = _area_scale(half_x, half_y)
    for i in range(max(1, round(_N_DEBRIS * sc))):
        x, y = _xy(rng, half_x, half_y, _LEGACY_MARGINS["debris"] if margin is None else margin)
        s = rng.uniform(0.12, 0.25)
        xml += f"""
    <body name="debris_{i}" pos="{x:.2f} {y:.2f} {_DEBRIS_HOLD_Z:.2f}">
      <freejoint name="debris_joint_{i}"/>
      <geom type="box" size="{s:.2f} {s:.2f} {s:.2f}" rgba="0.9 0.2 0.2 1" mass="1"/>
    </body>"""
    for i in range(max(1, round(2 * sc))):
        x, y = _xy(rng, half_x, half_y, _LEGACY_MARGINS["blocker"] if margin is None else margin)
        xml += (
            f'\n    <geom name="obs_{i}" type="box" pos="{x:.2f} {y:.2f} 0.4" '
            f'size="0.3 0.3 0.4" rgba="0.8 0.4 0.1 1"/>'
        )
    return xml


class ShadowWeaveEnv:
    """MuJoCo single-room environment.

    Primary method: ``step(action) -> obs`` where obs is a dict with:
        rgb:           (H, W, 3) uint8
        depth:         (H, W) float32 — normalised by cfg.shadow.max_range_m
        depth_metric:  (H, W) float32 — metres
        bev_occupancy: (S, S) float32 — ground-truth EGOCENTRIC occupancy
        collision:     bool — real geometric agent/obstacle overlap
        agent_pos:     (2,) float32
        agent_yaw:     float
        min_clearance: float — metres to the nearest obstacle surface
    """

    def __init__(self, cfg: DictConfig, render_rgb: bool = True) -> None:
        self.cfg = cfg
        self.render_rgb = render_rgb
        self._model: Any = None
        self._data: Any = None
        self._renderer: Any = None
        self._agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self._agent_yaw = 0.0
        # Per-episode room half-extents (X=width, Y=depth); reset() redraws them. Default to
        # the legacy fixed square so any pre-reset read is valid.
        self._half_x = self._half_y = float(cfg.sim.room_size) / 2.0
        self._qpos_addr = 0
        self._rng = np.random.default_rng(0)
        self._obstacle_geoms: list[int] = []
        self._frame = 0
        self._movers: list[dict] = []
        self._debris: list[dict] = []
        self._ego_grid_cache: Optional[tuple[np.ndarray, np.ndarray]] = None

        if not _MUJOCO_AVAILABLE:
            print("[ShadowWeaveEnv] mujoco not installed — running in dummy mode.")
            print("  Install with: pip install mujoco")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        if not _MUJOCO_AVAILABLE:
            return self._dummy_obs()

        # A local Generator keeps episodes reproducible without stomping on the
        # global numpy seed the way np.random.seed() did.
        self._rng = np.random.default_rng(seed)

        # Draw the room geometry FIRST (before obstacles) so obstacle placement can respect
        # it and the RNG stream is deterministic given the seed. Disjoint train/val seeds =>
        # no exact (half_x, half_y) recurs, so wall recall must be observation-driven.
        if self.cfg.sim.get("randomize_room", False):
            lo, hi = float(self.cfg.sim.room_half_min), float(self.cfg.sim.room_half_max)
            self._half_x = float(self._rng.uniform(lo, hi))
            self._half_y = float(self._rng.uniform(lo, hi)) if self.cfg.sim.get("room_aspect", True) else self._half_x
        else:
            self._half_x = self._half_y = float(self.cfg.sim.room_size) / 2.0
        # None => each builder uses its per-type _LEGACY_MARGINS value, reproducing the
        # legacy fixed-room scenes (and the published Aug 18 numbers) exactly.
        margin = (float(self.cfg.sim.get("obstacle_wall_margin", 0.6))
                  if self.cfg.sim.get("randomize_room", False) else None)

        difficulty = self.cfg.sim.difficulty
        builders = {
            "static": _static_objects_xml,
            "moving": _moving_objects_xml,
            "debris": _debris_objects_xml,
        }
        if difficulty not in builders:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        obj_xml = builders[difficulty](self._rng, self._half_x, self._half_y, margin)

        xml = _ROOM_XML_TEMPLATE.format(
            half_x=f"{self._half_x:.3f}", half_y=f"{self._half_y:.3f}",
            # One env step must advance exactly 1/fps of simulated time, or every
            # "k second" horizon label (and the falling-object lead time) is off by
            # the mismatch — 0.01 hardcoded here meant 30 steps = 0.9s, not 1s.
            timestep=1.0 / (self.cfg.sim.fps * self.cfg.sim.substeps),
            camera_height=self.cfg.sim.camera_height,
            fov=self.cfg.sim.camera_fov,
            zfar=max(self.cfg.shadow.max_range_m * 2.0, 20.0),
            objects_xml=obj_xml,
        )

        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)

        # Replacing the renderer without closing it leaked a GL context per episode.
        if self._renderer is not None:
            self._renderer.close()
        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w
        self._renderer = mujoco.Renderer(self._model, height=h, width=w)

        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "agent_joint")
        self._qpos_addr = int(self._model.jnt_qposadr[joint_id])
        self._cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "fp_cam")

        self._obstacle_geoms = self._find_obstacle_geoms()
        self._frame = 0
        self._movers = []
        self._debris = []
        self._init_movers()
        self._init_debris()
        self._update_scripted_bodies()

        self._agent_pos, self._agent_yaw = self._sample_free_spawn()
        self._apply_agent_pose()
        return self._render_obs()

    def step(self, action: Optional[np.ndarray] = None, n_substeps: Optional[int] = None) -> dict[str, Any]:
        if not _MUJOCO_AVAILABLE:
            return self._dummy_obs()
        if action is not None:
            self._move_agent(action)
        substeps = n_substeps if n_substeps is not None else self.cfg.sim.substeps
        for _ in range(substeps):
            mujoco.mj_step(self._model, self._data)
        self._frame += 1
        self._update_scripted_bodies(substeps)
        # Physics moved bodies; re-pin the kinematically driven agent.
        self._apply_agent_pose()
        return self._render_obs()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _find_obstacle_geoms(self) -> list[int]:
        out = []
        for gid in range(self._model.ngeom):
            if self._model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            body_id = self._model.geom_bodyid[gid]
            body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name.startswith(_OBSTACLE_PREFIXES) or body_name.startswith(_OBSTACLE_PREFIXES):
                out.append(gid)
        return out

    def _init_movers(self) -> None:
        """Set up kinematically driven movers that keep moving for the whole episode.

        The first version emitted an empty <actuator/> block and never touched qvel,
        so "moving" was identical to "static". Simply kicking qvel is not enough
        either: floor friction stops a cylinder in well under a second, so by the
        shortest 1s prediction horizon the scene is static again and the horizon axis
        carries no signal. These are driven directly and reflect off the walls.
        """
        self._movers = []
        # Iterate a generous cap (counts scale up with room area); missing joints are skipped.
        for i in range(_MAX_SCRIPTED):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"mover_joint_{i}")
            if jid < 0:
                continue
            adr = int(self._model.jnt_qposadr[jid])
            speed = self._rng.uniform(0.35, 0.7)
            angle = self._rng.uniform(0, 2 * math.pi)
            self._movers.append({
                "qadr": adr,
                "dofadr": int(self._model.jnt_dofadr[jid]),
                "pos": self._data.qpos[adr : adr + 2].copy(),
                "vel": np.array([speed * math.cos(angle), speed * math.sin(angle)]),
                "z": float(self._data.qpos[adr + 2]),
            })

    def _init_debris(self) -> None:
        """Stagger debris releases across the episode.

        A piece dropped from 3m lands in ~0.8s, so if everything is released at t=0
        the scene is settled before even the 1s horizon. Staggering means that at any
        moment some debris is aloft and due to land 1–10s from now, which is the
        signal the falling-object lead-time metric depends on.
        """
        self._debris = []
        horizon_s = max(self.cfg.world_model.prediction_horizons)
        # Stagger across the frames actually rolled and indexed into targets:
        # data.steps_per_episode observations, each looking up to max-horizon ahead.
        # Keying this to sim.max_episode_steps (500) instead gave span 800 while the
        # roll only reaches ~data.steps_per_episode + max_h*fps (~600), so debris
        # released past that never landed in any target — ~22% of the debris signal.
        span = int(self.cfg.data.steps_per_episode + horizon_s * self.cfg.sim.fps)
        for i in range(_MAX_SCRIPTED):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"debris_joint_{i}")
            if jid < 0:
                continue
            adr = int(self._model.jnt_qposadr[jid])
            self._debris.append({
                "qadr": adr,
                "dofadr": int(self._model.jnt_dofadr[jid]),
                "qpos": self._data.qpos[adr : adr + 7].copy(),
                "release": int(self._rng.uniform(0.05, 0.95) * span),
                "drop_z": float(self._rng.uniform(2.5, 4.0)),
            })

    def _update_scripted_bodies(self, n_substeps: Optional[int] = None) -> None:
        """Drive movers and hold un-released debris. Called after every mj_step."""
        # Advance by the substeps actually simulated this frame, or a caller
        # passing n_substeps desynchronises the scripted bodies from the physics.
        substeps = n_substeps if n_substeps is not None else self.cfg.sim.substeps
        dt = substeps * float(self._model.opt.timestep)
        halves = (self._half_x - 0.35, self._half_y - 0.35)

        for m in self._movers:
            m["pos"] = m["pos"] + m["vel"] * dt
            for ax in (0, 1):
                h = halves[ax]
                if m["pos"][ax] < -h:
                    m["pos"][ax] = -h
                    m["vel"][ax] = abs(m["vel"][ax])
                elif m["pos"][ax] > h:
                    m["pos"][ax] = h
                    m["vel"][ax] = -abs(m["vel"][ax])
            a = m["qadr"]
            self._data.qpos[a : a + 2] = m["pos"]
            self._data.qpos[a + 2] = m["z"]
            self._data.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]
            self._data.qvel[m["dofadr"] : m["dofadr"] + 6] = 0.0

        for d in self._debris:
            if self._frame < d["release"]:
                a = d["qadr"]
                self._data.qpos[a : a + 7] = d["qpos"]
                self._data.qvel[d["dofadr"] : d["dofadr"] + 6] = 0.0
            elif self._frame == d["release"]:
                a = d["qadr"]
                self._data.qpos[a + 2] = d["drop_z"]
                self._data.qvel[d["dofadr"] : d["dofadr"] + 6] = 0.0

    def _sample_free_spawn(self) -> tuple[np.ndarray, float]:
        hx, hy = self._half_x - 0.5, self._half_y - 0.5
        for _ in range(64):
            pos = np.array([self._rng.uniform(-hx, hx), self._rng.uniform(-hy, hy)], dtype=np.float32)
            if self._clearance(pos) > self.cfg.sim.agent_radius + 0.15:
                return pos, float(self._rng.uniform(0, 2 * math.pi))
        # Legacy fixed-room fallback was (0, -2); keep it so the fixed path reproduces the
        # published runs exactly. (0, 0) is only safe/centred for randomized rooms.
        if self.cfg.sim.get("randomize_room", False):
            return np.array([0.0, 0.0], dtype=np.float32), 0.0
        return np.array([0.0, -2.0], dtype=np.float32), 0.0

    # ------------------------------------------------------------------
    # Agent motion
    # ------------------------------------------------------------------

    def _move_agent(self, action: np.ndarray) -> None:
        """Apply (forward, strafe, turn) action in [-1,1] to agent pose."""
        a = np.asarray(action, dtype=np.float32).ravel()
        if a.size != 3:
            raise ValueError(f"action must be (forward, strafe, turn), got {a.size} values")
        if not np.isfinite(a).all():
            # np.clip propagates NaN, and a NaN pose then reads as "no obstacles
            # anywhere, clearance inf, no collision" — a diverged policy would
            # silently poison every rollout it touches. Fail loudly instead.
            raise ValueError(f"non-finite action {a} — policy has diverged")
        a = np.clip(a, -1.0, 1.0)
        speed = self.cfg.sim.agent_max_speed
        turn = self.cfg.sim.agent_max_turn_rate
        vfwd, vstrafe, d_yaw = float(a[0]) * speed, float(a[1]) * speed, float(a[2]) * turn

        self._agent_yaw = (self._agent_yaw + d_yaw) % (2 * math.pi)
        yaw = self._agent_yaw
        dx = vfwd * (-math.sin(yaw)) + vstrafe * math.cos(yaw)
        dy = vfwd * math.cos(yaw) + vstrafe * math.sin(yaw)

        hx, hy = self._half_x - 0.3, self._half_y - 0.3
        self._agent_pos[0] = float(np.clip(self._agent_pos[0] + dx, -hx, hx))
        self._agent_pos[1] = float(np.clip(self._agent_pos[1] + dy, -hy, hy))
        self._apply_agent_pose()

    def _apply_agent_pose(self) -> None:
        a = self._qpos_addr
        self._data.qpos[a] = self._agent_pos[0]
        self._data.qpos[a + 1] = self._agent_pos[1]
        self._data.qpos[a + 2] = self.cfg.sim.camera_height
        self._data.qpos[a + 3] = math.cos(self._agent_yaw / 2)
        self._data.qpos[a + 4] = 0.0
        self._data.qpos[a + 5] = 0.0
        self._data.qpos[a + 6] = math.sin(self._agent_yaw / 2)
        # The agent is kinematic, so zero its velocity to stop the integrator from
        # fighting the pose we just wrote.
        self._data.qvel[self._model.jnt_dofadr[
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "agent_joint")
        ] : ][:6] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def forward_vector(self) -> np.ndarray:
        return np.array([-math.sin(self._agent_yaw), math.cos(self._agent_yaw)], dtype=np.float32)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _planar_to_radial(self, shape: tuple[int, int]) -> np.ndarray:
        """Per-pixel factor converting MuJoCo's planar (optical-axis) GL depth to the
        radial (Euclidean) distance the BEV projector and _raycast_depth both assume:
        sqrt(1 + x^2 + y^2), with x, y the pixel directions in tan(FOV/2) space. Cached
        per resolution."""
        cache = getattr(self, "_p2r_cache", None)
        if cache is None or cache.shape != tuple(shape):
            h, w = shape
            fov = math.radians(self.cfg.sim.camera_fov)
            ndc_x = np.linspace(-1, 1, w, dtype=np.float32) * math.tan(fov / 2)
            ndc_y = np.linspace(1, -1, h, dtype=np.float32) * math.tan(fov / 2)
            gx, gy = np.meshgrid(ndc_x, ndc_y)
            cache = np.sqrt(1.0 + gx ** 2 + gy ** 2).astype(np.float32)
            self._p2r_cache = cache
        return cache

    def _render_obs(self) -> dict[str, Any]:
        # RGB costs a second full mjr_readPixels pass and is unused by data
        # generation and RL — only the dashboard needs it. Profiling put the two
        # passes at ~29% of rollout wall-clock.
        if self.render_rgb:
            self._renderer.update_scene(self._data, camera=self._cam_id)
            rgb = self._renderer.render().copy()
        else:
            rgb = _EMPTY_RGB

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self._data, camera=self._cam_id)
        depth_metric = self._renderer.render().astype(np.float32).copy()
        self._renderer.disable_depth_rendering()

        max_range = float(self.cfg.shadow.max_range_m)
        if not np.isfinite(depth_metric).all() or depth_metric.max() < 1e-3:
            # Depth rendering failed (missing ARB_clip_control on some macOS stacks).
            # Fall back to a raycast against the true geometry so the pipeline still
            # sees a geometrically correct depth map instead of a degenerate one.
            depth_metric = self._raycast_depth()   # already radial
        else:
            # MuJoCo's GL depth is PLANAR (distance along the optical axis), but the
            # BEV projector and the raycast fallback both treat depth as RADIAL
            # (Euclidean camera-to-surface distance). Uncorrected, observed occupancy
            # and visibility are wrong by up to 1/cos toward the frame edges (~28% at
            # the edge, ~41% in the corners for a 90 degree FOV) — exactly where the
            # shadows the system reasons about live. Convert planar -> radial.
            depth_metric = depth_metric * self._planar_to_radial(depth_metric.shape)

        # FIXED normalisation. Dividing by the per-frame max made the scale drift
        # frame to frame, so shadow values were not comparable across time.
        depth_norm = np.clip(depth_metric / max_range, 0.0, 1.0).astype(np.float32)

        collision, clearance = self._collision_state()
        return {
            "rgb": rgb,
            "depth": depth_norm,
            "depth_metric": depth_metric,
            "bev_occupancy": self.bev_occupancy(),
            "collision": collision,
            "min_clearance": clearance,
            "agent_pos": self._agent_pos.copy(),
            "agent_yaw": float(self._agent_yaw),
        }

    def _raycast_depth(self) -> np.ndarray:
        """Metric depth via mj_ray — fallback when GL depth rendering is unavailable."""
        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w
        max_range = float(self.cfg.shadow.max_range_m)
        fov = math.radians(self.cfg.sim.camera_fov)

        # Pinhole directions in camera frame, then rotated into world.
        ndc_x = np.linspace(-1, 1, w, dtype=np.float32) * math.tan(fov / 2)
        ndc_y = np.linspace(1, -1, h, dtype=np.float32) * math.tan(fov / 2)
        gx, gy = np.meshgrid(ndc_x, ndc_y)

        yaw = self._agent_yaw
        fwd = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        right = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])

        dirs = (fwd[None, None, :] + gx[..., None] * right + gy[..., None] * up)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        origin = np.array(
            [self._agent_pos[0], self._agent_pos[1], self.cfg.sim.camera_height], dtype=np.float64
        )
        out = np.full((h, w), max_range, dtype=np.float32)
        geomid = np.zeros(1, dtype=np.int32)
        for r in range(h):
            for c in range(w):
                d = mujoco.mj_ray(
                    self._model, self._data, origin,
                    dirs[r, c].astype(np.float64), None, 1, -1, geomid,
                )
                if d >= 0:
                    out[r, c] = min(float(d), max_range)
        return out

    # ------------------------------------------------------------------
    # Ground-truth egocentric BEV
    # ------------------------------------------------------------------

    def _ego_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Ego-frame (z_forward, x_right) cell centres, cached — pose-independent."""
        if self._ego_grid_cache is None:
            S = self.cfg.bev.size
            rng_m = self.cfg.bev.range_m
            cell = rng_m / S
            ez = np.linspace(rng_m - cell / 2, cell / 2, S, dtype=np.float32)
            ex = np.linspace(-rng_m / 2 + cell / 2, rng_m / 2 - cell / 2, S, dtype=np.float32)
            self._ego_grid_cache = np.meshgrid(ez, ex, indexing="ij")
        return self._ego_grid_cache

    def _bev_grid_world(self, pos: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """World XY of every BEV cell centre for a given pose. (S, S) each."""
        zz, xx = self._ego_grid()
        fx, fy = -math.sin(yaw), math.cos(yaw)
        rx, ry = math.cos(yaw), math.sin(yaw)
        return pos[0] + zz * fx + xx * rx, pos[1] + zz * fy + xx * ry

    def describe_scene(self) -> list[dict[str, Any]]:
        """Full geometry of the current scene, including the room shell.

        ``geom_snapshot`` deliberately carries only obstacles, packed as arrays, and
        sits on the hot path that renders prediction targets. Scene export needs
        everything — floor, walls, names, colours — so it gets its own accessor
        rather than widening that one.
        """
        if not _MUJOCO_AVAILABLE:
            return []
        out: list[dict[str, Any]] = []
        for gid in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if not name:
                body = mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, self._model.geom_bodyid[gid]
                )
                name = f"{body}_geom" if body else f"geom_{gid}"
            out.append({
                "name": name,
                "type": int(self._model.geom_type[gid]),
                "pos": np.array(self._data.geom_xpos[gid], dtype=np.float64),
                "mat": np.array(self._data.geom_xmat[gid], dtype=np.float64).reshape(3, 3),
                "size": np.array(self._model.geom_size[gid], dtype=np.float64),
                "rgba": np.array(self._model.geom_rgba[gid], dtype=np.float64),
                "dynamic": bool(self._model.body_dofnum[self._model.geom_bodyid[gid]] > 0),
            })
        return out

    def geom_snapshot(self) -> dict[str, np.ndarray]:
        """Freeze obstacle geometry for this instant.

        Lets a future frame's geometry be rendered into a *past* frame's egocentric
        grid, which is what the world model target needs: "what will be here, in the
        frame I am standing in now".
        """
        gids = self._obstacle_geoms
        snap = {
            "xpos": np.array([self._data.geom_xpos[g] for g in gids], dtype=np.float64),
            "xmat": np.array([self._data.geom_xmat[g] for g in gids], dtype=np.float64),
            "size": np.array([self._model.geom_size[g] for g in gids], dtype=np.float64),
            "type": np.array([self._model.geom_type[g] for g in gids], dtype=np.int32),
        }
        # The height gate depends only on this snapshot, so compute it here rather
        # than memoising per call site. An earlier version cached it in a dict keyed
        # by id(snap); CPython reuses ids after garbage collection, so a transient
        # snapshot could receive a previous one's gate and silently produce the wrong
        # ground truth (reproduced: three different snapshots shared one id).
        snap["active"] = self._height_gate(snap)
        return snap

    def _height_gate(self, snap: dict[str, np.ndarray]) -> np.ndarray:
        """Indices of geoms overlapping the body height band.

        This is what lets falling debris enter the target as it descends, and it
        usually rules out most geoms before any rasterisation happens.
        """
        n = len(snap["type"])
        if n == 0:
            return np.empty(0, dtype=np.int64)
        z_lo, z_hi = _BODY_BAND_LO, float(self.cfg.sim.camera_height) + _BODY_BAND_MARGIN
        half_z = np.array(
            [self._geom_half_height(int(snap["type"][i]), snap["size"][i]) for i in range(n)],
            dtype=np.float64,
        )
        z = snap["xpos"][:, 2]
        return np.nonzero((z + half_z >= z_lo) & (z - half_z <= z_hi))[0]

    def bev_occupancy(
        self,
        pos: Optional[np.ndarray] = None,
        yaw: Optional[float] = None,
        snapshot: Optional[dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Ground-truth egocentric BEV occupancy, (S, S) float32 in {0, 1}.

        This is the world model's supervision target. It includes obstacles the agent
        cannot currently see — predicting those is exactly the shadow-zone claim.

        Defaults to the current pose and current geometry. Pass ``snapshot`` from a
        later frame to build a prediction target.
        """
        if not _MUJOCO_AVAILABLE:
            S = self.cfg.bev.size
            return (np.random.rand(S, S) > 0.9).astype(np.float32)

        pos = self._agent_pos if pos is None else pos
        yaw = self._agent_yaw if yaw is None else yaw
        snap = self.geom_snapshot() if snapshot is None else snapshot

        wx, wy = self._bev_grid_world(pos, yaw)
        occ = np.zeros(wx.shape, dtype=np.float32)

        active = snap.get("active")
        if active is None:  # snapshot built before the gate existed
            active = self._height_gate(snap)
        for i in active:
            occ = np.maximum(
                occ,
                self._footprint(int(snap["type"][i]), snap["xpos"][i], snap["xmat"][i],
                                snap["size"][i], wx, wy),
            )
        return occ

    def _geom_half_height(self, gtype: int, size: np.ndarray) -> float:
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            return float(size[2])
        if gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            return float(size[1])
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0])
        if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return float(size[2])
        return float(np.max(size))

    def _footprint(
        self,
        gtype: int,
        pos: np.ndarray,
        xmat: np.ndarray,
        size: np.ndarray,
        wx: np.ndarray,
        wy: np.ndarray,
    ) -> np.ndarray:
        """Binary XY footprint of one geom sampled at world points (wx, wy)."""
        mat = np.asarray(xmat).reshape(3, 3)

        dx, dy = wx - pos[0], wy - pos[1]
        # Rotate into the geom's local frame (yaw component of its orientation).
        lx = mat[0, 0] * dx + mat[1, 0] * dy
        ly = mat[0, 1] * dx + mat[1, 1] * dy

        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            return ((np.abs(lx) <= size[0]) & (np.abs(ly) <= size[1])).astype(np.float32)
        if gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE,
                     mujoco.mjtGeom.mjGEOM_CAPSULE):
            return ((lx ** 2 + ly ** 2) <= size[0] ** 2).astype(np.float32)
        if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return (((lx / size[0]) ** 2 + (ly / size[1]) ** 2) <= 1.0).astype(np.float32)
        r = float(np.max(size[:2]))
        return ((lx ** 2 + ly ** 2) <= r ** 2).astype(np.float32)

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def _clearance(self, pos: np.ndarray) -> float:
        """Distance in metres from ``pos`` to the nearest obstacle surface."""
        if not self._obstacle_geoms:
            return float("inf")
        z_lo = _BODY_BAND_LO
        z_hi = float(self.cfg.sim.camera_height) + _BODY_BAND_MARGIN
        best = float("inf")
        for gid in self._obstacle_geoms:
            gp = self._data.geom_xpos[gid]
            size = self._model.geom_size[gid]
            gtype = self._model.geom_type[gid]
            half_z = self._geom_half_height(gtype, size)
            if gp[2] + half_z < z_lo or gp[2] - half_z > z_hi:
                continue
            mat = self._data.geom_xmat[gid].reshape(3, 3)
            dx, dy = pos[0] - gp[0], pos[1] - gp[1]
            lx = mat[0, 0] * dx + mat[1, 0] * dy
            ly = mat[0, 1] * dx + mat[1, 1] * dy
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                ox = max(abs(lx) - size[0], 0.0)
                oy = max(abs(ly) - size[1], 0.0)
                d = math.hypot(ox, oy)
            else:
                r = float(size[0]) if gtype != mujoco.mjtGeom.mjGEOM_BOX else float(np.max(size[:2]))
                d = max(math.hypot(lx, ly) - r, 0.0)
            best = min(best, d)
        return best

    def _collision_state(self) -> tuple[bool, float]:
        clearance = self._clearance(self._agent_pos)
        return bool(clearance < self.cfg.sim.agent_radius), float(clearance)

    # ------------------------------------------------------------------

    def _dummy_obs(self) -> dict[str, Any]:
        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w
        S = self.cfg.bev.size
        return {
            "rgb": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            "depth": np.random.rand(h, w).astype(np.float32),
            "depth_metric": np.random.rand(h, w).astype(np.float32) * self.cfg.shadow.max_range_m,
            "bev_occupancy": (np.random.rand(S, S) > 0.9).astype(np.float32),
            "collision": False,
            "min_clearance": 1.0,
            "agent_pos": self._agent_pos.copy(),
            "agent_yaw": float(self._agent_yaw),
        }


if __name__ == "__main__":
    from ..utils import load_config

    cfg = load_config()
    env = ShadowWeaveEnv(cfg)

    for difficulty in ["static", "moving", "debris"]:
        cfg.sim.difficulty = difficulty
        obs = env.reset(seed=42)
        print(f"\n── {difficulty} ──")
        for k, v in obs.items():
            shape = getattr(v, "shape", type(v).__name__)
            print(f"  {k}: {shape}")

        # The whole point: the observation must change as the agent moves.
        depths, bevs, cols = [], [], []
        for i in range(20):
            o = env.step(action=np.array([1.0, 0.0, 0.08], dtype=np.float32))
            depths.append(o["depth"])
            bevs.append(o["bev_occupancy"])
            cols.append(o["collision"])
        d = np.stack(depths)
        b = np.stack(bevs)
        print(f"  depth  range=[{d.min():.3f}, {d.max():.3f}]  std-over-time={d.std(axis=0).mean():.5f}")
        print(f"  bev    positive={b.mean():.4f}  std-over-time={b.std(axis=0).mean():.5f}")
        print(f"  collisions: {sum(cols)}/20   clearance={obs['min_clearance']:.2f}m")

    env.close()
    print("\nShadowWeaveEnv OK")
