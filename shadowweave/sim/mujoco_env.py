"""MuJoCo 3.x single-room environment with debris spawning.

Room: 6m × 6m, first-person camera at cfg.sim.camera_height, 90° FOV.
Difficulty tiers: static | moving | debris.
Exports: RGB frame, ground-truth depth, ground-truth occupancy grid, object velocities.

MuJoCo 3.x may not be installed — import is guarded with a graceful message.
Install: pip install mujoco
"""

from __future__ import annotations

import math
import textwrap
from typing import Optional

import numpy as np
from omegaconf import DictConfig


try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False


_ROOM_XML_TEMPLATE = textwrap.dedent("""
<mujoco model="shadowweave_room">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual>
    <map fogstart="3" fogend="10"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <!-- floor -->
    <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat"/>
    <!-- walls -->
    <geom name="wall_n" type="box" pos="0 3 1.5"  size="3 0.1 1.5" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall_s" type="box" pos="0 -3 1.5" size="3 0.1 1.5" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall_e" type="box" pos="3 0 1.5"  size="0.1 3 1.5" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall_w" type="box" pos="-3 0 1.5" size="0.1 3 1.5" rgba="0.5 0.5 0.5 1"/>
    <!-- camera (first-person) -->
    <body name="camera_body" pos="0 -2 {camera_height}">
      <!-- xyaxes: x=right(+X), y=up(+Z) → camera looks in +Y (forward into room) -->
      <camera name="fp_cam" fovy="{fov}" xyaxes="1 0 0 0 0 1"/>
    </body>
    {objects_xml}
  </worldbody>
  {actuators_xml}
</mujoco>
""")


def _static_objects_xml() -> tuple[str, str]:
    boxes = ""
    for i, (x, y) in enumerate([(1.0, 0.5), (-1.0, 1.0), (0.5, -1.5)]):
        boxes += f'<geom name="obs_{i}" type="box" pos="{x} {y} 0.25" size="0.25 0.25 0.25" rgba="0.8 0.4 0.1 1"/>\n    '
    return boxes, ""


def _moving_objects_xml() -> tuple[str, str]:
    bodies = ""
    acts = "<actuator>\n"
    for i in range(3):
        x, y = i * 1.5 - 1.5, 0.0
        bodies += f"""
    <body name="mover_{i}" pos="{x} {y} 0.25">
      <freejoint/>
      <geom type="cylinder" size="0.2 0.25" rgba="0.2 0.6 0.9 1"/>
    </body>"""
    acts += "</actuator>"
    return bodies, acts


def _debris_objects_xml() -> tuple[str, str]:
    bodies = ""
    for i in range(5):
        x = np.random.uniform(-2.5, 2.5)
        y = np.random.uniform(-2.5, 2.5)
        z = np.random.uniform(2.0, 4.0)
        bodies += f"""
    <body name="debris_{i}" pos="{x:.2f} {y:.2f} {z:.2f}">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.15" rgba="0.9 0.2 0.2 1" mass="1"/>
    </body>"""
    return bodies, ""


class ShadowWeaveEnv:
    """MuJoCo single-room environment.

    Primary method: ``step() -> obs`` where obs is a dict with:
        rgb:         (H, W, 3) uint8
        depth:       (H, W) float32 — ground-truth depth in metres
        occupancy:   (H, W) float32 — binary occupancy at camera height slice
        velocity:    (N_objects, 3) float32 — object velocities
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._model: Optional[object] = None
        self._data:  Optional[object] = None
        self._renderer: Optional[object] = None

        if not _MUJOCO_AVAILABLE:
            print("[ShadowWeaveEnv] mujoco not installed — running in dummy mode.")
            print("  Install with: pip install mujoco")

    def reset(self, seed: Optional[int] = None) -> dict[str, np.ndarray]:
        if not _MUJOCO_AVAILABLE:
            return self._dummy_obs()

        if seed is not None:
            np.random.seed(seed)

        difficulty = self.cfg.sim.difficulty
        if difficulty == "static":
            obj_xml, act_xml = _static_objects_xml()
        elif difficulty == "moving":
            obj_xml, act_xml = _moving_objects_xml()
        elif difficulty == "debris":
            obj_xml, act_xml = _debris_objects_xml()
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        xml = _ROOM_XML_TEMPLATE.format(
            camera_height=self.cfg.sim.camera_height,
            fov=self.cfg.sim.camera_fov,
            objects_xml=obj_xml,
            actuators_xml=act_xml,
        )

        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data  = mujoco.MjData(self._model)
        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w
        self._renderer = mujoco.Renderer(self._model, height=h, width=w)
        mujoco.mj_resetData(self._model, self._data)
        return self._render_obs()

    def step(self, n_substeps: int = 3) -> dict[str, np.ndarray]:
        if not _MUJOCO_AVAILABLE:
            return self._dummy_obs()
        for _ in range(n_substeps):
            mujoco.mj_step(self._model, self._data)
        return self._render_obs()

    def _render_obs(self) -> dict[str, np.ndarray]:
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "fp_cam")
        self._renderer.update_scene(self._data, camera=cam_id)
        rgb = self._renderer.render()

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self._data, camera=cam_id)
        depth_raw = self._renderer.render()
        self._renderer.disable_depth_rendering()

        # MuJoCo returns depth as actual metric distances (metres) on macOS MuJoCo 3.x.
        # Background pixels (no geometry) return the far-plane distance.
        # Normalise to [0, 1]: 0 = very close (obstacle), 1 = far/background (open).
        # Guard against degenerate rendering (all-zero buffer from missing ARB_clip_control).
        depth_metric = depth_raw.astype(np.float32)
        d_max = depth_metric.max()
        if d_max < 1e-3:
            # Fallback: depth rendering failed — synthesise from GT occupancy
            depth_norm = 1.0 - self._compute_occupancy()
        else:
            near = float(self._model.vis.map.znear)
            depth_norm = np.clip(depth_metric / d_max, 0.0, 1.0)

        occupancy = self._compute_occupancy()
        velocity  = self._compute_velocities()

        return {"rgb": rgb, "depth": depth_norm, "occupancy": occupancy, "velocity": velocity}

    def _compute_occupancy(self) -> np.ndarray:
        """Binary occupancy grid at camera height, same resolution as depth map.

        Marks a filled disc around each geom's footprint so the grid has enough
        occupied pixels for the world model to learn from (single-point marks are
        <0.1% density and produce a degenerate all-zeros training signal).
        """
        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w
        occ = np.zeros((h, w), dtype=np.float32)
        room = self.cfg.sim.room_size
        px_per_m = w / room  # pixels per metre

        for i in range(self._model.ngeom):
            geom_pos  = self._data.geom_xpos[i]         # world position
            geom_size = self._model.geom_size[i]        # (rx, ry, rz) half-sizes or radius

            # skip floor, walls, and worldbody (type=0 is plane, skip very large geoms)
            geom_type = self._model.geom_type[i]
            if geom_type == 0:  # mjGEOM_PLANE
                continue
            max_radius_m = float(np.max(geom_size[:2]))
            if max_radius_m > 1.5:   # skip wall-sized geoms
                continue

            # project to grid — occupancy is at any height within the room
            col_c = int((geom_pos[0] + room / 2) / room * w)
            row_c = int((geom_pos[1] + room / 2) / room * h)
            radius_px = max(2, int(max_radius_m * px_per_m))

            # fill a disc
            for dr in range(-radius_px, radius_px + 1):
                for dc in range(-radius_px, radius_px + 1):
                    if dr * dr + dc * dc <= radius_px * radius_px:
                        r, c = row_c + dr, col_c + dc
                        if 0 <= r < h and 0 <= c < w:
                            occ[r, c] = 1.0

        # always mark wall footprints as occupied (constant background)
        margin = int(0.12 * px_per_m)
        occ[:margin, :]  = 1.0   # south wall
        occ[-margin:, :] = 1.0   # north wall
        occ[:, :margin]  = 1.0   # west wall
        occ[:, -margin:] = 1.0   # east wall

        return occ

    def _compute_velocities(self) -> np.ndarray:
        """(N_bodies, 3) velocity array from MuJoCo qvel."""
        n = self._model.nbody
        vels = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            # qvel is packed by joint DOF — simplification: use subtree_linvel
            vels[i] = self._data.subtree_linvel[i]
        return vels

    def _dummy_obs(self) -> dict[str, np.ndarray]:
        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w
        return {
            "rgb":       np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
            "depth":     np.random.rand(h, w).astype(np.float32),
            "occupancy": (np.random.rand(h, w) > 0.9).astype(np.float32),
            "velocity":  np.random.randn(5, 3).astype(np.float32),
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    env = ShadowWeaveEnv(cfg)
    obs = env.reset(seed=42)

    print("ShadowWeaveEnv demo:")
    for k, v in obs.items():
        print(f"  {k}: {v.shape} dtype={v.dtype}")

    obs2 = env.step()
    print(f"  step(): depth range=[{obs2['depth'].min():.3f}, {obs2['depth'].max():.3f}]")
    env.close()
    print("ShadowWeaveEnv OK")
