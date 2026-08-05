"""Export MuJoCo rollouts to OpenUSD.

Why this exists
---------------
MuJoCo is cheap to roll out but renders flat, untextured frames. USD is the
interchange format Omniverse and Isaac Sim read, so exporting the *same* randomised
scene and the *same* agent trajectory means a rollout generated here can be re-rendered
photorealistically later without regenerating the episode or losing the correspondence
to its training targets. Cheap generation and high-fidelity rendering stop being an
either/or.

What is written
---------------
  * every geom — floor, walls, obstacles — as a typed USD prim with its MuJoCo
    transform, half-extents and colour
  * time samples on anything that moves, so the whole episode scrubs on a timeline
  * a camera prim following the agent's egocentric pose, so a re-render can reproduce
    the exact viewpoint the depth map came from

Stage conventions match MuJoCo: Z-up, metres. Getting either wrong silently produces
a scene that is rotated or 100x the wrong size when imported.

usd-core is an optional dependency (`pip install usd-core`); nothing else in the
pipeline imports this module.
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig

try:
    from pxr import Gf, Sdf, Usd, UsdGeom
    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False

# MuJoCo mjtGeom values, hard-coded so this module does not require mujoco to import.
_PLANE, _SPHERE, _CAPSULE, _CYLINDER, _BOX = 0, 2, 3, 5, 6


def _sanitise(name: str) -> str:
    """USD prim names must be valid identifiers."""
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return f"g_{s}" if not s or s[0].isdigit() else s


def _matrix(pos: np.ndarray, mat: np.ndarray) -> "Gf.Matrix4d":
    """MuJoCo (position, 3x3 rotation) → USD 4x4.

    Built through the explicit constructor, not by assigning ``m[r][c]``: indexing a
    Gf.Matrix4d returns a *copy* of the row, so element assignment writes into a
    temporary and is silently discarded, leaving every transform at identity. That
    produces a structurally valid stage with every prim stacked at the origin.

    MuJoCo's matrix is column-vector convention (columns are the local axes); USD is
    row-vector, so the rotation is transposed and the translation occupies the last
    row.
    """
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    return Gf.Matrix4d(
        float(m[0, 0]), float(m[1, 0]), float(m[2, 0]), 0.0,
        float(m[0, 1]), float(m[1, 1]), float(m[2, 1]), 0.0,
        float(m[0, 2]), float(m[1, 2]), float(m[2, 2]), 0.0,
        float(pos[0]), float(pos[1]), float(pos[2]), 1.0,
    )


class USDSceneExporter:
    """Records a rollout and writes it as a single USD stage.

    Primary methods: ``begin(env)``, ``record(env)`` per step, then ``save(path)``.
    """

    def __init__(self, cfg: DictConfig, fps: Optional[int] = None) -> None:
        if not _USD_AVAILABLE:
            raise ImportError(
                "usd-core required for USD export — pip install usd-core"
            )
        self.cfg = cfg
        self.fps = fps or cfg.sim.fps
        self._frames: list[dict[str, Any]] = []
        self._static: list[dict[str, Any]] = []

    def begin(self, env) -> None:
        self._static = env.describe_scene()
        self._frames = []

    def record(self, env) -> None:
        """Capture one frame: moving geometry plus the agent pose."""
        self._frames.append({
            "geoms": {g["name"]: (g["pos"].copy(), g["mat"].copy())
                      for g in env.describe_scene() if g["dynamic"]},
            "agent_pos": np.asarray(env._agent_pos, dtype=np.float64).copy(),
            "agent_yaw": float(env._agent_yaw),
        })

    # ------------------------------------------------------------------

    def _define(self, stage, path: str, geom: dict[str, Any]):
        """Create the typed prim. USD primitives are unit-sized, so MuJoCo's
        half-extents become scale rather than geometry."""
        t, size = geom["type"], geom["size"]
        if t == _BOX:
            prim = UsdGeom.Cube.Define(stage, path)
            prim.GetSizeAttr().Set(2.0)  # unit cube spans [-1, 1]
            scale = Gf.Vec3f(float(size[0]), float(size[1]), float(size[2]))
        elif t == _SPHERE:
            prim = UsdGeom.Sphere.Define(stage, path)
            prim.GetRadiusAttr().Set(float(size[0]))
            scale = None
        elif t in (_CYLINDER, _CAPSULE):
            cls = UsdGeom.Cylinder if t == _CYLINDER else UsdGeom.Capsule
            prim = cls.Define(stage, path)
            prim.GetRadiusAttr().Set(float(size[0]))
            prim.GetHeightAttr().Set(float(size[1]) * 2.0)  # MuJoCo stores half-height
            prim.GetAxisAttr().Set(UsdGeom.Tokens.z)
            scale = None
        elif t == _PLANE:
            # An infinite MuJoCo plane exports as a large thin slab; USD has no
            # native infinite plane and importers handle a slab predictably.
            prim = UsdGeom.Cube.Define(stage, path)
            prim.GetSizeAttr().Set(2.0)
            extent = float(size[0]) if size[0] > 0 else self.cfg.sim.room_size / 2
            scale = Gf.Vec3f(extent, extent, 0.01)
        else:
            prim = UsdGeom.Cube.Define(stage, path)
            prim.GetSizeAttr().Set(2.0)
            scale = Gf.Vec3f(*(float(max(s, 1e-3)) for s in size[:3]))

        rgba = geom["rgba"]
        prim.GetDisplayColorAttr().Set([Gf.Vec3f(float(rgba[0]), float(rgba[1]), float(rgba[2]))])
        if len(rgba) > 3 and float(rgba[3]) < 1.0:
            prim.GetDisplayOpacityAttr().Set([float(rgba[3])])
        if scale is not None:
            UsdGeom.Xformable(prim).AddScaleOp().Set(scale)
        return prim

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

        stage = Usd.Stage.CreateNew(str(path))
        # Z-up metres, matching MuJoCo. A mismatch here silently rotates or rescales
        # the whole scene on import.
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(max(len(self._frames) - 1, 0))
        stage.SetTimeCodesPerSecond(self.fps)

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Xform.Define(stage, "/World/Scene")

        # Static structure, plus a transform op on movers to hang time samples off.
        xform_ops = {}
        for geom in self._static:
            name = _sanitise(geom["name"])
            prim = self._define(stage, f"/World/Scene/{name}", geom)
            op = UsdGeom.Xformable(prim).AddTransformOp()
            if geom["dynamic"] and self._frames:
                xform_ops[geom["name"]] = op
            else:
                op.Set(_matrix(geom["pos"], geom["mat"]))

        for frame_idx, frame in enumerate(self._frames):
            for gname, (pos, mat) in frame["geoms"].items():
                if gname in xform_ops:
                    xform_ops[gname].Set(_matrix(pos, mat), Usd.TimeCode(frame_idx))

        self._write_camera(stage)
        stage.GetRootLayer().Save()
        return path

    def _write_camera(self, stage) -> None:
        """Egocentric camera, time-sampled along the agent trajectory."""
        cam = UsdGeom.Camera.Define(stage, "/World/AgentCamera")
        cam.GetFocalLengthAttr().Set(24.0)
        fov = float(self.cfg.sim.camera_fov)
        # horizontalAperture = 2 f tan(fov/2), so an importer recovers the same FOV.
        cam.GetHorizontalApertureAttr().Set(2 * 24.0 * math.tan(math.radians(fov) / 2))
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, float(self.cfg.shadow.max_range_m) * 2))

        op = UsdGeom.Xformable(cam).AddTransformOp()
        h = float(self.cfg.sim.camera_height)
        for i, frame in enumerate(self._frames):
            yaw = frame["agent_yaw"]
            # MuJoCo camera looks along +Y at yaw=0; USD cameras look down -Z. Rotate
            # so the exported view matches the depth map that trained the model.
            fwd = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, up)
            # Columns are the local axes, matching MuJoCo's convention so _matrix
            # transposes exactly once. A USD camera looks down its own -Z.
            rot = np.stack([right, up, -fwd], axis=1)
            pos = np.array([frame["agent_pos"][0], frame["agent_pos"][1], h])
            op.Set(_matrix(pos, rot), Usd.TimeCode(i))


def export_episode(
    cfg: DictConfig,
    path: str | pathlib.Path,
    n_steps: int = 60,
    seed: int = 0,
    difficulty: Optional[str] = None,
) -> pathlib.Path:
    """Roll one episode and write it to USD. Returns the written path."""
    from .mujoco_env import ShadowWeaveEnv
    from .synthetic_data import _make_policy

    if difficulty:
        cfg.sim.difficulty = difficulty
    env = ShadowWeaveEnv(cfg, render_rgb=False)
    obs = env.reset(seed=seed)

    exporter = USDSceneExporter(cfg)
    exporter.begin(env)
    policy = _make_policy(cfg.data.policy, np.random.default_rng(seed))
    for _ in range(n_steps):
        exporter.record(env)
        obs = env.step(action=policy.act(obs))
    env.close()
    return exporter.save(path)


def main() -> None:
    from ..utils import load_config

    ap = argparse.ArgumentParser(description="Export a MuJoCo rollout to OpenUSD")
    ap.add_argument("--out", type=str, default="results/scene.usda")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", type=str, default=None,
                    choices=["static", "moving", "debris"])
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    out = export_episode(cfg, args.out, args.steps, args.seed, args.difficulty)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    import tempfile

    from ..utils import load_config

    if not _USD_AVAILABLE:
        print("usd-core not installed — pip install usd-core")
        raise SystemExit(0)

    cfg = load_config()
    cfg.depth.output_h = cfg.depth.output_w = 64
    print(f"USD export demo (OpenUSD {'.'.join(map(str, Usd.GetVersion()))})")

    out = pathlib.Path(tempfile.mkdtemp()) / "scene.usda"
    export_episode(cfg, out, n_steps=20, seed=3, difficulty="debris")
    print(f"  wrote {out.name}  ({out.stat().st_size / 1024:.0f} KB)")

    stage = Usd.Stage.Open(str(out))
    prims = [p.GetPath().pathString for p in stage.Traverse()]
    print(f"  prims: {len(prims)}  e.g. {prims[1:4]}")
    print(f"  up-axis={UsdGeom.GetStageUpAxis(stage)}  "
          f"metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}  "
          f"timeCodes={stage.GetStartTimeCode():.0f}..{stage.GetEndTimeCode():.0f} "
          f"@ {stage.GetTimeCodesPerSecond():.0f}fps")

    cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/AgentCamera"))
    samples = cam.GetPrim().GetAttribute("xformOp:transform").GetTimeSamples()
    print(f"  camera time samples: {len(samples)}  (agent trajectory is animated)")
    assert len(samples) == 20

    moving = [p for p in stage.Traverse()
              if p.GetAttribute("xformOp:transform")
              and len(p.GetAttribute("xformOp:transform").GetTimeSamples()) > 1]
    print(f"  animated prims: {len(moving)}  (falling debris + camera)")
    print("USDSceneExporter OK")
