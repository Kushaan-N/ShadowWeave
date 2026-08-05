"""Tests for OpenUSD scene export.

These check *values*, not just that a file appeared. The first version produced a
structurally valid stage with the right prim count, time samples and metadata — and
every transform silently at identity, because assigning ``m[r][c]`` on a Gf.Matrix4d
writes into a copy of the row. A file that opens is not a file that is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pxr", reason="usd-core not installed")
pytest.importorskip("mujoco", reason="MuJoCo not installed")

from pxr import Usd, UsdGeom  # noqa: E402

from shadowweave.sim.mujoco_env import ShadowWeaveEnv  # noqa: E402
from shadowweave.sim.usd_export import (  # noqa: E402
    USDSceneExporter,
    _matrix,
    _sanitise,
    export_episode,
)


@pytest.fixture
def scene(cfg, tmp_path):
    """Export a short static episode and hand back the stage plus the truth."""
    c = cfg.copy()
    c.sim.difficulty = "static"
    env = ShadowWeaveEnv(c, render_rgb=False)
    env.reset(seed=7)

    ex = USDSceneExporter(c)
    ex.begin(env)
    poses = []
    for _ in range(4):
        ex.record(env)
        poses.append((env._agent_pos.copy(), env._agent_yaw))
        env.step(action=np.array([1.0, 0.0, 0.2], dtype=np.float32))
    path = ex.save(tmp_path / "s.usda")
    truth = {g["name"]: g for g in env.describe_scene()}
    env.close()
    return Usd.Stage.Open(str(path)), truth, poses, c


class TestMatrixConversion:
    def test_translation_survives(self):
        m = _matrix(np.array([1.0, -2.0, 3.5]), np.eye(3))
        assert (m[3][0], m[3][1], m[3][2]) == (1.0, -2.0, 3.5)

    def test_identity_rotation_is_identity(self):
        m = _matrix(np.zeros(3), np.eye(3))
        R = np.array([[m[r][c] for c in range(3)] for r in range(3)])
        assert np.allclose(R, np.eye(3))

    def test_rotation_is_transposed_once(self):
        """MuJoCo is column-vector, USD row-vector — exactly one transpose."""
        theta = 0.7
        mj = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
        m = _matrix(np.zeros(3), mj)
        R = np.array([[m[r][c] for c in range(3)] for r in range(3)])
        assert np.allclose(R, mj.T)

    def test_rotation_stays_orthonormal(self):
        theta = 1.1
        mj = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
        m = _matrix(np.array([5.0, 0.0, 1.0]), mj)
        R = np.array([[m[r][c] for c in range(3)] for r in range(3)])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)


class TestGeometryFidelity:
    def test_static_geom_positions_match_mujoco_exactly(self, scene):
        stage, truth, _, _ = scene
        for name, g in truth.items():
            if g["dynamic"]:
                continue
            prim = stage.GetPrimAtPath(f"/World/Scene/{_sanitise(name)}")
            assert prim, f"{name} missing from the stage"
            m = prim.GetAttribute("xformOp:transform").Get()
            got = np.array([m[3][0], m[3][1], m[3][2]])
            assert np.allclose(got, g["pos"], atol=1e-9), f"{name} at the wrong place"

    def test_not_everything_collapsed_to_the_origin(self, scene):
        """The exact failure mode of the original bug."""
        stage, truth, _, _ = scene
        origins = 0
        total = 0
        for name, g in truth.items():
            if g["dynamic"] or np.allclose(g["pos"], 0):
                continue
            m = stage.GetPrimAtPath(f"/World/Scene/{_sanitise(name)}").GetAttribute(
                "xformOp:transform").Get()
            total += 1
            if abs(m[3][0]) < 1e-9 and abs(m[3][1]) < 1e-9 and abs(m[3][2]) < 1e-9:
                origins += 1
        assert total > 0 and origins == 0

    def test_box_half_extents_become_scale(self, scene):
        stage, truth, _, _ = scene
        boxes = [(n, g) for n, g in truth.items() if g["type"] == 6 and not g["dynamic"]]
        assert boxes
        for name, g in boxes:
            prim = stage.GetPrimAtPath(f"/World/Scene/{_sanitise(name)}")
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            scale = [o for o in ops if "scale" in o.GetOpName()]
            assert scale, f"{name} has no scale op"
            s = scale[0].Get()
            assert np.allclose([s[0], s[1], s[2]], g["size"][:3], atol=1e-6)


class TestStageConventions:
    def test_z_up_and_metres(self, scene):
        """A mismatch here silently rotates or rescales the whole scene on import."""
        stage, _, _, _ = scene
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

    def test_timeline_matches_the_rollout(self, scene):
        stage, _, poses, c = scene
        assert stage.GetStartTimeCode() == 0
        assert stage.GetEndTimeCode() == len(poses) - 1
        assert stage.GetTimeCodesPerSecond() == c.sim.fps

    def test_has_a_default_prim(self, scene):
        """Without one, referencing the layer from another stage fails."""
        stage, _, _, _ = scene
        assert stage.GetDefaultPrim().IsValid()


class TestCamera:
    def test_camera_follows_the_agent(self, scene):
        stage, _, poses, c = scene
        attr = stage.GetPrimAtPath("/World/AgentCamera").GetAttribute("xformOp:transform")
        for i, (pos, _) in enumerate(poses):
            m = attr.Get(Usd.TimeCode(i))
            assert np.allclose([m[3][0], m[3][1]], pos, atol=1e-9)
            assert m[3][2] == pytest.approx(c.sim.camera_height)

    def test_camera_actually_moves(self, scene):
        stage, _, poses, _ = scene
        attr = stage.GetPrimAtPath("/World/AgentCamera").GetAttribute("xformOp:transform")
        first, last = attr.Get(Usd.TimeCode(0)), attr.Get(Usd.TimeCode(len(poses) - 1))
        assert abs(first[3][0] - last[3][0]) + abs(first[3][1] - last[3][1]) > 1e-6

    def test_camera_basis_is_a_rotation(self, scene):
        stage, _, poses, _ = scene
        m = stage.GetPrimAtPath("/World/AgentCamera").GetAttribute(
            "xformOp:transform").Get(Usd.TimeCode(0))
        R = np.array([[m[r][c] for c in range(3)] for r in range(3)])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0), "mirrored basis would flip the view"

    def test_aperture_encodes_the_configured_fov(self, scene):
        stage, _, _, c = scene
        cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/AgentCamera"))
        f = cam.GetFocalLengthAttr().Get()
        aperture = cam.GetHorizontalApertureAttr().Get()
        fov = 2 * np.degrees(np.arctan(aperture / (2 * f)))
        assert fov == pytest.approx(float(c.sim.camera_fov), abs=1e-3)


class TestDynamics:
    def test_falling_debris_is_animated(self, cfg, tmp_path):
        c = cfg.copy()
        path = export_episode(c, tmp_path / "d.usda", n_steps=25, seed=3,
                              difficulty="debris")
        stage = Usd.Stage.Open(str(path))
        animated = [
            p for p in stage.Traverse()
            if p.GetAttribute("xformOp:transform")
            and len(p.GetAttribute("xformOp:transform").GetTimeSamples()) > 1
        ]
        # Debris plus the camera.
        assert len(animated) > 1

    def test_static_scene_geoms_carry_no_time_samples(self, scene):
        """Time-sampling immovable walls would bloat the stage for nothing."""
        stage, truth, _, _ = scene
        for name, g in truth.items():
            if g["dynamic"]:
                continue
            attr = stage.GetPrimAtPath(
                f"/World/Scene/{_sanitise(name)}").GetAttribute("xformOp:transform")
            assert len(attr.GetTimeSamples()) == 0


class TestNaming:
    @pytest.mark.parametrize("raw,ok", [
        ("obs_0", "obs_0"), ("wall-n", "wall_n"), ("3debris", "g_3debris"), ("", "g_"),
    ])
    def test_prim_names_are_valid_identifiers(self, raw, ok):
        assert _sanitise(raw) == ok
