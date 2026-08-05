"""MuJoCo environment tests.

These are the regression guards for the root cause of the whole broken pipeline: the
sim produced an observation stream that never changed, so the dataset built from it
was constant and the world model learned nothing.

Skipped automatically when MuJoCo is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from shadowweave.sim.mujoco_env import ShadowWeaveEnv  # noqa: E402


@pytest.fixture(scope="module")
def sim_cfg():
    from shadowweave.utils import load_config

    c = load_config()
    c.depth.output_h = 64
    c.depth.output_w = 64
    c.bev.size = 32
    return c


def _rollout(cfg, difficulty, steps=25, seed=3):
    cfg.sim.difficulty = difficulty
    env = ShadowWeaveEnv(cfg)
    obs = env.reset(seed=seed)
    frames = [obs]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        frames.append(env.step(action=rng.uniform(-1, 1, 3).astype(np.float32)))
    env.close()
    return frames


class TestObservationVaries:
    """Depth had std 1.7e-9 across an entire episode; BEV targets had std exactly 0."""

    @pytest.mark.parametrize("difficulty", ["static", "moving", "debris"])
    def test_depth_changes_as_the_agent_moves(self, sim_cfg, difficulty):
        d = np.stack([f["depth"] for f in _rollout(sim_cfg, difficulty)])
        assert d.std(axis=0).mean() > 1e-3, f"depth is static (std={d.std(axis=0).mean():.2e})"

    @pytest.mark.parametrize("difficulty", ["static", "moving", "debris"])
    def test_bev_changes_as_the_agent_moves(self, sim_cfg, difficulty):
        b = np.stack([f["bev_occupancy"] for f in _rollout(sim_cfg, difficulty)])
        assert b.std(axis=0).mean() > 1e-3

    def test_agent_actually_moves(self, sim_cfg):
        pos = np.stack([f["agent_pos"] for f in _rollout(sim_cfg, "static")])
        assert float(np.abs(pos - pos[0]).max()) > 0.05


class TestDepthNormalisation:
    """Per-frame max normalisation made the depth scale drift between frames."""

    def test_depth_is_bounded(self, sim_cfg):
        for f in _rollout(sim_cfg, "static", steps=10):
            assert 0.0 <= float(f["depth"].min()) and float(f["depth"].max()) <= 1.0

    def test_scale_is_fixed_not_per_frame(self, sim_cfg):
        """The normalised map must be exactly metres/max_range_m.

        Checking the per-frame max cannot tell the two apart — the room always
        contains background beyond max_range, so a fixed scale saturates at 1.0 too.
        Tying `depth` to `depth_metric` pins the property directly: under per-frame
        normalisation the divisor changes every frame and this identity breaks.
        """
        max_range = float(sim_cfg.shadow.max_range_m)
        for f in _rollout(sim_cfg, "static", steps=12):
            expected = np.clip(f["depth_metric"] / max_range, 0.0, 1.0)
            assert np.allclose(f["depth"], expected, atol=1e-5)

    def test_same_geometry_gives_the_same_value_across_frames(self, sim_cfg):
        """A surface that has not moved must read the same on two different frames."""
        sim_cfg.sim.difficulty = "static"
        env = ShadowWeaveEnv(sim_cfg)
        first = env.reset(seed=7)
        # Turn a full circle and come back: same pose, so the same reading.
        for _ in range(8):
            env.step(action=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        for _ in range(8):
            env.step(action=np.array([0.0, 0.0, -1.0], dtype=np.float32))
        back = env.step(action=np.zeros(3, dtype=np.float32))
        env.close()
        assert np.allclose(first["depth"], back["depth"], atol=0.02)


class TestCollisionDetection:
    """`occupancy.max() > 0.5` was True on every step because walls were always set."""

    def test_not_always_colliding(self, sim_cfg):
        frames = _rollout(sim_cfg, "static", steps=30)
        assert not all(f["collision"] for f in frames), "collision True on every step"

    def test_clearance_is_finite_and_positive(self, sim_cfg):
        for f in _rollout(sim_cfg, "static", steps=10):
            assert np.isfinite(f["min_clearance"]) and f["min_clearance"] >= 0.0

    def test_collision_agrees_with_clearance(self, sim_cfg):
        for f in _rollout(sim_cfg, "static", steps=20):
            assert f["collision"] == (f["min_clearance"] < sim_cfg.sim.agent_radius)


class TestEgocentricTargets:
    def test_bev_follows_the_agent(self, sim_cfg):
        """A world-frame grid ignores the agent's pose; an egocentric one cannot."""
        sim_cfg.sim.difficulty = "static"
        env = ShadowWeaveEnv(sim_cfg)
        env.reset(seed=1)
        before = env.bev_occupancy().copy()
        for _ in range(15):
            env.step(action=np.array([1.0, 0.0, 0.25], dtype=np.float32))
        after = env.bev_occupancy().copy()
        env.close()
        assert not np.allclose(before, after), "BEV did not change after the agent moved"

    def test_future_geometry_renders_into_the_current_frame(self, sim_cfg):
        sim_cfg.sim.difficulty = "moving"
        env = ShadowWeaveEnv(sim_cfg)
        obs = env.reset(seed=5)
        pose = (obs["agent_pos"].copy(), obs["agent_yaw"])
        now = env.geom_snapshot()
        for _ in range(60):
            env.step(action=np.zeros(3, dtype=np.float32))
        later = env.geom_snapshot()

        a = env.bev_occupancy(pose[0], pose[1], now)
        b = env.bev_occupancy(pose[0], pose[1], later)
        env.close()
        # Same viewpoint, different world state → the target must differ.
        assert not np.allclose(a, b), "moving obstacles did not change the future target"


class TestDynamicsPersist:
    """Movers stopped from friction in <0.1s and debris landed in ~0.8s, so by the
    shortest 1s horizon every scene was static and the horizon axis carried no signal."""

    def test_movers_keep_moving_past_the_longest_horizon(self, sim_cfg):
        sim_cfg.sim.difficulty = "moving"
        env = ShadowWeaveEnv(sim_cfg)
        env.reset(seed=2)
        early = env.geom_snapshot()["xpos"].copy()
        for _ in range(300):  # 10s at 30fps
            env.step(action=np.zeros(3, dtype=np.float32))
        late = env.geom_snapshot()["xpos"].copy()
        env.close()
        assert float(np.abs(late - early).max()) > 0.5

    def test_debris_releases_are_staggered(self, sim_cfg):
        sim_cfg.sim.difficulty = "debris"
        env = ShadowWeaveEnv(sim_cfg)
        env.reset(seed=4)
        releases = sorted(d["release"] for d in env._debris)
        env.close()
        assert len(set(releases)) > 1, "all debris released at the same instant"
        assert max(releases) - min(releases) > 30, "releases not spread across the episode"
