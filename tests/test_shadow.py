"""Tests for the shadow-ray core and BEV projection.

Every test here pins an invariant that was silently violated in the original code.
"""

from __future__ import annotations

import torch

from shadowweave.shadow.bev import BEVProjector, bev_flow
from shadowweave.shadow.occupancy import GeometricOccupancyField
from shadowweave.shadow.raycast import ShadowRaycaster


class TestZoneDiscrimination:
    """The 9 zones collapsed to a single identical value (0.843 everywhere)."""

    def test_zones_are_not_uniform(self, cfg, wall_left_depth):
        grid = ShadowRaycaster(cfg).forward_from_depth(wall_left_depth).detach()
        spread = float(grid.max() - grid.min())
        assert spread > 0.2, f"zones barely differ (spread={spread:.4f})"

    def test_obstacle_side_reads_higher(self, cfg, wall_left_depth):
        grid = ShadowRaycaster(cfg).forward_from_depth(wall_left_depth).detach()[0]
        left, right = float(grid[0]), float(grid[-1])
        assert left > right * 1.5, f"left={left:.3f} not clearly above right={right:.3f}"

    def test_zone_weights_are_row_stochastic(self, cfg):
        w = ShadowRaycaster(cfg).zone_weights()
        assert torch.allclose(w.sum(dim=1), torch.ones(w.shape[0]), atol=1e-5)

    def test_mirrored_scene_mirrors_grid(self, cfg, wall_left_depth):
        rc = ShadowRaycaster(cfg)
        left = rc.forward_from_depth(wall_left_depth)[0]
        right = rc.forward_from_depth(torch.flip(wall_left_depth, dims=[-1]))[0]
        assert torch.allclose(left, torch.flip(right, dims=[0]), atol=0.06)


class TestShadowSemantics:
    """Empty space used to report MAXIMUM uncertainty — the semantics were inverted."""

    def test_empty_volume_casts_no_shadow(self, cfg, wall_left_depth):
        rc = ShadowRaycaster(cfg)
        empty = torch.zeros(1, 1, cfg.shadow.march_steps,
                            cfg.depth.output_h, cfg.depth.output_w)
        with torch.no_grad():
            grid = rc.forward(wall_left_depth, empty)
        assert float(grid.max()) < 0.05, f"empty space reads {float(grid.max()):.3f}"

    def test_full_volume_casts_shadow(self, cfg, wall_left_depth):
        rc = ShadowRaycaster(cfg)
        full = torch.ones(1, 1, cfg.shadow.march_steps,
                          cfg.depth.output_h, cfg.depth.output_w)
        with torch.no_grad():
            grid = rc.forward(wall_left_depth, full)
        assert float(grid.max()) > 0.3

    def test_near_obstacle_beats_far_obstacle(self, cfg):
        rc = ShadowRaycaster(cfg)
        h, w = cfg.depth.output_h, cfg.depth.output_w
        with torch.no_grad():
            near = rc.forward_from_depth(torch.full((1, 1, h, w), 0.1))
            far = rc.forward_from_depth(torch.full((1, 1, h, w), 0.9))
        assert float(near.mean()) > float(far.mean())

    def test_geometric_and_volumetric_paths_agree(self, cfg, wall_left_depth):
        rc = ShadowRaycaster(cfg)
        geo = rc.forward_from_depth(wall_left_depth)[0]
        vol = rc.forward(wall_left_depth, GeometricOccupancyField(cfg)(wall_left_depth))[0]
        assert torch.allclose(geo, vol, atol=0.15), f"{geo} vs {vol}"


class TestDifferentiability:
    def test_geometric_path_has_gradient(self, cfg, wall_left_depth):
        d = wall_left_depth.clone().requires_grad_(True)
        ShadowRaycaster(cfg).forward_from_depth(d).sum().backward()
        assert d.grad is not None and float(d.grad.abs().sum()) > 0

    def test_zone_assign_receives_gradient(self, cfg, wall_left_depth):
        rc = ShadowRaycaster(cfg)
        rc.forward_from_depth(wall_left_depth).sum().backward()
        assert rc.zone_assign.grad is not None
        assert float(rc.zone_assign.grad.abs().sum()) > 0


class TestBEVProjector:
    def test_shapes(self, cfg, wall_left_depth):
        occ, vis = BEVProjector(cfg)(wall_left_depth)
        S = cfg.bev.size
        assert occ.shape == (1, 1, S, S) and vis.shape == (1, 1, S, S)

    def test_behind_surface_is_shadow(self, cfg):
        """The defining property: what is behind an obstacle is unobserved."""
        proj = BEVProjector(cfg)
        h, w = cfg.depth.output_h, cfg.depth.output_w
        depth = torch.ones(1, 1, h, w)
        depth[0, 0, :, w // 3 : 2 * w // 3] = 0.4   # wall dead ahead
        _, vis = proj(depth)

        S = cfg.bev.size
        behind = float(vis[0, 0, : S // 4, S // 3 : 2 * S // 3].mean())
        infront = float(vis[0, 0, S // 2 :, S // 3 : 2 * S // 3].mean())
        assert behind < 0.1, f"behind the wall should be shadow, got {behind:.3f}"
        assert infront > 0.5, f"in front should be visible, got {infront:.3f}"

    def test_outside_fov_is_unobserved(self, cfg, wall_left_depth):
        proj = BEVProjector(cfg)
        _, vis = proj(wall_left_depth)
        outside = (proj.in_fov < 0.5)
        if outside.any():
            assert float(vis[0, 0][outside].max()) < 1e-6

    def test_open_scene_has_less_shadow_than_blocked(self, cfg):
        proj = BEVProjector(cfg)
        h, w = cfg.depth.output_h, cfg.depth.output_w
        open_s = proj.shadow_fraction(torch.ones(1, 1, h, w))
        blocked = proj.shadow_fraction(torch.full((1, 1, h, w), 0.15))
        assert float(open_s) < float(blocked)

    def test_differentiable(self, cfg, wall_left_depth):
        d = wall_left_depth.clone().requires_grad_(True)
        occ, vis = BEVProjector(cfg)(d)
        (occ.sum() + vis.sum()).backward()
        assert d.grad is not None and float(d.grad.abs().sum()) > 0

    def test_flow_is_zero_for_identical_frames(self, cfg):
        S = cfg.bev.size
        a = torch.rand(1, 1, S, S)
        assert float(bev_flow(a, a).abs().max()) == 0.0

    def test_flow_is_nonzero_when_scene_changes(self, cfg):
        S = cfg.bev.size
        a = torch.zeros(1, 1, S, S)
        b = torch.zeros(1, 1, S, S)
        b[..., 10:14, 10:14] = 1.0
        assert float(bev_flow(a, b).abs().max()) > 0
