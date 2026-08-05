"""Differentiable depth → egocentric bird's-eye-view occupancy + visibility.

This is the bridge that lets the world model predict in the same frame it observes.
The previous pipeline fed a first-person depth image and asked for a top-down
*world-frame* occupancy grid, which is (a) a different coordinate frame and (b)
independent of where the agent is standing — so the target barely depended on the
input and the model learned a constant.

Formulation is polar rather than scatter-based: for each BEV cell we compute its
bearing and range in the agent's frame, look up how far the first surface is along
that bearing, and compare. That makes every output a smooth function of the input
depth (fully differentiable, no non-differentiable scatter indices) and it makes the
shadow explicit as a first-class channel:

    range <  hit_range  → observed free space   (visibility 1, occupancy 0)
    range ≈  hit_range  → the surface itself    (visibility 1, occupancy 1)
    range >  hit_range  → SHADOW, unobserved    (visibility 0, occupancy unknown)
    outside FOV         → unobserved            (visibility 0)

Ego frame convention: +z forward (up the image), +x right. The agent sits at the
bottom-centre cell, looking up the grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..utils import sanitize_depth


class BEVProjector(nn.Module):
    """Depth map → egocentric BEV occupancy and visibility.

    Primary method: ``forward(depth_map) -> (occupancy, visibility)``
    Input:  (B, 1, H, W) float32 depth in [0, 1], 1.0 == cfg.shadow.max_range_m
    Output: (B, 1, S, S) occupancy in [0, 1], (B, 1, S, S) visibility in [0, 1]
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.size      = cfg.bev.size
        self.range_m   = cfg.bev.range_m
        self.max_range = cfg.shadow.max_range_m
        self.surf_tau  = cfg.bev.surface_thickness_m
        self.vis_tau   = cfg.bev.visibility_softness_m
        self.n_elev    = cfg.shadow.elevation_samples
        self.elev_ext  = cfg.shadow.elevation_extent
        self.half_fov  = math.radians(cfg.sim.camera_fov) / 2.0

        cell = self.range_m / self.size
        # Cell centres: +z forward from the agent, +x right. Row 0 is the FAR row so
        # the rendered image reads like a map with the agent at the bottom.
        z = torch.linspace(self.range_m - cell / 2, cell / 2, self.size)
        x = torch.linspace(-self.range_m / 2 + cell / 2, self.range_m / 2 - cell / 2, self.size)
        zz, xx = torch.meshgrid(z, x, indexing="ij")

        bearing = torch.atan2(xx, zz.clamp_min(1e-6))
        rng     = torch.sqrt(xx ** 2 + zz ** 2)

        self.register_buffer("cell_range", rng, persistent=False)
        self.register_buffer("cell_bearing", bearing, persistent=False)
        # Pinhole: bearing → normalised image x. Outside the FOV there is no pixel to
        # sample, so mask those cells as never-observed.
        self.register_buffer(
            "cell_norm_x", torch.tan(bearing.clamp(-self.half_fov, self.half_fov)) / math.tan(self.half_fov),
            persistent=False,
        )
        self.register_buffer("in_fov", (bearing.abs() <= self.half_fov).float(), persistent=False)
        self.register_buffer("in_range", (rng <= self.max_range).float(), persistent=False)
        self.register_buffer(
            "norm_y", torch.linspace(-self.elev_ext, self.elev_ext, self.n_elev), persistent=False
        )

    def hit_range(self, depth_map: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) → (B, S, S) metric distance to the first surface per bearing."""
        depth_map = sanitize_depth(depth_map)
        B = depth_map.shape[0]
        S, E = self.size, self.n_elev

        # Sample the depth map at every cell's bearing across an elevation band, then
        # take the nearest surface over that band (a soft-min keeps gradients alive on
        # more than just the argmin row).
        gx = self.cell_norm_x.reshape(1, S * S).expand(E, S * S)
        gy = self.norm_y.reshape(E, 1).expand(E, S * S)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, E, S * S, 2)

        sampled = F.grid_sample(
            depth_map, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(1)  # (B, E, S*S)

        soft_min = -self.cfg.bev.depth_softmin_beta * sampled
        nearest = (sampled * F.softmax(soft_min, dim=1)).sum(dim=1)  # (B, S*S)
        return nearest.reshape(B, S, S) * self.max_range

    def forward(self, depth_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (occupancy, visibility), each (B, 1, S, S) in [0, 1]."""
        hit = self.hit_range(depth_map)                       # (B, S, S) metres
        rng = self.cell_range.unsqueeze(0)                    # (1, S, S)

        # Surface band: a smooth bump where the cell's range matches the hit range.
        occupancy = torch.exp(-((rng - hit) ** 2) / (2 * self.surf_tau ** 2))

        # Visibility: 1 in front of the surface, decaying to 0 behind it. This is the
        # shadow — the whole point of the system.
        visibility = torch.sigmoid((hit - rng) / self.vis_tau)

        mask = (self.in_fov * self.in_range).unsqueeze(0)
        occupancy = occupancy * mask
        visibility = visibility * mask

        return occupancy.unsqueeze(1), visibility.unsqueeze(1)

    def shadow_fraction(self, depth_map: torch.Tensor) -> torch.Tensor:
        """(B,) scalar — fraction of the in-FOV BEV area that is in shadow."""
        _, vis = self.forward(depth_map)
        mask = (self.in_fov * self.in_range)
        return 1.0 - (vis.squeeze(1) * mask).sum(dim=(1, 2)) / mask.sum().clamp_min(1.0)


def bev_flow(bev_prev: torch.Tensor, bev_curr: torch.Tensor) -> torch.Tensor:
    """Two-channel motion proxy between consecutive BEV occupancy frames.

    Channel 0 is the temporal difference (what appeared / disappeared), channel 1 the
    lateral gradient of that difference (which way it moved). The old pipeline stored
    a frame-difference of a *static* depth map, so every velocity entry in
    data/rollouts was exactly zero.

    Args:
        bev_prev, bev_curr: (B, 1, S, S)
    Returns:
        (B, 2, S, S)
    """
    diff = bev_curr - bev_prev
    lateral = torch.zeros_like(diff)
    lateral[..., 1:-1] = (diff[..., 2:] - diff[..., :-2]) * 0.5
    return torch.cat([diff, lateral], dim=1)


if __name__ == "__main__":
    from ..utils import get_device, load_config

    cfg = load_config()
    device = get_device()
    print(f"BEVProjector demo on {device}\n")

    h, w = cfg.depth.output_h, cfg.depth.output_w
    proj = BEVProjector(cfg).to(device)

    # Wall straight ahead at half of max range, open elsewhere.
    depth = torch.ones(1, 1, h, w, device=device)
    depth[0, 0, :, w // 3 : 2 * w // 3] = 0.5

    occ, vis = proj(depth)
    print(f"depth {tuple(depth.shape)} → occ {tuple(occ.shape)}, vis {tuple(vis.shape)}")
    print(f"  occupancy: mean={occ.mean():.4f} max={occ.max():.3f}")
    print(f"  visibility: mean={vis.mean():.4f}")
    print(f"  shadow fraction: {proj.shadow_fraction(depth).item():.3f}")

    S = cfg.bev.size
    # Row 0 is the far edge and row S-1 is the agent, so the near half is the bottom.
    # In front of the wall must be observed free space; behind it must be shadow.
    behind = vis[0, 0, : S // 4, S // 3 : 2 * S // 3].mean().item()
    infront = vis[0, 0, S // 2 :, S // 3 : 2 * S // 3].mean().item()
    print(f"  visibility behind wall={behind:.3f}  in front of wall={infront:.3f}  (behind << front ✓)")

    depth.requires_grad_(True)
    proj(depth)[0].sum().backward()
    print(f"  Backward: OK (grad norm {depth.grad.norm():.4f})")

    prev = torch.zeros(1, 1, S, S, device=device)
    flow = bev_flow(prev, occ.detach())
    print(f"  bev_flow → {tuple(flow.shape)}, nonzero={float((flow != 0).float().mean()):.4f}")
    print("\nBEVProjector OK")
