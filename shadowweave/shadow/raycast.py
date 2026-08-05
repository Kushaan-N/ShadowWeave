"""Core shadow-ray propagation — differentiable raycast producing per-zone uncertainty.

Two forward paths:

  forward_from_depth(depth_map)              — fast geometric path (no occupancy CNN needed).
      Samples the depth map along each ray over a band of elevations, computes the
      fraction of the ray's range that lies behind the first visible surface.

  forward(depth_map, occupancy_volume)       — 3D volumetric marching path.
      Marches rays through the CNN latent occupancy volume in perspective-correct
      frustum coordinates, accumulating NeRF-style transmittance.

Both are differentiable end-to-end and share the same learnable zone_assign matrix.

Invariant (enforced by tests/test_raycast.py): a ray that never terminates casts no
shadow. Empty space must read as *low* uncertainty, not high — the previous
volumetric path inverted this and reported 0.84 uniformly for an empty volume.

Invariant: the 9 zones must actually discriminate. zone_assign is softmaxed over
*zones* (dim=1) with a temperature, so each ray distributes its contribution across
zones and each zone is a genuine local average. Normalising over rays (dim=0) with a
weak +2.0 bias left ~50% of every zone's weight on out-of-zone rays and collapsed all
9 zones to the same value.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..utils import sanitize_depth, supports_grid_sample_3d


class ShadowRaycaster(nn.Module):
    """Differentiable shadow-ray caster.

    Primary methods:
        forward_from_depth(depth_map) -> uncertainty_grid   ← use this first
        forward(depth_map, occupancy) -> uncertainty_grid   ← for joint training
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_rays    = cfg.shadow.num_rays
        self.march_steps = cfg.shadow.march_steps
        self.threshold   = cfg.shadow.density_threshold
        self.num_cells   = cfg.shadow.grid_cells
        self.fov_deg     = cfg.sim.camera_fov
        self.n_elevation = cfg.shadow.elevation_samples
        self.elev_extent = cfg.shadow.elevation_extent
        self.max_range   = cfg.shadow.max_range_m
        self.zone_temp   = cfg.shadow.zone_temperature

        # Learnable ray→zone assignment, softmaxed over zones (dim=1) at read time.
        # Init: each ray strongly prefers the zone its azimuth falls in, with a soft
        # tail into neighbours so gradients can move rays between zones.
        ray_idx  = torch.arange(self.num_rays, dtype=torch.float32)
        natural  = ray_idx / self.num_rays * self.num_cells - 0.5
        zone_idx = torch.arange(self.num_cells, dtype=torch.float32)
        logits   = -((natural[:, None] - zone_idx[None, :]) ** 2) / (2 * 0.75 ** 2)
        self.zone_assign = nn.Parameter(logits * cfg.shadow.zone_init_sharpness)

        # Ray geometry is fixed by the camera, so precompute it once as buffers
        # rather than rebuilding linspaces on every forward.
        half_fov = math.radians(self.fov_deg) / 2.0
        azimuths = torch.linspace(-half_fov, half_fov, self.num_rays)
        self.register_buffer("azimuths", azimuths, persistent=False)
        # Pinhole projection to normalised image x ∈ [−1, 1]
        self.register_buffer("norm_x", torch.tan(azimuths) / math.tan(half_fov), persistent=False)
        self.register_buffer(
            "norm_y", torch.linspace(-self.elev_extent, self.elev_extent, self.n_elevation),
            persistent=False,
        )
        self.register_buffer("z_vals", torch.linspace(-1.0, 1.0, self.march_steps), persistent=False)

    def zone_weights(self) -> torch.Tensor:
        """(num_rays, num_cells) — rows sum to 1, so every zone is a local average."""
        return F.softmax(self.zone_assign / self.zone_temp, dim=1)

    def _rays_to_zones(self, ray_uncertainty: torch.Tensor) -> torch.Tensor:
        """(B, R) ray values → (B, num_cells) zone means.

        Row-stochastic weights are a *sum* over rays, so divide by each zone's total
        mass to recover a mean; otherwise zone magnitude tracks ray count, not shadow.
        """
        w = self.zone_weights()                       # (R, C), rows sum to 1
        mass = w.sum(dim=0).clamp_min(1e-6)           # (C,)
        return (torch.matmul(ray_uncertainty, w) / mass).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Fast geometric path (2-D depth-map based, perspective-correct)
    # ------------------------------------------------------------------

    def forward_from_depth(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Geometric shadow from depth map — no 3D occupancy field needed.

        Args:
            depth_map: (B, 1, H, W) float32 in [0, 1], where 1.0 = max_range_m.

        Returns:
            uncertainty_grid: (B, num_cells) float32 in [0, 1].
                Value = fraction of the ray's range hidden behind the first surface,
                averaged over the elevation band. Close obstacle → large shadow.
        """
        # A NaN here would otherwise reach the audio output unflagged.
        depth_map = sanitize_depth(depth_map)
        B = depth_map.shape[0]
        R, E = self.num_rays, self.n_elevation

        # Sample a band of elevations, not just the equator. A single equator row
        # sits on the horizon and is dominated by the far wall / floor, which is
        # what saturated the grid to a uniform 0.993 on real sim frames.
        gx = self.norm_x.view(1, R).expand(E, R)
        gy = self.norm_y.view(E, 1).expand(E, R)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, E, R, 2)

        sampled = F.grid_sample(
            depth_map, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(1)  # (B, E, R)

        # Shadow fraction: everything beyond the first surface along this ray is
        # unobserved. depth is already normalised by max_range, so 1 − depth is the
        # occluded fraction of the ray's range.
        ray_shadow = (1.0 - sampled).clamp(0.0, 1.0).mean(dim=1)  # (B, R)
        return self._rays_to_zones(ray_shadow)

    # ------------------------------------------------------------------
    # 3-D volumetric path (for joint training with OccupancyField)
    # ------------------------------------------------------------------

    def _march(self, occupancy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """March rays through the occupancy frustum volume.

        The occupancy volume (B, 1, D, H, W) is in frustum space:
          D axis → normalised depth [−1, +1] (near → far)
          H axis → normalised elevation [−1, +1]
          W axis → normalised azimuth   [−1, +1]

        Returns:
            term_depth: (B, R) expected termination depth in [0, 1]
            opacity:    (B, R) total accumulated weight in [0, 1] — how much of the
                        ray was actually stopped by something
        """
        B = occupancy.shape[0]
        R, E, D = self.num_rays, self.n_elevation, self.march_steps
        device = occupancy.device

        x = self.norm_x.view(1, R, 1).expand(E, R, D)
        y = self.norm_y.view(E, 1, 1).expand(E, R, D)
        z = self.z_vals.view(1, 1, D).expand(E, R, D)
        points = torch.stack([x, y, z], dim=-1)                     # (E, R, D, 3)
        pts = points.unsqueeze(0).expand(B, E, R, D, 3)

        if supports_grid_sample_3d(device):
            sampled = F.grid_sample(
                occupancy, pts, mode="bilinear", padding_mode="zeros", align_corners=True
            )
        else:
            # MPS has no grid_sampler_3d kernel. Fall back to CPU only on MPS —
            # the old code took this round-trip unconditionally and throttled CUDA.
            sampled = F.grid_sample(
                occupancy.cpu(), pts.cpu(), mode="bilinear",
                padding_mode="zeros", align_corners=True,
            ).to(device)
        sampled = sampled.reshape(B, E, R, D).clamp(0.0, 1.0)

        # NeRF-style transmittance: weight_i = alpha_i * prod_{j<i}(1 - alpha_j)
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(sampled[..., :1]), 1.0 - sampled[..., :-1]], dim=-1),
            dim=-1,
        )
        weights = sampled * transmittance                            # (B, E, R, D)

        t_vals  = (self.z_vals + 1.0) / 2.0                          # [−1,1] → [0,1]
        opacity = weights.sum(dim=-1)                                # (B, E, R)
        # Expected depth conditioned on terminating; undefined for a ray that never
        # terminates, so normalise by opacity and let the caller gate on it.
        term_depth = (weights * t_vals.view(1, 1, 1, -1)).sum(dim=-1) / opacity.clamp_min(1e-6)

        return term_depth.mean(dim=1), opacity.mean(dim=1)           # (B, R) each

    def forward(self, depth_map: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        """Volumetric path — use for joint training with OccupancyField CNN.

        Args:
            depth_map: (B, 1, H, W) float32 — unused, kept for interface symmetry
            occupancy: (B, 1, D, H, W) float32 from OccupancyField

        Returns:
            uncertainty_grid: (B, num_cells) float32
        """
        term_depth, opacity = self._march(occupancy)

        # A ray that passes clean through casts NO shadow. Gating on opacity is the
        # fix for empty space previously reading as maximum uncertainty: with an
        # empty volume opacity → 0, so shadow → 0.
        ray_shadow = opacity * (1.0 - term_depth)
        return self._rays_to_zones(ray_shadow)


if __name__ == "__main__":
    import time

    from ..utils import get_device, load_config
    from .occupancy import GeometricOccupancyField, OccupancyField

    cfg = load_config()
    device = get_device()
    print(f"ShadowRaycaster demo on {device}\n")

    h, w = cfg.depth.output_h, cfg.depth.output_w
    raycaster = ShadowRaycaster(cfg).to(device)
    zone_labels = ["L-far", "L    ", "L-near", "FL   ", "F    ", "FR   ", "R-near", "R    ", "R-far"]

    def show(vals) -> None:
        for label, v in zip(zone_labels, vals):
            print(f"  {label}: {v:.3f}  {'█' * int(v * 30)}")

    print("── forward_from_depth (geometric fast path) ──")
    depth_map = torch.ones(1, 1, h, w, device=device) * 0.85
    depth_map[0, 0, :, : w // 3]  = 0.15   # close wall on left third
    depth_map[0, 0, 30:60, 55:75] = 0.25   # box in centre-left

    grid = raycaster.forward_from_depth(depth_map)
    vals = grid[0].detach().cpu().numpy()
    show(vals)
    print(f"  left/right contrast: {vals[0] / max(vals[8], 1e-6):.2f}x  (must be >> 1)")

    N = 50
    t0 = time.time()
    for _ in range(N):
        raycaster.forward_from_depth(depth_map)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  Throughput: {N / (time.time() - t0):.0f} Hz (target ≥15 Hz)")
    grid.sum().backward()
    print("  Backward: OK")

    print("\n── forward (volumetric, geometric occupancy) ──")
    geo = GeometricOccupancyField(cfg).to(device)
    grid2 = raycaster.forward(depth_map, geo(depth_map))
    show(grid2[0].detach().cpu().numpy())

    print("\n── forward (volumetric, empty volume) ──")
    empty = torch.zeros(1, 1, cfg.shadow.march_steps, h, w, device=device)
    grid3 = raycaster.forward(depth_map, empty)
    print(f"  max zone uncertainty on empty space: {grid3.max().item():.4f}  (must be ≈0)")

    print("\n── forward (volumetric, CNN OccupancyField) ──")
    occ_field = OccupancyField(cfg).to(device)
    grid4 = raycaster.forward(depth_map, occ_field(depth_map))
    grid4.sum().backward()
    print("  Backward: OK")
    print("\nShadowRaycaster OK")
