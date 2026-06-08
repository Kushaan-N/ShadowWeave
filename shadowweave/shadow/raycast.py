"""Core shadow-ray propagation — differentiable raycast producing per-zone uncertainty.

Two forward paths:

  forward_from_depth(depth_map)              — fast geometric path (no occupancy CNN needed).
      Samples the depth map at each ray's image-space column, computes shadow fraction.
      Used for: synthetic data generation, pipeline testing, local-agent inference.

  forward(depth_map, occupancy_volume)       — 3D volumetric marching path.
      Marches rays through the CNN latent occupancy volume in perspective-correct
      frustum coordinates. Used for: joint training with OccupancyField.

Both are differentiable end-to-end and share the same learnable zone_assign matrix.
Target throughput: ≥15Hz on a single GPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ShadowRaycaster(nn.Module):
    """Differentiable shadow-ray caster.

    Primary methods:
        forward_from_depth(depth_map) -> uncertainty_grid   ← use this first
        forward(depth_map, occupancy) -> uncertainty_grid   ← for joint training
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_rays     = cfg.shadow.num_rays
        self.march_steps  = cfg.shadow.march_steps
        self.threshold    = cfg.shadow.density_threshold
        self.num_cells    = cfg.shadow.grid_cells  # 9
        self.fov_deg      = cfg.sim.camera_fov     # 90

        # learnable zone assignment matrix (rays → zones), softmax-normalised
        self.zone_assign = nn.Parameter(
            torch.randn(self.num_rays, self.num_cells) * 0.1
        )

        # initialise zone_assign so rays are biased toward their natural azimuth zone
        # (soft prior: leftmost rays → zone 0, rightmost → zone 8)
        with torch.no_grad():
            for r in range(self.num_rays):
                zone_idx = int(r / self.num_rays * self.num_cells)
                self.zone_assign[r, zone_idx] += 2.0

    # ------------------------------------------------------------------
    # Fast geometric path (2-D depth-map based, perspective-correct)
    # ------------------------------------------------------------------

    def forward_from_depth(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Geometric shadow from depth map — no 3D occupancy field needed.

        Args:
            depth_map: (B, 1, H, W) float32 in [0, 1]

        Returns:
            uncertainty_grid: (B, 9) float32 in [0, 1]
                High value = close obstacle → large shadow behind it.
        """
        B, _, H, W = depth_map.shape
        device = depth_map.device

        fov_rad   = math.radians(self.fov_deg)
        half_fov  = fov_rad / 2.0

        # Azimuth angles for num_rays rays spread across the horizontal FOV
        azimuths = torch.linspace(-half_fov, half_fov, self.num_rays, device=device)

        # Normalised image-space x ∈ [−1, 1] via pinhole projection
        norm_x = torch.tan(azimuths) / math.tan(half_fov)
        norm_y = torch.zeros_like(norm_x)  # horizontal equator

        # Sample depth map at each ray's pixel column (bilinear, border padding)
        # grid: (B, 1, num_rays, 2)
        grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, self.num_rays, 2).expand(B, -1, -1, -1)
        sampled_depth = F.grid_sample(
            depth_map, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(1).squeeze(1)  # (B, num_rays)

        # Shadow fraction = proportion of depth range hidden behind this surface
        # Close obstacle (depth≈0) → shadow ≈ 1.0; far/absent (depth≈1) → shadow ≈ 0.0
        ray_uncertainty = (1.0 - sampled_depth).clamp(0.0, 1.0)  # (B, num_rays)

        # dim=0: normalise over rays so each zone column sums to 1 →
        # result is a proper weighted average of ray uncertainties per zone
        zone_weights     = F.softmax(self.zone_assign, dim=0)    # (num_rays, 9)
        uncertainty_grid = torch.matmul(ray_uncertainty, zone_weights)
        return uncertainty_grid.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # 3-D volumetric path (for joint training with OccupancyField)
    # ------------------------------------------------------------------

    def _perspective_ray_directions(self, device: torch.device) -> torch.Tensor:
        """Return (num_rays, 3) unit vectors in perspective-correct frustum space.

        Rays are distributed across the horizontal FOV.
        dir = (tan(azimuth), 0, 1) normalised — the z=1 forward convention
        maps directly to the perspective occupancy grid's depth axis.
        """
        fov_rad  = math.radians(self.fov_deg)
        half_fov = fov_rad / 2.0
        azimuths = torch.linspace(-half_fov, half_fov, self.num_rays, device=device)
        dx = torch.tan(azimuths)
        dy = torch.zeros_like(dx)
        dz = torch.ones_like(dx)
        dirs = torch.stack([dx, dy, dz], dim=-1)
        return F.normalize(dirs, dim=-1)  # (num_rays, 3)

    def _march(self, dirs: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        """March rays through the occupancy frustum volume.

        The occupancy volume (B, 1, D, H, W) is in frustum space:
          D axis → normalised depth [−1, +1] (near → far)
          H axis → normalised elevation [−1, +1]
          W axis → normalised azimuth   [−1, +1]

        Rays march from (norm_x, 0, −1) to (norm_x, 0, +1), where norm_x is
        determined by the ray's azimuth angle. Each march step advances along
        the depth (z) axis, keeping x and y fixed (perspective-correct).

        Returns:
            term_depth: (B, num_rays) normalised termination depth in [0, 1]
        """
        B = occupancy.shape[0]
        R = dirs.shape[0]
        device = occupancy.device

        fov_rad  = math.radians(self.fov_deg)
        half_fov = fov_rad / 2.0

        # Azimuth angles → normalised x position in [−1, +1]
        azimuths = torch.linspace(-half_fov, half_fov, R, device=device)
        norm_x   = torch.tan(azimuths) / math.tan(half_fov)  # (R,)

        # March step positions along the depth axis: z ∈ [−1, +1]
        z_vals = torch.linspace(-1.0, 1.0, self.march_steps, device=device)  # (D_steps,)

        # Build sampling grid: (R, D_steps, 3) with (x, y=0, z)
        x_grid = norm_x.view(R, 1).expand(R, self.march_steps)  # (R, D_steps)
        y_grid = torch.zeros_like(x_grid)
        z_grid = z_vals.view(1, self.march_steps).expand(R, self.march_steps)
        points = torch.stack([x_grid, y_grid, z_grid], dim=-1)  # (R, D_steps, 3)

        # Expand for batch
        pts_flat = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(
            B, R * self.march_steps, 1, 1, 3
        )

        # grid_sampler_3d_backward not on MPS → run on CPU, move result back
        cpu_occ  = occupancy.cpu()
        cpu_pts  = pts_flat.cpu()
        sampled  = F.grid_sample(
            cpu_occ, cpu_pts, mode="bilinear", padding_mode="zeros", align_corners=True
        ).to(device)  # (B, 1, R*D_steps, 1, 1)
        sampled  = sampled.reshape(B, R, self.march_steps)

        # Normalised depth values ∈ [0, 1] for the weighted sum
        t_vals = (z_vals + 1.0) / 2.0  # [−1,+1] → [0,1]

        # Soft volumetric termination (differentiable NeRF-style)
        weights = sampled * torch.cumprod(
            torch.cat([
                torch.ones(B, R, 1, device=device),
                1.0 - sampled[:, :, :-1],
            ], dim=-1),
            dim=-1,
        )
        term_depth = (weights * t_vals.view(1, 1, -1)).sum(dim=-1)  # (B, R)
        return term_depth

    def forward(self, depth_map: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        """Volumetric path — use for joint training with OccupancyField CNN.

        Args:
            depth_map: (B, 1, H, W) float32  — used only for device/shape
            occupancy: (B, 1, D, H, W) float32 from OccupancyField

        Returns:
            uncertainty_grid: (B, 9) float32
        """
        device = depth_map.device
        dirs       = self._perspective_ray_directions(device)
        term_depth = self._march(dirs, occupancy)              # (B, R)

        ray_uncertainty  = 1.0 - term_depth
        zone_weights     = F.softmax(self.zone_assign, dim=0)  # normalise over rays
        return torch.matmul(ray_uncertainty, zone_weights).clamp(0.0, 1.0)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib, time
    from .occupancy import GeometricOccupancyField, OccupancyField

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ShadowRaycaster demo on {device}\n")

    h, w = cfg.depth.output_h, cfg.depth.output_w
    raycaster = ShadowRaycaster(cfg).to(device)
    zone_labels = ["L-far", "L    ", "L-near", "FL   ", "F    ", "FR   ", "R-near", "R    ", "R-far"]

    # ── Geometric fast path ─────────────────────────────────────────
    print("── forward_from_depth (geometric fast path) ──")
    depth_map = torch.ones(1, 1, h, w, device=device) * 0.85
    depth_map[0, 0, :, : w // 3]  = 0.15   # close wall on left third
    depth_map[0, 0, 30:60, 55:75] = 0.25   # box in centre-left

    grid = raycaster.forward_from_depth(depth_map)
    vals = grid[0].detach().cpu().numpy()
    for label, v in zip(zone_labels, vals):
        bar = "█" * int(v * 30)
        print(f"  {label}: {v:.3f}  {bar}")

    # throughput
    N = 50
    t0 = time.time()
    for _ in range(N):
        g = raycaster.forward_from_depth(depth_map)
        if device == "mps": torch.mps.synchronize()
    hz = N / (time.time() - t0)
    print(f"\n  Throughput: {hz:.0f} Hz (target ≥15 Hz)")

    # backward
    grid.sum().backward()
    print("  Backward: OK")

    # ── Volumetric path (CNN occupancy) ─────────────────────────────
    print("\n── forward (volumetric, CNN OccupancyField) ──")
    occ_field = OccupancyField(cfg).to(device)
    depth_map2 = depth_map.detach().requires_grad_(False)
    occ_vol    = occ_field(depth_map2)
    grid2      = raycaster.forward(depth_map2, occ_vol)
    vals2      = grid2[0].detach().cpu().numpy()
    for label, v in zip(zone_labels, vals2):
        bar = "█" * int(v * 30)
        print(f"  {label}: {v:.3f}  {bar}")
    grid2.sum().backward()
    print("  Backward: OK")
    print("\nShadowRaycaster OK")
