"""Occupancy field implementations.

GeometricOccupancyField: deterministic depth→voxel projection, no training needed.
    Use this for bootstrapping the pipeline and generating synthetic training data.

OccupancyField: learned 3D CNN — replaces the geometric version after joint training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class GeometricOccupancyField(nn.Module):
    """Deterministic depth-map → voxel occupancy.

    A voxel at normalised depth d_vox is "occupied" (shadowed) if d_vox >= depth_surface.
    This directly encodes: anything behind the visible surface is in shadow.

    No parameters — no training required.

    Primary method: ``forward(depth_map) -> occupancy_volume``
    Input:  (B, 1, H, W) float32 depth in [0, 1]
    Output: (B, 1, D, H, W) float32 binary voxel occupancy
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.D = cfg.shadow.march_steps

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth_map.shape
        # voxel slab depths along the ray direction, (D,) in [0, 1]
        slabs = torch.linspace(0, 1, self.D, device=depth_map.device)
        # broadcast: slab (1,1,D,1,1) vs surface (B,1,1,H,W)
        occupied = (slabs.view(1, 1, self.D, 1, 1) >= depth_map.unsqueeze(2)).float()
        return occupied  # (B, 1, D, H, W)


class OccupancyField(nn.Module):
    """Learned 3D CNN occupancy field — encodes depth map into voxel density volume.

    Replaces GeometricOccupancyField for joint end-to-end training.
    Initialised with strongly negative bias so it starts near-empty (avoids the
    all-ones saturation that makes every zone look uncertain from the start).

    Primary method: ``forward(depth_map) -> occupancy_volume``
    Input:  (B, 1, H, W) float32 depth tensor
    Output: (B, 1, D, H, W) float32 voxel density in [0, 1]
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.shadow.march_steps

        self.lift = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, d, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 1, 3, padding=1),
        )
        # initialise final conv bias to -5 → sigmoid ≈ 0.007 (nearly empty space)
        # prevents all-ones saturation at the start of training
        nn.init.constant_(self.refine[-1].bias, -5.0)

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        vol = self.lift(depth_map)        # (B, D, H, W)
        vol = vol.unsqueeze(1)           # (B, 1, D, H, W)
        return torch.sigmoid(self.refine(vol))


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)
    h, w = cfg.depth.output_h, cfg.depth.output_w
    D = cfg.shadow.march_steps

    # Geometric field: obstacle close on left, open on right
    depth = torch.ones(1, 1, h, w) * 0.8
    depth[0, 0, :, : w // 3] = 0.2   # close obstacle on the left third

    geo = GeometricOccupancyField(cfg)
    vol = geo(depth)
    print(f"GeometricOccupancyField: {depth.shape} → {vol.shape}")
    left_occ  = vol[0, 0, :, :, : w // 3].mean().item()
    right_occ = vol[0, 0, :, :, 2 * w // 3 :].mean().item()
    print(f"  left col occupancy={left_occ:.3f}  right col occupancy={right_occ:.3f}  (left >> right ✓)")

    cnn = OccupancyField(cfg)
    out = cnn(depth)
    print(f"OccupancyField (CNN) initial mean density: {out.mean().item():.4f}  (expect ~0.007)")
    print("OccupancyField OK")
