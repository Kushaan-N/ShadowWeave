"""Physics-informed world model — U-Net backbone predicting future occupancy grids.

Architecture: ~50M param U-Net conditioned on current shadow map + optical flow velocities.
Fallback: ConvLSTM (same interface) if diffusion training does not converge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModel(nn.Module):
    """Predicts future occupancy grids from current shadow map + velocity field.

    Primary method: ``forward(shadow_map, velocity) -> predictions``

    Args:
        shadow_map: (B, 1, H, W) float32, current uncertainty map
        velocity:   (B, 2, H, W) float32, optical flow velocity field (u, v)

    Returns:
        predictions: (B, T, H, W) float32, predicted occupancy at each horizon in cfg
                     T = len(cfg.world_model.prediction_horizons)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.world_model.base_channels
        T = len(cfg.world_model.prediction_horizons)
        in_ch = 3  # shadow_map (1) + velocity (2)

        # encoder
        self.enc1 = DoubleConv(in_ch, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)

        # bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(c, T, 1)   # predict all horizons simultaneously

    def forward(self, shadow_map: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        x = torch.cat([shadow_map, velocity], dim=1)  # (B, 3, H, W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))  # (B, T, H, W)


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell — fallback if U-Net diffusion does not converge."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined).chunk(4, dim=1)
        i, f, g, o = (torch.sigmoid(g) for g in gates[:3]), torch.tanh(gates[2])
        i, f, o = [torch.sigmoid(g) for g in [gates[0], gates[1], gates[3]]]
        g = torch.tanh(gates[2])
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class WorldModelConvLSTM(nn.Module):
    """ConvLSTM fallback world model — same interface as WorldModel."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.world_model.base_channels
        T = len(cfg.world_model.prediction_horizons)
        self.cell = ConvLSTMCell(3, c)
        self.head = nn.Conv2d(c, T, 1)

    def forward(self, shadow_map: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        B, _, H, W = shadow_map.shape
        x = torch.cat([shadow_map, velocity], dim=1)
        h = torch.zeros(B, self.cell.hidden_ch, H, W, device=x.device)
        c = torch.zeros_like(h)
        h, c = self.cell(x, h, c)
        return torch.sigmoid(self.head(h))


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"WorldModel demo on {device}")

    h, w = cfg.depth.output_h, cfg.depth.output_w
    B, T = 2, len(cfg.world_model.prediction_horizons)

    model = WorldModel(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {param_count:.1f}M (target ~50M)")

    shadow = torch.rand(B, 1, h, w, device=device)
    vel    = torch.rand(B, 2, h, w, device=device)
    preds  = model(shadow, vel)
    print(f"  shadow: {shadow.shape}, vel: {vel.shape} → preds: {preds.shape}  (expected [{B},{T},{h},{w}])")

    loss = F.binary_cross_entropy(preds, torch.rand_like(preds))
    loss.backward()
    print(f"  BCE loss: {loss.item():.4f}, backward: OK")
    print("WorldModel OK")
