"""Physics-informed world model — U-Net backbone predicting future BEV occupancy.

Input is the egocentric BEV stack (observed occupancy, visibility/shadow mask, and
2-channel BEV flow); output is occupancy at each prediction horizon in the SAME
egocentric frame. Feeding the shadow mask in explicitly is what lets the model learn
that unobserved cells are where prediction actually matters.

Outputs are LOGITS. The previous version applied sigmoid inside forward() and the
training loop then called binary_cross_entropy on the result, which is numerically
fragile and incompatible with autocast; BCEWithLogits is both safer and lets us apply
a positive-class weight (BEV targets are only ~7% positive).

Fallback: ConvLSTM (same interface) if the U-Net does not converge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


def _n_input_channels(cfg: DictConfig) -> int:
    ch = list(cfg.bev.input_channels)
    n = 0
    n += 1 if "occupancy" in ch else 0
    n += 1 if "visibility" in ch else 0
    n += 2 if "flow" in ch else 0
    if n == 0:
        raise ValueError("bev.input_channels must select at least one channel")
    return n


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(groups, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(groups, out_ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModel(nn.Module):
    """Predicts future egocentric BEV occupancy from the current BEV observation.

    Primary method: ``forward(bev_stack) -> logits``

    Args:
        bev_stack: (B, C, S, S) float32 — occupancy / visibility / flow, per cfg
    Returns:
        logits: (B, T, S, S) float32, T = len(cfg.world_model.prediction_horizons)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.world_model.base_channels
        T = len(cfg.world_model.prediction_horizons)
        in_ch = _n_input_channels(cfg)

        self.enc1 = DoubleConv(in_ch, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(c, T, 1)

    def forward(self, bev_stack: torch.Tensor) -> torch.Tensor:
        # A 96x96 grid survives four 2x downsamples only because 96 = 32 * 3; assert
        # rather than let a mis-set bev.size silently produce a shape error deep in
        # the decoder skip connections.
        s = bev_stack.shape[-1]
        if s % 16 != 0:
            raise ValueError(f"bev.size must be divisible by 16 for the 4-level U-Net, got {s}")

        e1 = self.enc1(bev_stack)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    @torch.no_grad()
    def predict(self, bev_stack: torch.Tensor) -> torch.Tensor:
        """Probabilities in [0, 1] — the inference-time entry point."""
        return torch.sigmoid(self.forward(bev_stack))


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell — fallback if the U-Net does not converge."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i = torch.sigmoid(gates[0])
        f = torch.sigmoid(gates[1])
        g = torch.tanh(gates[2])
        o = torch.sigmoid(gates[3])
        c_next = f * c + i * g
        return o * torch.tanh(c_next), c_next


class WorldModelConvLSTM(nn.Module):
    """ConvLSTM fallback world model — same interface as WorldModel."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.world_model.base_channels
        T = len(cfg.world_model.prediction_horizons)
        self.cell = ConvLSTMCell(_n_input_channels(cfg), c)
        self.head = nn.Conv2d(c, T, 1)

    def forward(self, bev_stack: torch.Tensor) -> torch.Tensor:
        B, _, H, W = bev_stack.shape
        h = torch.zeros(B, self.cell.hidden_ch, H, W, device=bev_stack.device, dtype=bev_stack.dtype)
        c = torch.zeros_like(h)
        h, c = self.cell(bev_stack, h, c)
        return self.head(h)

    @torch.no_grad()
    def predict(self, bev_stack: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(bev_stack))


def build_world_model(cfg: DictConfig) -> nn.Module:
    arch = cfg.world_model.architecture
    if arch == "unet":
        return WorldModel(cfg)
    if arch == "convlstm":
        return WorldModelConvLSTM(cfg)
    raise ValueError(f"Unknown world_model.architecture: {arch}")


if __name__ == "__main__":
    import torch.nn.functional as F

    from ..utils import count_parameters, get_device, load_config

    cfg = load_config()
    device = get_device()
    print(f"WorldModel demo on {device}")

    S = cfg.bev.size
    B, T, C = 2, len(cfg.world_model.prediction_horizons), _n_input_channels(cfg)

    for arch in ["unet", "convlstm"]:
        cfg.world_model.architecture = arch
        model = build_world_model(cfg).to(device)
        print(f"\n── {arch} ──")
        print(f"  Parameters: {count_parameters(model)/1e6:.1f}M")

        x = torch.rand(B, C, S, S, device=device)
        logits = model(x)
        print(f"  in {tuple(x.shape)} → logits {tuple(logits.shape)}  (expected [{B},{T},{S},{S}])")

        target = (torch.rand_like(logits) > 0.93).float()
        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=torch.tensor(cfg.world_model.pos_weight, device=device)
        )
        loss.backward()
        print(f"  BCEWithLogits: {loss.item():.4f}, backward: OK")
        probs = model.predict(x)
        assert 0.0 <= probs.min() and probs.max() <= 1.0
        print(f"  predict() range: [{probs.min():.3f}, {probs.max():.3f}]")

    print("\nWorldModel OK")
