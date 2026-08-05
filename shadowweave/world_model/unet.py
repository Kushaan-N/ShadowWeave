"""Deterministic world model — U-Net backbone predicting future BEV occupancy.

This module was previously named ``diffusion.py``, which it never was: it is a plain
U-Net trained with BCE, plus a ConvLSTM fallback. The actual denoising diffusion model
now lives in ``ddpm.py`` and reuses ``CondUNet`` from here as its backbone.

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def n_input_channels(cfg: DictConfig) -> int:
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
        # Largest divisor of out_ch up to 8 — min(8, out_ch) crashes GroupNorm for
        # any channel count not divisible by 8 (e.g. base_channels=12).
        groups = next(g for g in (8, 4, 2, 1) if out_ch % g == 0)
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
        in_ch = n_input_channels(cfg)

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
        h, w = bev_stack.shape[-2], bev_stack.shape[-1]
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(
                f"spatial dims must be divisible by 16 for the 4-level U-Net, got {h}x{w}"
            )

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

    def loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted BCE plus a smoothness penalty across horizons.

        Shares a signature with DiffusionWorldModel.loss so the training loop is
        architecture-agnostic.
        """
        logits = self.forward(cond)
        pw = torch.tensor(self.cfg.world_model.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        probs = torch.sigmoid(logits)
        if probs.shape[1] > 1:
            # Occupancy should evolve smoothly between consecutive horizons.
            reg = (probs[:, 1:] - probs[:, :-1]).abs().mean()
        else:
            # mean() of an empty slice is NaN and would poison the whole run.
            reg = probs.new_zeros(())
        return (self.cfg.world_model.bce_weight * bce
                + self.cfg.world_model.physics_reg_weight * reg)


class TimestepEmbedding(nn.Module):
    """Sinusoidal position encoding of the diffusion step, then an MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class TimestepDoubleConv(nn.Module):
    """DoubleConv with the timestep embedding injected as a per-channel bias.

    The network has to behave differently at high noise (coarse layout) than at low
    noise (crisp edges), so it must be told which step it is denoising.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        groups = next(g for g in (8, 4, 2, 1) if out_ch % g == 0)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.emb = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.emb(temb)[:, :, None, None]
        return self.act(self.norm2(self.conv2(h)))


class CondUNet(nn.Module):
    """Timestep-conditioned U-Net used as the diffusion backbone.

    Same 4-level shape as WorldModel so the deterministic and diffusion variants are a
    fair ablation rather than a comparison of two different capacities.
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int,
                 time_dim: int) -> None:
        super().__init__()
        c = base_channels
        self.time = TimestepEmbedding(time_dim)

        self.enc1 = TimestepDoubleConv(in_channels, c, time_dim)
        self.enc2 = TimestepDoubleConv(c, c * 2, time_dim)
        self.enc3 = TimestepDoubleConv(c * 2, c * 4, time_dim)
        self.enc4 = TimestepDoubleConv(c * 4, c * 8, time_dim)
        self.bottleneck = TimestepDoubleConv(c * 8, c * 16, time_dim)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = TimestepDoubleConv(c * 16, c * 8, time_dim)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = TimestepDoubleConv(c * 8, c * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = TimestepDoubleConv(c * 4, c * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = TimestepDoubleConv(c * 2, c, time_dim)

        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(
                f"spatial dims must be divisible by 16 for the 4-level U-Net, got {h}x{w}"
            )
        temb = self.time(t)

        e1 = self.enc1(x, temb)
        e2 = self.enc2(self.pool(e1), temb)
        e3 = self.enc3(self.pool(e2), temb)
        e4 = self.enc4(self.pool(e3), temb)
        b = self.bottleneck(self.pool(e4), temb)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), temb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), temb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), temb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), temb)
        return self.head(d1)


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
        self.cell = ConvLSTMCell(n_input_channels(cfg), c)
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

    loss = WorldModel.loss


def build_world_model(cfg: DictConfig) -> nn.Module:
    arch = cfg.world_model.architecture
    if arch == "unet":
        return WorldModel(cfg)
    if arch == "convlstm":
        return WorldModelConvLSTM(cfg)
    if arch == "diffusion":
        from .ddpm import DiffusionWorldModel  # imported lazily to avoid a cycle
        return DiffusionWorldModel(cfg)
    raise ValueError(f"Unknown world_model.architecture: {arch}")


if __name__ == "__main__":
    import torch.nn.functional as F

    from ..utils import count_parameters, get_device, load_config

    cfg = load_config()
    device = get_device()
    print(f"WorldModel demo on {device}")

    S = cfg.bev.size
    B, T, C = 2, len(cfg.world_model.prediction_horizons), n_input_channels(cfg)

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
