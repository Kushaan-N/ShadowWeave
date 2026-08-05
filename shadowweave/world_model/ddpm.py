"""Conditional denoising diffusion over future BEV occupancy.

Why diffusion here, rather than as a label
------------------------------------------
The deterministic U-Net is trained with per-cell BCE, which is a mean-seeking loss.
Where the future is genuinely multimodal it converges to the conditional *mean* and
hedges — and this system has an unusually clean source of multimodality: cells the
sensor cannot see. Behind an obstacle there may or may not be something, and a person
in shadow may emerge left or right. Averaging those futures produces a uniform grey
smear over exactly the region the project exists to reason about.

A diffusion model samples from p(future | observation) instead of regressing its mean,
so each sample is a *coherent* hypothesis about what is behind the occlusion. That
also makes the shadow mask measurable rather than rhetorical: samples should agree
where the agent can see and disagree where it cannot. ``sample_statistics`` reports
exactly that ratio.

Formulation
-----------
Standard DDPM (Ho et al. 2020) with a cosine schedule (Nichol & Dhariwal 2021),
epsilon-parameterised, conditioned by channel-concatenation on the observed BEV stack.
Sampling uses DDIM (Song et al. 2021) so the step count is a runtime knob — the world
model runs at planning rate, not in the 20Hz audio loop, but the budget is not
unlimited either.

Occupancy is binary, and Gaussian diffusion assumes continuous data, so targets are
rescaled {0,1} -> [-1,1] and mapped back on sampling. A discrete (D3PM-style) variant
would be more principled; this is the standard, well-tested choice and keeps the
backbone shared with the deterministic model for a clean ablation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .unet import CondUNet, n_input_channels


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule. Linear schedules destroy low-resolution structure too early,
    which matters here because a 96x96 occupancy grid is mostly low-frequency."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()


class DiffusionWorldModel(nn.Module):
    """Conditional DDPM predicting future egocentric BEV occupancy.

    Primary methods:
        loss(cond, target)          -> scalar training loss
        predict(cond)               -> (B, T, S, S) probabilities (posterior mean)
        sample(cond, n_samples)     -> (n, B, T, S, S) distinct futures
        sample_statistics(cond, n)  -> per-cell mean and std across samples
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_horizons = len(cfg.world_model.prediction_horizons)
        self.cond_channels = n_input_channels(cfg)
        d = cfg.world_model.diffusion

        self.timesteps = int(d.timesteps)
        self.sample_steps = int(d.sample_steps)
        self.eta = float(d.ddim_eta)
        self.default_samples = int(d.predict_samples)

        # The backbone sees the noisy target and the conditioning side by side.
        self.net = CondUNet(
            in_channels=self.n_horizons + self.cond_channels,
            out_channels=self.n_horizons,
            base_channels=cfg.world_model.base_channels,
            time_dim=cfg.world_model.base_channels * 4,
        )

        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt(), persistent=False
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pm1(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

    @staticmethod
    def _from_pm1(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    def loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Epsilon-prediction MSE. Interface matches the deterministic model so the
        training loop does not need to know which architecture it is driving."""
        x0 = self._to_pm1(target)
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps = self.net(torch.cat([x_t, cond], dim=1), t)
        return F.mse_loss(eps, noise)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Logit-space view of the posterior mean, so callers expecting the
        deterministic model's logits keep working."""
        probs = self.predict(cond)
        return torch.log(probs.clamp(1e-6, 1 - 1e-6) / (1 - probs).clamp(1e-6, 1 - 1e-6))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ddim(self, cond: torch.Tensor, steps: int, generator=None) -> torch.Tensor:
        b = cond.shape[0]
        shape = (b, self.n_horizons, cond.shape[-2], cond.shape[-1])
        x = torch.randn(shape, device=cond.device, dtype=cond.dtype, generator=generator)

        times = torch.linspace(self.timesteps - 1, 0, steps, device=cond.device).long()
        for i, t in enumerate(times):
            t_batch = t.repeat(b)
            eps = self.net(torch.cat([x, cond], dim=1), t_batch)

            a_t = self.alphas_cumprod[t]
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1.0, 1.0)

            if i == len(times) - 1:
                x = x0
                break
            a_prev = self.alphas_cumprod[times[i + 1]]
            # eta=0 is deterministic DDIM; >0 reintroduces stochasticity per step.
            sigma = self.eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp_min(0).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).clamp_min(0).sqrt() * eps
            x = a_prev.sqrt() * x0 + dir_xt
            if self.eta > 0:
                x = x + sigma * torch.randn(shape, device=cond.device, generator=generator)
        return self._from_pm1(x)

    @torch.no_grad()
    def sample(
        self, cond: torch.Tensor, n_samples: int = 1, steps: Optional[int] = None,
        generator=None,
    ) -> torch.Tensor:
        """(n_samples, B, T, S, S) — each entry a coherent hypothesis, not an average."""
        steps = steps or self.sample_steps
        return torch.stack([self._ddim(cond, steps, generator) for _ in range(n_samples)])

    @torch.no_grad()
    def predict(self, cond: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        """Posterior mean over samples — the drop-in replacement for the
        deterministic model's sigmoid output."""
        n = n_samples or self.default_samples
        return self.sample(cond, n).mean(dim=0)

    @torch.no_grad()
    def sample_statistics(
        self, cond: torch.Tensor, n_samples: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(mean, std) across samples.

        The std map is the quantity the whole project is about: it should be near zero
        where the agent can see and large where it cannot.
        """
        s = self.sample(cond, n_samples)
        return s.mean(dim=0), s.std(dim=0)


if __name__ == "__main__":
    from ..utils import count_parameters, get_device, load_config

    cfg = load_config()
    cfg.world_model.architecture = "diffusion"
    cfg.world_model.base_channels = 16
    cfg.world_model.diffusion.sample_steps = 8
    device = get_device()
    print(f"DiffusionWorldModel demo on {device}")

    S = cfg.bev.size
    C = n_input_channels(cfg)
    T = len(cfg.world_model.prediction_horizons)

    model = DiffusionWorldModel(cfg).to(device)
    print(f"  Parameters: {count_parameters(model)/1e6:.1f}M")
    print(f"  timesteps={model.timesteps} sample_steps={model.sample_steps} eta={model.eta}")

    cond = torch.rand(2, C, S, S, device=device)
    target = (torch.rand(2, T, S, S, device=device) > 0.93).float()

    loss = model.loss(cond, target)
    loss.backward()
    print(f"  training loss (eps-MSE): {loss.item():.4f}, backward: OK")

    probs = model.predict(cond, n_samples=2)
    print(f"  predict → {tuple(probs.shape)}, range [{probs.min():.3f}, {probs.max():.3f}]")
    assert probs.shape == (2, T, S, S)
    assert 0.0 <= float(probs.min()) and float(probs.max()) <= 1.0

    # The point of the whole exercise: two samples must not be identical.
    s = model.sample(cond, n_samples=3)
    spread = float((s[0] - s[1]).abs().mean())
    print(f"  sample → {tuple(s.shape)}, mean |s0 − s1| = {spread:.4f}  (must be > 0)")
    assert spread > 0, "samples collapsed — this would be a deterministic model"

    mean, std = model.sample_statistics(cond, n_samples=3)
    print(f"  sample_statistics → mean {tuple(mean.shape)}, std mean={float(std.mean()):.4f}")
    print("DiffusionWorldModel OK")
