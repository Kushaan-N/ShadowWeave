"""Local reactive agent — MLP policy at 20Hz, trained with PPO.

Input:  current uncertainty vector (9,) + world model predictions (flattened)
Output: audio cue parameters per zone — direction, intensity, pitch (9×3 = 27 floats)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


class LocalAgent(nn.Module):
    """3-layer MLP reactive policy.

    Primary method: ``forward(obs) -> audio_params``
    Input:  obs (B, obs_dim) float32 — uncertainty grid + world model features
    Output: audio_params (B, 27) float32 — (direction, intensity, pitch) × 9 zones
    """

    def __init__(self, cfg: DictConfig, obs_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.agents.local.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d),       nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d),       nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, 27),      # 9 zones × 3 params
        )
        # separate value head for PPO critic
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim, d), nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Numpy in → numpy out, for the 20Hz control loop."""
        t = torch.from_numpy(obs).float().unsqueeze(0)
        return self.net(t).squeeze(0).numpy()


def compute_reward(
    collision: bool,
    path_efficiency: float,
    active_zones: int,
    cfg: DictConfig,
) -> float:
    """PPO reward function — see agent spec in CLAUDE.md."""
    r = 0.0
    if collision:
        r += cfg.agents.local.reward_collision_weight
    r += cfg.agents.local.reward_efficiency_weight * path_efficiency
    if active_zones > cfg.agents.local.max_simultaneous_cues:
        r += cfg.agents.local.reward_audio_clarity_weight * (active_zones - cfg.agents.local.max_simultaneous_cues)
    return r


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    T = len(cfg.world_model.prediction_horizons)
    # 9 current zones + T * 9 pooled world-model predictions (not raw H*W pixels)
    obs_dim = cfg.shadow.grid_cells + T * cfg.shadow.grid_cells

    print(f"LocalAgent obs_dim={obs_dim}")
    agent = LocalAgent(cfg, obs_dim)
    param_count = sum(p.numel() for p in agent.parameters()) / 1e6
    print(f"  Parameters: {param_count:.3f}M")

    dummy_obs = np.random.rand(obs_dim).astype(np.float32)
    audio_params = agent.act(dummy_obs)
    print(f"  obs: {dummy_obs.shape} → audio_params: {audio_params.shape}")
    print(f"  sample reward: {compute_reward(False, 0.8, 2, cfg):.2f}")
    print("LocalAgent OK")
