"""Local reactive agent - MLP policy at 20Hz, trained with PPO.

Input:  current uncertainty vector (9,) + world model zone predictions (T*9 floats)
Output: navigation actions - (forward_vel, strafe_vel, turn_rate); the environment
        clips each component to [-1, 1] before applying it.

Audio cues are computed downstream by CueMapper from the uncertainty grid; the agent
emits navigation only. The orchestrator used to treat this 3-vector as if it were a
27-dim audio parameter block, which is the contract break its own __main__ asserted
against and failed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

NAV_ACTION_DIM = 3


def observation_dim(cfg: DictConfig) -> int:
    """Single source of truth for the policy's observation width.

    run_eval computed this as ``grid_cells + T*H*W`` (65545) while the orchestrator
    fed ``grid_cells + T*grid_cells`` (45), so the policy raised a shape error the
    moment the two met.
    """
    n_zones = cfg.shadow.grid_cells
    T = len(cfg.world_model.prediction_horizons)
    return n_zones + T * n_zones


class LocalAgent(nn.Module):
    """MLP reactive policy with a value head for PPO.

    Primary method: forward(obs) -> nav_action
    Input:  obs (B, obs_dim) float32 - uncertainty grid + world model zone predictions
    Output: nav_action (B, 3) float32 - (forward_vel, strafe_vel, turn_rate), unbounded;
            the environment clips to [-1, 1] when applying the action.
    """

    def __init__(self, cfg: DictConfig, obs_dim: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim if obs_dim is not None else observation_dim(cfg)
        d = cfg.agents.local.hidden_dim
        n_layers = cfg.agents.local.num_layers

        layers: list[nn.Module] = [nn.Linear(self.obs_dim, d), nn.LayerNorm(d), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d), nn.LayerNorm(d), nn.GELU()]
        # No Tanh on the head: PPO's Gaussian is unsquashed, so a Tanh-bounded mean
        # combined with an unbounded sample only piled probability mass at ±1 and made
        # the deployed deterministic mean diverge from the optimised distribution. The
        # environment clips the action to [-1, 1] itself (mujoco_env._move_agent).
        layers += [nn.Linear(d, NAV_ACTION_DIM)]
        self.net = nn.Sequential(*layers)

        self.value_head = nn.Sequential(
            nn.Linear(self.obs_dim, d), nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Numpy in, numpy out, for the 20Hz control loop."""
        device = next(self.parameters()).device
        t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
        return self.net(t).squeeze(0).cpu().numpy()


def compute_reward(
    collision: bool,
    distance_moved: float,
    cfg: DictConfig,
    progress: float = 0.0,
    shadow_exposure: float = 0.0,
    audio_load: float = 0.0,
) -> float:
    """PPO reward.

    ``collision`` must be a real geometric overlap. train_rl.py previously derived it
    from ``occupancy.max() > 0.5``, and since walls were always marked 1.0 that was
    True on every step — the agent received a constant -10 and could not learn.

    ``audio_load`` is the fraction of zones loud enough to sound (>= audio.cue_threshold);
    reward_audio_clarity_weight is negative, so many zones sounding at once — hard to
    parse for a user navigating by ear — is penalised.
    """
    r = 0.0
    if collision:
        # A collision ends the episode (train_rl.step); it earns the penalty but not the
        # survival reward, so crashing is always worse than living.
        r += cfg.agents.local.reward_collision_weight
    else:
        r += cfg.agents.local.get("reward_alive_weight", 0.0)
    r += cfg.agents.local.reward_efficiency_weight * distance_moved
    r += cfg.agents.local.reward_progress_weight * progress
    r += cfg.agents.local.reward_shadow_weight * shadow_exposure
    r += cfg.agents.local.reward_audio_clarity_weight * audio_load
    return float(r)


if __name__ == "__main__":
    from ..utils import count_parameters, load_config

    cfg = load_config()
    obs_dim = observation_dim(cfg)
    print(f"LocalAgent obs_dim={obs_dim}  (9 zones + {len(cfg.world_model.prediction_horizons)}x9 horizons)")

    agent = LocalAgent(cfg)
    print(f"  Parameters: {count_parameters(agent)/1e6:.3f}M")

    dummy_obs = np.random.rand(obs_dim).astype(np.float32)
    nav_action = agent.act(dummy_obs)
    print(f"  obs: {dummy_obs.shape} -> nav_action: {nav_action.shape} (forward, strafe, turn)")
    assert nav_action.shape == (NAV_ACTION_DIM,)
    # The head is unbounded now (the env clips); just require finite output.
    assert np.all(np.isfinite(nav_action))

    print(f"  reward (clear, moving):    {compute_reward(False, 0.8, cfg, progress=0.1):.2f}")
    print(f"  reward (collision):        {compute_reward(True, 0.8, cfg, progress=0.1):.2f}")
    print(f"  reward (deep in shadow):   {compute_reward(False, 0.8, cfg, shadow_exposure=1.0):.2f}")
    print("LocalAgent OK")
