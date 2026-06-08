"""PPO/GRPO RL training entry point for the LocalAgent in MuJoCo."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from shadowweave.sim.mujoco_env import ShadowWeaveEnv
from shadowweave.shadow.occupancy import OccupancyField
from shadowweave.shadow.raycast import ShadowRaycaster
from shadowweave.world_model.diffusion import WorldModel
from shadowweave.agents.local_agent import LocalAgent, compute_reward


def make_obs(
    depth: np.ndarray,
    occ_field: OccupancyField,
    raycaster: ShadowRaycaster,
    world_model: WorldModel,
    device: str,
    cfg,
) -> np.ndarray:
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)
    occ_vol = occ_field(depth_t)
    uncertainty = raycaster(depth_t, occ_vol)[0].detach().cpu().numpy()

    vel_t   = torch.cat([depth_t, depth_t], dim=1)
    wm_pred = world_model(depth_t, vel_t)[0].detach().cpu().numpy()

    return np.concatenate([uncertainty, wm_pred.ravel()]).astype(np.float32)


def train_ppo(cfg, n_steps: int = 1_000_000) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        import gymnasium as gym
    except ImportError:
        raise ImportError("stable-baselines3 + gymnasium required — pip install stable-baselines3 gymnasium")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PPO training on {device}")

    H, W = cfg.depth.output_h, cfg.depth.output_w
    T = len(cfg.world_model.prediction_horizons)
    obs_dim = cfg.shadow.grid_cells + T * H * W

    occ_field  = OccupancyField(cfg).to(device).eval()
    raycaster  = ShadowRaycaster(cfg).to(device).eval()
    world_model = WorldModel(cfg).to(device).eval()

    # SB3 requires a gym.Env wrapper
    class ShadowWeaveGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.env = ShadowWeaveEnv(cfg)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(27,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            obs_dict = self.env.reset(seed=seed)
            return make_obs(obs_dict["depth"], occ_field, raycaster, world_model, device, cfg), {}

        def step(self, action):
            obs_dict = self.env.step()
            obs = make_obs(obs_dict["depth"], occ_field, raycaster, world_model, device, cfg)
            reward = compute_reward(
                collision=bool(obs_dict["occupancy"].max() > 0.5),
                path_efficiency=1.0,
                active_zones=int((obs[:cfg.shadow.grid_cells] > 0.1).sum()),
                cfg=cfg,
            )
            terminated = False
            truncated  = False
            return obs, reward, terminated, truncated, {}

        def close(self):
            self.env.close()

    model = PPO(
        "MlpPolicy",
        ShadowWeaveGymEnv(),
        learning_rate=cfg.agents.local.ppo_lr,
        gamma=cfg.agents.local.ppo_gamma,
        clip_range=cfg.agents.local.ppo_clip,
        verbose=1,
        device=device,
    )
    model.learn(total_timesteps=n_steps)
    ckpt_path = "checkpoints/local_agent/ppo_final.zip"
    pathlib.Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(ckpt_path)
    print(f"PPO training done. Saved to {ckpt_path}")


if __name__ == "__main__":
    cfg_path = pathlib.Path("shadowweave/configs/default.yaml")
    cfg = OmegaConf.load(cfg_path)

    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    train_ppo(cfg, n_steps)
