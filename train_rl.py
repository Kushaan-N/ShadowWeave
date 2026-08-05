"""PPO RL training entry point for the LocalAgent in MuJoCo.

The critical fix: the collision flag was ``obs["occupancy"].max() > 0.5``, and the
occupancy grid always marked the room's walls as 1.0, so this was True on every
single step of every episode. The agent received a constant -10 reward regardless of
what it did and could not possibly learn to avoid anything. Collision now comes from
the environment's real geometric clearance test.

Also switched to a vectorised env and a policy sized from cfg, and the world model is
loaded with the config it was actually trained under.

Launch:
    python train_rl.py --steps 1000000
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from omegaconf import DictConfig

from shadowweave.agents.local_agent import compute_reward, observation_dim
from shadowweave.shadow.bev import BEVProjector, bev_flow
from shadowweave.shadow.raycast import ShadowRaycaster
from shadowweave.shadow.zones import pool_predictions_to_zones_torch
from shadowweave.sim.mujoco_env import ShadowWeaveEnv
from shadowweave.utils import (
    config_from_checkpoint,
    get_device,
    load_checkpoint,
    load_config,
    seed_everything,
)
from shadowweave.world_model.diffusion import build_world_model


def load_world_model(cfg: DictConfig, device: torch.device):
    """Load the world model using the config embedded in its checkpoint.

    A bare state_dict carries no architecture, which is how a base_channels=32
    checkpoint ended up being loaded against a base_channels=64 config and raising a
    wall of size-mismatch errors.
    """
    ckpt = pathlib.Path(cfg.world_model.checkpoint_dir) / "best.pt"
    if not ckpt.exists():
        print(f"WARNING: no checkpoint at {ckpt} — world model using random weights")
        return build_world_model(cfg).to(device).eval()

    wm_cfg = config_from_checkpoint(ckpt, cfg)
    model = build_world_model(wm_cfg).to(device)
    try:
        model.load_state_dict(load_checkpoint(ckpt, map_location=device)["model"])
        print(f"Loaded world model from {ckpt} (base_channels={wm_cfg.world_model.base_channels})")
    except RuntimeError as e:
        print(f"WARNING: checkpoint incompatible ({str(e)[:120]}) — using random weights")
    return model.eval()


def make_env_class(cfg: DictConfig, device: torch.device):
    import gymnasium as gym

    obs_dim = observation_dim(cfg)
    projector = BEVProjector(cfg).to(device).eval()
    raycaster = ShadowRaycaster(cfg).to(device).eval()
    world_model = load_world_model(cfg, device)
    S = cfg.bev.size

    class ShadowWeaveGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, rank: int = 0):
            super().__init__()
            self.env = ShadowWeaveEnv(cfg)
            self.rank = rank
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            self._step_count = 0
            self._prev_pos = np.zeros(2, dtype=np.float32)
            self._prev_bev: torch.Tensor | None = None
            self._goal = np.zeros(2, dtype=np.float32)

        @torch.no_grad()
        def _make_obs(self, obs_dict) -> tuple[np.ndarray, float]:
            d = torch.from_numpy(obs_dict["depth"]).unsqueeze(0).unsqueeze(0).to(device)
            occ, vis = projector(d)
            uncertainty = raycaster.forward_from_depth(d)[0].cpu().numpy()

            flow = (bev_flow(self._prev_bev, occ) if self._prev_bev is not None
                    else torch.zeros(1, 2, S, S, device=device))
            self._prev_bev = occ

            stack = []
            ch = list(cfg.bev.input_channels)
            if "occupancy" in ch:
                stack.append(occ)
            if "visibility" in ch:
                stack.append(vis)
            if "flow" in ch:
                stack.append(flow)
            wm_pred = torch.sigmoid(world_model(torch.cat(stack, dim=1)))
            wm_zones = pool_predictions_to_zones_torch(wm_pred)[0].cpu().numpy().ravel()

            shadow_exposure = float(1.0 - vis.mean().item())
            return np.concatenate([uncertainty, wm_zones]).astype(np.float32), shadow_exposure

        def reset(self, *, seed=None, options=None):
            obs_dict = self.env.reset(seed=seed if seed is not None else self.rank)
            self._step_count = 0
            self._prev_bev = None
            self._prev_pos = obs_dict["agent_pos"].copy()
            half = cfg.sim.room_size / 2 - 0.5
            self._goal = np.array([-self._prev_pos[0], -self._prev_pos[1]], dtype=np.float32)
            self._goal = np.clip(self._goal, -half, half)
            obs, _ = self._make_obs(obs_dict)
            return obs, {}

        def step(self, action):
            obs_dict = self.env.step(action=np.asarray(action, dtype=np.float32))
            obs, shadow_exposure = self._make_obs(obs_dict)
            self._step_count += 1

            pos = obs_dict["agent_pos"]
            # Real geometric overlap from the environment, not a grid max().
            collision = bool(obs_dict["collision"])

            moved = float(np.linalg.norm(pos - self._prev_pos))
            max_dist = cfg.sim.agent_max_speed * np.sqrt(2)
            moved_norm = min(moved / (max_dist + 1e-8), 1.0)

            prev_d = float(np.linalg.norm(self._prev_pos - self._goal))
            cur_d = float(np.linalg.norm(pos - self._goal))
            progress = (prev_d - cur_d) / (max_dist + 1e-8)

            self._prev_pos = pos.copy()
            reward = compute_reward(collision, moved_norm, cfg,
                                    progress=progress, shadow_exposure=shadow_exposure)

            terminated = cur_d < 0.5
            if terminated:
                reward += 10.0
            truncated = self._step_count >= cfg.sim.max_episode_steps
            return obs, reward, terminated, truncated, {"collision": collision}

        def close(self):
            self.env.close()

    return ShadowWeaveGymEnv


def train_ppo(cfg: DictConfig, n_steps: int = 1_000_000) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError:
        raise ImportError("stable-baselines3 + gymnasium required — pip install stable-baselines3 gymnasium")

    device = get_device()
    seed_everything(cfg.seed)
    print(f"PPO training on {device}")

    env_cls = make_env_class(cfg, device)
    n_envs = int(cfg.agents.local.ppo_n_envs)
    env_fns = [(lambda i=i: env_cls(rank=i)) for i in range(n_envs)]

    if n_envs > 1:
        # MuJoCo renderers hold a GL context and torch holds device state; neither
        # survives a fork, and the workers die with a bare EOFError in the parent
        # that says nothing about the cause. Spawn gives each worker a clean
        # interpreter. Verified: fork fails here, spawn succeeds.
        venv = SubprocVecEnv(env_fns, start_method=cfg.agents.local.ppo_start_method)
    else:
        venv = DummyVecEnv(env_fns)

    ckpt_dir = pathlib.Path(cfg.agents.local.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # tensorboard is optional; SB3 raises outright if it is missing, which should not
    # be enough to kill a training run that is otherwise ready to go.
    try:
        import tensorboard  # noqa: F401
        tb_log = str(ckpt_dir / "tb")
    except ImportError:
        tb_log = None
        print("tensorboard not installed — skipping TB logging (pip install tensorboard)")

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=cfg.agents.local.ppo_lr,
        gamma=cfg.agents.local.ppo_gamma,
        clip_range=cfg.agents.local.ppo_clip,
        n_steps=cfg.agents.local.ppo_n_steps,
        policy_kwargs=dict(net_arch=[cfg.agents.local.hidden_dim] * cfg.agents.local.num_layers),
        tensorboard_log=tb_log,
        verbose=1,
        seed=cfg.seed,
        device=str(device),
    )
    model.learn(total_timesteps=n_steps)
    out = ckpt_dir / "ppo_final.zip"
    model.save(out)
    venv.close()
    print(f"PPO training done. Saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ShadowWeave local agent with PPO")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()
    train_ppo(load_config(overrides=args.overrides), args.steps)


if __name__ == "__main__":
    main()
