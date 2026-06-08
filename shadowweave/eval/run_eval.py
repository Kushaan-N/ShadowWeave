"""Evaluation harness — runs the full pipeline in MuJoCo and reports metrics."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..sim.mujoco_env import ShadowWeaveEnv
from ..shadow.occupancy import OccupancyField
from ..shadow.raycast import ShadowRaycaster
from ..world_model.diffusion import WorldModel
from ..agents.local_agent import LocalAgent
from ..agents.global_agent import GlobalAgent
from ..agents.orchestrator import Orchestrator
from .metrics import EvalMetrics


def run_eval(cfg: DictConfig, model_ckpt: str, n_episodes: int = 100) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = ShadowWeaveEnv(cfg)
    occ_field  = OccupancyField(cfg).to(device).eval()
    raycaster  = ShadowRaycaster(cfg).to(device).eval()

    world_model = WorldModel(cfg).to(device).eval()
    if pathlib.Path(model_ckpt).exists():
        world_model.load_state_dict(torch.load(model_ckpt, map_location=device))
        print(f"Loaded world model from {model_ckpt}")
    else:
        print(f"[WARNING] checkpoint not found at {model_ckpt} — using random weights")

    H, W = cfg.depth.output_h, cfg.depth.output_w
    T = len(cfg.world_model.prediction_horizons)
    obs_dim = cfg.shadow.grid_cells + T * H * W
    local_agent  = LocalAgent(cfg, obs_dim)
    global_agent = GlobalAgent(cfg)
    orch = Orchestrator(cfg, local_agent, global_agent)

    metrics = EvalMetrics()

    for tier in cfg.eval.difficulty_tiers:
        cfg.sim.difficulty = tier
        ep_in_tier = n_episodes // len(cfg.eval.difficulty_tiers)
        print(f"\nEvaluating difficulty={tier} ({ep_in_tier} episodes)")

        for ep in range(ep_in_tier):
            obs = env.reset(seed=ep)
            goal = (H - 4, W - 4)
            path_length = 0.0
            had_collision = False

            for step in range(200):
                depth_t = torch.from_numpy(obs["depth"]).unsqueeze(0).unsqueeze(0).to(device)
                occ_vol = occ_field(depth_t)
                uncertainty = raycaster(depth_t, occ_vol)[0].detach().cpu().numpy()

                vel_t = torch.from_numpy(obs["depth"]).unsqueeze(0).unsqueeze(0).to(device)
                vel_2ch = torch.cat([vel_t, vel_t], dim=1)
                wm_pred = world_model(depth_t, vel_2ch)[0].detach().cpu().numpy()

                metrics.log_prediction(wm_pred[2], obs["occupancy"])  # 5s horizon (index 2)

                audio_params, is_stop = orch.step(uncertainty, wm_pred.ravel())
                had_collision = had_collision or bool(obs["occupancy"].max() > 0.5 and uncertainty.max() > 0.8)
                path_length += 1.0

                obs = env.step()

            metrics.log_episode(had_collision, path_length, optimal_path_length=80.0)

    env.close()
    summary = metrics.summary()
    print("\nEval results:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    return summary


if __name__ == "__main__":
    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/world_model/best.pt"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    run_eval(cfg, ckpt, n_episodes=n_ep)
