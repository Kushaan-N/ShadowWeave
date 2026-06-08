"""Automated rollout generation for world model training datasets.

Generates (shadow_map, velocity, future_occupancy) tuples from randomised MuJoCo episodes.
Target: 50k+ pairs, saved as .npz files.

shadow_map = raw depth map (1, H, W) — direct world model input
velocity   = 2-channel frame-difference proxy for optical flow (2, H, W)
future_occupancy = ground-truth binary occupancy at each prediction horizon (horizons, H, W)
uncertainty_grid = 9-cell shadow uncertainty (9,) — for local agent obs / diagnostics
"""

from __future__ import annotations

import pathlib
import time
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from .mujoco_env import ShadowWeaveEnv
from ..shadow.raycast import ShadowRaycaster
from ..shadow.zones import pool_to_zones


class SyntheticDataGenerator:
    """Runs randomised MuJoCo episodes and saves rollout .npz files.

    Primary method: ``generate(n_episodes, output_dir)``
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.env = ShadowWeaveEnv(cfg)
        self.horizons = list(cfg.world_model.prediction_horizons)
        self.fps = cfg.sim.fps

        self.raycaster = ShadowRaycaster(cfg)
        self.raycaster.eval()

    @torch.no_grad()
    def _compute_uncertainty(self, depth: np.ndarray) -> np.ndarray:
        """depth: (H, W) float32 → uncertainty_grid: (9,) float32"""
        d = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        grid = self.raycaster.forward_from_depth(d)
        return grid[0].numpy()

    def generate(
        self,
        n_episodes: int,
        output_dir: str,
        split: str = "train",
        steps_per_episode: int = 300,
    ) -> None:
        out_dir = pathlib.Path(output_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        horizon_frames = [int(h * self.fps) for h in self.horizons]
        max_horizon_frames = max(horizon_frames)
        n_horizons = len(self.horizons)
        h, w = self.cfg.depth.output_h, self.cfg.depth.output_w

        t_start = time.time()
        total_steps = 0
        difficulties = ["static", "moving", "debris"]

        for ep in range(n_episodes):
            self.cfg.sim.difficulty = difficulties[ep % len(difficulties)]
            self.env.reset(seed=ep)

            # buffer full episode + lookahead frames
            frames: list[dict[str, np.ndarray]] = []
            for _ in range(steps_per_episode + max_horizon_frames + 1):
                frames.append(self.env.step())

            T = steps_per_episode

            shadow_maps      = np.zeros((T, 1, h, w),          dtype=np.float32)
            velocities       = np.zeros((T, 2, h, w),          dtype=np.float32)
            future_occupancy = np.zeros((T, n_horizons, h, w), dtype=np.float32)
            uncertainty_grid = np.zeros((T, 9),                dtype=np.float32)

            for t in range(T):
                depth_t = frames[t]["depth"]              # (H, W)
                shadow_maps[t, 0] = depth_t

                if t > 0:
                    diff = depth_t - frames[t - 1]["depth"]
                    # u = horizontal frame diff, v = vertical frame diff (proxy optical flow)
                    velocities[t, 0] = np.roll(diff, 1, axis=1) - diff   # horizontal gradient
                    velocities[t, 1] = np.roll(diff, 1, axis=0) - diff   # vertical gradient

                for hi, hf in enumerate(horizon_frames):
                    fi = min(t + hf, len(frames) - 1)
                    future_occupancy[t, hi] = frames[fi]["occupancy"]

                uncertainty_grid[t] = self._compute_uncertainty(depth_t)

            np.savez_compressed(
                out_dir / f"ep{ep:05d}.npz",
                shadow_map=shadow_maps,
                velocity=velocities,
                future_occupancy=future_occupancy,
                uncertainty_grid=uncertainty_grid,
            )
            total_steps += T

            if (ep + 1) % 5 == 0 or ep == 0:
                elapsed = time.time() - t_start
                rate = total_steps / elapsed
                print(f"  ep {ep+1:4d}/{n_episodes}  {rate:.0f} steps/s  difficulty={self.cfg.sim.difficulty}")

        print(f"\nDone: {n_episodes} episodes, {total_steps} steps → {out_dir}  ({time.time()-t_start:.1f}s)")
        self.env.close()


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    n_eps  = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    outdir = sys.argv[2] if len(sys.argv) > 2 else "./data/rollouts"

    print(f"Generating {n_eps} episodes → {outdir}/train")
    gen = SyntheticDataGenerator(cfg)
    gen.generate(n_eps, outdir, split="train", steps_per_episode=60)
    print("SyntheticDataGenerator OK")
