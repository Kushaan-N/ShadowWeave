"""Automated rollout generation for world model training datasets.

Per timestep it stores:
    bev_occupancy    (1, S, S)  OBSERVED occupancy, unprojected from depth
    bev_visibility   (1, S, S)  1 = observed free, 0 = shadow / outside FOV
    bev_flow         (2, S, S)  motion between consecutive observed BEV frames
    target           (H, S, S)  GROUND-TRUTH occupancy at each horizon, expressed in
                                the agent's frame AT TIME t
    uncertainty_grid (9,)       9-cell shadow uncertainty, for the agent / audio path

The target is rendered in the frame at time ``t`` rather than at ``t+h`` on purpose:
the agent plans from where it is standing now, so asking "what will be here, in my
current frame, h seconds from now" is well-posed and directly plannable. Rendering it
in the future frame would additionally require predicting its own future pose.

Why this file was rewritten: the previous generator called ``env.step()`` with no
action, so the camera never moved. Combined with a world-frame target that ignored
the agent's pose, every stored array was constant — measured velocity was exactly
zero everywhere and future_occupancy had std 0.0 across both time and horizons. The
world model could reach a low loss by emitting a constant image.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import time
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from ..console import Progress
from ..shadow.bev import BEVProjector, bev_flow
from ..shadow.raycast import ShadowRaycaster
from ..utils import get_device, load_config, seed_everything
from .mujoco_env import ShadowWeaveEnv


class ExplorePolicy:
    """Scripted wanderer that keeps the camera moving and mostly off the walls.

    Any policy would do — the requirement is only that the observation distribution
    is not a single frozen viewpoint.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._turn = 0.0
        self._hold = 0

    def act(self, obs: dict) -> np.ndarray:
        if self._hold <= 0:
            self._turn = float(self.rng.uniform(-1.0, 1.0))
            self._hold = int(self.rng.integers(8, 40))
        self._hold -= 1

        forward = 1.0
        # Back off and turn hard when something is close, so episodes spend their
        # time navigating rather than grinding into a wall.
        if obs["min_clearance"] < 0.6:
            forward = -0.3
            self._turn = math.copysign(1.0, self._turn if self._turn != 0 else 1.0)
            self._hold = max(self._hold, 10)

        strafe = float(self.rng.uniform(-0.3, 0.3))
        return np.array([forward, strafe, self._turn], dtype=np.float32)


def _make_policy(name: str, rng: np.random.Generator):
    if name == "explore":
        return ExplorePolicy(rng)
    if name == "random":
        return type("R", (), {"act": lambda self, obs: rng.uniform(-1, 1, 3).astype(np.float32)})()
    if name == "straight":
        return type("S", (), {"act": lambda self, obs: np.array([1, 0, 0.02], np.float32)})()
    raise ValueError(f"Unknown policy: {name}")


class SyntheticDataGenerator:
    """Runs randomised MuJoCo episodes and saves rollout .npz files.

    Primary method: ``generate(n_episodes, output_dir, split)``
    """

    def __init__(self, cfg: DictConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or get_device()
        # Rollouts never read the RGB frame, and rendering it doubles the readPixels cost.
        self.env = ShadowWeaveEnv(cfg, render_rgb=False)
        self.horizons = list(cfg.world_model.prediction_horizons)
        self.fps = cfg.sim.fps

        self.projector = BEVProjector(cfg).to(self.device).eval()
        self.raycaster = ShadowRaycaster(cfg).to(self.device).eval()

    @torch.no_grad()
    def _observe_batch(
        self, depths: list[np.ndarray], chunk: int = 64
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project a whole episode at once.

        Per-frame calls left the GPU idle between launches; batching turns hundreds of
        tiny kernels into a handful of large ones. Chunked so a long episode at a
        large bev.size cannot exhaust device memory.
        """
        occs, viss, uncs = [], [], []
        for i in range(0, len(depths), chunk):
            d = torch.from_numpy(np.stack(depths[i : i + chunk])).unsqueeze(1).to(self.device)
            occ, vis = self.projector(d)
            unc = self.raycaster.forward_from_depth(d)
            occs.append(occ[:, 0].cpu().numpy())
            viss.append(vis[:, 0].cpu().numpy())
            uncs.append(unc.cpu().numpy())
        return np.concatenate(occs), np.concatenate(viss), np.concatenate(uncs)

    def generate(
        self,
        n_episodes: int,
        output_dir: str,
        split: str = "train",
        steps_per_episode: Optional[int] = None,
        seed0: int = 0,
    ) -> None:
        out_dir = pathlib.Path(output_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        if n_episodes <= 0:
            # A SLURM shard whose arithmetic lands on zero episodes must no-op,
            # not crash in the summary print below on never-bound locals.
            print(f"  0 episodes requested for split '{split}' — nothing to do")
            return

        T = steps_per_episode or self.cfg.data.steps_per_episode
        horizon_frames = [int(h * self.fps) for h in self.horizons]
        max_h = max(horizon_frames)
        n_h = len(self.horizons)
        S = self.cfg.bev.size

        t_start = time.time()
        total_steps = 0
        difficulties = list(self.cfg.eval.difficulty_tiers)
        progress = Progress(n_episodes, f"  episodes[{split}]")

        for ep in range(n_episodes):
            seed = seed0 + ep
            self.cfg.sim.difficulty = difficulties[ep % len(difficulties)]
            rng = np.random.default_rng(seed)
            policy = _make_policy(self.cfg.data.policy, rng)

            obs = self.env.reset(seed=seed)

            # Roll the full episode plus the lookahead needed by the longest horizon.
            poses: list[tuple[np.ndarray, float]] = []
            snaps: list[dict] = []
            depths: list[np.ndarray] = []
            collisions: list[bool] = []

            for _ in range(T + max_h + 1):
                poses.append((obs["agent_pos"].copy(), obs["agent_yaw"]))
                snaps.append(self.env.geom_snapshot())
                depths.append(obs["depth"])
                collisions.append(obs["collision"])
                obs = self.env.step(action=policy.act(obs))

            bev_occ = np.zeros((T, 1, S, S), dtype=np.float32)
            bev_vis = np.zeros((T, 1, S, S), dtype=np.float32)
            bev_flw = np.zeros((T, 2, S, S), dtype=np.float32)
            target = np.zeros((T, n_h, S, S), dtype=np.float32)
            unc_grid = np.zeros((T, self.cfg.shadow.grid_cells), dtype=np.float32)
            coll = np.zeros((T,), dtype=np.bool_)

            occ_all, vis_all, unc_all = self._observe_batch(depths[:T])
            bev_occ[:, 0] = occ_all
            bev_vis[:, 0] = vis_all
            unc_grid[:] = unc_all
            coll[:] = collisions[:T]

            # Flow is a plain temporal difference, so the whole stack does at once.
            occ_t_all = torch.from_numpy(occ_all).unsqueeze(1)
            bev_flw[1:] = bev_flow(occ_t_all[:-1], occ_t_all[1:]).numpy()

            for t in range(T):
                pos_t, yaw_t = poses[t]
                for hi, hf in enumerate(horizon_frames):
                    fi = min(t + hf, len(snaps) - 1)
                    # Future geometry, current frame.
                    target[t, hi] = self.env.bev_occupancy(pos_t, yaw_t, snaps[fi])

            payload = dict(
                bev_occupancy=bev_occ,
                bev_visibility=bev_vis,
                bev_flow=bev_flw,
                target=target,
                uncertainty_grid=unc_grid,
                collision=coll,
                horizons=np.array(self.horizons, dtype=np.float32),
            )
            # Name by global seed, not local episode index: SLURM array shards all
            # start their loop at ep=0 and would otherwise overwrite each other.
            path = out_dir / f"ep{seed:06d}.npz"
            # Uncompressed: np.load(mmap_mode=...) is silently ignored for compressed
            # archives, so a compressed file forces a full decompress of the whole
            # episode on every single __getitem__.
            if self.cfg.data.compress:
                np.savez_compressed(path, **payload)
            else:
                np.savez(path, **payload)
            total_steps += T
            progress.update()

        progress.close()
        elapsed = time.time() - t_start
        print(f"\n  {n_episodes} episodes, {total_steps} steps → {out_dir}")
        print(f"  {elapsed:.1f}s at {total_steps/max(elapsed,1e-9):.1f} steps/s  "
              f"(last episode: target_pos={target.mean():.4f} shadow={1-bev_vis.mean():.3f})")
        self.env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate ShadowWeave rollouts")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    seed_everything(cfg.seed)
    out = args.out or cfg.data.root

    print(f"Generating {args.episodes} episodes → {out}/{args.split}")
    gen = SyntheticDataGenerator(cfg)
    gen.generate(args.episodes, out, split=args.split, steps_per_episode=args.steps, seed0=args.seed0)


if __name__ == "__main__":
    main()
