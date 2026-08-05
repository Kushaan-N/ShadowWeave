"""Evaluation harness — runs the full pipeline in MuJoCo and reports metrics.

The previous version could not run at all. It crashed three ways: the checkpoint's
architecture did not match default.yaml; ``obs_dim`` was computed as
``grid_cells + T*H*W`` (65545) while the orchestrator fed 45 values; and it passed a
flattened ``wm_pred.ravel()`` into an orchestrator that indexes it as (T, H, W).

It also measured the wrong things — IOU was scored against the *current* occupancy
rather than the future one it claimed to predict, and the collision flag was derived
from a grid whose walls were always set.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from omegaconf import DictConfig

from ..console import Progress, rule
from ..agents.global_agent import GlobalAgent
from ..agents.local_agent import LocalAgent
from ..agents.orchestrator import Orchestrator
from ..shadow.bev import BEVProjector, bev_flow
from ..shadow.raycast import ShadowRaycaster
from ..sim.mujoco_env import ShadowWeaveEnv
from ..utils import (
    config_from_checkpoint,
    get_device,
    load_checkpoint,
    load_config,
    seed_everything,
)
from ..world_model.diffusion import build_world_model
from .baselines import BaselineComparison, LeadTimeTracker, calibration_error
from .metrics import EvalMetrics


def _build_stack(cfg, occ, vis, flow):
    ch = list(cfg.bev.input_channels)
    parts = []
    if "occupancy" in ch:
        parts.append(occ)
    if "visibility" in ch:
        parts.append(vis)
    if "flow" in ch:
        parts.append(flow)
    return torch.cat(parts, dim=1)


def run_eval(
    cfg: DictConfig,
    model_ckpt: str,
    n_episodes: int = 30,
    policy_ckpt: str | None = None,
    allow_random_weights: bool = False,
) -> dict[str, float]:
    device = get_device()
    seed_everything(cfg.seed)
    print(f"Evaluating on {device}")

    env = ShadowWeaveEnv(cfg)
    projector = BEVProjector(cfg).to(device).eval()
    raycaster = ShadowRaycaster(cfg).to(device).eval()

    ckpt_path = pathlib.Path(model_ckpt)
    if ckpt_path.exists():
        wm_cfg = config_from_checkpoint(ckpt_path, cfg)
        world_model = build_world_model(wm_cfg).to(device)
        world_model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"])
        print(f"Loaded world model from {ckpt_path} "
              f"(base_channels={wm_cfg.world_model.base_channels})")
    elif allow_random_weights:
        print(f"[WARNING] checkpoint not found at {ckpt_path} — using random weights")
        world_model = build_world_model(cfg).to(device)
    else:
        # A typo'd path must not produce a complete, plausible-looking eval of
        # random weights whose only tell is one warning line in a batch log.
        raise SystemExit(
            f"checkpoint not found at {ckpt_path} — fix the path, or pass "
            "--allow-random-weights to evaluate an untrained model deliberately"
        )
    world_model.eval()

    local_agent = LocalAgent(cfg).to(device).eval()
    if policy_ckpt and pathlib.Path(policy_ckpt).exists():
        local_agent.load_state_dict(load_checkpoint(policy_ckpt, map_location=device)["model"])
        print(f"Loaded local agent from {policy_ckpt}")

    orch = Orchestrator(cfg, local_agent, GlobalAgent(cfg))
    horizons = list(cfg.world_model.prediction_horizons)
    metrics = EvalMetrics(horizons=horizons)
    baselines = BaselineComparison(horizons)
    lead_tracker = LeadTimeTracker()
    ece_samples: list[float] = []
    S = cfg.bev.size
    steps = cfg.eval.steps_per_episode
    fps = cfg.sim.fps

    # A horizon needs h*fps frames to elapse before it can be scored at all. With the
    # default 200-step episode the 10s horizon never came due and simply vanished
    # from the results with no indication anything was missing.
    unscorable = [h for h in horizons if int(h * fps) >= steps]
    if unscorable:
        needed = int(max(unscorable) * fps) + 1
        print(f"\n[WARNING] horizons {unscorable} need {needed} steps/episode but "
              f"eval.steps_per_episode={steps}.")
        print(f"          They will NOT appear in the results. To score them:"
              f"  --overrides eval.steps_per_episode={needed}")

    n_tiers = len(cfg.eval.difficulty_tiers)
    per_tier = max(n_episodes // n_tiers, 1)
    progress = Progress(per_tier * n_tiers, "  episodes")

    for tier in cfg.eval.difficulty_tiers:
        cfg.sim.difficulty = tier
        ep_in_tier = per_tier
        print(f"\n{rule(f'difficulty = {tier}  ({ep_in_tier} episodes)')}")

        for ep in range(ep_in_tier):
            obs = env.reset(seed=10_000 + ep)
            prev_bev = None
            n_collisions = 0
            path_length = 0.0
            start_pos = obs["agent_pos"].copy()
            # (issued_step, probs, pose_at_issue, shadow_at_issue, observed_at_issue)
            pending: list[tuple[int, np.ndarray, tuple, torch.Tensor, torch.Tensor]] = []

            for step in range(steps):
                t0 = time.perf_counter()
                d = torch.from_numpy(obs["depth"]).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    occ, vis = projector(d)
                    uncertainty = raycaster.forward_from_depth(d)[0].cpu().numpy()
                    flow = (bev_flow(prev_bev, occ) if prev_bev is not None
                            else torch.zeros(1, 2, S, S, device=device))
                    prev_bev = occ
                    wm_pred = torch.sigmoid(world_model(_build_stack(cfg, occ, vis, flow)))

                pred_np = wm_pred[0].cpu().numpy()
                out = orch.step(uncertainty, pred_np)
                metrics.log_latency((time.perf_counter() - t0) * 1000.0)

                # Score each prediction when it comes due, against the ground truth at
                # that moment — not against the present frame the way it used to.
                #
                # Critically, the truth must be rendered in the pose the agent held
                # when the prediction was ISSUED. The model is trained to answer "what
                # will be here, in the frame I am standing in now"; comparing against
                # obs["bev_occupancy"], which is in the pose at t+h, compares two
                # different coordinate frames (measured IOU 0.009 between them) and
                # drives every score to zero regardless of model quality.
                snapshot_now = env.geom_snapshot()
                for entry in list(pending):
                    issued, probs, (p_pos, p_yaw), p_shadow, p_obs = entry
                    ready = [i for i, h in enumerate(horizons) if issued + int(h * fps) == step]
                    if ready:
                        truth = torch.from_numpy(
                            env.bev_occupancy(p_pos, p_yaw, snapshot_now)
                        ).unsqueeze(0)
                        for hi in ready:
                            pred_t = torch.from_numpy(probs[hi]).unsqueeze(0)
                            metrics.log_prediction(pred_t, truth, p_shadow, horizon_idx=hi)
                            # Persistence is judged on what was visible when this
                            # prediction was made, not on the present frame.
                            baselines.log(hi, probs[hi], truth, p_shadow, observed_at_issue=p_obs)
                            ece_samples.append(calibration_error(pred_t, truth))
                    if step - issued >= int(max(horizons) * fps):
                        pending.remove(entry)
                pending.append((
                    step, pred_np,
                    (obs["agent_pos"].copy(), obs["agent_yaw"]),
                    (vis[0] < 0.5).cpu(),
                    BaselineComparison.observation_snapshot(occ[0]),
                ))

                lead_tracker.observe(step, pred_np, obs, horizons, fps)

                prev = obs["agent_pos"].copy()
                obs = env.step(action=out.nav_action)
                path_length += float(np.linalg.norm(obs["agent_pos"] - prev))
                n_collisions += int(obs["collision"])

            straight_line = float(np.linalg.norm(obs["agent_pos"] - start_pos))
            metrics.log_episode(
                had_collision=n_collisions > 0,
                path_length=max(path_length, 1e-6),
                optimal_path_length=max(straight_line, 1e-6),
                collision_rate_per_step=n_collisions / steps,
            )
            progress.update()

    progress.close()
    env.close()
    summary = metrics.summary()
    summary.update(baselines.summary())
    summary.update(lead_tracker.summary())
    if ece_samples:
        summary["calibration_error"] = float(np.mean(ece_samples))

    results_dir = pathlib.Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # Render the scored report inline rather than dumping raw key: value lines.
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts"))
        from report import render  # type: ignore
        print("\n" + render(summary, cfg))
    except Exception:
        print("\nEval results:")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")

    print(f"  results: {out_path}")
    print(f"  report:  python scripts/report.py --json {out_path} --markdown report.md")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the ShadowWeave pipeline")
    ap.add_argument("--ckpt", type=str, default="checkpoints/world_model/best.pt")
    ap.add_argument("--policy", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=9)
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--allow-random-weights", action="store_true",
                    help="evaluate an untrained world model instead of failing on a missing checkpoint")
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    run_eval(cfg, args.ckpt, n_episodes=args.episodes, policy_ckpt=args.policy,
             allow_random_weights=args.allow_random_weights)


if __name__ == "__main__":
    main()
