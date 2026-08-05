"""Preflight check — verify a GPU node can actually run ShadowWeave.

Run this on a compute node before submitting a long job. It catches the things that
otherwise fail an hour into training: no CUDA, MuJoCo unable to open an EGL context,
missing rollouts, a checkpoint that does not match the config.

    srun --gres=gpu:1 --pty python slurm/preflight.py
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

PASS, FAIL, WARN = "\033[92m✓\033[0m", "\033[91m✗\033[0m", "\033[93m!\033[0m"
_failures = 0
_warnings = 0


def check(name: str, fn, *, fatal: bool = True) -> object:
    global _failures, _warnings
    try:
        result = fn()
        detail = result if isinstance(result, str) else ""
        print(f"  {PASS} {name}  {detail}")
        return result
    except Exception as e:
        mark = FAIL if fatal else WARN
        print(f"  {mark} {name}  → {e.__class__.__name__}: {str(e)[:110]}")
        if fatal:
            _failures += 1
        else:
            _warnings += 1
        return None


def main() -> int:
    print("ShadowWeave preflight\n")

    print("Environment")
    check("python", lambda: sys.version.split()[0])
    check("MUJOCO_GL", lambda: os.environ.get("MUJOCO_GL", "(unset — will autodetect)"))
    check("SLURM job", lambda: os.environ.get("SLURM_JOB_ID", "(not under SLURM)"), fatal=False)

    print("\nTorch / GPU")

    def torch_info():
        import torch
        return f"torch {torch.__version__}"

    check("torch import", torch_info)

    def cuda_info():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — training will fall back to CPU")
        n = torch.cuda.device_count()
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
        return f"{n} GPU(s): {names}"

    check("cuda available", cuda_info, fatal=False)

    def matmul_check():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("skipped, no CUDA")
        a = torch.randn(1024, 1024, device="cuda")
        (a @ a).sum().item()
        free, total = torch.cuda.mem_get_info()
        return f"{(total - free) / 1e9:.1f}/{total / 1e9:.1f} GB used"

    check("cuda matmul", matmul_check, fatal=False)

    def amp_check():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("skipped, no CUDA")
        with torch.autocast("cuda"):
            x = torch.randn(64, 64, device="cuda")
            assert (x @ x).dtype == torch.float16
        return "fp16 autocast OK"

    check("amp autocast", amp_check, fatal=False)

    print("\nMuJoCo headless rendering")

    def mujoco_render():
        from shadowweave.utils import load_config
        from shadowweave.sim.mujoco_env import ShadowWeaveEnv

        cfg = load_config()
        env = ShadowWeaveEnv(cfg)
        obs = env.reset(seed=0)
        d = obs["depth"]
        env.close()
        if float(d.max()) <= 0.0:
            raise RuntimeError("depth buffer came back empty")
        return f"depth {d.shape} range [{d.min():.3f}, {d.max():.3f}]"

    check("render RGB+depth", mujoco_render)

    print("\nPipeline modules")

    def shadow_check():
        import torch
        from shadowweave.shadow.raycast import ShadowRaycaster
        from shadowweave.shadow.bev import BEVProjector
        from shadowweave.utils import get_device, load_config

        cfg = load_config()
        dev = get_device()
        depth = torch.ones(1, 1, cfg.depth.output_h, cfg.depth.output_w, device=dev)
        depth[..., : cfg.depth.output_w // 3] = 0.2
        with torch.no_grad():
            grid = ShadowRaycaster(cfg).to(dev).forward_from_depth(depth)
            occ, vis = BEVProjector(cfg).to(dev)(depth)
        spread = float(grid.max() - grid.min())
        if spread < 0.05:
            raise RuntimeError(f"zones not discriminating (spread={spread:.4f})")
        return f"zone spread={spread:.3f}, bev {tuple(occ.shape)}"

    check("shadow + BEV", shadow_check)

    def data_check():
        from shadowweave.utils import load_config

        cfg = load_config()
        root = pathlib.Path(os.environ.get("SW_DATA_ROOT", cfg.data.root))
        import numpy as np

        counts = {}
        for split in ("train", "val"):
            files = sorted((root / split).glob("*.npz"))
            # Every file, not files[0]: fresh 6-digit and stale 5-digit episodes
            # coexist after a regeneration, and either sort order can hide the
            # other. np.load only reads the zip directory, so this is cheap.
            for f in files:
                with np.load(f) as d:
                    if "bev_occupancy" not in d:
                        raise RuntimeError(
                            f"{f} uses the pre-BEV schema {sorted(d.files)} — "
                            "delete the stale rollouts and regenerate "
                            "(slurm/gen_data.sbatch or ./slurm/pipeline.sh)"
                        )
            counts[split] = len(files)
        if not counts["train"] and not counts["val"]:
            return "none yet — gen_data will create them"
        return f"train={counts['train']} val={counts['val']} episodes"

    # Stale/degenerate data must block submission, not warn: a green preflight
    # followed by a stage-2 crash costs the whole gen_data allocation.
    check("rollout data", data_check)

    def ckpt_check():
        from shadowweave.utils import config_from_checkpoint, load_checkpoint, load_config
        from shadowweave.world_model import build_world_model

        cfg = load_config()
        p = pathlib.Path(cfg.world_model.checkpoint_dir) / "best.pt"
        if not p.exists():
            return f"none at {p} (fine before the first run)"
        try:
            wm_cfg = config_from_checkpoint(p, cfg)
            model = build_world_model(wm_cfg)
            model.load_state_dict(load_checkpoint(p)["model"])
        except Exception as e:
            raise RuntimeError(
                f"{p} exists but cannot be loaded ({type(e).__name__}) — a stale "
                "checkpoint here silently degrades RL/eval to random weights; "
                "delete it or retrain"
            ) from e
        return f"loads with base_channels={wm_cfg.world_model.base_channels}"

    check("world model checkpoint", ckpt_check)

    print()
    if _failures:
        print(f"\033[91m{_failures} fatal issue(s)\033[0m — fix before submitting.")
        return 1
    if _warnings:
        print(f"\033[93m{_warnings} warning(s)\033[0m — usable, but check the notes above.")
        return 0
    print("\033[92mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
