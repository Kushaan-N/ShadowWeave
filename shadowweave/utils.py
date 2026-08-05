"""Shared utilities: device selection, seeding, config loading, checkpoint I/O."""

from __future__ import annotations

import os
import pathlib
import random
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

_CONFIG_PATH = pathlib.Path(__file__).parent / "configs" / "default.yaml"


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Best available device. SW_DEVICE env var and ``prefer`` override autodetection.

    Order is cuda > mps > cpu. On a SLURM GPU node CUDA must win, so this never
    hard-codes mps-first the way the per-module scripts used to.
    """
    choice = prefer or os.environ.get("SW_DEVICE")
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_grid_sample_3d(device: torch.device) -> bool:
    """MPS has no grid_sampler_3d kernel; CUDA and CPU do."""
    return device.type != "mps"


def sanitize_depth(depth: torch.Tensor) -> torch.Tensor:
    """Clamp depth to [0, 1] and replace non-finite values with 0.0.

    A single NaN from a flaky sensor or a monocular depth model otherwise propagates
    straight through the raycaster into the uncertainty grid and out to the audio the
    user is navigating by, with nothing signalling that the frame was bad.

    Non-finite maps to 0.0 — "something is right in front of me" — because that reads
    as maximum uncertainty and trips the orchestrator's stop override. Mapping it to
    1.0 ("clear ahead") would be the unsafe direction to fail in.
    """
    return torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def load_config(path: Optional[str] = None, overrides: Optional[list[str]] = None) -> DictConfig:
    """Load default.yaml, then apply ``key=value`` dotlist overrides from the CLI."""
    cfg = OmegaConf.load(path or _CONFIG_PATH)
    if overrides:
        # Struct mode makes a typo'd key an error instead of a silently-ignored
        # new entry — a misspelled hyperparameter on a cluster job otherwise
        # trains with defaults and one only finds out after the run.
        OmegaConf.set_struct(cfg, True)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN autotuning picks different algorithms per run, which perturbs
        # results even with a fixed seed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: str | pathlib.Path,
    model: torch.nn.Module,
    cfg: DictConfig,
    *,
    epoch: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    metrics: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Save weights *with* the config that produced them.

    Bare state_dicts are why checkpoints/world_model/best.pt (base_channels=32)
    cannot be loaded against default.yaml (base_channels=64) — nothing recorded
    the architecture. Every checkpoint now carries its own config.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 2,
        "model": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    # Write-then-rename: a job killed mid-save must not leave a truncated checkpoint
    # where the resume path expects a valid one.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | pathlib.Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint, normalising legacy bare-state_dict files to the v2 layout."""
    raw = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "format_version" in raw:
        return raw
    # Legacy: the whole file is the state_dict, architecture unknown. Say so
    # loudly — otherwise the failure surfaces later as raw missing-key soup with
    # no hint that the file itself is the problem.
    print(f"[load_checkpoint] WARNING: {path} is a legacy bare-state_dict checkpoint "
          "with no embedded config — the architecture it was trained with is "
          "unknown; if loading fails with missing/mismatched keys, retrain it")
    return {"format_version": 1, "model": raw, "config": None, "epoch": 0, "metrics": {}}


def config_from_checkpoint(path: str | pathlib.Path, fallback: DictConfig) -> DictConfig:
    """Config a checkpoint was trained with, or ``fallback`` for legacy files."""
    ckpt = load_checkpoint(path)
    if ckpt.get("config") is None:
        return fallback
    return OmegaConf.create(ckpt["config"])  # type: ignore[return-value]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DistributedDataParallel and torch.compile wrappers.

    torch.compile returns an OptimizedModule whose state_dict keys are prefixed with
    ``_orig_mod.``. Saving that produces a checkpoint no plain model can load, and it
    also breaks EMA key matching — so every save/EMA path must unwrap first.
    """
    seen = 0
    while seen < 4:  # DDP(compile(model)) is two layers; the bound is a safety net
        inner = getattr(model, "module", None) or getattr(model, "_orig_mod", None)
        if inner is None:
            break
        model = inner
        seen += 1
    return model


def setup_headless_rendering() -> str:
    """Pick a MuJoCo GL backend that works without an X server.

    Compute nodes have no display, so MuJoCo's default backend fails there.
    EGL renders off-screen on the GPU; osmesa is the software fallback.
    """
    if "MUJOCO_GL" in os.environ:
        return os.environ["MUJOCO_GL"]
    if os.uname().sysname == "Darwin":
        backend = "cgl"
    elif os.environ.get("DISPLAY"):
        backend = "glfw"
    else:
        backend = "egl"
    os.environ["MUJOCO_GL"] = backend
    return backend
