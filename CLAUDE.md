# ShadowWeave — CLAUDE.md

## What this project is

Multi-agent navigation system that gives users spatial awareness without visual input. Output is **3D spatial audio** (HRTF-based). Designed for blind users and disaster-response scenarios.

**Core novel mechanism**: differentiable shadow-ray propagation through a latent occupancy field. Rays that terminate early define "shadow zones" — high-uncertainty regions behind obstacles. These drive a physics-informed world model and a multi-agent planner. The user hears the shape of their environment.

**Not**: SLAM, standard object detection + avoidance, haptics hardware project, real-time robot control.

---

## Repo layout

```
shadowweave/           ← main package
├── ingestion/         ← camera abstraction + Depth Anything V2 wrapper
├── shadow/            ← core shadow-ray module (THE novel contribution)
├── world_model/       ← U-Net diffusion model, dataset, training loop
├── agents/            ← local MLP agent (20Hz PPO) + global A* planner (2Hz) + orchestrator
├── audio/             ← HRTF spatial audio engine + uncertainty → cue mapping
├── sim/               ← MuJoCo 3.x environment + synthetic data generator
├── dashboard/         ← Gradio split-screen live demo
├── eval/              ← metrics + eval harness
└── configs/           ← default.yaml (all hyperparameters — no magic numbers in code)
train_rl.py            ← PPO/GRPO RL entry point
```

---

## Build order (strict — do not skip ahead)

Each step must produce a runnable artifact before the next begins.

1. `sim/mujoco_env.py` — MuJoCo room renders + exports RGB + ground-truth depth
2. `shadow/raycast.py` — depth map → 9-cell uncertainty grid (≥15Hz on GPU)
3. `world_model/` — trains on synthetic rollouts, predicts occupancy at t+1/3/5/10s
4. `agents/local_agent.py` — PPO in sim, collision rate < 20%
5. `agents/global_agent.py` — A* waypoints from world model output
6. `audio/` — 9-cell uncertainty vector → directional HRTF audio cues
7. `dashboard/app.py` — Gradio split-screen wiring everything together
8. `eval/` — latency optimization + eval harness

---

## Coding conventions

- Every module: clean `__init__` + one primary method (e.g. `ShadowRaycaster.forward(depth_map) -> uncertainty_grid`)
- Every module: `if __name__ == "__main__":` block that demos it with dummy/random input
- All hyperparameters in `configs/default.yaml` via OmegaConf — zero magic numbers in code
- Type hints on all function signatures (Python 3.10+)
- Log training metrics to `wandb`
- No comments explaining WHAT — only WHY (hidden constraints, subtle invariants)
- No docstrings beyond a one-liner if truly needed

---

## Performance targets

| Metric | Target |
|---|---|
| Full pipeline latency (camera → audio) | < 100ms |
| Shadow-ray throughput | ≥ 15Hz |
| World model prediction IOU (5s ahead) | > 60% |
| Agent collision rate (hard tier) | < 10% |
| Falling-beam prediction lead time | ≥ 3s before impact |
| Dashboard render rate | ≥ 5Hz |

---

## Tech stack

| Component | Library |
|---|---|
| Deep learning | PyTorch 2.x + CUDA (24GB VRAM assumed) |
| Depth estimation | Depth Anything V2 (HuggingFace) |
| Object detection | Grounding DINO + SAM 2 |
| Physics sim | MuJoCo 3.x (primary), PyBullet (fallback) |
| World model | Custom U-Net ~50M params, ConvLSTM fallback |
| RL training | Stable-Baselines3 PPO or custom GRPO |
| Graph planning | NetworkX |
| Spatial audio | sounddevice + numpy + MIT KEMAR HRTF |
| Optical flow | RAFT (torchvision) |
| Visualization | Plotly, Matplotlib |
| Dashboard | Gradio |
| Inference optimization | ONNX + INT8 quantization |
| Config management | Hydra / OmegaConf |

---

## MuJoCo note

MuJoCo 3.x may not be installed. All sim imports are guarded with graceful fallback messages. To install: `pip install mujoco`.

---

## Key invariants

- Shadow zones = rays that terminate early through the occupancy field. This is NOT the same as detected obstacle bounding boxes.
- 9-cell spatial grid maps 1-to-1 to 9 directional audio zones (left-far → right-far).
- Absence of audio cue = safe path. Don't break this convention.
- Orchestrator override: if max zone uncertainty > 0.85, pause all nav suggestions and emit a distinct "stop" pattern.
- World model fallback: if diffusion doesn't converge, swap to ConvLSTM — same interface.
