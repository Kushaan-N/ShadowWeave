# ShadowWeave

Multi-agent navigation that conveys spatial awareness through 3D spatial audio, for
blind users and disaster-response scenarios. No haptics, no screens required.

**Core mechanism.** Differentiable rays are cast through a learned occupancy field.
Rays that terminate early define *shadow zones* — regions the sensor cannot see into.
Those shadow zones are propagated forward by a physics-informed world model and turned
into directional HRTF audio, so the user hears not just what is there but what *might
be* in the space they cannot observe.

This is deliberately **not** SLAM and not object detection + avoidance. The novel claim
is uncertainty propagation through unobserved space.

---

## Quick start

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate shadowweave
pip install -e .

python scripts/pipeline_test.py         # end-to-end smoke test, no training needed
python scripts/benchmark.py             # size batch/workers before booking GPU time
pytest -q                               # 84 tests
```

Every module also runs standalone with a demo:

```bash
python -m shadowweave.shadow.raycast    # zone uncertainty + throughput
python -m shadowweave.shadow.bev        # BEV occupancy / shadow projection
python -m shadowweave.sim.mujoco_env    # sim observations across all three tiers
python -m shadowweave.audio.hrtf        # spatialisation + phase-continuity checks
```

## Dashboard

```bash
python -m shadowweave.dashboard.app --live     # http://localhost:7860
```

---

## Running on a GPU cluster (SLURM)

```bash
# 0. one-time: point slurm/env.sh at your conda install / CUDA module
srun --gres=gpu:1 --pty python slurm/preflight.py   # verify the node before queuing

# 1. whole pipeline as a dependency chain
./slurm/pipeline.sh

# or stage by stage
sbatch slurm/gen_data.sbatch                        # 8-way array, sharded rollouts
sbatch slurm/train_worldmodel.sbatch                # add --gres=gpu:4 for DDP
sbatch slurm/train_rl.sbatch
sbatch slurm/eval.sbatch
sbatch slurm/sweep.sbatch                           # hyperparameter array job
```

Everything is overridable without editing files:

```bash
SW_ENV=my-env SW_CUDA_MODULE=cuda/12.6 sbatch slurm/train_worldmodel.sbatch
SW_OVERRIDES="world_model.base_channels=96 world_model.lr=2e-4" sbatch slurm/train_worldmodel.sbatch
```

Notes:
- `MUJOCO_GL=egl` is set automatically for headless off-screen rendering. Set
  `SW_MUJOCO_GL=osmesa` if the node has no EGL.
- `train_worldmodel.sbatch` traps `SIGUSR1` 120s before the time limit, requeues
  itself, and resumes from `last.pt` — long runs survive the queue limit.
- Multi-GPU is automatic: the script counts visible GPUs and switches to `torchrun`.
- `WANDB_MODE=offline` by default since compute nodes are usually firewalled; run
  `wandb sync wandb/` afterwards.

---

## Architecture

```
camera / MuJoCo ──> depth (128x128, metres / max_range_m)
                      │
        ┌─────────────┴─────────────┐
        │                           │
   ShadowRaycaster            BEVProjector
   9 azimuth zones            egocentric BEV:
   uncertainty (9,)           occupancy + visibility (shadow)
        │                           │
        │                     WorldModel (U-Net)
        │                     occupancy at t+1/3/5/10s
        │                           │
        └──────────> Orchestrator <─┘
                      │        │
              CueMapper        GlobalAgent (A*)
              9 -> 27 audio    waypoints, shadow-penalised
                      │
                 HRTFEngine ──> stereo out
```

Everything downstream of depth is differentiable, so the raycaster's ray→zone
assignment trains end to end.

### Frames and conventions

| Quantity | Frame | Meaning |
|---|---|---|
| `depth` | first-person image | `0` = at the camera, `1` = `shadow.max_range_m` (**fixed** scale) |
| `bev_occupancy` | egocentric BEV | agent at bottom-centre, `+z` up the grid |
| `bev_visibility` | egocentric BEV | `1` = observed free, `0` = **shadow** or outside FOV |
| `target` | egocentric BEV **at time t** | ground-truth occupancy at `t+h`, in the current pose's frame |
| `uncertainty_grid` | 9 azimuth zones | left→right across the FOV, 1-to-1 with audio zones |

The world model input and target share a frame, and the target moves with the agent —
so what the agent observes genuinely determines what it must predict.

### Key invariants

- Shadow zones are rays that terminate early. Empty space casts **no** shadow.
- Absence of an audio cue means safe. Never break this.
- At most `agents.local.max_simultaneous_cues` zones sound at once.
- Max zone uncertainty > `orchestrator.uncertainty_stop_threshold` → stop pattern,
  navigation output zeroed.
- Checkpoints embed the config that produced them; always load through
  `utils.config_from_checkpoint`.

---

## Repo layout

```
shadowweave/
├── ingestion/     camera + Depth Anything V2
├── shadow/        raycast.py (core), bev.py, occupancy.py, zones.py, visualizer.py
├── world_model/   diffusion.py (U-Net + ConvLSTM), dataset.py, train.py
├── agents/        local_agent.py, global_agent.py, orchestrator.py
├── audio/         hrtf.py, cues.py
├── sim/           mujoco_env.py, synthetic_data.py
├── dashboard/     app.py (Gradio)
├── eval/          metrics.py, baselines.py, run_eval.py
└── configs/       default.yaml — all hyperparameters
slurm/             sbatch scripts, env.sh, preflight.py
scripts/           pipeline_test.py (smoke), benchmark.py (throughput)
tests/             pytest suite
```

All hyperparameters live in `configs/default.yaml`. Override anything from the CLI:

```bash
python -m shadowweave.world_model.train --overrides world_model.lr=3e-4 bev.size=128
```

---

## Performance targets

| Metric | Target | Status |
|---|---|---|
| Full pipeline latency (camera → audio) | < 100ms | ✅ 2.2ms p50, 4.5ms p95 |
| Shadow-ray throughput | ≥ 15Hz | ✅ ~1700Hz |
| World model IOU @ 5s | > 60% | ⏳ needs a full training run |
| Agent collision rate (hard tier) | < 10% | ⏳ needs PPO training |
| Falling-object lead time | ≥ 3s | ⏳ needs a full training run |
| Dashboard render rate | ≥ 5Hz | ✅ `gr.Timer` at 5Hz |

Latency measured on Apple M-series MPS; a CUDA node is faster.

### Reading the eval output

IOU alone is not interpretable here — BEV targets are ~7% positive, so predicting
*nothing* already scores well on near-empty frames, and echoing the current
observation scores well wherever nothing moves. `run_eval` therefore reports every
horizon against two baselines:

| column | meaning |
|---|---|
| `persist` | echo the currently observed occupancy, unchanged |
| `empty` | predict free space everywhere |
| `gain` | model minus the strongest baseline — **this is the number that matters** |

A negative gain means the world model has not learned dynamics, however good its raw
IOU looks. The same comparison is reported restricted to shadow cells
(`shadow_iou_*`), which is where the project's actual claim lives.

`falling_detection_rate` must be read together with `falling_false_alarm_rate`: a
model that predicts "occupied everywhere" detects every falling object trivially.
`falling_detection_margin` is the difference and is the honest figure.
`calibration_error` catches a model that is confidently wrong — which matters more
than usual for a system whose entire output is an uncertainty signal.

---

## Data

Rollouts are generated, not shipped:

```bash
python -m shadowweave.sim.synthetic_data --episodes 400 --split train
python -m shadowweave.sim.synthetic_data --episodes 80 --split val --seed0 900000
```

Files are named by global seed so SLURM array shards never collide, and written
uncompressed so the loader can memory-map them (`np.load(mmap_mode=...)` is silently
ignored for compressed `.npz`, which otherwise forces a full decompress per sample).

> **Rollouts generated before the egocentric-BEV change are unusable** — their targets
> were world-frame and constant. `RolloutDataset` detects the old schema and tells you
> to regenerate.

---

## Status

The pipeline runs end to end, and the shadow-ray core, sim, and data path are verified
by the test suite. The world model, PPO policy, and the accuracy-dependent targets
above still need a full training run on a GPU node — start with `./slurm/pipeline.sh`.
