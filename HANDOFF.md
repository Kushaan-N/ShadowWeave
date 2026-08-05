# ShadowWeave — cluster handoff

**Read this before touching anything.** Written for an assistant picking the project up
on a GPU cluster with no prior context. `README.md` explains what the system *is*; this
explains what state it is in, what will break, and the exact order to run things.

---

## 1. State of the project in one paragraph

Everything is built and tested; **nothing has been trained**. 210 tests pass, 16 module
demos run, every stage has executed at least once on CPU/MPS — but there is not a single
accuracy number in the repo. Every performance target still reads "needs a training
run." The job on this cluster is to produce those numbers, not to add features.

The development machine was an Apple laptop with **no NVIDIA GPU**, so three things have
never run on real hardware and are the most likely sources of surprise:

| never exercised | why it matters |
|---|---|
| **NCCL** multi-GPU DDP | only 2-rank gloo/CPU was verified |
| **CUDA AMP** (`torch.autocast` + `GradScaler`) | only the disabled path ran |
| **EGL** headless MuJoCo rendering | macOS used CGL; EGL is a different driver path |

`slurm/preflight.py` checks all three. Run it first.

---

## 2. Two landmines that do not travel with git

`.gitignore` excludes `data/` and `checkpoints/`, so the clone is clean — but if you
copied the working directory instead of cloning, **delete both**:

```bash
rm -rf data/rollouts checkpoints/world_model    # keep data/kemar — it IS tracked
```

- `data/rollouts/` on the dev machine holds the **pre-BEV schema** (`shadow_map`,
  `future_occupancy`, `velocity`) — a degenerate dataset whose targets had std 0.0.
  `RolloutDataset` detects and rejects it, but only after you have waited for the job.
- `checkpoints/world_model/*.pt` are from a June notebook at `base_channels=32` with no
  embedded config. Unloadable. `preflight.py` flags them.

`data/kemar/` (37 HRTF `.wav` files, 152 KB) **is** tracked and must survive — without it
the audio engine silently degrades to amplitude panning.

---

## 3. Setup

Either environment works. **venv is usually the better fit on HPC** — the cluster
already provides a Python module, PyTorch's pip wheels ship their own CUDA runtime, and
there is no conda install to manage. Conda is only genuinely needed if you want its
`mesalib` for the *osmesa* software-rendering fallback; EGL on a GPU node comes from the
NVIDIA driver, not from the Python environment.

### Option A — venv (recommended)

```bash
module load python/3.11          # whatever your cluster provides; needs >= 3.10
./scripts/setup_venv.sh          # detects driver CUDA, picks the matching wheel index
source .venv/bin/activate
```

The script reads `nvidia-smi` and selects `cu124` / `cu121` / `cu118` accordingly. A
wheel newer than the driver supports is the usual cause of a silent
`cuda_available=False` on an otherwise healthy node. Override if you know better:

```bash
SW_CUDA=cu121 ./scripts/setup_venv.sh
SW_VENV=/scratch/$USER/venv ./scripts/setup_venv.sh    # keep it off a small $HOME
```

**Run it on a GPU node, not the login node** — on a login node there is no `nvidia-smi`,
so it falls back to CPU wheels and every job afterwards runs on CPU.

### Option B — conda

```bash
conda env create -f environment.yml       # name: shadowweave
conda activate shadowweave
pip install -e ".[sim,viz,audio,depth,dev,usd]"
```

### Either way

```bash
$EDITOR slurm/env.sh                      # point at your cluster, once
srun --gres=gpu:1 --pty python slurm/preflight.py
```

`slurm/env.sh` picks the environment up automatically: it uses `./.venv` if present (or
`$SW_VENV`), otherwise conda. The lines that usually need changing:

```bash
SW_PYTHON_MODULE="${SW_PYTHON_MODULE:-}"          # e.g. python/3.11
SW_CUDA_MODULE="${SW_CUDA_MODULE:-cuda/12.4}"     # `module avail cuda` to check
SW_ENV="${SW_ENV:-shadowweave}"                   # conda env name, if using conda
```

If the venv lives outside the repo, export `SW_VENV` before every `sbatch` (or set it in
`env.sh`).

Preflight must report CUDA available, AMP working, and a non-empty MuJoCo depth buffer.
If depth rendering fails, set `SW_MUJOCO_GL=osmesa` and re-run — that path is software
rendering and is the one case where conda's `mesalib` genuinely helps.

## 4. Storage — check this before generating data

At default settings the dataset is **~85 GB**:

| | |
|---|---|
| per episode (300 steps, 96×96 BEV, 4 horizons) | 88 MB |
| train 400 eps | 35 GB |
| val 80 eps | 7 GB |
| `.memmap` sidecars (a full second copy, created on first load) | ×2 |
| **total** | **~85 GB** |

The sidecars exist because `np.load(mmap_mode=...)` is silently ignored for `.npz`;
each field is extracted once to a real `.npy` so a sample costs one page-in instead of
decompressing a whole episode. That trade is deliberate — do not "fix" it by enabling
`data.compress`, which would make every `__getitem__` O(episode).

If quota is tight, in order of preference:

```bash
SW_OVERRIDES="bev.size=48"          # ~21 GB, 4x smaller, still divisible by 16
SW_STEPS=150                        # ~42 GB, half the timesteps per episode
SW_EPISODES=200                     # ~43 GB, fewer episodes (weakest option — less data)
```

`bev.size` **must be divisible by 16** (four 2× downsamples in the U-Net). 48, 64, 96,
128 are all valid; the model raises a clear error otherwise.

---

## 5. The run, in order

```bash
# everything, as a dependency chain (each stage waits for the previous to succeed)
./slurm/pipeline.sh
```

Or stage by stage, which is what you want the first time:

```bash
# 1. data — 8-way array, ~35 GB, shards are disjoint by seed
sbatch slurm/gen_data.sbatch
SW_SPLIT=val SW_EPISODES=80 sbatch --array=0-3 slurm/gen_data.sbatch

# 2. world model — add --gres=gpu:4 and it switches to torchrun/DDP automatically
sbatch slurm/train_worldmodel.sbatch

# 2b. the diffusion variant, for the ablation
SW_CKPT_DIR=checkpoints/world_model_diffusion \
SW_OVERRIDES="world_model.architecture=diffusion" \
  sbatch slurm/train_worldmodel.sbatch

# 3. PPO — REQUIRES a world model checkpoint; it refuses to start without one
sbatch slurm/train_rl.sbatch

# 4. eval + the reasoning benchmark
SW_EVAL_EPISODES=90 SW_OVERRIDES="eval.steps_per_episode=301" sbatch slurm/eval.sbatch
python -m shadowweave.eval.reasoning --split val
python scripts/report.py --markdown results/report.md
```

### Verify after each stage — do not chain blindly

```bash
# after data generation: the #1 historical failure was a degenerate dataset
python - <<'EOF'
import numpy as np, glob
f = sorted(glob.glob("data/rollouts/train/*.npz"))
print(f"{len(f)} episodes")
d = np.load(f[0])
print("keys:", sorted(d.files))                      # must contain bev_occupancy
print("target std over time :", d["target"].std(axis=0).mean())   # must be > 1e-3
print("target std over horiz:", d["target"].std(axis=1).mean())   # >0 for moving/debris
print("flow nonzero frac    :", (d["bev_flow"] != 0).mean())      # must be > 0
EOF

# after training
python -c "
from shadowweave.utils import load_checkpoint
c = load_checkpoint('checkpoints/world_model/best.pt')
print('epoch', c['epoch'], c['metrics'])"
```

If `target std over time` is ~0 or `flow nonzero` is 0, **stop** — the agent is not
moving and the model will learn a constant. That exact failure is what the whole
`sim/synthetic_data.py` rewrite fixed.

---

## 6. Traps that will cost you hours

**The 10s horizon is silently never scored.** A horizon needs `h × fps` frames to elapse.
At `fps=30`, the 10s horizon needs **301** steps/episode but `eval.steps_per_episode`
defaults to 200 — so `iou_10s` just does not appear in the results. `run_eval` now warns,
but pass `eval.steps_per_episode=301` to actually get it.

**PPO must use spawn, not fork.** A forked worker inherits MuJoCo's GL context and torch
device state; neither survives, and the parent dies with a bare `EOFError` naming no
cause. `agents.local.ppo_start_method: "spawn"` is already set — do not change it. Each
worker rebuilds the world model, so budget roughly `model_size × ppo_n_envs` extra RSS.

**`batch_size` larger than the split silently does nothing.** `drop_last=True` yields
zero batches, every epoch is a no-op, and the reported loss is 0.0. Training now raises
instead — if you see that error, lower `world_model.batch_size` or generate more data.

**Requeue is automatic but resumes only from `last.pt`.** `train_worldmodel.sbatch` traps
`SIGUSR1` 120 s before the limit, requeues, and resumes. To start genuinely fresh set
`SW_FRESH=1`, otherwise it silently continues the previous run.

**W&B defaults to offline** (compute nodes are usually firewalled). Sync afterwards:
```bash
wandb sync wandb/
```

**Non-finite input fails toward "stop", not "clear".** If depth returns NaN, the
uncertainty grid reads 1.0 and the orchestrator trips its stop override. That is
deliberate — silence means *safe* to a blind user. Do not "fix" it to fail quietly.

---

## 7. Overriding anything without editing files

Two mechanisms, both composable:

```bash
# SLURM-level (see slurm/env.sh for the full list)
SW_ENV=myenv SW_CUDA_MODULE=cuda/12.6 SW_DATA_ROOT=/scratch/$USER/rollouts \
SW_CKPT_DIR=/scratch/$USER/ckpt sbatch slurm/train_worldmodel.sbatch

# config-level — any key in shadowweave/configs/default.yaml, dotted
SW_OVERRIDES="world_model.base_channels=96 world_model.lr=2e-4 world_model.batch_size=64" \
  sbatch slurm/train_worldmodel.sbatch

# same thing outside SLURM
python -m shadowweave.world_model.train --overrides world_model.epochs=50 bev.size=64
```

Full `SW_*` list: `SW_ENV SW_CONDA_BASE SW_CUDA_MODULE SW_MUJOCO_GL SW_ROOT
SW_DATA_ROOT SW_CKPT_DIR SW_CKPT SW_OVERRIDES SW_EPISODES SW_VAL_EPISODES SW_SPLIT
SW_STEPS SW_SKIP_DATA SW_FRESH SW_RL_STEPS SW_RL_ENVS SW_EVAL_EPISODES`

**Size the run before booking hours:**
```bash
srun --gres=gpu:1 --pty python scripts/benchmark.py --data data/rollouts
```
Prints fwd+bwd time and peak memory per batch size, per-stage inference latency, and
data-loading throughput. Pick the largest batch that fits with headroom. On the dev
machine loading ran ~37× faster than the model, so it is compute-bound — `num_workers=8`
is likely already plenty.

---

## 8. What "done" looks like

The targets, and where each number comes from:

| metric | target | produced by |
|---|---|---|
| World model IOU @ 5s | > 0.60 | `run_eval` → `results/eval_summary.json` |
| Gain over persistence baseline | **> 0** | same — *this matters more than raw IOU* |
| Shadow-masked IOU @ 5s | > 0.40 | same |
| Collision rate | < 0.10 | same, after PPO |
| Falling-object lead time | ≥ 3 s | same |
| Reasoning gain over best baseline | > 0 | `python -m shadowweave.eval.reasoning` |
| Shadow diversity ratio (diffusion only) | > 1 | `run_eval` with a diffusion checkpoint |

**Read gain, not IOU.** BEV targets are ~7% positive, so predicting nothing already
scores well and echoing the current frame scores well wherever nothing moves. A negative
`model_gain_over_best_baseline_5s` means the model has not learned dynamics however good
its raw IOU looks.

Render everything:
```bash
python scripts/report.py --markdown results/report.md   # scored table vs targets
python scripts/visualize.py                             # BEV figures → results/figures/
```

---

## 9. Conventions to not break

- **All hyperparameters live in `configs/default.yaml`.** No magic numbers in code.
- **Every module has a `__main__` demo**, and `tests/test_demos.py` asserts none is
  broken. If you add a module with a demo, add it to `DEMO_MODULES`.
- **Checkpoints embed the config that produced them.** Always load through
  `utils.config_from_checkpoint`; never assume `default.yaml` matches a checkpoint.
- **Both world models expose `loss(cond, target)` and `predict(cond)`.** Training, eval,
  RL and the dashboard never branch on architecture — keep it that way.
- **The numpy and torch zone poolers must stay numerically identical.** PPO builds
  observations with one and the orchestrator uses the other; they once differed by 0.50
  and the policy was trained on a different representation than it ran on.
- **Commit one file per commit, no `Co-Authored-By` or attribution trailers, straight to
  `main`.** No feature branches.

Run before every commit:
```bash
make test && make demos
```

---

## 10. If you have spare cluster time

In priority order, once real numbers exist:

1. **The unet-vs-diffusion ablation.** Both are trained by the same command with one
   override. Diffusion exists specifically because BCE is mean-seeking and blurs
   multimodal futures — which in this system are concentrated in the shadow. The
   `shadow_diversity_ratio` is the measurable form of that claim: >1 means the model tied
   its uncertainty to visibility, ≈1 means the samples are just noisy everywhere.
2. **`sbatch slurm/sweep.sbatch`** — 6-config array over base_channels / lr / pos_weight,
   each with its own checkpoint dir. `scripts/report.py --compare` diffs the results.
3. **Real-video ingestion.** `ingestion/depth.py` wraps Depth Anything V2 and works, but
   only ever sees sim frames. Pointing it at real footage → BEV + shadow → curated clips
   would close the "everything is synthetic" gap.
4. **MPPI/CEM planning through the world model.** Predictions are currently pooled to 36
   numbers and fed to a reactive policy — a feature extractor, not a world model. Rolling
   the diffusion model forward under candidate action sequences and choosing actions safe
   across *all* sampled futures would be risk-averse planning under occlusion.
