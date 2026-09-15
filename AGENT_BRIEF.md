# ShadowWeave — briefing for an agent running checks

## What this project is
ShadowWeave is a **world model for eyes-free navigation via 3D spatial audio**. Pipeline:
monocular depth → 9-zone shadow-ray uncertainty + egocentric BEV occupancy (with a
visibility channel) → **world model forecasts BEV occupancy at 1/3/5/10 s INTO occluded
("shadow") space** → 20 Hz PPO local agent (reactive collision-avoider) + 2 Hz A* global
planner → HRTF spatial audio. Target venue: a NeurIPS **world-model workshop**.

**The core claim** = the world model predicts occupancy in space the sensor never observed,
beating trivial baselines. The headline metric is
`model_shadow_gain_over_best_baseline_5s` (model shadow-IOU minus the best of
persistence/empty), which is **> 0** (≈ +0.32 on policy rollouts, ≈ +0.36 on the fixed val
set). Raw IOU alone is misleading — BEV is ~7% positive.

## Where things live
- **Code (git repo):** `/work/pi_sniekum_umass_edu/kn/ShadowWeave` — origin
  `github.com/Kushaan-N/ShadowWeave`, branch `main`. ⚠️ `/work` is ~99% full; do NOT write
  large artifacts here.
- **Scratch workspace (`$WS`):** `/scratch4/workspace/knaskar_umass_edu-shadowweave`
  — all large artifacts. **Expires ~2026-09-06** (no recovery); regenerable inputs are fine
  to lose. ⚠️ The paper checkpoint is **`world_model/best.pt` (epoch 8)** — `final.pt` is
  the post-early-stop overfit checkpoint (val_loss 0.408 vs 0.300). A weights-only copy of
  best.pt is already preserved at `/work/.../ShadowWeave/checkpoints/world_model/best_weights.pt`.
  - **venv:** `$WS/sw-venv/bin/python` (Python 3.11, torch cu126 — works on all Unity GPUs incl. GTX 1080 Ti / sm_61).
  - **data:** `$WS/rollouts/{train,val}/*.npz`; clean per-difficulty val sets at
    `$WS/rollouts_val_{static,moving,debris}/val/`.
  - **world-model checkpoints:** `$WS/checkpoints/world_model/` (full: `best.pt` 497 MB with
    optimizer, `final.pt` 124 MB weights-only), `world_model_noshadow` (no-visibility ablation),
    `world_model_noflow` and `world_model_diffusion` (running now).
  - **fixed-val / decomposition results:** `$WS/results_noshadow/`,
    `$WS/val_pertier/{static,moving,debris}.json` (full model) + `val_pertier/novis/`,
    `$WS/val_decomp/{full,novis}/{static,moving,debris}.json`, and (pending)
    `$WS/results_noflow/`, `$WS/results_diffusion/`.
- **In the repo directory (small, durable on /work):** trained PPO policy
  `checkpoints/local_agent/best.pt`; rollout-eval summaries `results/eval_summary.json`
  (full pipeline), `results_smoke/`, `results_rl_smoke/`; figures in `figures/`;
  write-ups `RESULTS.md`, `report.md`, adversarial-review record `REVIEW.md`.
  Note `checkpoints/` is **gitignored**; the eval summaries are git-tracked via explicit
  `.gitignore` exceptions (added 2026-08-20) — figures and write-ups are tracked.

## Hard execution constraints
- **You are on the LOGIN NODE.** Never run GPU/CUDA/MuJoCo-render compute or `sbatch` GPU
  jobs and expect them to run here. GPU work goes through SLURM: partition `gpu`, account
  `pi_sniekum_umass_edu` (NOT the default `pi_andrewlan`). Env for jobs: `SBATCH_PARTITION=gpu
  SBATCH_ACCOUNT=pi_sniekum_umass_edu`, plus `WS`, `SW_VENV`, `SW_DATA_ROOT`, `SW_CKPT_DIR`,
  `SW_PYTHON_MODULE=python/3.11.7`, `SW_CUDA_MODULE=cuda/12.6`.
- **Login-node-safe checks** (do these): reading files, git, config parsing, and **CPU-only
  torch inference** on a few batches — set `CUDA_VISIBLE_DEVICES=""` and pass `--max-batches N`.
  No MuJoCo GL (it hangs/needs a GPU).
- The launcher `run_*.sh` scripts (repo root) set the env and `sbatch` the SLURM jobs; they are
  untracked helpers.

## How to run checks (all login-node-safe unless noted)
```bash
PY=/scratch4/workspace/knaskar_umass_edu-shadowweave/sw-venv/bin/python
WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export CUDA_VISIBLE_DEVICES=""            # force CPU on the login node

# syntax
$PY -m py_compile <file.py>

# unit tests — full pytest HANGS on MuJoCo GL; deselect sim/env or use a timeout
timeout 200 $PY -m pytest -q tests/test_contracts.py -k "not sim and not env and not mujoco"

# read a results summary
$PY scripts/report.py --json results/eval_summary.json

# fixed-val shadow-IOU for any world-model checkpoint (quick check)
$PY scripts/eval_val_shadow.py --ckpt $WS/checkpoints/world_model/best.pt \
    --data $WS/rollouts --split val --batch-size 16 --max-batches 4

# static/dynamic decomposition + calibration-in-shadow
$PY scripts/eval_val_decomp.py --ckpt $WS/checkpoints/world_model/best.pt \
    --data $WS/rollouts_val_moving --max-batches 5
```
Full-scale versions of the last two run as GPU batch jobs (`slurm/val_shadow.sbatch`,
`slurm/val_decomp.sbatch`); launch via `run_val_pertier.sh` / `run_val_decomp.sh`.

## Key metrics to sanity-check
- **Core:** `model_shadow_gain_over_best_baseline_5s` > 0 (in any `eval_summary.json`).
- **Navigation:** `collision_rate` (lower better; trained ≈ 0.367 vs untrained 0.60),
  `path_efficiency`.
- **Decomposition (`eval_val_decomp.py`):** `static_coverage` (amodal completion ≈ 0.85–0.93,
  flat over horizons), `dynamic_iou` (forecasting; small, DECAYS with horizon — that decay is
  the signature of real forecasting), `shadow_fp_rate` (low), and `ece_shadow` vs `ece_observed`
  (model is well-calibrated in shadow, ECE ≈ 0.05).

## State of the study (2026-08-20)
- **World model works;** PPO was previously broken (trained worse than untrained) and was fixed
  by making the local agent a reactive collision-avoider (reward redesign; see the 6 commits
  around `0ef222f`..`69a94fd`). Current: collision 0.367, path_eff 0.773, shadow gain intact.
- **Ablations:** visibility channel is **redundant** (no-vis ≈ full, confirmed 3 ways);
  **no-flow** and **diffusion** are training now (`run_ablations.sh`, chained one-GPU-at-a-time;
  results will land in `$WS/results_noflow` and `$WS/results_diffusion`). Frame the diffusion
  comparison on `shadow_diversity_ratio` + calibration, NOT a raw IOU horse-race (it averages
  DDIM samples and is not capacity-matched).

## Gotchas
- `masked_iou` returns **1.0 on an empty masked union** (empty-credit) — never per-sample-mean it
  over rare sub-masks; micro-average (pool counts) instead.
- The **pooled** shadow gain is confounded by the 4 static walls that dominate shadow positives;
  use `eval_val_decomp.py` to separate completion from forecasting.
- Commit style for this repo: **one file per commit, no `Co-Authored-By` trailer.**
- Jobs default to the wrong SLURM account unless `SBATCH_ACCOUNT=pi_sniekum_umass_edu` is set.
