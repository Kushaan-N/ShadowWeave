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

> **Running this on a GPU cluster?** Read [`HANDOFF.md`](HANDOFF.md) first —
> setup, the exact job order, storage requirements, and the traps that cost hours.

---

## Quick start

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate shadowweave
pip install -e .

make smoke                              # end-to-end smoke test, no training needed
make bench                              # size batch/workers before booking GPU time
make test                               # 210 tests
```

`make help` lists everything:

| target | what it does |
|---|---|
| `setup` `test` `demos` | install, run the suite, run every module demo |
| `smoke` `bench` `preflight` | end-to-end check, throughput sizing, node readiness |
| `data` `train` `rl` `eval` | generate rollouts, train the world model, PPO, evaluate |
| `report` `viz` | scored results table, BEV prediction figures |
| `dashboard` | live Gradio demo |
| `clean` `clean-data` | drop caches / drop rollouts and memmap sidecars |

After a run:

```bash
python scripts/report.py                       # scored table vs the project targets
python scripts/report.py --markdown report.md  # same, as markdown
python scripts/report.py --compare results/a.json results/b.json
python scripts/visualize.py                    # BEV prediction figures -> results/figures/
```

Every module also runs standalone with a demo (`make demos` runs all of them, and
`tests/test_demos.py` asserts none is left broken):

```bash
python -m shadowweave.shadow.raycast    # zone uncertainty + throughput
python -m shadowweave.shadow.bev        # BEV occupancy / shadow projection
python -m shadowweave.sim.mujoco_env    # sim observations across all three tiers
python -m shadowweave.audio.hrtf        # spatialisation + phase-continuity checks
python -m shadowweave.eval.baselines    # baselines, lead time, calibration
python -m shadowweave.world_model.ddpm  # diffusion sampling + sample diversity
python -m shadowweave.eval.reasoning    # spatial-reasoning question generation
```

## Dashboard

```bash
make dashboard          # or: python -m shadowweave.dashboard.app --live
```

Serves on http://localhost:7860 — camera feed, shadow fan, egocentric BEV with the
planned path, and a zone bar chart, refreshed by `gr.Timer` at 5Hz.

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

### Two world models

`world_model.architecture` selects between them; both expose the same
`loss(cond, target)` / `predict(cond)` interface, so training, eval, RL and the
dashboard are agnostic to which one is loaded.

| | `unet` (default) | `diffusion` |
|---|---|---|
| objective | weighted BCE + horizon smoothness | epsilon-MSE (DDPM, cosine schedule) |
| output | one deterministic occupancy map | samples from p(future \| observation) |
| inference | single forward pass | DDIM, `sample_steps` is the latency knob |

**Why diffusion is here and not just on the label.** BCE is mean-seeking. Where the
future is genuinely multimodal the deterministic model converges to the conditional
*mean* and hedges — and this system has an unusually clean source of multimodality:
cells the sensor cannot see. Behind an obstacle there may or may not be something;
averaging those futures produces a grey smear over exactly the region the project
exists to reason about. Diffusion draws coherent hypotheses instead.

That makes the claim measurable rather than rhetorical. `shadow_diversity` reports the
ratio of sample disagreement inside shadow to disagreement in observed space:

- **ratio > 1** — the model has tied its uncertainty to visibility
- **ratio ≈ 1** — the samples are just noisy everywhere

A deterministic model cannot be scored on this at all; its std is identically zero.

```bash
make train                                              # deterministic U-Net
python -m shadowweave.world_model.train --overrides world_model.architecture=diffusion
```

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
- At most `agents.local.max_simultaneous_cues` zones sound at once, and a falling
  object always keeps a slot regardless of how low its uncertainty is.
- Max zone uncertainty > `orchestrator.uncertainty_stop_threshold` → stop pattern,
  navigation output zeroed.
- **Fail toward "stop", never toward "clear".** A non-finite depth frame or
  uncertainty grid is treated as maximum danger, not as silence — silence means
  *safe* to someone navigating by ear. `utils.sanitize_depth` and the orchestrator's
  finiteness check both enforce this.
- Checkpoints embed the config that produced them; always load through
  `utils.config_from_checkpoint`.
- The numpy and torch zone poolers must stay numerically identical — PPO builds
  observations with one and the orchestrator uses the other.

---

## Repo layout

```
shadowweave/
├── ingestion/     camera.py, depth.py (Depth Anything V2)
├── shadow/        raycast.py (core), bev.py, occupancy.py, zones.py, visualizer.py
├── world_model/   unet.py (U-Net + ConvLSTM), ddpm.py (diffusion), dataset.py, train.py
├── agents/        local_agent.py, global_agent.py, orchestrator.py
├── audio/         hrtf.py, cues.py
├── sim/           mujoco_env.py, synthetic_data.py, usd_export.py (OpenUSD)
├── dashboard/     app.py (Gradio)
├── eval/          metrics.py, baselines.py, reasoning.py, run_eval.py
├── configs/       default.yaml — all hyperparameters
├── console.py     terminal tables/progress (degrades to plain text in SLURM logs)
└── utils.py       device, seeding, config loading, checkpoint I/O
slurm/             sbatch scripts, env.sh, preflight.py
scripts/           pipeline_test.py, benchmark.py, report.py, visualize.py
data/kemar/        MIT KEMAR HRTF impulse responses (tracked; 152KB)
tests/             pytest suite — see below
```

The test suite is organised by what each file protects:

| file | protects |
|---|---|
| `test_shadow.py` | zone discrimination, shadow semantics, BEV projection |
| `test_contracts.py` | interfaces where two modules previously disagreed on shape or meaning |
| `test_pipeline.py` | dataset loading, checkpoint round-trip, metrics |
| `test_sim.py` | the sim actually varies over time; real collisions; fixed depth scale |
| `test_optimizations.py` | fast paths stay numerically equal to the slow ones |
| `test_verification.py` | regressions found by adversarial review |
| `test_robustness.py` | DDP, degenerate sensor input, hostile configs, shard arithmetic |
| `test_demos.py` | every module's `__main__` still runs |
| `test_diffusion.py` | the diffusion model samples rather than collapsing; shadow-diversity metric |
| `test_usd_export.py` | exported scene geometry matches MuJoCo exactly, not just "a file appeared" |
| `test_reasoning.py` | the benchmark's control is trivially passable and its hard questions are not |

All hyperparameters live in `configs/default.yaml`. Override anything from the CLI:

```bash
python -m shadowweave.world_model.train --overrides world_model.lr=3e-4 bev.size=128
```

---

## Performance targets

| Metric | Target | Status |
|---|---|---|
| Full pipeline latency (camera → audio) | < 100ms | ✅ 2.4ms p50, 3.3ms p95 |
| Shadow-ray throughput | ≥ 15Hz | ✅ ~1700Hz |
| Dashboard render rate | ≥ 5Hz | ✅ `gr.Timer` at 5Hz |
| World model IOU @ 5s | > 60% | ⏳ needs a full training run |
| Agent collision rate (hard tier) | < 10% | ⏳ needs PPO training |
| Falling-object lead time | ≥ 3s | ⏳ needs a full training run |

Latency measured on Apple M-series MPS; a CUDA node is faster. The figures are
**steady state** — `pipeline_test.py` excludes 3 warmup frames, because lazy imports,
scipy's FFT plan and shader compilation land on the first call and cost ~1s. Including
them made the reported p95 depend on `--frames` rather than on the pipeline.

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

## Spatial audio

The 9-cell uncertainty grid maps 1-to-1 onto 9 azimuth zones, and `CueMapper` turns it
into 27 audio parameters (direction, intensity, pitch per zone) that `HRTFEngine`
convolves into stereo.

The MIT KEMAR impulse responses are **committed under `data/kemar/`** (37 files,
152KB), so spatialisation works out of the box — `HRTFEngine.is_spatialised` is `True`
on a fresh clone with no download step. Without them the engine falls back to
constant-power panning, which still separates left from right but loses the
elevation and front/back cues.

KEMAR only measures one side of the head; left-side azimuths are produced by swapping
the ear channels. The data is free to use provided the authors are cited — see
`data/kemar/README.md` for the citation.

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

### Spatial-reasoning benchmark

```bash
make reason        # or: python -m shadowweave.eval.reasoning --split val
```

Auto-generates questions with *verifiable* answers from simulator ground truth —
`"Is there an obstacle to your left that you cannot currently see?"`,
`"Will the path ahead be blocked in 5 seconds?"`,
`"How many separate obstacles are hidden?"` — and scores the world model against two
trivial strategies. This is only possible because the scenes are simulated: real
footage has no ground truth for what is behind a crate.

Questions split into two classes, and the split is the point:

| class | who can answer | role |
|---|---|---|
| `perception_only` | a camera alone | **control** — must be ~1.0 for everyone, or the question set is broken |
| `needs_world_model` | only a model that predicts into shadow | where all the signal is |

Two baselines bracket the problem: `blind_optimistic` assumes shadow is empty,
`blind_pessimistic` assumes it is full. Each wins a different subset depending on
whether the shadow happens to be occupied, so beating only one proves nothing —
`reasoning_gain_over_best_baseline` requires clearing **both**.

```
  predictor          control  needs world model  count MAE
  model                1.000              0.587     601.87
  blind_optimistic     1.000              0.297       3.07
  blind_pessimistic    1.000              0.560       1.77
```

A large `count MAE` is diagnostic rather than incidental: it means the model is
emitting speckle instead of objects, which IOU alone will not reveal.

### OpenUSD export

Rollouts can be written as an OpenUSD twin alongside the `.npz` training data:

```bash
python -m shadowweave.sim.synthetic_data --episodes 400 --split train --usd-dir data/usd
python -m shadowweave.sim.usd_export --out scene.usda --difficulty debris   # one episode
```

MuJoCo is cheap to roll out but renders flat, untextured frames. USD is what Omniverse
and Isaac Sim read, so the export carries the *same* randomised scene, the *same* agent
trajectory and an egocentric camera prim — letting an episode be re-rendered
photorealistically later without regenerating it or breaking correspondence to its
training targets. Stage is Z-up in metres, matching MuJoCo.

Requires `usd-core` (`pip install -e ".[usd]"`); nothing else in the pipeline imports it.

> **Rollouts generated before the egocentric-BEV change are unusable** — their targets
> were world-frame and constant. `RolloutDataset` detects the old schema and tells you
> to regenerate.

---

## Status

The pipeline runs end to end and every stage has been executed at least once:
rollout generation, single-GPU training, 2-rank distributed training (gloo/CPU), PPO
with both a single env and subprocess workers, evaluation, and reporting. 150 tests
pass, along with all 14 module demos.

Still unproven, and the reason to book a GPU node:

- **The accuracy targets.** IOU@5s, collision rate and lead time all need a real
  training run. Start with `./slurm/pipeline.sh`.
- **NVIDIA-specific paths.** NCCL and CUDA AMP have only been exercised through their
  CPU/gloo equivalents — there is no NVIDIA hardware in the development environment.
  Run `slurm/preflight.py` on a compute node first; it checks CUDA, AMP, headless
  EGL rendering and the rollout schema before a long job is queued.

Anything in `data/rollouts/` or `checkpoints/` from before the egocentric-BEV change
is unusable and is detected rather than silently consumed.
