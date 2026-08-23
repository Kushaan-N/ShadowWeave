# ShadowWeave: Calibrated Amodal Occupancy Completion into Occluded Space from a Single Depth Frame

*Full draft, 2026-08-21. Completion-first framing (see `DRAFT.md`, `RESULTS.md`). All numbers
traceable to `RESULTS.md`; re-verify at freeze. Prose is workshop-length and will be tightened
to the CFP page limit once available.*

---

## Abstract

A mobile agent must reason about space it cannot currently see — the region occluded by the
walls and obstacles it is navigating around ("shadow"). We study how well a model can
**complete** occupancy in that occluded region from a **single monocular-depth frame**, and,
crucially, whether it knows *when it is guessing*. ShadowWeave projects one depth frame into an
egocentric bird's-eye-view (BEV) occupancy grid, marks the unobserved cells with an explicit
shadow mask, and predicts occupancy inside that mask together with a per-cell uncertainty. On a
MuJoCo indoor benchmark, the model beats a persistence baseline by a micro-averaged shadow-IoU
gain of **+0.31 to +0.36 at a 5 s horizon** (95% per-episode bootstrap CIs clear of zero at
every horizon), far exceeds an observation-blind dataset prior even at the prior's best
threshold (0.61–0.72 IoU vs 0.08–0.10), and its uncertainty is **better calibrated inside
shadow (ECE 0.043–0.047) than in directly observed space (0.058–0.086)**. We introduce a decomposition that separates *amodal
completion* of persistent hidden structure from *temporal forecasting* of dynamics, and use it to
make an honest claim: the gain is almost entirely completion — the pure-forecasting residual sits
at the noise floor. Randomizing room geometry per episode leaves the gain statistically unchanged
on rooms the model never saw, ruling out memorization of a fixed layout. The completed grid and
its uncertainty drive an eyes-free spatial-audio navigation interface at 7 ms (p95)
perception-to-planning latency, well inside a 20 Hz budget. We release the evaluation methodology — an empty-credit-immune
micro-averaged gain, a completion/forecasting decomposition, and per-episode confidence
intervals — as a reusable protocol for honest occupancy-completion claims.

---

## 1. Introduction

Agents act in space they cannot see. A person stepping into a corridor, a robot rounding a
shelf, a blind traveler crossing an unfamiliar room — each must anticipate structure that is
occluded by the very obstacles they are navigating around. Sensing provides a partial view; the
rest must be inferred. For safety-critical, eyes-free use, that inference is only useful if it is
accompanied by an honest estimate of its own reliability: a confident hallucination of clear
space behind a wall is worse than a calibrated "I don't know."

Prior work addresses parts of this problem but not their conjunction. Semantic scene completion,
from MonoScene through VoxFormer and VisHall3D, hallucinates dense geometry beyond the visible
frontier from a single image, but targets outdoor driving voxels scored by mIoU alone, with no
notion of calibrated confidence in the hallucinated region. Occupancy world models such as
OccWorld forecast the *temporal* evolution of multi-frame occupancy for autonomous driving.
Closest to our setting, ProxMaP completes indoor proximal occupancy from a single top-down view
to improve navigation — but it emits a point estimate, with no explicit occlusion mask, no
calibrated uncertainty in the hidden cells, and no attribution of *where* its predictive power
comes from.

We present **ShadowWeave**, which from a single monocular-depth frame produces (i) an egocentric
BEV occupancy grid, (ii) an explicit shadow mask marking the cells the sensor cannot see, and
(iii) a calibrated per-cell occupancy probability inside that mask. We evaluate it with a
protocol designed to resist the standard failure modes of occupancy metrics: an
empty-credit-immune, micro-averaged shadow-IoU gain over a persistence baseline; per-episode
bootstrap confidence intervals; and a static/dynamic/never decomposition that attributes the gain
to *completion* of persistent hidden structure versus *forecasting* of dynamics. Downstream, the
completed grid and its uncertainty drive an A* planner and a nine-zone HRTF spatial-audio
interface for eyes-free navigation.

Our contributions are:

- **A calibrated amodal-completion result.** From one depth frame, the model completes occluded
  occupancy with a micro-averaged shadow gain of +0.31–0.36 at 5 s (CIs clear of zero at every
  horizon), and its in-shadow uncertainty is *better* calibrated (ECE 0.043–0.047) than in
  observed space — evidence that the model conveys, rather than merely produces, hidden-space
  predictions.
- **A decomposition methodology** that separates completion from forecasting. It reveals that the
  gain is spatial completion of persistent hidden structure; the pure-forecasting residual is at
  the noise floor (fixed static +0.000; moving tier +0.007 [+0.002, +0.012] at 5 s — statistically
  significant but practically small, and decaying with horizon as real forecasting should). We
  report this negative honestly: most pooled occupancy numbers in the literature are vulnerable to
  exactly the confound this decomposition dissects.
- **A memorization confound-kill.** Randomizing room geometry per episode leaves the gain
  statistically unchanged on never-seen rooms (+0.306 fixed vs +0.309 [+0.294, +0.324]
  randomized), showing the completion is per-scene inference, not a memorized constant layout.
- **A real-time eyes-free system** that turns completed occupancy and its uncertainty into an
  A* path and nine-zone spatial-audio cues at 7 ms p95 perception-to-planning latency.

---

## 2. Method

Fig. `architecture.png` summarizes the pipeline: a single depth frame is lifted to an
egocentric BEV with an explicit shadow mask, a single-frame U-Net completes occupancy with
per-cell probabilities, and the completed grid feeds the planning and audio layers
(system context).

### 2.1 From a depth frame to an egocentric BEV with a shadow mask

At each step the agent observes a single monocular-depth image (in simulation, the renderer's
depth; in deployment, a monocular-depth network such as Depth-Anything-V2-Small). A differentiable
projector lifts the depth map into an egocentric bird's-eye-view (BEV) occupancy grid of size
96×96 covering 5 m ahead, together with a **visibility** grid: cells along a ray up to the first
returned surface are marked observed-free, the surface cell observed-occupied, and everything
beyond the surface **shadow** — geometry the sensor cannot see. Thresholding visibility at 0.5
yields the binary shadow mask that defines the region of interest for every metric in this paper.
Optical flow between consecutive BEV frames provides an optional motion channel.

### 2.2 World model

A U-Net (31M parameters) takes the single-frame BEV stack and predicts occupancy at horizons of
1, 3, 5, and 10 s as per-cell probabilities. It is trained with binary cross-entropy (positive
class up-weighted, since BEV targets are ≈7% occupied) on 400 training episodes, with early
stopping and exponentially-averaged (EMA) weights; we report the epoch-8 checkpoint (validation
loss 0.300, validation IoU@5s 0.665). Ablations (§4.6) show the visibility and flow channels are
redundant — the completion signal is carried by the occupancy channel alone — so the deployable
model reduces to depth→occupancy→completion. The input is a **single frame**: the model has no
multi-frame memory, which is what makes the completion-vs-forecasting distinction (§3.2) sharp.

### 2.3 Uncertainty in shadow

The model's per-cell occupancy probability *is* its uncertainty estimate; the claim we test in
§4.5 is that this probability is **calibrated inside the shadow region specifically**, not merely
grid-wide (where easy free space dominates). Calibrating uncertainty in the occluded region is the
property that makes the output safe to consume downstream.

### 2.4 Downstream eyes-free interface (system context)

The completed occupancy grid and its uncertainty drive an A* global planner (which owns
goal-seeking and pays an extra cost to route through unobserved cells) and a reactive local
collision-avoidance policy. Occupancy and uncertainty in nine azimuth zones are rendered as
HRTF-spatialized audio cues for eyes-free navigation. We describe this interface for context and
report its latency and navigation behavior honestly (§4.7); the audio interface itself is a
system demonstration, not a user-evaluated result.

---

## 3. Evaluation methodology

Occupancy metrics are easy to report and easy to inflate. Because our claim lives entirely inside
a sparse, mostly-empty sub-region (shadow is ~92% never-occupied), we adopt a protocol built to
resist the specific ways such a claim can look better than it is. We consider this protocol a
contribution in its own right.

### 3.1 The empty-credit trap and micro-averaging

The natural masked IoU scores an empty prediction against an empty masked region as a perfect 1.0
("correctly predicted nothing here"). Since most shadow cells are legitimately empty, this inflates
both the model and the baselines, and the inflation is not uniform across sub-masks. We therefore
report **micro-averaged** shadow IoU: pool integer intersection and union counts across all frames
and divide once, so rare sub-masks cannot each contribute a free 1.0. Empty denominators report as
undefined, never as 1.0. The defended quantity is the **gain** over a persistence baseline (the
observed frame carried forward), not the absolute level.

### 3.2 Completion vs. forecasting decomposition

Within the shadow mask, we classify each cell by its occupancy pattern across the prediction
horizons — valid because all horizons are rendered from the same issue pose, so static geometry
projects to identical cells at every horizon:

- **STATIC** — occupied at *all* horizons: persistent hidden structure (walls, static obstacles,
  settled debris); ≈7% of shadow. Recall here measures **amodal completion**.
- **DYNAMIC** — occupied at *some but not all* horizons: transient-in-window structure (arrivals,
  departures, pass-throughs); ≤1.5% of shadow. IoU here measures genuine **forecasting** beyond
  persistence.
- **NEVER** — occupied at *no* horizon: free hidden space; ≈92% of shadow. The false-positive
  test.

Reporting completion and forecasting separately is what lets us state honestly that the gain is
completion, not clairvoyance — a distinction the pooled shadow-IoU hides.

### 3.3 Per-episode bootstrap confidence intervals

Frames within an episode share geometry and pose and are correlated, so resampling frames would
understate uncertainty. We resample **episodes** with replacement (B = 10,000), recompute the
pooled micro-gain per replicate, and take the paired model-minus-persistence difference within each
replicate (the two are positively correlated across episodes). We report the resulting 2.5/97.5
percentile interval for every headline gain.

### 3.4 Calibration in shadow

We measure expected calibration error (ECE) restricted to shadow cells, contrasted with observed
cells and with a walls-excluded variant, using equal-mass binning (a fine 1000-bin histogram merged
into 15 equal-count macro-bins, because in-shadow probabilities cluster near the prior and
equal-width bins would be mostly empty). ECE-in-shadow close to or below ECE-in-observed is the
evidence that uncertainty propagates faithfully into unseen space.

---

## 4. Experiments

### 4.1 Setup

We use a MuJoCo single-room benchmark with three difficulty tiers — **static** (fixed obstacles),
**moving** (kinematically driven movers), and **debris** (objects that fall and settle) — with 400
training and 80 validation episodes on disjoint seeds; the per-tier decomposition and CI
analyses use clean single-tier validation sets of 30 episodes (9,000 frames) each, and the
randomized-geometry validation (§4.4) uses 80 episodes. The BEV is 96×96 over 5 m; horizons are
1/3/5/10 s. Unless noted, fixed-geometry results use a 6×6 m room; §4.4 randomizes geometry.
Reproducibility is pinned: the fixed-geometry scene generation is verified bit-exact against a
golden hash, and a full rollout re-run reproduces every accuracy metric bit-identically
(wall-clock latency, which is hardware-dependent, is the one exception).

### 4.2 Shadow-gain headline

The model predicts occupancy in cells the sensor cannot see, beating persistence at every horizon.
Micro-averaged shadow gain at 5 s, per tier, with 95% bootstrap CIs:

| tier (fixed val, @5 s) | micro shadow gain | 95% CI |
|---|:---:|:---:|
| static | +0.306 | [+0.288, +0.327] |
| moving | +0.363 | [+0.349, +0.378] |
| debris | +0.315 | [+0.281, +0.350] |

Every interval is clear of zero at every horizon, so the gain is statistically significant rather
than sampling noise (Fig. `shadow_gain_ci.png`). In policy rollouts the frame-averaged (macro)
shadow gain is +0.325/+0.322/+0.320/+0.178 at 1/3/5/10 s.

**Observation-blind prior.** Persistence could be a weak baseline for *completion*, so we also
evaluate a pose-marginal dataset prior — the mean training-set occupancy in the egocentric frame,
using no observation at all — swept over thresholds and reported at its most favorable one. It
reaches only 0.08–0.10 micro shadow IoU across tiers, and 0.09 on the randomized-geometry set:
to recall hidden structure (0.87–0.92) it must blanket 72–79% of genuinely empty hidden space
with false positives, whereas the model achieves 0.61–0.72 IoU at a 1.4–3.4% false-positive
rate. Completion is therefore observation-conditional
inference, not a memorized average room; together with the geometry-randomization result (§4.4)
this rules out both memorized layouts and memorized marginals. (Raw macro shadow-IoU levels — model 0.744 vs
persistence 0.424 at 5 s — are empty-credit inflated and are reported only for reference; the
micro-gain is the defended number.)

### 4.3 What the gain is: completion, not clairvoyance

Decomposing the shadow mask (§3.2) attributes the gain almost entirely to amodal completion of
persistent hidden structure:

| component (fixed val @5 s) | model | persistence |
|---|:---:|:---:|
| persistent-structure recall (amodal completion) | 0.83–0.93 | 0.37–0.51 |
| dynamic IoU, moving tier (1s→10s) | 0.092 → 0.031 | 0.060 → 0.020 |
| dynamic IoU, debris tier | 0.020 → 0.019 | 0.006 → 0.023 |
| false-positive rate on never-occupied cells | 0.014–0.034 | 0.022–0.027 |

Excluding all persistent structure, the residual "nostatic" gain sits at the noise floor: fixed
static +0.000, moving +0.007 (CI [+0.002, +0.012]), randomized-geometry static +0.002
([+0.001, +0.004]) at 5 s. The honest claim is therefore **amodal completion of hidden
structure**, with a small but statistically significant dynamic-forecasting signal on the moving
tier that decays with horizon as genuine forecasting should. We do not claim dynamic forecasting on
debris, where the dynamic signal is at the noise floor and persistence matches the model beyond
1 s.

### 4.4 Robustness to room geometry (memorization confound-kill)

A fixed room shell could let the model memorize a constant wall prior. We regenerated the data with
per-episode randomized room width and depth (each drawn i.i.d. in [5, 8] m, obstacle counts scaled
to hold the ~7% positive fraction), retrained the model from scratch, and evaluated on 80 held-out
randomized rooms:

| static tier, @5 s | micro shadow gain | 95% CI | completion recall |
|---|:---:|:---:|:---:|
| fixed 6×6 geometry | +0.306 | [+0.288, +0.327] | 0.829 |
| **randomized geometry** | **+0.309** | **[+0.294, +0.324]** | 0.854 |

The gain does not drop — the intervals overlap almost entirely, and completion recall is if
anything slightly higher on never-seen rooms. Memorizing a constant shell contributes essentially
nothing; completion is per-scene inference.

### 4.5 Calibration in shadow

At the 5 s horizon, in-shadow ECE is 0.043–0.047 across tiers — *better* than in observed space
(0.058–0.086) — with Brier 0.022–0.036 (Fig. `reliability.png`); excluding the persistent
structure entirely, in-shadow ECE remains 0.055–0.056 on the tiers with dynamic content (on the
static tier the structure-excluded region has no positive labels, so no calibration curve exists
there), so the calibration is not carried by the easy always-occupied cells. The model is underconfident in the mid-range but places little mass
there. Calibration is not an artifact of the fixed room: on randomized geometry,
in-shadow ECE is 0.050–0.067 vs observed 0.056–0.077, with shadow becoming the better-calibrated
region at the longer 5–10 s horizons.

### 4.6 Ablations: occupancy alone carries the signal

Dropping either auxiliary input channel and retraining leaves the gain intact:

| variant | shadow gain @5 s (micro, static/moving/debris) | rollout shadow gain @5 s |
|---|---|---|
| full (occupancy+visibility+flow) | +0.306 / +0.363 / +0.315 | +0.320 |
| no visibility | +0.309 / +0.360 / +0.313 | — |
| no flow | — | +0.381 |

The no-visibility per-tier gains are within CI of the full model; the no-flow rollout gain is if
anything higher. Both auxiliary channels are redundant, which simplifies the input and removes
optical-flow estimation from the runtime critical path.

### 4.7 System: latency, anticipation, navigation

**Latency.** The perception-to-planning forward path (BEV projection, raycaster, world-model
forward, orchestrator; batch 1, deterministic U-Net) runs at 5.97 ms p50 / 7.00 ms p95 — well
inside a 50 ms (20 Hz) budget. The monocular-depth stage is excluded and is not free: measured in
isolation, Depth-Anything-V2-Small takes 49.9 ms p50 / 50.7 ms p95 on the evaluation-era GPU
(GTX 1080 Ti, 480×640 input) — comparable to the entire budget on that hardware. Run
sequentially, the measured end-to-end latency including depth is 54.4 ms (18 Hz) on the same
GPU; a deployed system therefore runs depth asynchronously at its own ~20 Hz rate while the
7 ms completion path consumes the most recent depth frame, at the cost of one frame of depth
staleness. Audio synthesis remains excluded.

**Anticipating occupancy revealed in shadow (a falling-object *proxy*).** We measure whether the
model flags occupancy that materializes in unobserved space before it is revealed. Cell-level
detection is 0.787 at a horizon-granular mean lead of 2.62 s. A component-level variant that
scores each connected arrival region gives detection 0.752 and exposes what the cell-level
false-alarm rate (0.021) hides: component-level precision is only 0.714. Three definitional
caveats apply: events are counted per due prediction (one object aloft is scored at every issue
step and horizon it is due, not once per object); "arrival" keys on unobserved-at-issue rather
than newly-occupied cells, so occluded static geometry revealed by agent motion also counts and
the metric pools all tiers; and precision is an unmatched majority-overlap over predicted
components, sensitive to fragmentation. Lead time is bounded by the {1,3,5,10} s horizon grid —
an interface limit — and the ≥3 s target is not met. We report this metric as a diagnostic, not
a strength.

**Navigation.** With the completed grid feeding the planner, the trained reactive policy achieves a
per-episode collision rate of 0.367 (vs 0.60 untrained) and path straightness 0.773, but does not
meet the ≤0.10 collision target. We present navigation as an honest system demonstration.

### 4.8 Generative variant (ongoing work)

We additionally trained a conditional-diffusion (DDPM) world model as an alternative to the
deterministic U-Net, motivated by the fact that BCE is mean-seeking and hedges into a grey smear
exactly where the future is multimodal — the shadow — whereas a generative model can sample
coherent alternatives and expose its disagreement as uncertainty concentrated in the occluded
region. A full calibrated evaluation (in-shadow sample-diversity ratio and calibration, the fair
comparison given that BCE and DDPM optimize different objectives) is ongoing: iterated DDIM
sampling exceeds our batch-eval budget, and we leave a like-for-like generative comparison to
future work. The deterministic model's calibrated per-cell probabilities already carry the
uncertainty narrative of this paper.

---

## 5. Related work

Predicting the structure of unobserved space has been approached from three angles. Semantic scene
completion, from SSCNet through MonoScene (Cao & de Charette, 2022) and its monocular successors
VoxFormer (Li et al., 2023) and VisHall3D (Lu et al., 2025), infers dense voxel geometry — and even
hallucinates geometry beyond the visible frontier — from a single image, but targets outdoor
driving voxels evaluated purely by mIoU/IoU, without calibrated uncertainty in the occluded region.
Occupancy world models such as OccWorld (Zheng et al., 2024), together with occupancy-forecasting
benchmarks (Occ3D; Tian et al., 2023) and self-supervised raycasting forecasters (Khurana et al.,
2023), instead predict the *temporal* evolution of multi-frame occupancy for autonomous vehicles.
Closest in spirit, ProxMaP (Sharma et al., 2023) completes indoor proximal occupancy from a single
view for navigation, yet emits only point estimates. ShadowWeave is distinct in delivering
egocentric BEV amodal completion into explicitly masked occluded space from a single
monocular-depth frame, with calibrated uncertainty inside the hidden region, and a
completion-versus-forecasting decomposition showing the gain is spatial completion rather than
dynamics.

---

## 6. Limitations

Our evaluation is in simulation; ground-truth occluded occupancy is unavailable on real footage
without instrumentation, so no quantitative real-world claim is made. The perception front-end
nonetheless accepts real monocular depth (Depth-Anything-V2), so sim-to-real transfer of the
completion model is the natural next step. The model performs
amodal *completion*, not multi-step *forecasting*: the pure-dynamics residual is at the noise floor,
and we frame the contribution accordingly. Navigation does not meet its collision target and the
audio interface is not user-evaluated; both are presented as system context. Falling-object lead
time is bounded by the discrete horizon grid.

---

## 7. Conclusion

Calibrated amodal completion of occluded occupancy is achievable from a single monocular-depth
frame, transfers to room geometries the model never saw, and can be delivered within a real-time
perception-to-planning budget for an eyes-free interface. Equally important is *how* we measure it:
an empty-credit-immune micro-gain, per-episode confidence intervals, and a decomposition that
separates completion from forecasting keep the claim honest and expose exactly where the predictive
power lies. We hope the protocol is reused: many occupancy-completion results would read differently
under it.

---

## References

See `RELATED_WORK.md` for verified BibTeX (ProxMaP, VisHall3D, OccWorld, MonoScene, VoxFormer,
Occ3D, Khurana et al.).
