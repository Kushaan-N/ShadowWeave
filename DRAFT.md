# ShadowWeave — paper draft skeleton (completion-first framing)

Working draft, 2026-08-21. Framing locked to **amodal occupancy completion into occluded
space with calibrated uncertainty**, not temporal forecasting (see RESULTS.md, memory
`paper-framing-completion`). Numbers below are the defended micro-gains with CIs; verify
against RESULTS.md before submission. Venue-agnostic; tighten to the CFP page limit later.

---

## Title (options)

1. **ShadowWeave: Calibrated Amodal Occupancy Completion into Occluded Space from a Single Depth Frame**
2. Seeing Around Corners: Calibrated Completion of Occluded Occupancy for Eyes-Free Navigation
3. What Lies in Shadow: Amodal BEV Occupancy Completion with Uncertainty You Can Trust

*(1 is the safe, descriptive choice; leads with the two differentiators — amodal completion + calibration — and the single-depth-frame constraint.)*

---

## Abstract (~200 words, drafted)

A mobile agent must reason about space it cannot currently see — the region occluded by
walls and obstacles ("shadow"). We ask how well a model can **complete** occupancy in that
occluded region from a **single monocular-depth frame**, and — crucially — whether it knows
*when it is guessing*. ShadowWeave projects one depth frame into an egocentric bird's-eye-view
(BEV) occupancy grid, marks the unobserved cells with an explicit shadow mask, and predicts
occupancy inside that mask together with a per-cell uncertainty. On a MuJoCo indoor
benchmark, the model beats a persistence baseline by a micro-averaged shadow-IoU gain of
**+0.31 to +0.36 @5 s** (95 % bootstrap CIs over episodes clear of zero at every horizon),
and its uncertainty is **better calibrated inside shadow (ECE 0.043–0.047) than in directly
observed space**. We introduce a decomposition that separates *amodal completion* of
persistent hidden structure from *temporal forecasting* of dynamics, and use it to make an
honest claim: the gain is almost entirely completion — the pure-forecasting residual is at
the noise floor. Randomizing room geometry per episode leaves the gain unchanged on rooms
the model never saw, ruling out memorization. The completed grid and its uncertainty drive
an eyes-free spatial-audio navigation interface within a 7 ms perception-to-planning budget.

---

## 1. Introduction (drafted skeleton — workshop length)

**¶1 — Motivation.** Agents act in space they cannot see. A person stepping into a corridor,
a robot rounding a shelf, a blind traveler crossing a room — all must anticipate structure
occluded by the very obstacles they navigate around. Sensing gives a partial view; the rest
must be *inferred*. For safety-critical, eyes-free use the inference is only useful if it
also reports its own reliability: a confident hallucination behind a wall is worse than an
honest "unknown."

**¶2 — Gap.** Prior work touches pieces of this. Semantic scene completion (MonoScene, VoxFormer,
VisHall3D) hallucinates geometry beyond the visible frontier but targets outdoor voxels
scored by mIoU alone, with no notion of calibrated confidence in the hallucinated region.
Occupancy world models (OccWorld) forecast the *temporal* evolution of multi-frame occupancy
for driving. Closest to us, ProxMaP completes indoor proximal occupancy from a single view
for navigation — but emits a point estimate. None (a) works from a single monocular-depth
frame, (b) attaches *calibrated* uncertainty to the occluded region specifically, or (c)
separates completion from forecasting to state honestly where the predictive power lies.

**¶3 — What we do.** We present ShadowWeave, which from one depth frame produces an egocentric
BEV occupancy grid, an explicit shadow (visibility) mask, and a calibrated per-cell
occupancy probability inside the shadow. We evaluate with an empty-credit-immune,
micro-averaged shadow-IoU gain over persistence, per-episode bootstrap CIs, and a
static/dynamic/never decomposition that attributes the gain to completion vs. forecasting.
The completed grid feeds an A* planner and a nine-zone HRTF spatial-audio interface for
eyes-free navigation.

**¶4 — Findings / contributions.**
- **A calibrated amodal-completion result.** +0.31–0.36 @5 s micro shadow gain (CIs clear of
  zero), with in-shadow ECE 0.043–0.047 — *better* calibrated than observed space.
- **A decomposition methodology** separating completion from forecasting, which reveals the
  gain is spatial completion of persistent hidden structure; the pure-forecasting residual is
  ~0 (fixed static +0.000; moving +0.007 [+0.002, +0.012], small but significant and decaying
  with horizon). We report this negative honestly — most pooled occupancy numbers in the
  literature are vulnerable to exactly the confound we dissect.
- **A memorization confound-kill.** Per-episode randomized room geometry leaves the gain
  statistically unchanged (+0.306 fixed vs +0.309 [+0.294, +0.324] on never-seen rooms).
- **A real-time eyes-free system.** Completed occupancy + uncertainty → A* + HRTF audio,
  7 ms p95 perception-to-planning (depth acquisition and audio synthesis excluded).

---

## 2. Section outline (to flesh out)

**2. Method.**
- Depth → egocentric BEV occupancy projection; shadow mask from visibility (`vis<0.5`).
- World model: U-Net (31M), single-frame input stack (occupancy [+visibility+flow, both shown
  redundant]); BCE with pos_weight; EMA weights, epoch-8 `best.pt`.
- Uncertainty = the model's per-cell occupancy probability; calibration measured in shadow.
- Downstream interface (brief, system-context): A* over completed grid; 9-zone HRTF cues.

**3. Evaluation methodology (a contribution in its own right).**
- The empty-credit trap in masked IoU (empty-on-empty → 1.0) → micro-averaging over pooled
  counts.
- Static / dynamic / never decomposition within the shadow mask (completion vs forecasting).
- Per-episode cluster bootstrap CIs (resample episodes, not frames; paired model−persistence).
- Calibration: equal-mass ECE in shadow vs observed vs walls-excluded.

**4. Results.**
- 4.1 Shadow-gain headline + per-tier CIs (Table).
- 4.2 Completion vs forecasting (decomposition table; the honest negative).
- 4.3 Robustness to room geometry (randgeom, confound-kill).
- 4.4 Calibration in shadow (reliability figure).
- 4.5 Ablations: occupancy alone carries the signal (visibility, flow redundant).
- 4.6 System: latency budget; falling-object anticipation (cell + object-level, with caveats);
  navigation (honest — misses collision target).
- 4.7 (optional) Diffusion variant: sample diversity / calibration, pending numbers.

**5. Limitations & honest negatives.** Synthetic-only (real-footage qualitative panel);
completion not forecasting; navigation below target; falling-lead-time bounded by horizon grid.

**6. Related work.** From RELATED_WORK.md — SSC lineage, occupancy world models, ProxMaP;
differentiate on single-depth-frame input, calibrated in-shadow uncertainty, completion/
forecasting decomposition.

**7. Conclusion.** Calibrated amodal completion into occluded space is achievable from one
depth frame and transfers across geometry; a measurement methodology that separates
completion from forecasting keeps the claim honest.

---

## Figures (have / need)
- HAVE: `shadow_gain_ci.png` (gain + CIs, randgeom overlay) → Fig for 4.1/4.3.
- HAVE: `reliability.png` (calibration in shadow) → Fig for 4.4.
- HAVE: `decomp_curves.png` → Fig for 4.2.
- HAVE: `prediction_0x.png` panels → qualitative completion figure.
- NEED: real-footage depth→shadow→BEV→completion panel (blocked on a clip).
- NEED: system/architecture diagram (1 figure).

## Open TODO before submission
- Tighten to CFP page limit (need CFP).
- Real-footage panel (need clip).
- Diffusion row (pending eval; cut if weak).
- Confirm every number against RESULTS.md at freeze; tag `neurips-ws-2026-submission`.
