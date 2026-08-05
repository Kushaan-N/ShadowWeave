"""Spatial-reasoning benchmark auto-generated from simulator ground truth.

What this measures
------------------
Whether the world model has learned anything a camera could not have told it.

Every question is generated with a *verifiable* answer, derived from the same BEV
targets and visibility masks the model is trained against. That is only possible
because the scenes are simulated: real footage has no ground truth for "what is behind
that crate", so a benchmark like this cannot be built from it at all.

Questions are deliberately split into two classes:

  requires_world_model=False   answerable from the current observation alone.
      These are the control. A perception-only baseline should score near-perfect —
      if it does not, the questions are measuring noise rather than reasoning.

  requires_world_model=True    about occluded space, or about the future.
      A camera cannot answer these. Only a model that predicts into shadow can.

Isolating the contribution
--------------------------
The same ``answer_question`` function scores the model and the baselines, so the
comparison is exact. Two baselines bracket the problem:

  blind_optimistic    assumes shadow is empty  — aces "is the way clear?", fails to
                      anticipate anything hidden
  blind_pessimistic   assumes shadow is full   — aces "is something hidden?", fails
                      counting and direction-choosing

Either is trivially reachable without learning anything, and each wins a different
subset of questions. A model that has genuinely learned occlusion has to beat *both*
at once. Reporting only against one would be easy to game.
"""

from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..shadow.zones import zone_bounds

ZONE_NAMES = ["far left", "left", "near left", "front left", "directly ahead",
              "front right", "near right", "right", "far right"]

_BOOL, _COUNT, _ZONE = "bool", "count", "zone"


@dataclass
class Question:
    """One benchmark item with a ground-truth answer."""
    text: str
    kind: str
    answer: Any
    answer_type: str
    horizon_idx: int
    requires_world_model: bool
    region: str = ""
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Region masks over the egocentric BEV
# ----------------------------------------------------------------------

def zone_mask(size: int, zone_idx: int, n_zones: int = 9) -> np.ndarray:
    """Column strip for one azimuth zone — the same partition the audio uses."""
    c0, c1 = zone_bounds(size, n_zones)[zone_idx]
    m = np.zeros((size, size), dtype=bool)
    m[:, c0:c1] = True
    return m


def ahead_mask(size: int, n_zones: int = 9) -> np.ndarray:
    """The corridor the agent would walk into: the three central zones, near half.

    Row 0 is the far edge and row size-1 is the agent, so "near" is the bottom.
    """
    c0 = zone_bounds(size, n_zones)[n_zones // 2 - 1][0]
    c1 = zone_bounds(size, n_zones)[n_zones // 2 + 1][1]
    m = np.zeros((size, size), dtype=bool)
    m[size // 2 :, c0:c1] = True
    return m


def _components(binary: np.ndarray) -> int:
    """Count separate obstacles. Falls back to a coarse estimate without scipy."""
    try:
        from scipy import ndimage
        return int(ndimage.label(binary)[1])
    except ImportError:
        return int(binary.any())


# ----------------------------------------------------------------------
# Answering — one implementation, used for ground truth AND for every predictor
# ----------------------------------------------------------------------

def answer_question(
    q: Question,
    occupancy: np.ndarray,
    visibility: np.ndarray,
    threshold: float = 0.5,
    observed: Optional[np.ndarray] = None,
) -> Any:
    """Answer ``q`` from an occupancy stack, whoever produced it.

    Using one function for the ground truth, the model and the baselines is what makes
    the comparison exact — no predictor gets a subtly different question.

    Args:
        occupancy:  (H, S, S) probabilities per horizon
        visibility: (S, S) in [0, 1]; 0 = shadow
    """
    S = occupancy.shape[-1]
    occ = occupancy[q.horizon_idx] > threshold
    shadow = visibility < 0.5

    if q.kind == "visible_presence":
        # Graded against the observation every predictor shares, so this is a true
        # control: it validates the question set rather than ranking the predictors.
        # Grading it against t+h occupancy instead made a correct baseline score 0.60
        # and the benchmark look broken.
        grid = (np.asarray(observed) > threshold) if observed is not None else occ
        return bool((grid & zone_mask(S, q.meta["zone"]) & ~shadow).any())
    if q.kind == "occluded_presence":
        return bool((occ & zone_mask(S, q.meta["zone"]) & shadow).any())
    if q.kind == "occluded_count":
        return _components(occ & shadow)
    if q.kind == "path_blocked_future":
        return bool((occ & ahead_mask(S)).any())
    if q.kind == "emergence":
        # Something arriving in the corridor that is hidden right now.
        return bool((occ & ahead_mask(S) & shadow).any())
    if q.kind == "clearest_direction":
        scores = [float((occ & zone_mask(S, z)).mean()) for z in range(len(ZONE_NAMES))]
        return ZONE_NAMES[int(np.argmin(scores))]
    raise ValueError(f"unknown question kind: {q.kind}")


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------

class QuestionGenerator:
    """Builds verifiable questions from one frame's ground truth.

    Primary method: ``generate(target, visibility, horizons) -> list[Question]``
    """

    def __init__(self, horizons: list[float], seed: int = 0) -> None:
        self.horizons = list(horizons)
        self.rng = random.Random(seed)

    def generate(
        self,
        target: np.ndarray,
        visibility: np.ndarray,
        observed: Optional[np.ndarray] = None,
        max_questions: int = 6,
    ) -> list[Question]:
        """Args:
            target:     (H, S, S) ground-truth occupancy per horizon
            visibility: (S, S), 0 = shadow
            observed:   (S, S) the BEV actually seen this frame, for the control
        """
        S = target.shape[-1]
        out: list[Question] = []

        def gt(kind: str, hi: int, **meta) -> Any:
            probe = Question("", kind, None, "", hi, False, meta=meta)
            return answer_question(probe, target, visibility, observed=observed)

        # -- control: answerable from the current observation --------------
        zone = self.rng.randrange(len(ZONE_NAMES))
        out.append(Question(
            text=f"Is there an obstacle {ZONE_NAMES[zone]} that you can currently see?",
            kind="visible_presence", answer=gt("visible_presence", 0, zone=zone),
            answer_type=_BOOL, horizon_idx=0, requires_world_model=False,
            region=ZONE_NAMES[zone], meta={"zone": zone},
        ))

        # -- occlusion: a camera cannot answer these -----------------------
        zone = self.rng.randrange(len(ZONE_NAMES))
        out.append(Question(
            text=f"Is there an obstacle {ZONE_NAMES[zone]} that you cannot currently see?",
            kind="occluded_presence", answer=gt("occluded_presence", 0, zone=zone),
            answer_type=_BOOL, horizon_idx=0, requires_world_model=True,
            region=ZONE_NAMES[zone], meta={"zone": zone},
        ))
        out.append(Question(
            text="How many separate obstacles are hidden in the space you cannot see?",
            kind="occluded_count", answer=gt("occluded_count", 0),
            answer_type=_COUNT, horizon_idx=0, requires_world_model=True,
        ))

        # -- temporal: about the future ------------------------------------
        # These stay requires_world_model=True even at horizon 0: their ground truth
        # covers a whole region, and that region can contain occluded cells, so a
        # camera cannot answer them at any horizon. Marking them perception-answerable
        # at h=0 dragged the control class down to 0.80 and made a fair benchmark look
        # broken.
        hi = self.rng.randrange(len(self.horizons))
        h = self.horizons[hi]
        out.append(Question(
            text=f"Will the path directly ahead be blocked in {h:g} seconds?",
            kind="path_blocked_future", answer=gt("path_blocked_future", hi),
            answer_type=_BOOL, horizon_idx=hi, requires_world_model=True,
            region="directly ahead", meta={"horizon_s": h},
        ))
        out.append(Question(
            text=f"Which direction will be clearest in {h:g} seconds?",
            kind="clearest_direction", answer=gt("clearest_direction", hi),
            answer_type=_ZONE, horizon_idx=hi, requires_world_model=True,
            meta={"horizon_s": h},
        ))

        # -- the hardest one: hidden now, in your way later ----------------
        out.append(Question(
            text=(f"Will an obstacle you cannot currently see move into the path "
                  f"ahead within {h:g} seconds?"),
            kind="emergence", answer=gt("emergence", hi),
            answer_type=_BOOL, horizon_idx=hi, requires_world_model=True,
            region="directly ahead", meta={"horizon_s": h},
        ))

        self.rng.shuffle(out)
        return out[:max_questions]


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------

def blind_optimistic(observed: np.ndarray, visibility: np.ndarray, n_horizons: int) -> np.ndarray:
    """What a camera alone implies: shadow is empty, and nothing changes."""
    occ = np.asarray(observed).copy()
    occ[visibility < 0.5] = 0.0
    return np.repeat(occ[None], n_horizons, axis=0)


def blind_pessimistic(observed: np.ndarray, visibility: np.ndarray, n_horizons: int) -> np.ndarray:
    """Safety-conservative: treat everything unseen as blocked."""
    occ = np.asarray(observed).copy()
    occ[visibility < 0.5] = 1.0
    return np.repeat(occ[None], n_horizons, axis=0)


BASELINES = {"blind_optimistic": blind_optimistic, "blind_pessimistic": blind_pessimistic}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

class ReasoningBenchmark:
    """Accumulates answers from several predictors and scores them per question class.

    Primary methods: ``record(...)`` per frame, then ``summary()``.
    """

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], list[float]] = {}
        self._by_class: dict[tuple[str, bool], list[float]] = {}
        self._count_err: dict[str, list[float]] = {}
        self.n_questions = 0

    @staticmethod
    def _correct(q: Question, predicted: Any) -> float:
        if q.answer_type == _COUNT:
            return float(int(predicted) == int(q.answer))
        return float(predicted == q.answer)

    def record(
        self,
        questions: list[Question],
        predictions: dict[str, np.ndarray],
        visibility: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> None:
        """Args:
            predictions: {name: (H, S, S) occupancy} — the model and every baseline
        """
        for q in questions:
            self.n_questions += 1
            for name, occ in predictions.items():
                got = answer_question(q, occ, visibility, observed=observed)
                self._hits.setdefault((name, q.kind), []).append(self._correct(q, got))
                self._by_class.setdefault((name, q.requires_world_model), []).append(
                    self._correct(q, got))
                if q.answer_type == _COUNT:
                    self._count_err.setdefault(name, []).append(
                        float(abs(int(got) - int(q.answer))))

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"reasoning_n_questions": float(self.n_questions)}
        for (name, kind), v in sorted(self._hits.items()):
            out[f"reasoning_{name}_{kind}"] = float(np.mean(v))
        for (name, needs_wm), v in sorted(self._by_class.items()):
            tag = "needs_world_model" if needs_wm else "perception_only"
            out[f"reasoning_{name}_{tag}_acc"] = float(np.mean(v))
        for name, v in sorted(self._count_err.items()):
            out[f"reasoning_{name}_count_mae"] = float(np.mean(v))

        # The headline: on questions a camera cannot answer, how far ahead of the
        # best trivial strategy is the model? Beating only one baseline is easy.
        key = "reasoning_model_needs_world_model_acc"
        if key in out:
            rivals = [v for k, v in out.items()
                      if k.endswith("_needs_world_model_acc") and "model" != k.split("_")[1]
                      and not k.startswith("reasoning_model_")]
            if rivals:
                out["reasoning_gain_over_best_baseline"] = out[key] - max(rivals)
        return out


def evaluate(cfg, model, dataset, max_samples: int = 200, seed: int = 0) -> dict[str, float]:
    """Score a world model against the trivial strategies over a rollout split.

    Runs offline on stored samples rather than inside a live rollout: each sample
    already carries the observation, the visibility mask and the full horizon stack,
    which is exactly what a question needs. Scoring inside the rollout loop only has
    the single horizon that came due, which is what made the control question
    unanswerable there.
    """
    import torch

    horizons = list(cfg.world_model.prediction_horizons)
    gen = QuestionGenerator(horizons, seed=seed)
    bench = ReasoningBenchmark()
    idxs = np.linspace(0, len(dataset) - 1, min(max_samples, len(dataset))).astype(int)

    for i in idxs:
        sample = dataset[int(i)]
        x, y = sample["input"], sample["target"].numpy()
        observed = x[0].numpy()
        visibility = x[1].numpy() if x.shape[0] > 1 else np.ones_like(observed)

        # Dataset tensors are always CPU; the model may not be.
        device = next(model.parameters()).device
        with torch.no_grad():
            pred = model.predict(x.unsqueeze(0).to(device))[0].cpu().numpy()

        qs = gen.generate(y, visibility, observed=observed)
        bench.record(qs, {
            "model": pred,
            **{n: f(observed, visibility, len(horizons)) for n, f in BASELINES.items()},
        }, visibility, observed=observed)
    return bench.summary()


def main() -> None:
    import argparse

    import torch

    from ..utils import config_from_checkpoint, get_device, load_checkpoint, load_config
    from ..world_model import build_world_model
    from ..world_model.dataset import RolloutDataset
    from ..console import header, table

    ap = argparse.ArgumentParser(description="Spatial-reasoning benchmark")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    device = get_device()
    ckpt = pathlib.Path(args.ckpt or (pathlib.Path(cfg.world_model.checkpoint_dir) / "best.pt"))
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} — train one first (make train)")

    wm_cfg = config_from_checkpoint(ckpt, cfg)
    model = build_world_model(wm_cfg).to(device).eval()
    model.load_state_dict(load_checkpoint(ckpt, map_location=device)["model"])
    ds = RolloutDataset(cfg, args.data or cfg.data.root, split=args.split)

    s = evaluate(cfg, model, ds, args.samples)
    print(header("Spatial-reasoning benchmark",
                 f"{int(s['reasoning_n_questions'])} questions over {args.split}"))
    rows = []
    for name in ["model", *BASELINES]:
        rows.append([
            name,
            f"{s.get(f'reasoning_{name}_perception_only_acc', float('nan')):.3f}",
            f"{s.get(f'reasoning_{name}_needs_world_model_acc', float('nan')):.3f}",
            f"{s.get(f'reasoning_{name}_count_mae', float('nan')):.2f}",
        ])
    print()
    print(table(rows, ["predictor", "control", "needs world model", "count MAE"]))
    print(f"\n  gain over the best trivial strategy: "
          f"{s.get('reasoning_gain_over_best_baseline', 0):+.3f}")
    print("\n  The control must be ~1.0 for everyone — it verifies the question set,")
    print("  not the model. All the signal is in the 'needs world model' column.")


if __name__ == "__main__":
    import sys

    # With arguments, score a real checkpoint; without, run the self-contained demo
    # so `make demos` and tests/test_demos.py keep working.
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)

    from ..utils import load_config

    cfg = load_config()
    S = 48
    horizons = list(cfg.world_model.prediction_horizons)
    H = len(horizons)
    print("Spatial-reasoning benchmark demo\n")

    rng = np.random.default_rng(0)
    # A scene the agent can only half see: visible obstacle on the left, a hidden one
    # behind it, and something that drifts into the corridor later.
    visibility = np.ones((S, S), dtype=np.float32)
    visibility[: S // 2, : S // 3] = 0.0            # shadow behind the left obstacle
    target = np.zeros((H, S, S), dtype=np.float32)
    target[:, S // 2 - 4 : S // 2, 4:10] = 1.0      # the visible obstacle
    target[:, 6:12, 6:12] = 1.0                     # hidden behind it
    target[2:, S // 2 + 4 : S // 2 + 10, S // 2 - 3 : S // 2 + 3] = 1.0  # arrives later

    observed = (target[0] * (visibility > 0.5)).astype(np.float32)

    gen = QuestionGenerator(horizons, seed=1)
    questions = gen.generate(target, visibility, observed=observed, max_questions=6)
    print(f"generated {len(questions)} questions from one frame:")
    for q in questions:
        flag = "world-model" if q.requires_world_model else "perception "
        print(f"  [{flag}] {q.text}\n      ground truth: {q.answer}")

    # An oracle-ish "model" that sees the truth, versus the two trivial strategies.
    preds = {
        "model": target,
        "blind_optimistic": blind_optimistic(observed, visibility, H),
        "blind_pessimistic": blind_pessimistic(observed, visibility, H),
    }

    bench = ReasoningBenchmark()
    for _ in range(40):
        bench.record(gen.generate(target, visibility, observed=observed), preds,
                     visibility, observed=observed)

    print("\naccuracy by question class:")
    s = bench.summary()
    for name in ("model", "blind_optimistic", "blind_pessimistic"):
        p = s.get(f"reasoning_{name}_perception_only_acc", float("nan"))
        w = s.get(f"reasoning_{name}_needs_world_model_acc", float("nan"))
        print(f"  {name:18s} perception-only={p:.3f}   needs-world-model={w:.3f}")
    print(f"\n  gain over the best baseline: {s.get('reasoning_gain_over_best_baseline', 0):+.3f}")

    # The control questions must be easy for everyone, or the benchmark is broken.
    assert s["reasoning_blind_optimistic_perception_only_acc"] > 0.9, \
        "a camera should ace the questions a camera can answer"
    # And neither trivial strategy should reach the model on the hard ones.
    assert s["reasoning_model_needs_world_model_acc"] > \
        max(s["reasoning_blind_optimistic_needs_world_model_acc"],
            s["reasoning_blind_pessimistic_needs_world_model_acc"])
    print("\nReasoningBenchmark OK")
