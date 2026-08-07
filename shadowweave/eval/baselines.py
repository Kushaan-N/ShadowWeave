"""Reference baselines and the falling-object lead-time metric.

An IOU number on its own is uninterpretable. BEV targets are ~7% positive, so a model
that predicts *nothing* already scores well on the many near-empty frames, and one
that simply echoes what it currently observes scores well on the static tier where
nothing moves. Reporting the world model against these makes the number mean
something: the claim is that it beats persistence, especially inside shadow.

Baselines
---------
empty       predict free space everywhere — the trivial floor
persistence echo the currently observed occupancy, unchanged, at every horizon.
            This is the one that matters: beating it is the minimum bar for
            "the world model learned dynamics rather than an autoencoder".
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .metrics import iou, masked_iou


class BaselineComparison:
    """Scores reference predictors on exactly the samples the model is scored on.

    Primary methods: ``set_observation(occ)``, ``log(...)``, then ``summary()``.
    """

    def __init__(self, horizons: list[float]) -> None:
        self.horizons = list(horizons)
        self._scores: dict[tuple[str, int], list[float]] = defaultdict(list)
        self._shadow: dict[tuple[str, int], list[float]] = defaultdict(list)
        self._model: dict[int, list[float]] = defaultdict(list)
        self._model_shadow: dict[int, list[float]] = defaultdict(list)

    @staticmethod
    def observation_snapshot(occupancy: torch.Tensor) -> torch.Tensor:
        """Freeze the observed BEV so it can be scored later.

        Persistence must be the observation from when the prediction was ISSUED,
        because that is the frame the ground truth is rendered in. Using the
        observation available at due-time compares an ego-BEV in the pose at t+h
        against a truth in the pose at t — the same frame mismatch that broke the
        main eval path, which made the baseline artificially *weak* and so flattered
        the model. Measured effect of the fix: persistence IOU 0.109 -> 0.146, i.e. a
        harder and more honest bar.
        """
        return occupancy.detach().float().cpu().reshape(1, *occupancy.shape[-2:]).clone()

    def log(
        self,
        horizon_idx: int,
        model_pred: np.ndarray,
        truth: torch.Tensor,
        shadow_mask: torch.Tensor | None = None,
        observed_at_issue: torch.Tensor | None = None,
    ) -> None:
        pred_t = torch.from_numpy(np.asarray(model_pred)).unsqueeze(0)
        self._model[horizon_idx].append(float(iou(pred_t, truth)))
        if shadow_mask is not None:
            self._model_shadow[horizon_idx].append(
                float(masked_iou(pred_t, truth, shadow_mask.bool()))
            )

        candidates: dict[str, torch.Tensor] = {"empty": torch.zeros_like(truth)}
        if observed_at_issue is not None:
            candidates["persistence"] = observed_at_issue

        for name, pred in candidates.items():
            self._scores[(name, horizon_idx)].append(float(iou(pred, truth)))
            if shadow_mask is not None:
                self._shadow[(name, horizon_idx)].append(
                    float(masked_iou(pred, truth, shadow_mask.bool()))
                )

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for (name, h), vals in sorted(self._scores.items()):
            label = f"{self.horizons[h]:g}s" if h < len(self.horizons) else f"h{h}"
            out[f"baseline_{name}_iou_{label}"] = float(np.mean(vals))
        for (name, h), vals in sorted(self._shadow.items()):
            label = f"{self.horizons[h]:g}s" if h < len(self.horizons) else f"h{h}"
            out[f"baseline_{name}_shadow_iou_{label}"] = float(np.mean(vals))

        # The headline comparison: model minus the strongest baseline, per horizon.
        for h, model_vals in sorted(self._model.items()):
            label = f"{self.horizons[h]:g}s" if h < len(self.horizons) else f"h{h}"
            best = max(
                (float(np.mean(v)) for (n, hh), v in self._scores.items() if hh == h),
                default=0.0,
            )
            out[f"model_gain_over_best_baseline_{label}"] = float(np.mean(model_vals)) - best

        # Same comparison restricted to shadow — where the project's actual claim lives.
        # Both shadow IOUs are inflated by empty-vs-empty credit, so the raw numbers are
        # not comparable by eye; the gain (model minus the strongest baseline) is.
        for h, model_vals in sorted(self._model_shadow.items()):
            label = f"{self.horizons[h]:g}s" if h < len(self.horizons) else f"h{h}"
            best = max(
                (float(np.mean(v)) for (n, hh), v in self._shadow.items() if hh == h),
                default=0.0,
            )
            out[f"model_shadow_gain_over_best_baseline_{label}"] = float(np.mean(model_vals)) - best
        return out


class LeadTimeTracker:
    """Measures how far ahead a falling object is flagged before it lands.

    ``EvalMetrics.log_lead_time`` existed but nothing ever called it, so the project's
    "falling-object lead time >= 3s" target was never actually measured.

    Method: watch the ground-truth BEV for cells that are free now and occupied later
    (an object arriving). For the earliest horizon at which the model already puts
    probability above threshold on that cell, the lead time is that horizon.
    """

    def __init__(self, threshold: float = 0.5, min_new_cells: int = 8) -> None:
        self.threshold = threshold
        self.min_new_cells = min_new_cells
        self._pending: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.lead_times: list[float] = []
        self.missed = 0
        # A model that simply predicts "occupied everywhere" would score a perfect
        # detection rate. Tracking how often it also flags cells where nothing
        # arrives is what stops that from looking like success.
        self._false_alarms: list[float] = []

    def observe(
        self,
        step: int,
        pred: np.ndarray,
        obs: dict,
        horizons: list[float],
        fps: int,
    ) -> None:
        """Feed one frame: the model's prediction stack and the current GT occupancy."""
        current = np.asarray(obs["bev_occupancy"]) > 0.5
        max_frames = int(max(horizons) * fps)

        # Rebuilt rather than mutated with list.remove: removing a tuple that holds
        # numpy arrays only works because CPython's comparison has an identity fast
        # path, and an equal-but-not-identical tuple raises "truth value of an array
        # is ambiguous". Filtering sidesteps that entirely.
        keep: list[tuple[int, np.ndarray, np.ndarray]] = []
        for issued, probs, past_occ in self._pending:
            elapsed = step - issued
            # Cells that were free when the prediction was made and are occupied now.
            arrived = current & ~past_occ
            if arrived.sum() < self.min_new_cells:
                if elapsed < max_frames:
                    keep.append((issued, probs, past_occ))
                continue

            # Did any horizon at or beyond this delay already flag those cells?
            flagged = None
            for hi, h in enumerate(horizons):
                if int(h * fps) < elapsed:
                    continue
                if (probs[hi][arrived] > self.threshold).mean() > 0.5:
                    flagged = h
                    # Same prediction, cells where nothing arrived: the false-alarm
                    # counterpart of this detection.
                    quiet = ~current & ~past_occ
                    if quiet.any():
                        self._false_alarms.append(
                            float((probs[hi][quiet] > self.threshold).mean())
                        )
                    break
            if flagged is not None:
                self.lead_times.append(float(flagged))
            else:
                self.missed += 1

        self._pending = keep
        self._pending.append((step, np.asarray(pred), current))

    @staticmethod
    def _grid2d(a) -> np.ndarray:
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        a = np.asarray(a)
        return a.reshape(a.shape[-2], a.shape[-1])

    def log_due(self, horizon_s: float, model_pred, truth, observed_at_issue) -> None:
        """Pose-consistent lead-time logging for one horizon as it comes due.

        All three grids MUST be in the same (issue) pose: ``truth`` is the future
        occupancy rendered in the pose the prediction was issued from, ``observed_at_issue``
        is what was observed then, and ``model_pred`` is the prediction for this horizon.
        This supersedes ``observe`` in the eval path: observe compared the current-pose
        obs["bev_occupancy"] against issue-pose predictions, so as the agent moved/turned
        between issue and detection it measured lead time across mismatched ego frames —
        the same frame mismatch the IOU path documents fixing. run_eval already renders
        issue-pose truth for IOU; this reuses it.

        An "arrival" is a cell occupied in the future truth but not observed-occupied at
        issue; if the prediction put probability on those cells the object was flagged
        ``horizon_s`` seconds ahead.
        """
        pred = self._grid2d(model_pred)
        truth_b = self._grid2d(truth) > self.threshold
        obs_b = self._grid2d(observed_at_issue) > self.threshold
        arrived = truth_b & (~obs_b)
        if int(arrived.sum()) < self.min_new_cells:
            return
        if float((pred[arrived] > self.threshold).mean()) > 0.5:
            self.lead_times.append(float(horizon_s))
        else:
            self.missed += 1
        quiet = (~truth_b) & (~obs_b)
        if bool(quiet.any()):
            self._false_alarms.append(float((pred[quiet] > self.threshold).mean()))

    def summary(self) -> dict[str, float]:
        if not self.lead_times and not self.missed:
            return {}
        n_events = len(self.lead_times) + self.missed
        out = {
            "falling_events_detected": float(len(self.lead_times)),
            "falling_detection_rate": len(self.lead_times) / max(n_events, 1),
        }
        if self.lead_times:
            out["falling_lead_time_s"] = float(np.mean(self.lead_times))
            out["falling_lead_time_min_s"] = float(np.min(self.lead_times))
        if self._false_alarms:
            fa = float(np.mean(self._false_alarms))
            out["falling_false_alarm_rate"] = fa
            # Read the detection rate against this: a high detection rate paired with
            # a high false-alarm rate means the model flags everything, not that it
            # anticipates anything.
            out["falling_detection_margin"] = out["falling_detection_rate"] - fa
        return out


def shadow_diversity(
    sample_std: torch.Tensor, shadow_mask: torch.Tensor
) -> dict[str, float]:
    """Where does a generative world model disagree with itself?

    This is the measurable form of the project's central claim. A model that has
    learned what occlusion *means* should produce samples that agree on the cells the
    agent can see and diverge on the cells it cannot. The ratio is the headline: > 1
    means uncertainty is concentrated in shadow, ~1 means the samples are just noisy
    everywhere and the model has not tied its uncertainty to visibility.

    A deterministic model cannot be scored on this at all — its std is identically 0,
    which is precisely the limitation that motivates sampling.

    Args:
        sample_std:  (..., S, S) per-cell std across samples
        shadow_mask: (..., S, S) bool, True where unobserved
    """
    std = sample_std.detach().float()
    mask = shadow_mask.detach().bool()
    # The mask is per-cell (1, S, S) while std carries a horizon axis (T, S, S).
    # Comparing ndim alone misses that — both are 3-D — so compare full shapes.
    if mask.shape != std.shape:
        mask = mask.expand_as(std)

    in_shadow = std[mask]
    observed = std[~mask]
    if in_shadow.numel() == 0 or observed.numel() == 0:
        return {}

    s, o = float(in_shadow.mean()), float(observed.mean())
    return {
        "sample_std_shadow": s,
        "sample_std_observed": o,
        "shadow_diversity_ratio": s / max(o, 1e-6),
    }


def calibration_error(probs: torch.Tensor, truth: torch.Tensor, n_bins: int = 10) -> float:
    """Expected calibration error.

    For a system whose whole premise is conveying *uncertainty*, a prediction of 0.8
    should be right about 80% of the time. A confidently wrong model is worse than an
    accurate one here, and IOU alone cannot see that.
    """
    p = probs.detach().flatten().float()
    t = (truth.detach().flatten() > 0.5).float().to(p.device)
    # Built on p's device: a CPU edge tensor compared against a CUDA p raises.
    edges = torch.linspace(0, 1, n_bins + 1, device=p.device)
    ece = 0.0
    n = p.numel()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        ece += float(m.sum()) / n * abs(float(p[m].mean()) - float(t[m].mean()))
    return ece


if __name__ == "__main__":
    horizons = [1, 3, 5, 10]

    print("BaselineComparison")
    bc = BaselineComparison(horizons)
    rng = np.random.default_rng(0)
    for _ in range(40):
        truth = (torch.rand(1, 32, 32) > 0.92).float()
        # What was visible when the prediction was issued — a partial, stale view.
        observed = BaselineComparison.observation_snapshot(
            truth * (torch.rand(1, 32, 32) > 0.3).float()
        )
        for hi in range(len(horizons)):
            model = (truth * 0.9 + torch.rand(1, 32, 32) * 0.2).clamp(0, 1)[0].numpy()
            bc.log(hi, model, truth, shadow_mask=torch.rand(1, 32, 32) > 0.5,
                   observed_at_issue=observed)
    for k, v in bc.summary().items():
        print(f"  {k}: {v:+.4f}")

    print("\nLeadTimeTracker")
    tracker = LeadTimeTracker(min_new_cells=4)
    occ = np.zeros((32, 32), dtype=np.float32)
    for step in range(0, 200, 10):
        if step >= 90:                      # an object 'lands' partway through
            occ = np.zeros((32, 32), dtype=np.float32)
            occ[10:14, 10:14] = 1.0
        # Model foresees it from early on.
        pred = np.zeros((4, 32, 32), dtype=np.float32)
        pred[:, 10:14, 10:14] = 0.9
        tracker.observe(step, pred, {"bev_occupancy": occ}, horizons, fps=30)
    for k, v in tracker.summary().items():
        print(f"  {k}: {v:.3f}")

    print("\ncalibration_error")
    perfect = torch.rand(4096)
    print(f"  well-calibrated: {calibration_error(perfect, (torch.rand(4096) < perfect).float()):.4f}")
    print(f"  overconfident:   {calibration_error(torch.full((4096,), 0.95), torch.zeros(4096)):.4f}")
    print("\nbaselines OK")
