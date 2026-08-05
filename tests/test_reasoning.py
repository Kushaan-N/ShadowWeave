"""Tests for the spatial-reasoning benchmark.

A benchmark that nobody can pass is as useless as one everybody passes, so most of
these check the benchmark itself rather than any model: the control must be trivially
answerable, the hard questions must be genuinely unanswerable without predicting into
shadow, and neither trivial strategy may dominate both classes at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadowweave.eval.reasoning import (
    BASELINES,
    ZONE_NAMES,
    Question,
    QuestionGenerator,
    ReasoningBenchmark,
    ahead_mask,
    answer_question,
    blind_optimistic,
    blind_pessimistic,
    zone_mask,
)

S = 48
HORIZONS = [1, 3, 5, 10]
H = len(HORIZONS)


@pytest.fixture
def scene():
    """Half-seen room: a visible obstacle, one hidden behind it, one arriving later."""
    visibility = np.ones((S, S), dtype=np.float32)
    visibility[: S // 2, : S // 3] = 0.0
    target = np.zeros((H, S, S), dtype=np.float32)
    target[:, S // 2 - 4 : S // 2, 4:10] = 1.0        # visible
    target[:, 6:12, 6:12] = 1.0                       # hidden behind it
    target[2:, S // 2 + 4 : S // 2 + 10, S // 2 - 3 : S // 2 + 3] = 1.0  # arrives later
    observed = (target[0] * (visibility > 0.5)).astype(np.float32)
    return target, visibility, observed


class TestRegionMasks:
    def test_zones_tile_the_grid_without_overlap(self):
        total = np.zeros((S, S), dtype=int)
        for z in range(len(ZONE_NAMES)):
            total += zone_mask(S, z).astype(int)
        assert total.max() == 1, "zones overlap"
        assert total.min() == 1, "zones leave a gap"

    def test_ahead_is_central_and_near(self):
        m = ahead_mask(S)
        assert m.any()
        assert not m[: S // 2].any(), "the corridor should be the near half"
        assert not m[:, 0].any() and not m[:, -1].any(), "should not touch the edges"


class TestGroundTruthIsSelfConsistent:
    def test_answering_truth_with_truth_reproduces_the_answer(self, scene):
        """The generator's stored answer must equal what the shared answering
        function returns on the ground truth — otherwise predictors are graded
        against a question nobody asked."""
        target, visibility, observed = scene
        gen = QuestionGenerator(HORIZONS, seed=0)
        for _ in range(10):
            for q in gen.generate(target, visibility, observed=observed):
                got = answer_question(q, target, visibility, observed=observed)
                assert got == q.answer, f"{q.kind} disagrees with its own ground truth"

    def test_every_question_has_a_typed_answer(self, scene):
        target, visibility, observed = scene
        for q in QuestionGenerator(HORIZONS, seed=1).generate(
            target, visibility, observed=observed, max_questions=6
        ):
            assert q.text.endswith("?")
            assert q.answer is not None
            assert q.answer_type in ("bool", "count", "zone")
            if q.answer_type == "zone":
                assert q.answer in ZONE_NAMES


class TestBenchmarkIntegrity:
    def _run(self, scene, n=60):
        target, visibility, observed = scene
        gen = QuestionGenerator(HORIZONS, seed=2)
        bench = ReasoningBenchmark()
        preds = {
            "model": target,  # an oracle
            **{k: f(observed, visibility, H) for k, f in BASELINES.items()},
        }
        for _ in range(n):
            qs = gen.generate(target, visibility, observed=observed)
            bench.record(qs, preds, visibility, observed=observed)
        return bench.summary()

    def test_control_is_answerable_by_everyone(self, scene):
        """If a camera cannot pass the camera-answerable questions, the benchmark is
        measuring noise. This caught a real bug: grading the control against t+h
        occupancy instead of the observation dropped a correct baseline to 0.60."""
        s = self._run(scene)
        for name in ["model", *BASELINES]:
            assert s[f"reasoning_{name}_perception_only_acc"] > 0.95, name

    def test_hard_questions_are_hard_without_a_world_model(self, scene):
        s = self._run(scene)
        for name in BASELINES:
            assert s[f"reasoning_{name}_needs_world_model_acc"] < 0.95, (
                f"{name} answers occlusion questions too well — they are not testing "
                f"occlusion")

    def test_neither_trivial_strategy_wins_universally(self):
        """Which trivial strategy looks good depends entirely on whether the shadow
        happens to be occupied — so neither is a safe assumption, and a model has to
        beat both across scenes rather than getting lucky on one.

        Asserted over two scenes rather than one: within a single scene whichever
        assumption matches that scene's shadow wins every question kind, which is a
        property of the scene, not of the strategies.
        """
        def run(shadow_occupied: bool) -> dict:
            vis = np.ones((S, S), dtype=np.float32)
            vis[:, : S // 3] = 0.0                      # left band unobserved
            target = np.zeros((H, S, S), dtype=np.float32)
            target[:, S // 2 :, S // 2 :] = 1.0         # something visible on the right
            if shadow_occupied:
                target[:, 10:20, 2:10] = 1.0           # and something hidden
            observed = (target[0] * (vis > 0.5)).astype(np.float32)
            gen = QuestionGenerator(HORIZONS, seed=5)
            bench = ReasoningBenchmark()
            preds = {k: f(observed, vis, H) for k, f in BASELINES.items()}
            for _ in range(60):
                bench.record(gen.generate(target, vis, observed=observed), preds,
                             vis, observed=observed)
            return bench.summary()

        occupied = run(True)
        empty = run(False)
        key = "reasoning_{}_needs_world_model_acc"
        # Shadow full -> assuming it is full wins. Shadow empty -> the reverse.
        assert occupied[key.format("blind_pessimistic")] > occupied[key.format("blind_optimistic")]
        assert empty[key.format("blind_optimistic")] > empty[key.format("blind_pessimistic")]

    def test_oracle_beats_both_baselines(self, scene):
        s = self._run(scene)
        assert s["reasoning_gain_over_best_baseline"] > 0

    def test_question_count_is_reported(self, scene):
        assert self._run(scene, n=10)["reasoning_n_questions"] > 0


class TestBaselineSemantics:
    def test_optimistic_clears_shadow(self, scene):
        _, visibility, observed = scene
        occ = blind_optimistic(observed, visibility, H)
        assert occ[0][visibility < 0.5].max() == 0.0

    def test_pessimistic_fills_shadow(self, scene):
        _, visibility, observed = scene
        occ = blind_pessimistic(observed, visibility, H)
        assert occ[0][visibility < 0.5].min() == 1.0

    def test_baselines_are_static_across_horizons(self, scene):
        """Neither strategy predicts anything, so all horizons are identical — that is
        what makes them the floor a real model must clear."""
        _, visibility, observed = scene
        for f in BASELINES.values():
            occ = f(observed, visibility, H)
            assert np.array_equal(occ[0], occ[-1])


class TestAnswering:
    def test_emergence_needs_both_shadow_and_the_corridor(self):
        # Shadow the LEFT half, not the far half: the corridor is the *near* half
        # (rows S/2..S, cols 16..31), so a far-half shadow cannot overlap it at all
        # and the "hidden AND in the way" case would be unconstructible.
        vis = np.ones((S, S), dtype=np.float32)
        vis[:, : S // 2] = 0.0
        occ = np.zeros((H, S, S), dtype=np.float32)
        q = Question("", "emergence", None, "bool", 0, True)

        occ[0][ahead_mask(S) & (vis > 0.5)] = 1.0        # in corridor, but visible
        assert answer_question(q, occ, vis) is False

        occ[:] = 0.0
        occ[0][~ahead_mask(S) & (vis < 0.5)] = 1.0       # hidden, but not in the way
        assert answer_question(q, occ, vis) is False

        occ[:] = 0.0
        occ[0][ahead_mask(S) & (vis < 0.5)] = 1.0        # hidden AND in the way
        assert answer_question(q, occ, vis) is True

    def test_counting_separates_objects(self):
        vis = np.zeros((S, S), dtype=np.float32)          # all shadow
        occ = np.zeros((H, S, S), dtype=np.float32)
        occ[0, 4:8, 4:8] = 1.0
        occ[0, 20:24, 20:24] = 1.0
        q = Question("", "occluded_count", None, "count", 0, True)
        assert answer_question(q, occ, vis) == 2

    def test_count_penalises_speckle(self):
        """An undertrained model outputs noise, not objects; the count metric should
        say so loudly rather than average it away."""
        vis = np.zeros((S, S), dtype=np.float32)
        rng = np.random.default_rng(0)
        occ = (rng.random((H, S, S)) > 0.5).astype(np.float32)
        q = Question("", "occluded_count", None, "count", 0, True)
        assert answer_question(q, occ, vis) > 20

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown question kind"):
            answer_question(Question("", "nope", None, "bool", 0, True),
                            np.zeros((H, S, S)), np.ones((S, S)))
