"""Tests for the LearnedPlanner and the weight search.

Covers: determinism of the planner given fixed weights, feature-vector shape
and bounds, reproducibility of the search from a seed, disjoint train/held-out
split, and that the learned planner integrates with run_episode like the
baselines. No grasp constants or physics are touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import learn
from src.planner import (
    LEARNED_N_FEATURES,
    LearnedPlanner,
    item_features,
)
from src.robot import Gripper
from src.scene import make_scene
from src.sim import run_episode


def test_learned_planner_requires_correct_weight_count() -> None:
    """Constructing with the wrong number of weights raises ValueError."""
    with pytest.raises(ValueError):
        LearnedPlanner((1.0, 2.0))  # too few
    # Correct count is accepted.
    LearnedPlanner((0.1,) * LEARNED_N_FEATURES)


def test_item_features_shape_and_bounds() -> None:
    """Feature vector has the declared length; coverage/crowding in [0, 1]."""
    scene = make_scene(seed=3, n_items=15, clutter=0.9)
    gripper = Gripper(x=0.0, y=0.0, holding=None)
    for it in scene.items:
        feats = item_features(scene, gripper, it)
        assert len(feats) == LEARNED_N_FEATURES
        cov, dist, crowd = feats
        assert 0.0 <= cov <= 1.0
        assert dist >= 0.0
        assert 0.0 <= crowd <= 1.0


def test_learned_planner_is_deterministic_given_weights() -> None:
    """Same weights + same scene/seed -> identical episode result, repeatably."""
    w = (0.8, 0.3, 0.5)
    a = run_episode(make_scene(seed=7, n_items=15, clutter=0.9),
                    LearnedPlanner(w), seed=7)
    b = run_episode(make_scene(seed=7, n_items=15, clutter=0.9),
                    LearnedPlanner(w), seed=7)
    assert a == b


def test_learned_planner_carries_no_rng_or_state() -> None:
    """Two fresh planners with equal weights pick the same action on a scene."""
    w = (0.5, 0.5, 0.5)
    scene = make_scene(seed=2, n_items=12, clutter=0.6)
    g = Gripper(x=10.0, y=10.0, holding=None)
    a1 = LearnedPlanner(w).next_action(scene, g)
    a2 = LearnedPlanner(w).next_action(scene, g)
    assert a1 == a2


def test_different_weights_can_change_choice() -> None:
    """Weights actually matter: some weighting differs from another's pick.

    Not every scene distinguishes them, so scan a few seeds and assert at least
    one scene yields a different first grasp under two very different weightings.
    """
    w_cov = (1.0, 0.0, 0.0)   # purely least-covered
    w_dist = (0.0, 1.0, 0.0)  # purely nearest
    differed = False
    for seed in range(8):
        scene = make_scene(seed=seed, n_items=15, clutter=0.9)
        g = Gripper(x=0.0, y=0.0, holding=None)
        if (LearnedPlanner(w_cov).next_action(scene, g)
                != LearnedPlanner(w_dist).next_action(scene, g)):
            differed = True
            break
    assert differed


def test_learned_planner_clears_a_simple_desk() -> None:
    """The learned policy clears every item on a low-clutter desk."""
    result = run_episode(make_scene(seed=1, n_items=10, clutter=0.3),
                         LearnedPlanner((1.0, 1.0, 1.0)), seed=1)
    assert result.items_sorted == result.items_total == 10


# --------------------------------------------------------------------------- #
# Weight search (src.learn)
# --------------------------------------------------------------------------- #
def test_split_seeds_are_disjoint() -> None:
    """Training and held-out seed lists never overlap."""
    train, holdout = learn.split_seeds(n_train=15, n_holdout=15, base_seed=0)
    assert len(train) == 15 and len(holdout) == 15
    assert set(train).isdisjoint(set(holdout))


def test_search_is_reproducible_from_seed() -> None:
    """Same search_seed + same training config -> identical best weights."""
    train, _ = learn.split_seeds(5, 5)
    r1 = learn.search_weights(train, n_items=10, clutter=0.6, iters=20, search_seed=42)
    r2 = learn.search_weights(train, n_items=10, clutter=0.6, iters=20, search_seed=42)
    assert r1.weights == r2.weights
    assert r1.train_ticks == r2.train_ticks


def test_search_does_not_worsen_training_score() -> None:
    """Hill-climb's best-so-far history is non-increasing (it never regresses)."""
    train, _ = learn.split_seeds(5, 5)
    res = learn.search_weights(train, n_items=10, clutter=0.6, iters=30, search_seed=1)
    for earlier, later in zip(res.history, res.history[1:]):
        assert later <= earlier + 1e-9


def test_evaluate_holdout_includes_learned_and_baselines() -> None:
    """Held-out evaluation returns the learned policy plus all five baselines."""
    _, holdout = learn.split_seeds(5, 5)
    rows = learn.evaluate_holdout((1.0, 1.0, 1.0), holdout, n_items=10, clutter=0.6)
    names = {r.name for r in rows}
    assert "learned" in names
    assert {"random", "nearest", "leastcov", "nearretry", "occlusion"} <= names
    assert all(r.mean > 0.0 for r in rows)


def test_run_is_reproducible() -> None:
    """The full train+held-out run is reproducible from the search seed."""
    a = learn.run(n_items=10, clutter=0.6, n_train=5, n_holdout=5, iters=15, search_seed=3)
    b = learn.run(n_items=10, clutter=0.6, n_train=5, n_holdout=5, iters=15, search_seed=3)
    assert a[0].weights == b[0].weights
    assert [(r.name, r.mean) for r in a[1]] == [(r.name, r.mean) for r in b[1]]


# --------------------------------------------------------------------------- #
# App integration: learned planner is selectable like the others
# --------------------------------------------------------------------------- #
def test_learned_is_selectable_in_app() -> None:
    """The app exposes 'learned' and builds a LearnedPlanner for it."""
    from app import logic

    assert "learned" in logic.available_planners()
    planner = logic.make_planner("learned", 0)
    assert isinstance(planner, LearnedPlanner)
    # A deterministic one-trial run through the app layer.
    r = logic.run_trial(logic.TrialConfig("learned", seed=1, n_items=10, clutter=0.3))
    assert r.items_sorted == r.items_total == 10


# --------------------------------------------------------------------------- #
# Live View: a learned trial can be traced and rendered
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rotate", [False, True])
def test_learned_trial_replays_and_renders(rotate) -> None:
    """A 'learned' trial builds a trace, matches the sim, and renders frames.

    Regression test: build_trace previously only knew the five parameterless
    planners, so replaying 'learned' raised KeyError. replay_trial now injects
    the app's planner factory, so the learned planner (with its configured
    weights) is reconstructed for the replay exactly as the batch runner builds
    it. Works in both axis-aligned and rotated shape modes.
    """
    from app import logic

    rr = logic.replay_trial(
        logic.TrialConfig("learned", seed=1, n_items=12, clutter=0.6, rotate=rotate)
    )
    assert rr.frames, "learned replay produced no frames"
    # Trace must match the paired EpisodeResult (same weights as the runner).
    last = rr.frames[-1]
    assert last.sorted_count == rr.result.items_sorted
    assert last.failed_count == rr.result.failed_grasps
    assert last.push_count == rr.result.pushes
    assert last.tick == rr.result.steps
    # Frames render to RGB images without error.
    images = logic.render_frames(rr)
    assert len(images) == len(rr.frames)
    assert all(img.rgb.shape[2] == 3 for img in images)


def test_build_trace_make_planner_override_matches_builtin() -> None:
    """A make_planner override reproduces the built-in factory for a baseline.

    Passing an override equivalent to the built-in factory yields an identical
    trace, so the new parameter is behavior-preserving for existing planners.
    """
    from src import trace as trace_core

    a = trace_core.build_trace("occlusion", seed=3, n_items=12, clutter=0.9)
    b = trace_core.build_trace(
        "occlusion", seed=3, n_items=12, clutter=0.9,
        make_planner=lambda name, seed: trace_core.PLANNER_FACTORIES[name](seed),
    )
    assert a == b


# --------------------------------------------------------------------------- #
# Live View uses the OPTIMIZED weights, not the all-ones default (regression)
# --------------------------------------------------------------------------- #
def test_live_view_learned_uses_optimized_weights() -> None:
    """A learned replay must use the optimized weights, not all-ones.

    Regression: the app previously constructed the learned planner with a local
    all-1.0 default, so Live View replayed a different policy than intended.
    This asserts (a) the replay renders without raising, and (b) the planner
    the trace/render path uses has the optimized weights (the single-source
    DEFAULT_LEARNED_WEIGHTS), NOT (1.0, 1.0, 1.0).
    """
    from app import logic
    from src.planner import DEFAULT_LEARNED_WEIGHTS

    # The app constructs the 'learned' planner exactly as the replay path does.
    planner = logic.make_planner("learned", 0)
    assert tuple(planner.weights) == tuple(DEFAULT_LEARNED_WEIGHTS)
    assert tuple(planner.weights) != (1.0, 1.0, 1.0)
    # Pin the actual optimized vector so a silent weight change is caught.
    assert tuple(round(w, 4) for w in planner.weights) == (0.1175, 0.0888, -0.7255)

    # (a) The learned trial replays and renders without raising.
    rr = logic.replay_trial(
        logic.TrialConfig("learned", seed=1, n_items=12, clutter=0.6)
    )
    images = logic.render_frames(rr)
    assert images and all(img.rgb.shape[2] == 3 for img in images)
