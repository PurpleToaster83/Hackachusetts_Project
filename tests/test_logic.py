"""Tests for the headless logic layer (app.logic).

Covers determinism, metrics parity with the engine, the CI formula, the
fresh-planner-per-trial guarantee, input clamping and errors, the import
boundary (no streamlit/pygame/render), and the output-location guard.
"""

from __future__ import annotations

import importlib
import math
import sys

import pytest

from app import logic
from src.scene import make_scene
from src.sim import run_episode


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _experiment(seeds, planners, n_items=12, clutter=0.6):
    """Build an ExperimentConfig for the given seeds and planners."""
    return logic.ExperimentConfig(
        n_items=n_items, clutter=clutter, seeds=list(seeds), planners=list(planners)
    )


# --------------------------------------------------------------------------- #
# (3) CI formula
# --------------------------------------------------------------------------- #
def test_ci95_matches_hand_computation() -> None:
    """ci95 equals 1.96 * sample_std(ddof=1) / sqrt(n) on a known list."""
    values = [10.0, 12.0, 14.0, 16.0, 18.0]
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    expected = 1.96 * math.sqrt(var) / math.sqrt(n)
    assert logic.ci95(values) == pytest.approx(expected)


def test_ci95_under_two_samples_is_zero() -> None:
    """Fewer than two samples yields a 0.0 interval (undefined)."""
    assert logic.ci95([5.0]) == 0.0
    assert logic.ci95([]) == 0.0


def test_ci95_matches_engine_definition() -> None:
    """ci95 agrees with src.compare._ci95 (the engine's definition)."""
    from src.compare import _ci95 as engine_ci

    values = [3.0, 7.0, 7.0, 19.0, 2.0, 11.0]
    assert logic.ci95(values) == pytest.approx(engine_ci(values))


def test_overlap_rule() -> None:
    """overlap is True for touching/overlapping intervals, False when clear."""
    a = logic.Stat(mean=10.0, ci=2.0)  # [8, 12]
    b = logic.Stat(mean=13.0, ci=2.0)  # [11, 15] -> overlap
    c = logic.Stat(mean=20.0, ci=1.0)  # [19, 21] -> clear of a
    assert logic.overlap(a, b) is True
    assert logic.overlap(a, c) is False
    # Symmetric.
    assert logic.overlap(b, a) is True
    assert logic.overlap(c, a) is False


# --------------------------------------------------------------------------- #
# (1) Determinism
# --------------------------------------------------------------------------- #
def test_run_experiment_is_deterministic() -> None:
    """Equal configs yield identical rows and identical stats."""
    cfg = _experiment(seeds=range(4), planners=["occlusion", "nearest"])
    a = logic.run_experiment(cfg)
    b = logic.run_experiment(cfg)
    assert a.rows == b.rows
    assert a.stats == b.stats


def test_run_trial_is_deterministic() -> None:
    """A single TrialConfig run twice yields an identical EpisodeResult."""
    tc = logic.TrialConfig(planner_name="occlusion", seed=7, n_items=15, clutter=0.9)
    assert logic.run_trial(tc) == logic.run_trial(tc)


# --------------------------------------------------------------------------- #
# (2) Metrics parity with a direct engine run
# --------------------------------------------------------------------------- #
def test_trial_result_matches_direct_run_episode() -> None:
    """A TrialRow's EpisodeResult equals a direct run_episode (all 5 fields)."""
    tc = logic.TrialConfig(planner_name="nearretry", seed=5, n_items=15, clutter=0.7)
    via_logic = logic.run_trial(tc)

    scene = make_scene(seed=5, n_items=15, clutter=0.7)
    planner = logic.make_planner("nearretry", 5)
    direct = run_episode(scene, planner, seed=5)

    assert via_logic == direct


# --------------------------------------------------------------------------- #
# (4) Fresh planner per trial (no state leak)
# --------------------------------------------------------------------------- #
def test_no_state_leak_across_trials_for_stateful_planner() -> None:
    """A stateful planner in a batch matches independent per-seed runs.

    OcclusionPlanner carries mutable attempt/retry state. If run_experiment
    reused a planner instance across seeds, batch results would diverge from
    running each seed with its own fresh planner. They must match.
    """
    seeds = range(6)
    cfg = _experiment(seeds=seeds, planners=["occlusion"], n_items=20, clutter=0.9)
    batch = logic.run_experiment(cfg)

    for row in batch.rows:
        independent = logic.run_trial(row.config)
        assert row.result == independent


def test_aggregate_stats_matches_hand_aggregation() -> None:
    """aggregate_stats means equal a direct mean over the trial results."""
    cfg = _experiment(seeds=range(5), planners=["nearest"], n_items=10, clutter=0.5)
    result = logic.run_experiment(cfg)
    stat = result.stats[0]

    ticks = [float(r.result.steps) for r in result.rows]
    assert stat.n_trials == 5
    assert stat.ticks.mean == pytest.approx(sum(ticks) / len(ticks))
    assert stat.ticks.ci == pytest.approx(logic.ci95(ticks))


# --------------------------------------------------------------------------- #
# (5) Clamping and ValueError
# --------------------------------------------------------------------------- #
def test_clutter_is_clamped() -> None:
    """clutter below 0 or above 1 is clamped into [0, 1]."""
    assert logic.clamp_clutter(-0.5) == 0.0
    assert logic.clamp_clutter(1.5) == 1.0
    assert logic.clamp_clutter(0.3) == 0.3
    # build_scene applies the clamp (make_scene also clips, so scenes are equal).
    assert logic.build_scene(1, 10, -0.5) == logic.build_scene(1, 10, 0.0)


def test_n_items_is_clamped() -> None:
    """n_items is clamped into the supported range before make_scene."""
    assert logic.clamp_n_items(-3) == logic.N_ITEMS_MIN
    assert logic.clamp_n_items(999) == logic.N_ITEMS_MAX
    assert logic.clamp_n_items(12) == 12
    scene = logic.build_scene(2, 999, 0.5)
    assert len(scene.items) == logic.N_ITEMS_MAX


def test_make_planner_rejects_unknown_name() -> None:
    """make_planner raises ValueError naming the bad planner."""
    with pytest.raises(ValueError, match="unknown planner"):
        logic.make_planner("does-not-exist", 0)


def test_run_experiment_rejects_unknown_planner() -> None:
    """run_experiment validates planner names before running any trial."""
    cfg = _experiment(seeds=[0, 1], planners=["nearest", "bogus"])
    with pytest.raises(ValueError, match="unknown planner"):
        logic.run_experiment(cfg)


def test_available_planners_is_stable() -> None:
    """available_planners returns the documented five names in a stable order."""
    names = logic.available_planners()
    assert names == ["random", "nearest", "leastcov", "nearretry", "occlusion"]
    assert logic.available_planners() == names


# --------------------------------------------------------------------------- #
# (6) Import boundary
# --------------------------------------------------------------------------- #
def test_logic_imports_no_ui_or_pygame_modules() -> None:
    """Importing app.logic must not pull in streamlit, pygame, or render."""
    # Drop any previously-imported copies, then re-import fresh.
    for mod in ("streamlit", "pygame", "src.render"):
        sys.modules.pop(mod, None)
    importlib.reload(logic)
    assert "streamlit" not in sys.modules
    assert "pygame" not in sys.modules
    assert "src.render" not in sys.modules


# --------------------------------------------------------------------------- #
# (7) Output-location guard
# --------------------------------------------------------------------------- #
def test_output_guard_accepts_paths_under_out() -> None:
    """A relative path resolves under out/; an explicit out/ path is accepted."""
    resolved = logic.resolve_output_path("dataset/episodes.csv")
    assert resolved.is_relative_to(logic.OUT_DIR.resolve())
    assert logic.resolve_output_path("out/plot.png").is_relative_to(
        logic.OUT_DIR.resolve()
    )


def test_output_guard_rejects_paths_outside_out() -> None:
    """Paths under data/ or robotics/, or escaping out/, are rejected."""
    # Relative paths targeting a read-only input tree are rejected outright.
    with pytest.raises(ValueError, match="read-only input tree"):
        logic.resolve_output_path("data/bridge-v2/data.csv")
    with pytest.raises(ValueError, match="read-only input tree"):
        logic.resolve_output_path("robotics/oxe/data.csv")
    # A path escaping out/ via .. is rejected as outside out/.
    with pytest.raises(ValueError, match="outside out/"):
        logic.resolve_output_path("../escape.txt")
    # An absolute path outside out/ is rejected too.
    with pytest.raises(ValueError, match="outside out/"):
        logic.resolve_output_path("/tmp/escape.txt")
