"""Parity and determinism tests for the headless trace core (src.trace).

These lock in the behavior-preserving promotion of build_trace out of
src.render: the trace must reproduce run_episode's outcome exactly, and
render.py must still expose the moved names (unchanged public API).
"""

from __future__ import annotations

import pytest

from src import trace as trace_core
from src.scene import make_scene
from src.sim import run_episode

# A spread of seeds, planners, and scene configs to exercise the stepping core.
_SEEDS = (0, 1, 3, 7, 11)
_PLANNERS = ("random", "nearest", "leastcov", "nearretry", "occlusion")
_CONFIGS = ((8, 0.3), (15, 0.9), (20, 0.9))


@pytest.mark.parametrize("planner_name", _PLANNERS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_trace_final_counters_match_episode_result(planner_name, seed) -> None:
    """Property 2: trace final counters equal the paired EpisodeResult.

    For each seed/planner the trace's last frame must report the same sorted,
    failed, push, and tick totals as a direct run_episode on a fresh scene with
    a fresh planner and the same seed — proving the promotion changed nothing.
    """
    n_items, clutter = 15, 0.9
    result = run_episode(
        make_scene(seed=seed, n_items=n_items, clutter=clutter),
        trace_core.PLANNER_FACTORIES[planner_name](seed),
        seed=seed,
    )
    trace = trace_core.build_trace(
        planner_name, seed=seed, n_items=n_items, clutter=clutter
    )
    last = trace.frames[-1]
    assert last.sorted_count == result.items_sorted
    assert last.failed_count == result.failed_grasps
    assert last.push_count == result.pushes
    assert last.tick == result.steps


@pytest.mark.parametrize("n_items,clutter", _CONFIGS)
def test_trace_parity_across_configs(n_items, clutter) -> None:
    """Parity holds across several (n_items, clutter) configs for occlusion."""
    for seed in _SEEDS:
        result = run_episode(
            make_scene(seed=seed, n_items=n_items, clutter=clutter),
            trace_core.PLANNER_FACTORIES["occlusion"](seed),
            seed=seed,
        )
        trace = trace_core.build_trace(
            "occlusion", seed=seed, n_items=n_items, clutter=clutter
        )
        last = trace.frames[-1]
        assert last.sorted_count == result.items_sorted
        assert last.failed_count == result.failed_grasps
        assert last.push_count == result.pushes
        assert last.tick == result.steps


def test_build_trace_is_deterministic() -> None:
    """Property: equal config yields identical trace frames (value-equal)."""
    a = trace_core.build_trace("occlusion", seed=5, n_items=20, clutter=0.9)
    b = trace_core.build_trace("occlusion", seed=5, n_items=20, clutter=0.9)
    assert a == b


def test_render_public_api_reexports_trace_names() -> None:
    """Property 7.4: render.py re-exports the moved names (API unchanged)."""
    from src import render

    for name in (
        "PLANNER_FACTORIES",
        "ItemState",
        "ContainerState",
        "TickFrame",
        "Trace",
        "build_trace",
    ):
        assert getattr(render, name) is getattr(trace_core, name)


def test_trace_core_imports_no_pygame() -> None:
    """The trace core must be headless: importing it pulls in no pygame."""
    import sys

    sys.modules.pop("pygame", None)
    import importlib

    importlib.reload(trace_core)
    assert "pygame" not in sys.modules
