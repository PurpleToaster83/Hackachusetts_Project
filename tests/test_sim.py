"""Tests for the episode runner and the random planner."""

from __future__ import annotations

import numpy as np

from src.planner import RandomPlanner
from src.scene import (
    DESK_H,
    DESK_W,
    Item,
    Scene,
    covered_fraction,
    make_scene,
    occluders,
)
from src.sim import EpisodeResult, run_episode


def _spaced_scene(n_items: int) -> Scene:
    """A hand-built scene whose small items sit far apart, so nothing overlaps.

    Items are 20x20 placed on a grid with 60 px spacing, guaranteeing zero
    coverage regardless of the scene generator's random placement.
    """
    items = [
        Item(
            id=i,
            kind="pen",
            x=40.0 + (i % 8) * 60.0,
            y=40.0 + (i // 8) * 60.0,
            w=20.0,
            h=20.0,
            z=float(i),
        )
        for i in range(n_items)
    ]
    return Scene(
        width=DESK_W,
        height=DESK_H,
        items=items,
        containers=[],
        clutter=0.0,
        rng=np.random.default_rng(0),
    )


def _run(seed: int, n_items: int, clutter: float) -> EpisodeResult:
    """Build a fresh scene and planner for one seeded episode."""
    scene = make_scene(seed=seed, n_items=n_items, clutter=clutter)
    planner = RandomPlanner(seed=seed)
    return run_episode(scene, planner, seed=seed)


def test_same_seed_is_deterministic() -> None:
    """Identical seeds produce identical episode results."""
    a = _run(seed=3, n_items=15, clutter=0.6)
    b = _run(seed=3, n_items=15, clutter=0.6)
    assert a == b


def test_empty_scene_finishes_in_zero_steps() -> None:
    """A scene with no items takes no steps and sorts nothing."""
    scene = Scene(
        width=DESK_W,
        height=DESK_H,
        items=[],
        containers=[],
        clutter=0.0,
        rng=np.random.default_rng(0),
    )
    result = run_episode(scene, RandomPlanner(seed=0), seed=0)
    assert result == EpisodeResult(
        items_sorted=0, items_total=0, steps=0, failed_grasps=0, pushes=0
    )


def test_no_overlap_sorts_every_item() -> None:
    """When nothing overlaps, every item is grasped and the desk clears.

    Coverage is 0 for every item, so grasp probability is exactly 1.0 and no
    attempt can fail; the run must clear the desk in exactly n steps.

    Note: make_scene(clutter=0) does not guarantee a non-overlapping layout, so
    this uses a hand-built spaced scene to pin down the property.
    """
    n_items = 15
    scene = _spaced_scene(n_items)
    # Precondition: the layout really has no occluded items.
    assert occluders(scene) == {}
    for item in scene.items:
        assert covered_fraction(scene, item.id) == 0.0

    result = run_episode(scene, RandomPlanner(seed=11), seed=11)
    assert result.items_sorted == result.items_total == n_items
    assert result.failed_grasps == 0
    assert result.steps == n_items  # every grasp succeeds, so no wasted steps
