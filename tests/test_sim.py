"""Tests for the episode runner and the random planner."""

from __future__ import annotations

import numpy as np

from src.planner import RandomPlanner
from src.robot import FAILED_GRASP_TICKS, Grasp, Gripper, Push
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


class GraspOnlyPlanner:
    """Always grasps the first remaining item. Deterministic, no pushes."""

    def next_action(self, scene: Scene, gripper: Gripper) -> Grasp | None:
        """Grasp the first remaining item, or None when the desk is clear."""
        if not scene.items:
            return None
        return Grasp(item_id=scene.items[0].id)


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
    """Identical seeds produce identical episode results, pushes included."""
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
    """When nothing overlaps, a grasp-only run clears the whole desk.

    Coverage is 0 for every item, so grasp probability is exactly 1.0 and no
    attempt can fail. A grasp-only planner keeps the run deterministic so the
    sorting guarantee is testable.

    Note: make_scene(clutter=0) does not guarantee a non-overlapping layout, so
    this uses a hand-built spaced scene to pin down the property.
    """
    n_items = 15
    scene = _spaced_scene(n_items)
    # Precondition: the layout really has no occluded items.
    assert occluders(scene) == {}
    for item in scene.items:
        assert covered_fraction(scene, item.id) == 0.0

    result = run_episode(scene, GraspOnlyPlanner(), seed=11)
    assert result.items_sorted == result.items_total == n_items
    assert result.failed_grasps == 0
    assert result.pushes == 0


def test_pushes_are_counted() -> None:
    """A push action increments the push counter in the result."""

    class PushThenGrasp:
        """Push the only item once, then grasp it."""

        def __init__(self) -> None:
            self.done_push = False

        def next_action(self, scene: Scene, gripper: Gripper):
            if not scene.items:
                return None
            item_id = scene.items[0].id
            if not self.done_push:
                self.done_push = True
                return Push(item_id=item_id, dx=40.0, dy=0.0)
            return Grasp(item_id=item_id)

    scene = _spaced_scene(1)
    result = run_episode(scene, PushThenGrasp(), seed=0)
    assert result.pushes == 1
    assert result.items_sorted == 1


def test_travel_cost_shows_up_in_steps() -> None:
    """A far-apart grasp-only run costs more ticks than items grasped.

    With a spaced grid the gripper travels between items, so ticks exceed the
    item count even though every grasp succeeds on the first try.
    """
    n_items = 15
    scene = _spaced_scene(n_items)
    result = run_episode(scene, GraspOnlyPlanner(), seed=0)
    assert result.items_sorted == n_items
    assert result.failed_grasps == 0
    assert result.steps > n_items  # travel between items adds ticks


def test_failed_grasp_costs_failed_grasp_ticks() -> None:
    """One failed grasp charges travel plus FAILED_GRASP_TICKS, nothing else.

    Item B is fully covered by A (grasp prob 0.2). Seed 0's first rng draw is
    0.637 >= 0.2, so the single grasp attempt fails. The planner then stops, so
    the only cost is the travel to B's center plus the failed-grasp penalty.
    """

    class GraspBOnce:
        """Attempt to grasp item id 0 exactly once, then stop."""

        def __init__(self) -> None:
            self.done = False

        def next_action(self, scene: Scene, gripper: Gripper):
            if self.done:
                return None
            self.done = True
            return Grasp(item_id=0)

    b = Item(id=0, kind="pen", x=120.0, y=0.0, w=40.0, h=40.0, z=0.0)
    a = Item(id=1, kind="mug", x=120.0, y=0.0, w=40.0, h=40.0, z=1.0)
    scene = Scene(
        width=DESK_W,
        height=DESK_H,
        items=[b, a],
        containers=[],
        clutter=0.0,
        rng=np.random.default_rng(0),
    )
    # Precondition: B is fully covered.
    assert covered_fraction(scene, 0) == 1.0

    result = run_episode(scene, GraspBOnce(), seed=0)
    assert result.failed_grasps == 1
    assert result.items_sorted == 0
    # Gripper starts at (0, 0); B is at (120, 0) -> 120 px -> 1 travel tick.
    assert result.steps == 1 + FAILED_GRASP_TICKS
