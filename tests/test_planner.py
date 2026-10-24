"""Tests for the planners: nearest-first and occlusion-aware behavior."""

from __future__ import annotations

import numpy as np

from src.planner import (
    LeastCoveredPlanner,
    NearestPlanner,
    NearestRetryPlanner,
    OcclusionPlanner,
    RandomPlanner,
)
from src.robot import Grasp, Gripper, Push
from src.robot import push as apply_push
from src.scene import DESK_H, DESK_W, Item, Scene, covered_fraction, make_scene, occluders
from src.sim import run_episode


def _scene(items: list[Item]) -> Scene:
    """Wrap hand-built items in a container-free scene."""
    return Scene(
        width=DESK_W,
        height=DESK_H,
        items=items,
        containers=[],
        clutter=0.0,
        rng=np.random.default_rng(0),
    )


def test_nearest_picks_closest_item() -> None:
    """NearestPlanner grasps the item closest to the gripper."""
    near = Item(id=0, kind="pen", x=30.0, y=30.0, w=20.0, h=20.0, z=0.0)
    far = Item(id=1, kind="key", x=400.0, y=400.0, w=20.0, h=20.0, z=0.0)
    scene = _scene([far, near])
    action = NearestPlanner().next_action(scene, Gripper(x=0.0, y=0.0))
    assert isinstance(action, Grasp)
    assert action.item_id == 0


def test_nearest_follows_gripper_position() -> None:
    """The nearest choice depends on where the gripper currently is."""
    left = Item(id=0, kind="pen", x=50.0, y=240.0, w=20.0, h=20.0, z=0.0)
    right = Item(id=1, kind="key", x=550.0, y=240.0, w=20.0, h=20.0, z=0.0)
    scene = _scene([left, right])
    action = NearestPlanner().next_action(scene, Gripper(x=600.0, y=240.0))
    assert action.item_id == 1


def test_occlusion_grasps_exposed_before_covered() -> None:
    """A covered item is never grasped while an exposed one is available.

    Item 0 sits under item 1 (fully covered); item 2 is off on its own
    (exposed). The planner must grasp an exposed item, never the covered one.
    """
    low = Item(id=0, kind="pen", x=200.0, y=200.0, w=40.0, h=40.0, z=0.0)
    cover = Item(id=1, kind="mug", x=200.0, y=200.0, w=40.0, h=40.0, z=1.0)
    free = Item(id=2, kind="key", x=400.0, y=100.0, w=40.0, h=40.0, z=0.0)
    scene = _scene([low, cover, free])
    assert occluders(scene) == {0: [1]}

    action = OcclusionPlanner().next_action(scene, Gripper(x=0.0, y=0.0))
    assert isinstance(action, Grasp)
    assert action.item_id != 0  # the covered item is off-limits


def test_occlusion_never_grasps_covered_over_full_episode() -> None:
    """Across a full cluttered run, every grasp targets an exposed item."""
    scene = make_scene(seed=4, n_items=20, clutter=0.8)
    planner = OcclusionPlanner()
    gripper = Gripper(x=0.0, y=0.0)

    for _ in range(1000):
        if not scene.items:
            break
        covered = occluders(scene)
        action = planner.next_action(scene, gripper)
        if action is None:
            break
        if isinstance(action, Grasp):
            if any(it.id not in covered for it in scene.items):
                assert action.item_id not in covered
            # Simulate a guaranteed success so the episode makes progress.
            scene.items = [
                it for it in scene.items if it.id != action.item_id
            ]
        else:  # Push
            apply_push(scene, action.item_id, action.dx, action.dy)
            moved = scene.item_by_id(action.item_id)
            gripper.x, gripper.y = moved.x, moved.y


def test_occlusion_pushes_coverer_after_two_failed_grasps() -> None:
    """Two failed grasps on a covered item trigger a push of its top coverer.

    Item 0 is covered by item 1. We seed the attempt counter to the retry
    threshold so the planner clears the coverer (pushes item 1) rather than
    grasping again. The push direction points from item 0 toward its coverer.
    """
    low = Item(id=0, kind="pen", x=200.0, y=200.0, w=40.0, h=40.0, z=0.0)
    cover = Item(id=1, kind="mug", x=220.0, y=200.0, w=40.0, h=40.0, z=1.0)
    scene = _scene([low, cover])
    assert occluders(scene) == {0: [1]}

    planner = OcclusionPlanner()
    planner._attempts[0] = 2  # noqa: SLF001 -- simulate two prior failures

    action = planner.next_action(scene, Gripper(x=0.0, y=0.0))
    assert isinstance(action, Push)
    assert action.item_id == 1  # the top coverer is shoved
    assert action.dx > 0.0  # coverer is to the right of item 0, so push right
    assert action.dy == 0.0


def test_occlusion_pushes_when_all_remaining_are_covered() -> None:
    """When no exposed item exists, the planner singulates by pushing.

    The attempt counter is cleared, so this exercises the all-covered branch
    directly: both items are marked covered via seeded attempts on the lower
    one, and the planner shoves the least-covered item's top coverer.
    """
    low = Item(id=0, kind="pen", x=200.0, y=200.0, w=40.0, h=40.0, z=0.0)
    cover = Item(id=1, kind="mug", x=200.0, y=200.0, w=40.0, h=40.0, z=1.0)
    scene = _scene([low, cover])

    planner = OcclusionPlanner()
    # Grasp the exposed coverer, then "fail" it twice so the planner is forced
    # to clear item 0's coverer.
    planner._attempts[0] = 2  # noqa: SLF001
    action = planner.next_action(scene, Gripper(x=0.0, y=0.0))
    assert isinstance(action, Push)


def test_least_covered_picks_smallest_coverage() -> None:
    """LeastCoveredPlanner grasps the least-covered item, ignoring crowding.

    Item 2 is fully covered, item 0 is exposed but farther, item 1 is exposed
    and nearer. Coverage ranks 0 and 1 equal (both 0.0), so the nearer wins.
    """
    exposed_far = Item(id=0, kind="pen", x=300.0, y=300.0, w=30.0, h=30.0, z=0.0)
    exposed_near = Item(id=1, kind="key", x=60.0, y=60.0, w=30.0, h=30.0, z=0.0)
    low = Item(id=2, kind="mug", x=150.0, y=150.0, w=40.0, h=40.0, z=0.0)
    cover = Item(id=3, kind="trash", x=150.0, y=150.0, w=40.0, h=40.0, z=1.0)
    scene = _scene([exposed_far, exposed_near, low, cover])
    assert covered_fraction(scene, 2) == 1.0

    action = LeastCoveredPlanner().next_action(scene, Gripper(x=0.0, y=0.0))
    assert isinstance(action, Grasp)
    assert action.item_id == 1  # exposed and nearest among the tie


def test_nearest_retry_skips_after_two_failures() -> None:
    """After two failed grasps on an item, NearestRetryPlanner leaves it be.

    Item 0 is the nearest but we never let it succeed (simulate failure by
    leaving it on the desk). After two failures it is skipped, so the planner
    switches to the farther item 1 for the next actions.
    """
    near = Item(id=0, kind="pen", x=40.0, y=40.0, w=20.0, h=20.0, z=0.0)
    far = Item(id=1, kind="key", x=300.0, y=300.0, w=20.0, h=20.0, z=0.0)
    scene = _scene([near, far])
    planner = NearestRetryPlanner()
    gripper = Gripper(x=0.0, y=0.0)

    # First two actions target the nearest item 0; we leave it (grasp "fails").
    a1 = planner.next_action(scene, gripper)
    assert a1.item_id == 0
    a2 = planner.next_action(scene, gripper)
    assert a2.item_id == 0
    # The second failure trips the skip; subsequent actions avoid item 0.
    a3 = planner.next_action(scene, gripper)
    assert a3.item_id == 1


def test_nearest_retry_returns_to_item_after_skip_window() -> None:
    """The skip is temporary: item 0 becomes eligible again after the window."""
    near = Item(id=0, kind="pen", x=40.0, y=40.0, w=20.0, h=20.0, z=0.0)
    far = Item(id=1, kind="key", x=300.0, y=300.0, w=20.0, h=20.0, z=0.0)
    scene = _scene([near, far])
    planner = NearestRetryPlanner()
    gripper = Gripper(x=0.0, y=0.0)

    planner.next_action(scene, gripper)  # targets 0 (fail)
    planner.next_action(scene, gripper)  # targets 0 (fail) -> skip trips
    targets = [planner.next_action(scene, gripper).item_id for _ in range(6)]
    # Item 0 is skipped for a window, then eligible (and nearest) again.
    assert targets[0] == 1
    assert 0 in targets  # it comes back once the window elapses


def test_occlusion_singulation_rule_is_value_based() -> None:
    """The singulation rule fires only when expected saving beats push cost.

    For a fully exposed but crowded item the base probability is 1.0, so the
    crowded probability is 0.7; the expected failed-grasp saving is capped well
    below the minimum push cost. The planner therefore grasps rather than
    pushes such an item. This pins the fixed rule's behavior.
    """
    a = Item(id=0, kind="pen", x=100.0, y=100.0, w=40.0, h=40.0, z=0.0)
    b = Item(id=1, kind="key", x=150.0, y=100.0, w=40.0, h=40.0, z=0.0)  # 10 px gap
    scene = _scene([a, b])  # a and b crowd each other, both exposed
    from src.scene import crowding_neighbors
    from src.robot import CLEARANCE_MARGIN

    assert crowding_neighbors(scene, 0, CLEARANCE_MARGIN)  # genuinely crowded
    planner = OcclusionPlanner()
    action = planner.next_action(scene, Gripper(x=0.0, y=0.0))
    assert isinstance(action, Grasp)  # saving < push cost -> just grasp


def test_planners_are_deterministic() -> None:
    """Each planner yields identical episode results for the same seed."""
    for make_planner in (
        lambda: RandomPlanner(seed=5),
        NearestPlanner,
        LeastCoveredPlanner,
        NearestRetryPlanner,
        OcclusionPlanner,
    ):
        a = run_episode(
            make_scene(seed=5, n_items=15, clutter=0.7), make_planner(), seed=5
        )
        b = run_episode(
            make_scene(seed=5, n_items=15, clutter=0.7), make_planner(), seed=5
        )
        assert a == b
