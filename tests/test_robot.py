"""Tests for the robot model: grasp probability, travel cost, push primitive."""

from __future__ import annotations

from math import isclose

import numpy as np

from src.robot import (
    CLEARANCE_MARGIN,
    CLEARANCE_PENALTY,
    FAILED_GRASP_TICKS,
    MAX_GRASP_PROB,
    MAX_SPEED,
    MIN_GRASP_PROB,
    Gripper,
    effective_grasp_prob,
    grasp_success_prob,
    push,
    travel_ticks,
)
from src.scene import DESK_H, DESK_W, Item, Scene, covered_fraction


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


def test_exposed_item_grasps_certainly() -> None:
    """A fully exposed item (coverage 0) has probability 1.0."""
    assert grasp_success_prob(0.0) == 1.0


def test_base_formula_midrange() -> None:
    """In-range coverage follows 1 - 0.8 * covered_fraction."""
    assert isclose(grasp_success_prob(0.5), 0.6)


def test_clamped_to_floor() -> None:
    """Heavy coverage is clamped to the 0.05 floor, never below."""
    # 1 - 0.8 * 1.0 = 0.2, still above the floor.
    assert isclose(grasp_success_prob(1.0), 0.2)
    # Beyond-1 coverage would dip under the floor without clamping.
    assert grasp_success_prob(2.0) == MIN_GRASP_PROB


def test_clamped_to_ceiling() -> None:
    """Negative coverage cannot push probability above 1.0."""
    assert grasp_success_prob(-1.0) == MAX_GRASP_PROB


def test_gripper_defaults_to_empty() -> None:
    """A gripper starts holding nothing."""
    g = Gripper(x=0.0, y=0.0)
    assert g.holding is None


def test_failed_grasp_penalty_is_three_ticks() -> None:
    """The failed-grasp penalty constant is three ticks."""
    assert FAILED_GRASP_TICKS == 3


def test_travel_cost_grows_with_distance() -> None:
    """Farther moves cost at least as many ticks, strictly more past a cap."""
    near = travel_ticks(MAX_SPEED)  # exactly one tick of travel
    far = travel_ticks(MAX_SPEED * 5)  # five ticks
    assert near == 1
    assert far == 5
    assert far > near
    # Monotonic: a longer move never costs fewer ticks.
    assert travel_ticks(10.0) <= travel_ticks(200.0) <= travel_ticks(400.0)


def test_travel_cost_is_at_least_one_tick() -> None:
    """Even a zero-distance action costs a tick."""
    assert travel_ticks(0.0) == 1
    assert travel_ticks(-5.0) == 1


def test_push_reduces_covered_fraction_below() -> None:
    """Pushing a covering item off its target drops the target's coverage.

    Target B sits under cover A (higher z) with heavy overlap. Shoving A far to
    the side must reduce B's covered fraction.
    """
    b = Item(id=0, kind="pen", x=100.0, y=100.0, w=40.0, h=40.0, z=0.0)
    a = Item(id=1, kind="mug", x=100.0, y=100.0, w=40.0, h=40.0, z=1.0)
    scene = _scene([b, a])

    before = covered_fraction(scene, 0)
    assert before > 0.0  # precondition: B really is covered

    push(scene, item_id=1, dx=200.0, dy=0.0)
    after = covered_fraction(scene, 0)
    assert after < before
    assert after == 0.0  # pushed fully clear


def test_push_clamps_to_desk() -> None:
    """A push cannot shove an item off the desk."""
    item = Item(id=0, kind="key", x=50.0, y=50.0, w=20.0, h=20.0, z=0.0)
    scene = _scene([item])
    push(scene, item_id=0, dx=-1000.0, dy=-1000.0)
    moved = scene.item_by_id(0)
    assert moved.x == item.w / 2.0
    assert moved.y == item.h / 2.0


def test_clearance_penalty_constant() -> None:
    """The crowding penalty and margin match the spec."""
    assert CLEARANCE_PENALTY == 0.7
    assert CLEARANCE_MARGIN == 15.0


def test_crowded_item_takes_the_penalty() -> None:
    """A neighbour within the margin multiplies grasp probability by 0.7.

    Two uncovered items with a 10 px gap (< 15 px margin). Each is exposed, so
    the base probability is 1.0; crowding drops it to 0.7.
    """
    a = Item(id=0, kind="pen", x=100.0, y=100.0, w=40.0, h=40.0, z=0.0)
    # a's right edge is at 120; put b's left edge at 130 (center 150) so the
    # gap is 10 px < 15 px margin.
    b = Item(id=1, kind="key", x=150.0, y=100.0, w=40.0, h=40.0, z=0.0)
    scene = _scene([a, b])
    assert covered_fraction(scene, 0) == 0.0  # exposed
    assert isclose(effective_grasp_prob(scene, 0), CLEARANCE_PENALTY)


def test_uncrowded_item_has_no_penalty() -> None:
    """A lone exposed item keeps the full probability."""
    a = Item(id=0, kind="pen", x=100.0, y=100.0, w=40.0, h=40.0, z=0.0)
    far = Item(id=1, kind="key", x=400.0, y=400.0, w=40.0, h=40.0, z=0.0)
    scene = _scene([a, far])
    assert effective_grasp_prob(scene, 0) == 1.0


def test_push_removes_the_clearance_penalty() -> None:
    """Shoving the crowding neighbour away restores the full probability."""
    a = Item(id=0, kind="pen", x=100.0, y=100.0, w=40.0, h=40.0, z=0.0)
    b = Item(id=1, kind="key", x=150.0, y=100.0, w=40.0, h=40.0, z=0.0)  # 10 px gap
    scene = _scene([a, b])
    assert isclose(effective_grasp_prob(scene, 0), CLEARANCE_PENALTY)

    # Push b well clear of a's 15 px margin.
    push(scene, item_id=1, dx=200.0, dy=0.0)
    assert effective_grasp_prob(scene, 0) == 1.0