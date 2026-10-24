"""Tests for the scene model: determinism, overlap math, clutter behavior."""

from __future__ import annotations

import numpy as np

from src.scene import (
    DESK_H,
    DESK_W,
    Item,
    Scene,
    make_scene,
    occluders,
)


def _empty_scene(items: list[Item]) -> Scene:
    """A Scene wrapping hand-built items, with no containers."""
    return Scene(
        width=DESK_W,
        height=DESK_H,
        items=items,
        containers=[],
        clutter=0.0,
        rng=np.random.default_rng(0),
    )


def test_same_seed_is_deterministic() -> None:
    """A given seed fully determines the scene, including occluders."""
    a = make_scene(seed=7, n_items=20, clutter=0.6)
    b = make_scene(seed=7, n_items=20, clutter=0.6)
    # Scene excludes the rng from equality, so value equality holds.
    assert a == b
    assert a.items == b.items
    assert occluders(a) == occluders(b)


def test_different_seed_differs() -> None:
    """Different seeds should not produce identical item layouts."""
    a = make_scene(seed=1, n_items=20, clutter=0.6)
    b = make_scene(seed=2, n_items=20, clutter=0.6)
    assert a.items != b.items


def test_overlap_fraction_hand_built() -> None:
    """A square covering exactly a quarter of a target counts as occluding.

    Target B: 40x40 centered at (100, 100), area 1600 px^2, z=0.
    Cover A: 40x40 centered at (120, 120), z=1. The overlap is the
    20x20 bottom-right corner = 400 px^2, i.e. fraction 0.25 exactly.
    0.25 is not strictly greater than the 0.25 threshold, so B is not occluded.
    """
    b = Item(id=0, kind="pen", x=100.0, y=100.0, w=40.0, h=40.0, z=0.0)
    a = Item(id=1, kind="mug", x=120.0, y=120.0, w=40.0, h=40.0, z=1.0)
    scene = _empty_scene([b, a])
    # Exactly at threshold -> excluded.
    assert occluders(scene) == {}

    # Nudge the cover to raise overlap above 0.25: shift so overlap is
    # 22x22 = 484 px^2 -> fraction 0.3025 > 0.25.
    a2 = Item(id=1, kind="mug", x=118.0, y=118.0, w=40.0, h=40.0, z=1.0)
    scene2 = _empty_scene([b, a2])
    assert occluders(scene2) == {0: [1]}


def test_lower_item_does_not_occlude() -> None:
    """Only items with higher z occlude; a lower item never does."""
    top = Item(id=0, kind="pen", x=100.0, y=100.0, w=40.0, h=40.0, z=5.0)
    under = Item(id=1, kind="mug", x=100.0, y=100.0, w=40.0, h=40.0, z=1.0)
    scene = _empty_scene([top, under])
    result = occluders(scene)
    # The fully-covered lower item is occluded; the top item is not.
    assert result == {1: [0]}


def test_occluders_omits_uncovered_items() -> None:
    """Items nothing covers are absent, so len() counts covered items."""
    far = Item(id=0, kind="pen", x=50.0, y=50.0, w=20.0, h=20.0, z=0.0)
    lonely = Item(id=1, kind="key", x=400.0, y=400.0, w=20.0, h=20.0, z=1.0)
    scene = _empty_scene([far, lonely])
    result = occluders(scene)
    assert result == {}
    assert len(result) == 0


def test_more_clutter_covers_more_items() -> None:
    """Clutter 0.9 yields more covered items than 0.1, averaged over seeds."""
    seeds = range(20)
    n_items = 20

    def avg_covered(clutter: float) -> float:
        total = 0
        for s in seeds:
            scene = make_scene(seed=s, n_items=n_items, clutter=clutter)
            total += len(occluders(scene))
        return total / len(seeds)

    high = avg_covered(0.9)
    low = avg_covered(0.1)
    assert high > low
