"""Tests for the rotated-rectangle footprint path in src.scene.

Covers the convex polygon overlap for rotated items, the on-edge clipping
degeneracy that was fixed, monotonic overlap decrease as identical squares
rotate apart, covered_fraction bounds, rotated-scene determinism, and the
higher-z-only rule for occluders. Axis-aligned behavior is covered by
tests/test_scene.py and must remain unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.scene import (
    DESK_H,
    DESK_W,
    Item,
    Scene,
    _overlap_area,
    covered_fraction,
    make_scene,
    occluders,
)


def _scene(items: list[Item]) -> Scene:
    """Wrap hand-built items in a container-free Scene."""
    return Scene(
        width=DESK_W, height=DESK_H, items=items, containers=[],
        clutter=0.0, rng=np.random.default_rng(0),
    )


# --------------------------------------------------------------------------- #
# Overlap area: known rotated cases, parity at angle 0, symmetry
# --------------------------------------------------------------------------- #
def test_overlap_matches_aabb_at_angle_zero() -> None:
    """With both items axis-aligned the polygon overlap equals the AABB area."""
    b = Item(id=0, kind="pen", x=100, y=100, w=40, h=40, z=0.0)
    a = Item(id=1, kind="mug", x=120, y=120, w=40, h=40, z=1.0)
    assert _overlap_area(a, b) == pytest.approx(400.0)  # 20x20 corner
    same = Item(id=2, kind="pen", x=100, y=100, w=40, h=40, z=0.0)
    assert _overlap_area(b, same) == pytest.approx(1600.0)  # full


def test_overlap_on_edge_case_has_no_degenerate_area() -> None:
    """A 45deg diamond sharing a corner region clips cleanly to the right area.

    This is the case that previously produced a duplicate intersection vertex.
    The correct overlap is the 400 px^2 triangle bounded by x=120, y=120 and
    the diamond's two near edges.
    """
    b = Item(id=0, kind="pen", x=100, y=100, w=40, h=40, z=0.0)
    diamond = Item(id=1, kind="mug", x=120, y=120, w=40, h=40, z=1.0,
                   angle=math.pi / 4)
    assert _overlap_area(b, diamond) == pytest.approx(400.0, abs=1e-6)


def test_overlap_is_symmetric_for_rotated_items() -> None:
    """overlap(a, b) == overlap(b, a) for rotated footprints."""
    a = Item(id=0, kind="pen", x=100, y=100, w=48, h=30, z=0.0, angle=0.6)
    b = Item(id=1, kind="mug", x=118, y=108, w=40, h=36, z=1.0, angle=-0.4)
    assert _overlap_area(a, b) == pytest.approx(_overlap_area(b, a))


def test_disjoint_rotated_items_do_not_overlap() -> None:
    """Far-apart rotated items have zero overlap."""
    a = Item(id=0, kind="pen", x=80, y=80, w=40, h=40, z=0.0, angle=0.5)
    b = Item(id=1, kind="mug", x=400, y=400, w=40, h=40, z=1.0, angle=1.2)
    assert _overlap_area(a, b) == 0.0


def test_full_overlap_never_exceeds_item_area() -> None:
    """Overlap of a concentric rotated square never exceeds the box area."""
    base = Item(id=0, kind="pen", x=200, y=200, w=40, h=40, z=0.0)
    for deg in range(0, 91, 15):
        rot = Item(id=1, kind="mug", x=200, y=200, w=40, h=40, z=1.0,
                   angle=math.radians(deg))
        ov = _overlap_area(base, rot)
        assert ov <= base.area + 1e-6
        assert ov >= 0.0


# --------------------------------------------------------------------------- #
# Monotonic overlap decrease as identical squares rotate apart
# --------------------------------------------------------------------------- #
def test_overlap_decreases_as_identical_squares_rotate_apart() -> None:
    """Two concentric identical squares: overlap falls as the angle grows 0->45.

    At 0 deg they coincide (full area). As one rotates toward 45 deg the
    intersection shrinks monotonically to the classic two-squares minimum.
    """
    base = Item(id=0, kind="pen", x=200, y=200, w=40, h=40, z=0.0)
    prev = None
    for deg in (0, 10, 20, 30, 40, 45):
        rot = Item(id=1, kind="mug", x=200, y=200, w=40, h=40, z=1.0,
                   angle=math.radians(deg))
        ov = _overlap_area(base, rot)
        if prev is not None:
            assert ov <= prev + 1e-6, f"overlap rose at {deg} deg"
        prev = ov
    # Full at 0, strictly less by 45.
    full = _overlap_area(base, Item(2, "pen", 200, 200, 40, 40, 0.0))
    assert prev < full


# --------------------------------------------------------------------------- #
# covered_fraction stays in [0, 1]
# --------------------------------------------------------------------------- #
def test_covered_fraction_in_unit_range_for_rotated_scene() -> None:
    """Every item's covered fraction stays within [0, 1] on a rotated scene."""
    scene = make_scene(seed=4, n_items=25, clutter=0.9, rotate=True)
    for it in scene.items:
        cf = covered_fraction(scene, it.id)
        assert 0.0 <= cf <= 1.0


def test_covered_fraction_capped_with_multiple_rotated_coverers() -> None:
    """Many higher rotated items covering one target cap the fraction at 1.0."""
    target = Item(id=0, kind="pen", x=200, y=200, w=40, h=40, z=0.0)
    coverers = [
        Item(id=i, kind="mug", x=200, y=200, w=40, h=40, z=float(i),
             angle=math.radians(15 * i))
        for i in range(1, 5)
    ]
    scene = _scene([target, *coverers])
    assert covered_fraction(scene, 0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Determinism of rotated scenes
# --------------------------------------------------------------------------- #
def test_rotated_scene_is_deterministic() -> None:
    """A fixed seed fully determines a rotated scene, angles included."""
    a = make_scene(seed=7, n_items=20, clutter=0.6, rotate=True)
    b = make_scene(seed=7, n_items=20, clutter=0.6, rotate=True)
    assert a == b
    assert a.items == b.items
    assert [it.angle for it in a.items] == [it.angle for it in b.items]
    assert occluders(a) == occluders(b)


def test_rotated_scene_has_nonzero_angles() -> None:
    """rotate=True actually assigns rotation; rotate=False leaves angles at 0."""
    rot = make_scene(seed=3, n_items=15, clutter=0.6, rotate=True)
    assert any(it.angle != 0.0 for it in rot.items)
    plain = make_scene(seed=3, n_items=15, clutter=0.6)
    assert all(it.angle == 0.0 for it in plain.items)


# --------------------------------------------------------------------------- #
# occluders lists only higher-z items
# --------------------------------------------------------------------------- #
def test_occluders_only_lists_higher_z_rotated() -> None:
    """Only higher-z rotated items occlude; a lower one never does."""
    top = Item(id=0, kind="pen", x=200, y=200, w=40, h=40, z=5.0,
               angle=math.radians(20))
    under = Item(id=1, kind="mug", x=205, y=203, w=40, h=40, z=1.0,
                 angle=math.radians(50))
    scene = _scene([top, under])
    result = occluders(scene)
    # The lower item is occluded by the higher one; the top item is not listed.
    assert 1 in result
    assert result[1] == [0]
    assert 0 not in result


def test_occluders_deterministic_on_rotated_scene() -> None:
    """occluders over a rotated scene is stable for a fixed seed."""
    scene = make_scene(seed=11, n_items=20, clutter=0.9, rotate=True)
    assert occluders(scene) == occluders(make_scene(
        seed=11, n_items=20, clutter=0.9, rotate=True))
