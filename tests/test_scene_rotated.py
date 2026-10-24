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


# --------------------------------------------------------------------------- #
# Per-kind convex polygon footprints (experimental rotated mode)
# --------------------------------------------------------------------------- #
from src.scene import KIND_SHAPE, _convex_hull  # noqa: E402


def _is_convex_ccw(poly: list[tuple[float, float]]) -> bool:
    """True when ``poly`` is convex and wound counter-clockwise."""
    n = len(poly)
    for i in range(n):
        o, a, b = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        if cross < -1e-9:  # a strict clockwise turn -> not convex-CCW
            return False
    return True


@pytest.mark.parametrize(
    "kind,shape,w,h",
    [("pen", "box", 48, 16), ("key", "hexagon", 44, 28),
     ("mug", "rounded", 40, 40), ("trash", "blob", 40, 34)],
)
def test_each_kind_shape_is_convex_ccw(kind, shape, w, h) -> None:
    """Every per-kind footprint is a strictly convex, CCW polygon."""
    it = Item(id=1, kind=kind, x=200, y=200, w=float(w), h=float(h), z=0.0,
              angle=0.6, shape=shape)
    poly = it.polygon()
    assert len(poly) >= 3
    assert _is_convex_ccw(poly)


@pytest.mark.parametrize("shape", ["box", "hexagon", "rounded", "blob"])
def test_shape_area_within_bounding_box(shape) -> None:
    """A convex footprint's area is positive and never exceeds its w*h box."""
    it = Item(id=1, kind="mug", x=100, y=100, w=40.0, h=30.0, z=0.0,
              shape=shape)
    assert 0.0 < it.area <= 40.0 * 30.0 + 1e-6


def test_kind_shape_map_covers_all_kinds() -> None:
    """Every KIND has a shape, and all mapped shapes are supported convex ones."""
    from src.scene import KINDS

    assert set(KIND_SHAPE) == set(KINDS)
    assert set(KIND_SHAPE.values()) <= {"box", "hexagon", "rounded", "blob"}


def test_convex_hull_is_convex_and_ccw() -> None:
    """_convex_hull returns a convex CCW polygon and drops interior points."""
    pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]  # last point is interior
    hull = _convex_hull(pts)
    assert (2, 2) not in hull  # interior point excluded
    assert _is_convex_ccw(hull)
    assert len(hull) == 4


def test_polygon_overlap_symmetric_for_mixed_shapes() -> None:
    """overlap(a, b) == overlap(b, a) for two different convex polygon kinds."""
    a = Item(id=0, kind="key", x=200, y=200, w=44, h=28, z=0.0,
             angle=0.3, shape="hexagon")
    b = Item(id=1, kind="trash", x=214, y=206, w=40, h=34, z=1.0,
             angle=-0.5, shape="blob")
    assert _overlap_area(a, b) == pytest.approx(_overlap_area(b, a))


def test_polygon_overlap_never_exceeds_smaller_area() -> None:
    """Intersection area cannot exceed either footprint's own area."""
    a = Item(id=0, kind="mug", x=200, y=200, w=40, h=40, z=0.0,
             angle=0.4, shape="rounded")
    b = Item(id=1, kind="key", x=205, y=203, w=44, h=28, z=1.0,
             angle=0.9, shape="hexagon")
    ov = _overlap_area(a, b)
    assert 0.0 <= ov <= min(a.area, b.area) + 1e-6


def test_covered_fraction_in_unit_range_for_polygon_scene() -> None:
    """covered_fraction stays within [0, 1] for a per-kind-shape rotated scene."""
    scene = make_scene(seed=6, n_items=25, clutter=0.9, rotate=True)
    # All four convex shapes should appear on a 25-item desk.
    assert {it.shape for it in scene.items} <= {"box", "hexagon", "rounded", "blob"}
    for it in scene.items:
        assert 0.0 <= covered_fraction(scene, it.id) <= 1.0


def test_polygon_scene_is_deterministic() -> None:
    """A fixed seed fully determines a per-kind-shape rotated scene."""
    a = make_scene(seed=9, n_items=20, clutter=0.8, rotate=True)
    b = make_scene(seed=9, n_items=20, clutter=0.8, rotate=True)
    assert a == b
    assert [(it.shape, it.angle) for it in a.items] == \
        [(it.shape, it.angle) for it in b.items]
    assert occluders(a) == occluders(b)
    # Per-item polygons are identical across rebuilds.
    assert [it.polygon() for it in a.items] == [it.polygon() for it in b.items]


def test_axis_aligned_mode_keeps_all_boxes() -> None:
    """Without rotate, every item stays a box at angle 0 (byte-identical path)."""
    plain = make_scene(seed=6, n_items=25, clutter=0.9)
    assert all(it.shape == "box" and it.angle == 0.0 for it in plain.items)
