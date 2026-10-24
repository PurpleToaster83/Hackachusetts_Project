"""Scene model for the Messy Desk simulation.

Defines the desk, the items scattered on it, and the sorting containers.
All geometry is in pixels on a 640 x 480 desk. All randomness comes from a
seeded ``numpy.random.Generator`` so that a given seed is fully reproducible.

The simulation core is headless: this module must never import pygame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Desk dimensions in pixels.
DESK_W: int = 640
DESK_H: int = 480

# Item kinds and the container each kind should be sorted into.
KINDS: tuple[str, ...] = ("pen", "key", "mug", "trash")

# Overlap of an item that must be covered for it to count as occluded.
OCCLUSION_THRESHOLD: float = 0.25


@dataclass
class Item:
    """A desk item to be sorted.

    Coordinates are the center of the item in pixels. ``w`` and ``h`` are the
    item's size in pixels and ``z`` is a stacking height (larger is on top).
    """

    id: int
    kind: str
    x: float  # center x, pixels
    y: float  # center y, pixels
    w: float  # width, pixels
    h: float  # height, pixels
    z: float  # stacking height; higher z sits on top
    angle: float = 0.0  # footprint rotation in radians; 0.0 == axis-aligned

    @property
    def area(self) -> float:
        """Item footprint area in square pixels (rotation preserves area)."""
        return self.w * self.h

    def polygon(self) -> list[tuple[float, float]]:
        """Footprint corners (CCW) in pixels, rotated by ``angle`` about center.

        A rotated rectangle. With ``angle == 0.0`` the corners are exactly the
        axis-aligned box corners, so box-based behavior is unchanged.
        """
        hw, hh = self.w / 2.0, self.h / 2.0
        # Local corners CCW: BL, BR, TR, TL.
        local = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        ca = np.cos(self.angle)
        sa = np.sin(self.angle)
        return [
            (self.x + lx * ca - ly * sa, self.y + lx * sa + ly * ca)
            for lx, ly in local
        ]

    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box (left, top, right, bottom) of the footprint.

        For ``angle == 0.0`` this is the item's own box; for a rotated item it
        is the AABB enclosing the rotated polygon (used as a cheap broad-phase).
        """
        xs = [px for px, _ in self.polygon()]
        ys = [py for _, py in self.polygon()]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class Container:
    """A drop target for one kind of item. Position is its center in pixels."""

    kind: str
    x: float  # center x, pixels
    y: float  # center y, pixels
    w: float  # width, pixels
    h: float  # height, pixels


@dataclass
class Scene:
    """A desk populated with items and sorting containers."""

    width: int
    height: int
    items: list[Item]
    containers: list[Container]
    clutter: float
    # The generator carries no value identity, so exclude it from equality and
    # repr. Two scenes built from the same seed then compare equal by value.
    rng: np.random.Generator = field(repr=False, compare=False)

    def item_by_id(self, item_id: int) -> Item:
        """Return the item with the given id."""
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)


def _shoelace_area(poly: list[tuple[float, float]]) -> float:
    """Unsigned area of a simple polygon (pixels^2) via the shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _clip_convex(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Clip CCW convex ``subject`` by CCW convex ``clip`` (Sutherland-Hodgman).

    Returns the vertices of the intersection polygon (possibly empty). Both
    inputs must be convex and counter-clockwise.
    """
    output = list(subject)
    n = len(clip)
    for i in range(n):
        if not output:
            break
        cx1, cy1 = clip[i]
        cx2, cy2 = clip[(i + 1) % n]
        ex, ey = cx2 - cx1, cy2 - cy1  # clip edge direction

        def inside(p: tuple[float, float]) -> float:
            # >= 0 means left of (or on) the CCW edge, i.e. inside.
            return ex * (p[1] - cy1) - ey * (p[0] - cx1)

        inp = output
        output = []
        m = len(inp)
        for j in range(m):
            cur = inp[j]
            prv = inp[j - 1]
            d_cur = inside(cur)
            d_prv = inside(prv)
            if d_cur >= 0:
                if d_prv < 0:
                    output.append(_edge_intersect(prv, cur, (cx1, cy1), (cx2, cy2)))
                output.append(cur)
            elif d_prv >= 0:
                output.append(_edge_intersect(prv, cur, (cx1, cy1), (cx2, cy2)))
    return output


def _edge_intersect(p1, p2, q1, q2) -> tuple[float, float]:
    """Intersection point of line p1->p2 with the infinite line q1->q2."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0.0:
        return p2  # parallel; degenerate, fall back to the segment end
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_num / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _overlap_area(a: Item, b: Item) -> float:
    """Area of the overlap between two item footprints, in square pixels.

    Works for rotated rectangles (and any convex footprint) via convex polygon
    clipping. With both items axis-aligned (``angle == 0.0``) it equals the AABB
    overlap, so box-based behavior is unchanged. A quick AABB reject skips the
    clip when the bounding boxes do not touch.
    """
    al, at, ar, ab = a.bounds()
    bl, bt, br, bb = b.bounds()
    if min(ar, br) <= max(al, bl) or min(ab, bb) <= max(at, bt):
        return 0.0
    inter = _clip_convex(a.polygon(), b.polygon())
    return _shoelace_area(inter)


def make_scene(
    seed: int, n_items: int, clutter: float, rotate: bool = False
) -> Scene:
    """Build a randomized cluttered desk.

    ``seed`` fully determines the scene. ``n_items`` is the number of items to
    scatter. ``clutter`` in [0, 1] pulls items toward the desk center so higher
    values produce more overlap. All sizes and positions are in pixels.

    ``rotate`` is an EXPERIMENTAL opt-in (spike): when True each item is given a
    random footprint ``angle`` in radians, turning the boxes into rotated
    rectangles. It defaults to False, and when False NO extra random draw is
    made, so the rng draw order -- and therefore every existing scene and
    downstream simulation result -- is byte-for-byte identical to before.
    """
    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    clutter = float(np.clip(clutter, 0.0, 1.0))

    rng = np.random.default_rng(seed)

    # Containers: one per kind, laid out along the right margin of the desk.
    cont_w, cont_h = 70.0, 90.0
    margin = 15.0
    containers: list[Container] = []
    for i, kind in enumerate(KINDS):
        cx = DESK_W - cont_w / 2.0 - margin
        slot = (DESK_H - margin) / len(KINDS)
        cy = margin + slot * i + slot / 2.0
        containers.append(Container(kind=kind, x=cx, y=cy, w=cont_w, h=cont_h))

    # Item placement region: keep clear of the container column on the right.
    region_w = DESK_W - cont_w - 3 * margin
    cx_desk = region_w / 2.0
    cy_desk = DESK_H / 2.0

    items: list[Item] = []
    for i in range(n_items):
        kind = KINDS[int(rng.integers(0, len(KINDS)))]
        w = float(rng.uniform(24.0, 48.0))
        h = float(rng.uniform(24.0, 48.0))

        # Spread of placement shrinks as clutter rises, concentrating overlap.
        spread = 1.0 - 0.7 * clutter
        x = float(cx_desk + rng.uniform(-1.0, 1.0) * (region_w / 2.0 - w / 2.0) * spread)
        y = float(cy_desk + rng.uniform(-1.0, 1.0) * (DESK_H / 2.0 - h / 2.0) * spread)

        # Clamp centers so the full item stays on the placement region.
        x = float(np.clip(x, w / 2.0, region_w - w / 2.0))
        y = float(np.clip(y, h / 2.0, DESK_H - h / 2.0))

        # Stacking order: later items tend to sit on top. Jitter avoids ties.
        z = float(i + rng.uniform(0.0, 0.5))

        # EXPERIMENTAL: only draw an angle when rotation is requested, so the
        # default (rotate=False) draw order is unchanged from the box model.
        angle = float(rng.uniform(0.0, np.pi)) if rotate else 0.0
        items.append(Item(id=i, kind=kind, x=x, y=y, w=w, h=h, z=z, angle=angle))

    return Scene(
        width=DESK_W,
        height=DESK_H,
        items=items,
        containers=containers,
        clutter=clutter,
        rng=rng,
    )


def covered_fraction(scene: Scene, item_id: int) -> float:
    """Fraction of ``item_id``'s area covered by items stacked above it.

    Only items with higher ``z`` count as covering. The returned value is the
    summed overlap fraction, capped at 1.0.
    """
    target = scene.item_by_id(item_id)
    area = target.area
    if area <= 0.0:
        return 0.0
    covered = 0.0
    for other in scene.items:
        if other.id == item_id:
            continue
        if other.z <= target.z:
            continue
        covered += _overlap_area(other, target)
    return float(min(covered / area, 1.0))


def _aabb_gap(a: Item, b: Item) -> float:
    """Smallest pixel gap between two items' axis-aligned bounding boxes.

    Returns 0.0 when the boxes touch or overlap, otherwise the larger of the
    per-axis gaps (so a diagonal separation is measured by whichever axis is
    clear). This is the original box-based crowding measure.
    """
    al, at, ar, ab = a.bounds()
    bl, bt, br, bb = b.bounds()
    gap_x = max(bl - ar, al - br, 0.0)
    gap_y = max(bt - ab, at - bb, 0.0)
    return max(gap_x, gap_y)


def _seg_seg_distance(p1, p2, q1, q2) -> float:
    """Shortest distance between two 2D segments (0.0 if they intersect)."""
    import math as _math

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1]

    def sub(u, v):
        return (u[0] - v[0], u[1] - v[1])

    def clamp01(x):
        return max(0.0, min(1.0, x))

    def pt_seg(pt, a, b):
        ab = sub(b, a)
        denom = dot(ab, ab)
        if denom == 0.0:
            return _math.hypot(pt[0] - a[0], pt[1] - a[1])
        tt = clamp01(dot(sub(pt, a), ab) / denom)
        proj = (a[0] + tt * ab[0], a[1] + tt * ab[1])
        return _math.hypot(pt[0] - proj[0], pt[1] - proj[1])

    return min(
        pt_seg(p1, q1, q2), pt_seg(p2, q1, q2),
        pt_seg(q1, p1, p2), pt_seg(q2, p1, p2),
    )


def _poly_gap(a: Item, b: Item) -> float:
    """Smallest pixel gap between two convex footprints (0.0 if overlapping).

    Used when either item is rotated. When they overlap the gap is 0.0;
    otherwise it is the minimum distance between their edges.
    """
    if _overlap_area(a, b) > 0.0:
        return 0.0
    pa = a.polygon()
    pb = b.polygon()
    na, nb = len(pa), len(pb)
    best = float("inf")
    for i in range(na):
        e1a, e1b = pa[i], pa[(i + 1) % na]
        for j in range(nb):
            e2a, e2b = pb[j], pb[(j + 1) % nb]
            d = _seg_seg_distance(e1a, e1b, e2a, e2b)
            if d < best:
                best = d
    return best


def _edge_gap(a: Item, b: Item) -> float:
    """Smallest pixel gap between two item footprints.

    For axis-aligned items (both ``angle == 0.0``) this uses the original
    AABB gap, so box-based crowding is unchanged. If either item is rotated it
    falls back to the convex polygon gap.
    """
    if a.angle == 0.0 and b.angle == 0.0:
        return _aabb_gap(a, b)
    return _poly_gap(a, b)


def crowding_neighbors(scene: Scene, item_id: int, margin: float) -> list[int]:
    """Ids of items whose box comes within ``margin`` pixels of ``item_id``.

    An item crowds ``item_id`` when the gap between their bounding boxes is less
    than ``margin`` pixels (touching or overlapping counts). The item itself is
    never listed. Used to decide the grasp clearance penalty.
    """
    target = scene.item_by_id(item_id)
    crowding: list[int] = []
    for other in scene.items:
        if other.id == item_id:
            continue
        if _edge_gap(target, other) < margin:
            crowding.append(other.id)
    return sorted(crowding)


def is_crowded(scene: Scene, item_id: int, margin: float) -> bool:
    """True when any other item sits within ``margin`` pixels of ``item_id``."""
    return bool(crowding_neighbors(scene, item_id, margin))


def occluders(scene: Scene) -> dict[int, list[int]]:
    """Map each item id to the ids of items that cover more than 25% of it.

    An item A occludes B when ``A.z > B.z`` and the overlap area divided by B's
    area exceeds :data:`OCCLUSION_THRESHOLD`. Items that nothing covers are
    omitted entirely, so ``len(occluders(scene))`` counts covered items.
    """
    result: dict[int, list[int]] = {}
    for target in scene.items:
        area = target.area
        if area <= 0.0:
            continue
        covering: list[int] = []
        for other in scene.items:
            if other.id == target.id:
                continue
            if other.z <= target.z:
                continue
            if _overlap_area(other, target) / area > OCCLUSION_THRESHOLD:
                covering.append(other.id)
        if covering:
            result[target.id] = sorted(covering)
    return result
