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

    @property
    def area(self) -> float:
        """Item footprint area in square pixels."""
        return self.w * self.h

    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds as (left, top, right, bottom) in pixels."""
        return (
            self.x - self.w / 2.0,
            self.y - self.h / 2.0,
            self.x + self.w / 2.0,
            self.y + self.h / 2.0,
        )


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


def _overlap_area(a: Item, b: Item) -> float:
    """Area of the axis-aligned overlap between two items, in square pixels."""
    al, at, ar, ab = a.bounds()
    bl, bt, br, bb = b.bounds()
    dx = min(ar, br) - max(al, bl)
    dy = min(ab, bb) - max(at, bt)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


def make_scene(seed: int, n_items: int, clutter: float) -> Scene:
    """Build a randomized cluttered desk.

    ``seed`` fully determines the scene. ``n_items`` is the number of items to
    scatter. ``clutter`` in [0, 1] pulls items toward the desk center so higher
    values produce more overlap. All sizes and positions are in pixels.
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
        items.append(Item(id=i, kind=kind, x=x, y=y, w=w, h=h, z=z))

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


def _edge_gap(a: Item, b: Item) -> float:
    """Smallest pixel gap between two items' bounding boxes.

    Returns 0.0 when the boxes touch or overlap, otherwise the shortest
    distance between their edges (the larger of the per-axis gaps, so a
    diagonal separation is measured by whichever axis is clear).
    """
    al, at, ar, ab = a.bounds()
    bl, bt, br, bb = b.bounds()
    gap_x = max(bl - ar, al - br, 0.0)
    gap_y = max(bt - ab, at - bb, 0.0)
    return max(gap_x, gap_y)


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
