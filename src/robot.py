"""Robot model for the Messy Desk simulation.

Pure, headless helpers for the gripper, its primitives, and the grasp model.
No pygame, no global random state. Geometry is in pixels; time is in ticks at
15 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.scene import DESK_H, DESK_W, Scene, covered_fraction, is_crowded

# Floor on grasp success so a fully covered item is still occasionally liftable.
MIN_GRASP_PROB: float = 0.05
MAX_GRASP_PROB: float = 1.0

# Timing and motion limits for the kinematic gripper.
TICK_HZ: int = 15  # simulation ticks per second
MAX_SPEED: float = 120.0  # pixels the gripper may travel in one tick

# Extra ticks charged when a grasp attempt fails (fumble and reset).
FAILED_GRASP_TICKS: int = 3

# Grasping an item with another item crowding its edges is harder: if any other
# item's box is within CLEARANCE_MARGIN pixels, the grasp probability is scaled
# by CLEARANCE_PENALTY. Singulating (pushing the crowd away) removes this.
CLEARANCE_MARGIN: float = 15.0  # pixels
CLEARANCE_PENALTY: float = 0.7  # multiplier applied when crowded


@dataclass
class Gripper:
    """The kinematic gripper. Position is in pixels; ``holding`` is an item id.

    ``holding`` is ``None`` when the gripper is empty, otherwise the id of the
    item currently held.
    """

    x: float  # pixels
    y: float  # pixels
    holding: int | None = None


@dataclass
class Grasp:
    """An action: attempt to grasp the item with ``item_id``."""

    item_id: int


@dataclass
class Push:
    """An action: shove ``item_id`` by ``(dx, dy)`` pixels."""

    item_id: int
    dx: float  # pixels
    dy: float  # pixels


def grasp_success_prob(covered_fraction: float) -> float:
    """Probability a grasp succeeds given how covered the item is.

    ``covered_fraction`` is the fraction of the item's area hidden by items
    stacked above it (0.0 fully exposed, 1.0 fully covered). The base model is
    ``1 - 0.8 * covered_fraction``, clamped to [0.05, 1.0].
    """
    prob = 1.0 - 0.8 * covered_fraction
    return float(min(max(prob, MIN_GRASP_PROB), MAX_GRASP_PROB))


def effective_grasp_prob(scene: Scene, item_id: int) -> float:
    """Grasp probability for ``item_id`` including the clearance penalty.

    Combines the coverage model with the crowding penalty: the base probability
    from :func:`grasp_success_prob` on the item's covered fraction, multiplied
    by :data:`CLEARANCE_PENALTY` when another item sits within
    :data:`CLEARANCE_MARGIN` pixels of its edges. The result is floored at
    :data:`MIN_GRASP_PROB`.
    """
    base = grasp_success_prob(covered_fraction(scene, item_id))
    if is_crowded(scene, item_id, CLEARANCE_MARGIN):
        base *= CLEARANCE_PENALTY
    return float(max(base, MIN_GRASP_PROB))


def travel_ticks(distance: float) -> int:
    """Ticks to move ``distance`` pixels at the capped speed, at least 1.

    The gripper moves at most :data:`MAX_SPEED` pixels per tick, so a move of
    ``distance`` pixels costs ``ceil(distance / MAX_SPEED)`` ticks. Any move
    (and any in-place action) costs at least one tick.
    """
    if distance <= 0.0:
        return 1
    return max(1, int(np.ceil(distance / MAX_SPEED)))


def push(scene: Scene, item_id: int, dx: float, dy: float) -> None:
    """Shove an item by ``(dx, dy)`` pixels, clamped to the desk.

    Moving the item changes its overlap with everything it was covering, so the
    covered fraction of items beneath it drops (to zero once it clears them).
    The scene is mutated in place.
    """
    item = scene.item_by_id(item_id)
    item.x = float(np.clip(item.x + dx, item.w / 2.0, DESK_W - item.w / 2.0))
    item.y = float(np.clip(item.y + dy, item.h / 2.0, DESK_H - item.h / 2.0))
