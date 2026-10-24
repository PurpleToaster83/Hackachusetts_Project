"""Episode runner for the Messy Desk simulation.

Runs a planner against a scene one grasp attempt per step. Headless: no pygame,
all randomness from a seeded ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robot import grasp_success_prob
from src.scene import Scene, covered_fraction


@dataclass
class EpisodeResult:
    """Outcome of one episode. All counts are in whole steps/items."""

    items_sorted: int  # items successfully grasped and removed
    items_total: int  # items present at the start
    steps: int  # grasp attempts taken (ticks)
    failed_grasps: int  # attempts that did not remove an item
    pushes: int  # push primitives used (unused by current planners)


def run_episode(scene: Scene, planner, logger=None, seed: int = 0) -> EpisodeResult:
    """Run ``planner`` against ``scene`` until sorted or the step budget runs out.

    Each step the planner returns an item id to attempt. The grasp succeeds with
    probability ``grasp_success_prob(covered_fraction)`` for that item, drawn
    from a seeded generator. On success the item is removed and counts as
    sorted; on failure ``failed_grasps`` increments. The run stops when every
    item is sorted or ``steps`` reaches ``10 * items_total``.

    ``scene`` is mutated: sorted items are removed from ``scene.items``.
    ``logger`` is accepted for interface stability and is currently unused.
    """
    rng = np.random.default_rng(seed)

    items_total = len(scene.items)
    max_steps = 10 * items_total

    items_sorted = 0
    steps = 0
    failed_grasps = 0
    pushes = 0

    while scene.items and steps < max_steps:
        item_id = planner.next_action(scene)
        if item_id is None:
            break
        steps += 1

        prob = grasp_success_prob(covered_fraction(scene, item_id))
        if rng.random() < prob:
            scene.items = [it for it in scene.items if it.id != item_id]
            items_sorted += 1
        else:
            failed_grasps += 1

    return EpisodeResult(
        items_sorted=items_sorted,
        items_total=items_total,
        steps=steps,
        failed_grasps=failed_grasps,
        pushes=pushes,
    )
