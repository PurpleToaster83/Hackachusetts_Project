"""Episode runner for the Messy Desk simulation.

Runs a planner against a scene one action per iteration. Each action advances
the gripper and the clock: steps count ticks at 15 Hz, scaled by how far the
gripper travels. Headless: no pygame, all randomness from a seeded
``numpy.random.Generator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.robot import (
    FAILED_GRASP_TICKS,
    Grasp,
    Gripper,
    Push,
    effective_grasp_prob,
    push,
    travel_ticks,
)
from src.scene import Scene


@dataclass
class EpisodeResult:
    """Outcome of one episode. Counts are whole items; ``steps`` is in ticks."""

    items_sorted: int  # items successfully grasped and removed
    items_total: int  # items present at the start
    steps: int  # ticks elapsed (travel + action time at 15 Hz)
    failed_grasps: int  # grasp attempts that did not remove an item
    pushes: int  # push primitives used


def _travel_to(gripper: Gripper, x: float, y: float) -> int:
    """Move the gripper to (x, y) and return the ticks the move cost."""
    distance = math.hypot(x - gripper.x, y - gripper.y)
    ticks = travel_ticks(distance)
    gripper.x = x
    gripper.y = y
    return ticks


def run_episode(
    scene: Scene,
    planner,
    logger=None,
    seed: int = 0,
    max_ticks: int | None = None,
) -> EpisodeResult:
    """Run ``planner`` against ``scene`` until sorted or the step budget runs out.

    Each iteration the planner returns an action: a :class:`~src.robot.Grasp`
    of an item id or a :class:`~src.robot.Push` of an item id with a direction.
    The gripper first travels to the target item; that travel costs ticks
    proportional to distance (capped speed per tick, 15 Hz).

    A grasp succeeds with probability
    ``effective_grasp_prob(scene, item_id)`` for the target (coverage model plus
    the crowding clearance penalty), drawn from a seeded generator. On success
    the item is removed and counts as sorted; a failed grasp costs
    :data:`~src.robot.FAILED_GRASP_TICKS` extra ticks and increments
    ``failed_grasps``. A push moves the item (reducing what it covered) and
    increments ``pushes``.

    The run stops when every item is sorted or ``steps`` reaches the tick
    budget. The budget is ``max_ticks`` when given, otherwise
    ``10 * items_total`` ticks. ``scene`` and the gripper are mutated in place.

    When ``logger`` is given (a :class:`~src.log.EpisodeLogger`), one frame is
    recorded per action: the gripper state after travelling to the target, the
    action type and target, and the reward (1.0 for a successful grasp, else
    0.0). The logger stamps ``timestamp`` from the current tick at 15 Hz.
    """
    rng = np.random.default_rng(seed)

    items_total = len(scene.items)
    max_steps = max_ticks if max_ticks is not None else 10 * items_total

    gripper = Gripper(x=0.0, y=0.0, holding=None)

    items_sorted = 0
    steps = 0
    failed_grasps = 0
    pushes = 0

    while scene.items and steps < max_steps:
        action = planner.next_action(scene, gripper)
        if action is None:
            break

        item = scene.item_by_id(action.item_id)
        # Travel to the item first; travel costs ticks by distance.
        steps += _travel_to(gripper, item.x, item.y)

        if isinstance(action, Grasp):
            prob = effective_grasp_prob(scene, action.item_id)
            success = rng.random() < prob
            if success:
                scene.items = [
                    it for it in scene.items if it.id != action.item_id
                ]
                items_sorted += 1
            else:
                failed_grasps += 1
                steps += FAILED_GRASP_TICKS  # a failed grasp burns extra ticks
            if logger is not None:
                logger.log_frame(
                    tick=steps,
                    state_x=gripper.x,
                    state_y=gripper.y,
                    holding=gripper.holding,
                    action_type="grasp",
                    action_x=item.x,
                    action_y=item.y,
                    reward=1.0 if success else 0.0,
                )
        elif isinstance(action, Push):
            push(scene, action.item_id, action.dx, action.dy)
            pushes += 1
            if logger is not None:
                logger.log_frame(
                    tick=steps,
                    state_x=gripper.x,
                    state_y=gripper.y,
                    holding=gripper.holding,
                    action_type="push",
                    action_x=action.dx,
                    action_y=action.dy,
                    reward=0.0,
                )
            # Follow the item to where it was shoved.
            gripper.x = item.x
            gripper.y = item.y
        else:  # pragma: no cover - guards against a bad planner
            raise TypeError(f"unknown action: {action!r}")

    return EpisodeResult(
        items_sorted=items_sorted,
        items_total=items_total,
        steps=steps,
        failed_grasps=failed_grasps,
        pushes=pushes,
    )
