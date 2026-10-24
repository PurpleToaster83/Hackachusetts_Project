"""Headless per-tick trace core for the Messy Desk simulation.

Promoted out of :mod:`src.render` so the trace structures and the replay
stepping logic live in the headless simulation core. This module imports NO
pygame and no renderer, so both :mod:`src.render` (for its animation) and the
Streamlit app's logic layer can build traces without pulling in pygame.

:func:`build_trace` mirrors :func:`src.sim.run_episode` exactly: the same
``numpy.random.default_rng(seed)`` draw order for grasp outcomes, the same
``effective_grasp_prob`` model, the same ``push`` primitive, and the same tick
accounting (travel ticks by distance; a failed grasp adds
:data:`~src.robot.FAILED_GRASP_TICKS`), with the same ``10 * items_total``
default budget. It therefore reproduces the sim's outcome for a seed rather
than inventing one; a parity test pins this down. Geometry is in pixels, time
in ticks at 15 Hz.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Allow running within a repo checkout without an installed package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.planner import (  # noqa: E402
    LeastCoveredPlanner,
    NearestPlanner,
    NearestRetryPlanner,
    OcclusionPlanner,
    RandomPlanner,
)
from src.robot import (  # noqa: E402
    FAILED_GRASP_TICKS,
    Grasp,
    Gripper,
    Push,
    effective_grasp_prob,
    push,
    travel_ticks,
)
from src.scene import Scene, make_scene  # noqa: E402

# Name -> (seed) -> planner. Only RandomPlanner consumes the seed; the others
# are deterministic given the scene. Shared by the trace builder and the
# renderer (which re-exports this).
PLANNER_FACTORIES: dict[str, object] = {
    "random": lambda seed: RandomPlanner(seed=seed),
    "nearest": lambda seed: NearestPlanner(),
    "leastcov": lambda seed: LeastCoveredPlanner(),
    "nearretry": lambda seed: NearestRetryPlanner(),
    "occlusion": lambda seed: OcclusionPlanner(),
}


# --------------------------------------------------------------------------- #
# Trace: replay an episode, recording what to draw each tick
# --------------------------------------------------------------------------- #
@dataclass
class ItemState:
    """A drawable snapshot of one item (center, size, kind, z, angle).

    ``angle`` is the footprint rotation in radians (0.0 for axis-aligned items);
    it lets the renderer draw the true rotated footprint. It never affects the
    simulation, which reads geometry from the live Scene, not from the trace.
    """

    id: int
    kind: str
    x: float
    y: float
    w: float
    h: float
    z: float
    angle: float = 0.0


@dataclass
class ContainerState:
    """A drawable snapshot of one container box."""

    kind: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class TickFrame:
    """Everything needed to draw one simulated tick of one planner's run.

    ``gx``/``gy`` is the gripper center in pixels at this tick. ``event`` is
    one of ``""``, ``"grasp"``, ``"fail"``, ``"push"`` and drives the event
    highlight. ``event_item`` is the item the event acts on (or -1). Counters
    are cumulative up to this tick.
    """

    gx: float
    gy: float
    items: list[ItemState]
    event: str
    event_item: int
    tick: int
    sorted_count: int
    failed_count: int
    push_count: int


@dataclass
class Trace:
    """A full per-tick trace for one planner, plus the static containers."""

    planner_name: str
    containers: list[ContainerState]
    frames: list[TickFrame] = field(default_factory=list)


def _item_states(scene: Scene) -> list[ItemState]:
    """Snapshot the scene's items as drawable states, sorted by z (low first)."""
    states = [
        ItemState(
            id=it.id, kind=it.kind, x=it.x, y=it.y, w=it.w, h=it.h, z=it.z,
            angle=getattr(it, "angle", 0.0),
        )
        for it in scene.items
    ]
    states.sort(key=lambda s: s.z)
    return states


def build_trace(
    planner_name: str, seed: int, n_items: int, clutter: float
) -> Trace:
    """Replay one planner on a fresh scene, recording a per-tick trace.

    Reproduces :func:`src.sim.run_episode`: same ``default_rng(seed)`` draw
    order for grasp outcomes, same ``effective_grasp_prob`` model, same push
    primitive, same tick accounting (travel ticks by distance, a failed grasp
    adds :data:`~src.robot.FAILED_GRASP_TICKS`). Each action emits one
    "settled" tick frame after travel; travel itself is interpolated at draw
    time, so the trace stores the start/target gripper positions implicitly via
    consecutive frames.
    """
    scene = make_scene(seed=seed, n_items=n_items, clutter=clutter)
    planner = PLANNER_FACTORIES[planner_name](seed)
    rng = np.random.default_rng(seed)

    containers = [
        ContainerState(kind=c.kind, x=c.x, y=c.y, w=c.w, h=c.h)
        for c in scene.containers
    ]
    trace = Trace(planner_name=planner_name, containers=containers)

    items_total = len(scene.items)
    max_steps = 10 * items_total

    gripper = Gripper(x=0.0, y=0.0, holding=None)
    steps = 0
    sorted_count = 0
    failed_count = 0
    push_count = 0

    # Opening frame: nothing happened yet.
    trace.frames.append(
        TickFrame(
            gx=gripper.x,
            gy=gripper.y,
            items=_item_states(scene),
            event="",
            event_item=-1,
            tick=0,
            sorted_count=0,
            failed_count=0,
            push_count=0,
        )
    )

    while scene.items and steps < max_steps:
        action = planner.next_action(scene, gripper)
        if action is None:
            break
        item = scene.item_by_id(action.item_id)

        distance = math.hypot(item.x - gripper.x, item.y - gripper.y)
        steps += travel_ticks(distance)
        gripper.x, gripper.y = item.x, item.y

        event = ""
        event_item = action.item_id
        if isinstance(action, Grasp):
            prob = effective_grasp_prob(scene, action.item_id)
            if rng.random() < prob:
                scene.items = [
                    it for it in scene.items if it.id != action.item_id
                ]
                sorted_count += 1
                event = "grasp"
            else:
                failed_count += 1
                steps += FAILED_GRASP_TICKS
                event = "fail"
        elif isinstance(action, Push):
            push(scene, action.item_id, action.dx, action.dy)
            push_count += 1
            gripper.x, gripper.y = item.x, item.y
            event = "push"

        trace.frames.append(
            TickFrame(
                gx=gripper.x,
                gy=gripper.y,
                items=_item_states(scene),
                event=event,
                event_item=event_item,
                tick=steps,
                sorted_count=sorted_count,
                failed_count=failed_count,
                push_count=push_count,
            )
        )

    return trace
