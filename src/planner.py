"""Planners for the Messy Desk simulation.

A planner decides the next action: grasp an item, or push one in a direction.
Stateful planners carry their own seeded ``numpy.random.Generator`` so choices
are reproducible and independent of the scene's generator. Headless: no pygame.
"""

from __future__ import annotations

import math

import numpy as np

from src.robot import (
    CLEARANCE_MARGIN,
    CLEARANCE_PENALTY,
    FAILED_GRASP_TICKS,
    Grasp,
    Gripper,
    Push,
    grasp_success_prob,
    travel_ticks,
)
from src.scene import (
    DESK_H,
    DESK_W,
    Item,
    Scene,
    covered_fraction,
    crowding_neighbors,
    occluders,
)

# How far a random push shoves an item, in pixels.
PUSH_DISTANCE: float = 60.0

# Fraction of actions that are pushes rather than grasps.
PUSH_PROB: float = 0.2

# Grasp attempts on one item before the occlusion planner clears its coverer.
RETRY_BEFORE_PUSH: int = 2

# NearestRetryPlanner: attempts before an item is skipped, and for how long.
RETRY_SKIP_AFTER: int = 2
RETRY_SKIP_FOR: int = 5


def _distance(gripper: Gripper, item: Item) -> float:
    """Euclidean distance from the gripper to an item center, in pixels."""
    return math.hypot(item.x - gripper.x, item.y - gripper.y)


def _nearest(items: list[Item], gripper: Gripper) -> Item:
    """Return the item closest to the gripper, breaking ties by id."""
    return min(items, key=lambda it: (_distance(gripper, it), it.id))


def _expected_fail_ticks(prob: float) -> float:
    """Expected wasted ticks from failed grasps before one success.

    With success probability ``prob`` the number of failures before a success
    is geometric with mean ``(1 - prob) / prob``; each failure costs
    :data:`~src.robot.FAILED_GRASP_TICKS` ticks.
    """
    prob = max(prob, 1e-6)
    return FAILED_GRASP_TICKS * (1.0 - prob) / prob


class RandomPlanner:
    """Picks a random remaining item and randomly grasps or pushes it."""

    def __init__(self, seed: int = 0) -> None:
        """Seed the planner's own generator so choices are reproducible."""
        self.rng = np.random.default_rng(seed)

    def next_action(self, scene: Scene, gripper: Gripper) -> Grasp | Push | None:
        """Return a random grasp or push, or None if no items remain.

        ``gripper`` is accepted for interface parity but unused: this baseline
        ignores robot state and chooses uniformly at random.
        """
        if not scene.items:
            return None
        idx = int(self.rng.integers(0, len(scene.items)))
        item_id = scene.items[idx].id

        if self.rng.random() < PUSH_PROB:
            # Push in one of the four axis-aligned directions.
            direction = int(self.rng.integers(0, 4))
            dx, dy = (
                (PUSH_DISTANCE, 0.0),
                (-PUSH_DISTANCE, 0.0),
                (0.0, PUSH_DISTANCE),
                (0.0, -PUSH_DISTANCE),
            )[direction]
            return Push(item_id=item_id, dx=dx, dy=dy)
        return Grasp(item_id=item_id)


class NearestPlanner:
    """Always grasps the nearest remaining item, ignoring coverage."""

    def next_action(self, scene: Scene, gripper: Gripper) -> Grasp | None:
        """Grasp the remaining item closest to the gripper, or None if empty."""
        if not scene.items:
            return None
        return Grasp(item_id=_nearest(scene.items, gripper).id)


class LeastCoveredPlanner:
    """Grasps the least-covered item, breaking ties by distance to the gripper.

    A pure coverage-greedy baseline: it always reaches for the item with the
    smallest covered fraction (the easiest grasp by the coverage model),
    ignoring crowding. Ties in coverage go to the nearer item, then lower id.
    """

    def next_action(self, scene: Scene, gripper: Gripper) -> Grasp | None:
        """Grasp the least-covered remaining item, or None if the desk is clear."""
        if not scene.items:
            return None
        target = min(
            scene.items,
            key=lambda it: (
                covered_fraction(scene, it.id),
                _distance(gripper, it),
                it.id,
            ),
        )
        return Grasp(item_id=target.id)


class NearestRetryPlanner:
    """Nearest-first, but gives up on an item that keeps failing.

    Grasps the nearest remaining item. After :data:`RETRY_SKIP_AFTER` failed
    grasps on the same item, it skips that item for the next
    :data:`RETRY_SKIP_FOR` actions so the gripper makes progress elsewhere
    before returning to it. Coverage is otherwise ignored.
    """

    def __init__(self) -> None:
        """Track per-item failures and how long each stays skipped."""
        self._fails: dict[int, int] = {}
        self._skip_until: dict[int, int] = {}
        self._action_index: int = 0
        self._last_grasp_id: int | None = None

    def _record_prior_grasp(self, scene: Scene) -> None:
        """Bump the failure count if the last grasped item is still present."""
        if self._last_grasp_id is None:
            return
        present = any(it.id == self._last_grasp_id for it in scene.items)
        if present:
            fails = self._fails.get(self._last_grasp_id, 0) + 1
            self._fails[self._last_grasp_id] = fails
            if fails >= RETRY_SKIP_AFTER:
                self._skip_until[self._last_grasp_id] = (
                    self._action_index + RETRY_SKIP_FOR
                )
                self._fails[self._last_grasp_id] = 0
        else:
            self._fails.pop(self._last_grasp_id, None)
            self._skip_until.pop(self._last_grasp_id, None)
        self._last_grasp_id = None

    def next_action(self, scene: Scene, gripper: Gripper) -> Grasp | None:
        """Grasp the nearest non-skipped item, or the nearest if all skipped."""
        self._record_prior_grasp(scene)
        if not scene.items:
            return None
        self._action_index += 1

        available = [
            it
            for it in scene.items
            if self._skip_until.get(it.id, -1) < self._action_index
        ]
        # If everything is currently skipped, fall back to the whole set so the
        # planner never stalls.
        candidates = available if available else scene.items
        target = _nearest(candidates, gripper)
        self._last_grasp_id = target.id
        return Grasp(item_id=target.id)


class OcclusionPlanner:
    """Grasps exposed, uncrowded items first; singulates when it pays off.

    Priority order each call:

    1. If the item last grasped failed twice and still has a coverer, push that
       coverer away (the stubborn-retry case).
    2. Among items nothing covers, pick the nearest. If that target is crowded
       (a neighbor within :data:`~src.robot.CLEARANCE_MARGIN` px), decide
       whether to singulate using an expected-value rule (below). Otherwise
       grasp it.
    3. If every remaining item is covered, push the least-covered item's top
       coverer away.

    Singulation rule (fixed, not tuned to results): a crowded target has grasp
    probability ``p_crowded`` (coverage model x CLEARANCE_PENALTY); clearing the
    crowd would restore ``p_clear`` (coverage model only). The expected ticks
    wasted on failed grasps before a success is ``FAILED_GRASP_TICKS *
    (1 - p) / p``. Clearing saves the difference of those expectations. The
    push itself costs travelling to the neighbour plus one tick. Push only when
    the expected saving strictly exceeds that push cost; otherwise just grasp.
    """

    def __init__(self) -> None:
        """Track per-item grasp attempts so repeated failures trigger a push."""
        self._attempts: dict[int, int] = {}
        self._last_grasp_id: int | None = None

    def _record_prior_grasp(self, scene: Scene) -> None:
        """Count a retry when the last grasped item is still on the desk.

        A failed grasp leaves the item in place, so if the item we grasped last
        turn is still present, that grasp failed and we bump its attempt count.
        A sorted item is gone, so its counter is retired.
        """
        if self._last_grasp_id is None:
            return
        present = any(it.id == self._last_grasp_id for it in scene.items)
        if present:
            self._attempts[self._last_grasp_id] = (
                self._attempts.get(self._last_grasp_id, 0) + 1
            )
        else:
            self._attempts.pop(self._last_grasp_id, None)
        self._last_grasp_id = None

    def _push_away(self, pivot: Item, mover: Item, magnitude: float) -> Push:
        """Push ``mover`` directly away from ``pivot`` by ``magnitude`` pixels."""
        dx = mover.x - pivot.x
        dy = mover.y - pivot.y
        norm = math.hypot(dx, dy)
        ux, uy = (1.0, 0.0) if norm == 0.0 else (dx / norm, dy / norm)
        return Push(item_id=mover.id, dx=ux * magnitude, dy=uy * magnitude)

    def _push_coverer(self, scene: Scene, item: Item, coverer_ids: list[int]) -> Push:
        """Shove ``item``'s top coverer (highest z) away by about one width."""
        coverers = [scene.item_by_id(cid) for cid in coverer_ids]
        top = max(coverers, key=lambda it: (it.z, it.id))
        return self._push_away(item, top, top.w)

    def _singulation_pays(
        self, scene: Scene, target: Item, gripper: Gripper
    ) -> bool:
        """True when clearing ``target``'s crowd saves more ticks than it costs.

        Compares expected failed-grasp ticks crowded vs cleared against the
        push cost (travel to the neighbour plus one tick). Fixed rule; see the
        class docstring.
        """
        p_clear = grasp_success_prob(covered_fraction(scene, target.id))
        p_crowded = max(p_clear * CLEARANCE_PENALTY, 0.0)
        saving = _expected_fail_ticks(p_crowded) - _expected_fail_ticks(p_clear)

        neighbor_ids = crowding_neighbors(scene, target.id, CLEARANCE_MARGIN)
        neighbor = _nearest(
            [scene.item_by_id(nid) for nid in neighbor_ids], gripper
        )
        push_cost = travel_ticks(_distance(gripper, neighbor)) + 1
        return saving > push_cost

    def _declutter_push(self, scene: Scene, target: Item, gripper: Gripper) -> Push:
        """Shove ``target``'s nearest crowding neighbour clear of the margin.

        Moves the neighbour away from the target by its width plus the clearance
        margin, enough to leave the penalty zone.
        """
        neighbor_ids = crowding_neighbors(scene, target.id, CLEARANCE_MARGIN)
        neighbor = _nearest(
            [scene.item_by_id(nid) for nid in neighbor_ids], gripper
        )
        return self._push_away(target, neighbor, neighbor.w + CLEARANCE_MARGIN)

    def next_action(self, scene: Scene, gripper: Gripper) -> Grasp | Push | None:
        """Grasp the nearest exposed item, singulating when it pays off."""
        self._record_prior_grasp(scene)
        if not scene.items:
            return None

        covered = occluders(scene)  # item id -> coverer ids, covered items only

        # 1. Stubborn retry: a twice-failed item with a coverer gets cleared.
        for item in scene.items:
            if self._attempts.get(item.id, 0) >= RETRY_BEFORE_PUSH:
                coverer_ids = covered.get(item.id)
                if coverer_ids:
                    self._attempts[item.id] = 0
                    return self._push_coverer(scene, item, coverer_ids)

        uncovered = [it for it in scene.items if it.id not in covered]
        if uncovered:
            target = _nearest(uncovered, gripper)
            # 2. Crowded target: singulate only when the expected saving wins.
            if crowding_neighbors(scene, target.id, CLEARANCE_MARGIN) and (
                self._singulation_pays(scene, target, gripper)
            ):
                return self._declutter_push(scene, target, gripper)
            self._last_grasp_id = target.id
            return Grasp(item_id=target.id)

        # 3. Everything is covered: singulate the least-covered item's coverer.
        least_covered_id = min(
            covered, key=lambda iid: (len(covered[iid]), iid)
        )
        item = scene.item_by_id(least_covered_id)
        return self._push_coverer(scene, item, covered[least_covered_id])


# Diagonal of the desk, in pixels; used to normalize the distance feature into
# roughly [0, 1] so the learned weights are on a comparable scale.
_DESK_DIAG: float = math.hypot(DESK_W, DESK_H)

# Number of features the LearnedPlanner scores. Keep in sync with
# ``item_features`` and the weight-search dimension.
LEARNED_N_FEATURES: int = 3

# The optimized LearnedPlanner weights (w0=coverage, w1=distance, w2=crowding).
# SINGLE SOURCE OF TRUTH: every path that wants the trained policy -- the app
# planner registry and the LearnedPlanner default -- reads this one constant so
# the batch runner and the replay/render path can never diverge. Produced by
# the reproducible search in ``src.learn`` (search_seed=0, 15 items, clutter
# 0.9, 20 train / 20 held-out seeds, 200 iters).
DEFAULT_LEARNED_WEIGHTS: tuple[float, ...] = (0.1175, 0.0888, -0.7255)


def item_features(scene: Scene, gripper: Gripper, item: Item) -> tuple[float, ...]:
    """Feature vector for one candidate item, from existing engine helpers.

    All features reuse the engine's own geometry -- no new physics:

    1. ``covered_fraction`` in [0, 1] (how occluded the item is).
    2. normalized gripper distance in [0, ~1] (``_distance`` / desk diagonal).
    3. normalized crowding in [0, 1] (crowding-neighbour count / remaining items).

    Lower values are "easier/closer/clearer"; the learned weights decide how to
    trade them off.
    """
    cov = covered_fraction(scene, item.id)
    dist = _distance(gripper, item) / _DESK_DIAG
    n_other = max(len(scene.items) - 1, 1)
    crowd = len(crowding_neighbors(scene, item.id, CLEARANCE_MARGIN)) / n_other
    return (cov, dist, crowd)


class LearnedPlanner:
    """Grasps the item with the best learned weighted-feature score.

    Scores every remaining item as ``dot(weights, item_features(...))`` and
    grasps the LOWEST-scoring item (features are "cost-like": lower coverage,
    distance, and crowding are better), breaking ties by item id. Grasp-only,
    like :class:`NearestPlanner`/:class:`LeastCoveredPlanner`; it adds no new
    physics and reuses :func:`item_features`.

    The weights are a fixed policy supplied at construction (e.g. found by the
    weight search in :mod:`src.learn`). With fixed weights the planner is fully
    deterministic -- it carries no rng and no mutable state -- so a given scene
    and gripper always yield the same choice.
    """

    def __init__(self, weights: tuple[float, ...] | None = None) -> None:
        """Store the fixed feature weights.

        Defaults to :data:`DEFAULT_LEARNED_WEIGHTS` (the optimized policy) when
        ``weights`` is None, so an unparameterized LearnedPlanner runs the
        trained policy rather than an arbitrary all-ones one.
        """
        if weights is None:
            weights = DEFAULT_LEARNED_WEIGHTS
        if len(weights) != LEARNED_N_FEATURES:
            raise ValueError(
                f"expected {LEARNED_N_FEATURES} weights, got {len(weights)}"
            )
        self.weights: tuple[float, ...] = tuple(float(w) for w in weights)

    def _score(self, scene: Scene, gripper: Gripper, item: Item) -> float:
        """Weighted-feature score for one item (lower is preferred)."""
        feats = item_features(scene, gripper, item)
        return sum(w * f for w, f in zip(self.weights, feats))

    def next_action(self, scene: Scene, gripper: Gripper) -> Grasp | None:
        """Grasp the lowest-scoring remaining item, or None if the desk is clear."""
        if not scene.items:
            return None
        target = min(
            scene.items,
            key=lambda it: (self._score(scene, gripper, it), it.id),
        )
        return Grasp(item_id=target.id)
