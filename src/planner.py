"""Planners for the Messy Desk simulation.

A planner decides which item to attempt next. Each planner carries its own
seeded ``numpy.random.Generator`` so its choices are reproducible and
independent of the scene's generator. Headless: no pygame.
"""

from __future__ import annotations

import numpy as np

from src.scene import Scene


class RandomPlanner:
    """Picks a uniformly random remaining item each step."""

    def __init__(self, seed: int = 0) -> None:
        """Seed the planner's own generator so choices are reproducible."""
        self.rng = np.random.default_rng(seed)

    def next_action(self, scene: Scene) -> int | None:
        """Return the id of a random remaining item, or None if none remain."""
        if not scene.items:
            return None
        idx = int(self.rng.integers(0, len(scene.items)))
        return scene.items[idx].id
