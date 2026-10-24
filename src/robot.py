"""Grasp model for the Messy Desk simulation.

Pure, headless helpers for the robot's grasping behavior. No pygame, no global
random state.
"""

from __future__ import annotations

# Floor on grasp success so a fully covered item is still occasionally liftable.
MIN_GRASP_PROB: float = 0.05
MAX_GRASP_PROB: float = 1.0


def grasp_success_prob(covered_fraction: float) -> float:
    """Probability a grasp succeeds given how covered the item is.

    ``covered_fraction`` is the fraction of the item's area hidden by items
    stacked above it (0.0 fully exposed, 1.0 fully covered). The base model is
    ``1 - 0.8 * covered_fraction``, clamped to [0.05, 1.0].
    """
    prob = 1.0 - 0.8 * covered_fraction
    return float(min(max(prob, MIN_GRASP_PROB), MAX_GRASP_PROB))
