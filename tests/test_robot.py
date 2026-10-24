"""Tests for the grasp model."""

from __future__ import annotations

from math import isclose

from src.robot import MAX_GRASP_PROB, MIN_GRASP_PROB, grasp_success_prob


def test_exposed_item_grasps_certainly() -> None:
    """A fully exposed item (coverage 0) has probability 1.0."""
    assert grasp_success_prob(0.0) == 1.0


def test_base_formula_midrange() -> None:
    """In-range coverage follows 1 - 0.8 * covered_fraction."""
    assert isclose(grasp_success_prob(0.5), 0.6)


def test_clamped_to_floor() -> None:
    """Heavy coverage is clamped to the 0.05 floor, never below."""
    # 1 - 0.8 * 1.0 = 0.2, still above the floor.
    assert isclose(grasp_success_prob(1.0), 0.2)
    # Beyond-1 coverage would dip under the floor without clamping.
    assert grasp_success_prob(2.0) == MIN_GRASP_PROB


def test_clamped_to_ceiling() -> None:
    """Negative coverage cannot push probability above 1.0."""
    assert grasp_success_prob(-1.0) == MAX_GRASP_PROB
