"""Compare the five planners on randomized cluttered desks.

Runs random, nearest, least-covered, nearest-retry, and occlusion planners
across a sweep of seeds, item counts, and clutter levels, and prints a plain
text table with 95% confidence intervals. No plots, no dataset logging: this is
a quick numeric read on how the planners differ.

Usage:
    python src/compare.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as ``python src/compare.py`` by putting the repo root on path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.planner import (
    LeastCoveredPlanner,
    NearestPlanner,
    NearestRetryPlanner,
    OcclusionPlanner,
    RandomPlanner,
)
from src.scene import make_scene
from src.sim import run_episode

# Sweep configuration.
SEEDS = range(30)
ITEM_COUNTS = (5, 15, 25)
CLUTTERS = (0.3, 0.9)

# Budget for "items sorted in time": 40 ticks at 15 items, 2.7 ticks per item.
TICKS_PER_ITEM = 2.7

# 95% normal-approximation interval half-width uses this z-score.
Z_95 = 1.96

PLANNERS = (
    ("random", lambda seed: RandomPlanner(seed=seed)),
    ("nearest", lambda seed: NearestPlanner()),
    ("leastcov", lambda seed: LeastCoveredPlanner()),
    ("nearretry", lambda seed: NearestRetryPlanner()),
    ("occlusion", lambda seed: OcclusionPlanner()),
)


def budget_for(n_items: int) -> int:
    """Tick budget for ``n_items`` items (40 at 15 items, 2.7 ticks/item)."""
    return round(TICKS_PER_ITEM * n_items)


def _mean(values: list[float]) -> float:
    """Arithmetic mean, 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _ci95(values: list[float]) -> float:
    """Half-width of the 95% interval: 1.96 * std / sqrt(n).

    Uses the sample standard deviation (n-1). Returns 0.0 for fewer than two
    samples, where the interval is undefined.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return Z_95 * math.sqrt(var) / math.sqrt(n)


@dataclass
class Cell:
    """Averaged metrics for one (planner, items, clutter) combination."""

    ticks_mean: float
    ticks_ci: float
    failed_mean: float
    failed_ci: float
    pushes_mean: float
    sorted_in_budget: float
    n_items: int


def evaluate(make_planner, n_items: int, clutter: float) -> Cell:
    """Average one planner's metrics over all seeds at a scene configuration.

    Ticks-to-clear, failed grasps, and pushes come from an unbudgeted run (the
    sim's default ``10 * n_items`` cap). Items sorted in budget comes from a
    second run capped at :func:`budget_for` ticks.
    """
    budget = budget_for(n_items)
    ticks, failed, pushes, sorted_in_budget = [], [], [], []

    for seed in SEEDS:
        full = run_episode(
            make_scene(seed=seed, n_items=n_items, clutter=clutter),
            make_planner(seed),
            seed=seed,
        )
        ticks.append(float(full.steps))
        failed.append(float(full.failed_grasps))
        pushes.append(float(full.pushes))

        capped = run_episode(
            make_scene(seed=seed, n_items=n_items, clutter=clutter),
            make_planner(seed),
            seed=seed,
            max_ticks=budget,
        )
        sorted_in_budget.append(float(capped.items_sorted))

    return Cell(
        ticks_mean=_mean(ticks),
        ticks_ci=_ci95(ticks),
        failed_mean=_mean(failed),
        failed_ci=_ci95(failed),
        pushes_mean=_mean(pushes),
        sorted_in_budget=_mean(sorted_in_budget),
        n_items=n_items,
    )


def print_table() -> None:
    """Run the full sweep and print one table per clutter level."""
    header = (
        f"{'planner':<10} {'items':>5} {'ticks/clear (95% CI)':>22} "
        f"{'failed (95% CI)':>18} {'pushes':>7} {'sorted@budget':>14}"
    )
    for clutter in CLUTTERS:
        budgets = ", ".join(f"{n}->{budget_for(n)}t" for n in ITEM_COUNTS)
        print(
            f"\n=== clutter {clutter} | {len(list(SEEDS))} seeds "
            f"| budgets {budgets} ==="
        )
        print(header)
        print("-" * len(header))
        for name, make_planner in PLANNERS:
            for n_items in ITEM_COUNTS:
                c = evaluate(make_planner, n_items, clutter)
                ticks_col = f"{c.ticks_mean:>7.1f} +/- {c.ticks_ci:<5.1f}"
                failed_col = f"{c.failed_mean:>6.2f} +/- {c.failed_ci:<5.2f}"
                print(
                    f"{name:<10} {n_items:>5} {ticks_col:>22} "
                    f"{failed_col:>18} {c.pushes_mean:>7.2f} "
                    f"{c.sorted_in_budget:>6.1f}/{n_items:<6}"
                )


if __name__ == "__main__":
    print_table()
