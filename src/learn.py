"""Weight search for the LearnedPlanner (reproducible, train/held-out split).

Searches a small set of feature weights for :class:`~src.planner.LearnedPlanner`
on a set of TRAINING seeds, then evaluates the resulting FIXED policy on a
separate set of HELD-OUT seeds and compares it to the five hand-written
planners. Everything flows from a single seed so a run is fully reproducible.

No new physics and no changes to grasp constants: fitness is the mean
ticks-to-clear from the real :func:`~src.sim.run_episode`, exactly like the
other evaluations. Headless; imports no pygame.

Usage:
    python -m src.learn                 # default search + held-out table
    python -m src.learn --iters 200 --train 20 --holdout 20
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.planner import (  # noqa: E402
    LEARNED_N_FEATURES,
    LeastCoveredPlanner,
    LearnedPlanner,
    NearestPlanner,
    NearestRetryPlanner,
    OcclusionPlanner,
    RandomPlanner,
)
from src.scene import make_scene  # noqa: E402
from src.sim import run_episode  # noqa: E402

# 95% normal-approximation interval half-width (matches compare/run_eval).
Z_95 = 1.96


def _mean(values: list[float]) -> float:
    """Arithmetic mean, 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _ci95(values: list[float]) -> float:
    """Half-width of the 95% interval: 1.96 * sample_std(n-1) / sqrt(n)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return Z_95 * math.sqrt(var) / math.sqrt(n)


def mean_ticks(
    weights: tuple[float, ...], seeds: list[int], n_items: int, clutter: float,
    rotate: bool = False,
) -> float:
    """Mean ticks-to-clear for a LearnedPlanner(weights) over ``seeds``.

    Builds a FRESH scene per seed and a FRESH planner (the LearnedPlanner is
    stateless given fixed weights, but fresh-per-seed keeps the pattern). Lower
    is better. Deterministic in its arguments.
    """
    ticks: list[float] = []
    for seed in seeds:
        scene = make_scene(seed=seed, n_items=n_items, clutter=clutter, rotate=rotate)
        result = run_episode(scene, LearnedPlanner(weights), seed=seed)
        ticks.append(float(result.steps))
    return _mean(ticks)


@dataclass
class SearchResult:
    """Outcome of a weight search: the best weights and their training score."""

    weights: tuple[float, ...]
    train_ticks: float
    history: list[float]  # best-so-far training score per iteration


def search_weights(
    train_seeds: list[int],
    n_items: int,
    clutter: float,
    iters: int = 150,
    search_seed: int = 0,
    sigma: float = 0.5,
    rotate: bool = False,
) -> SearchResult:
    """Evolutionary hill-climb over LearnedPlanner weights on TRAINING seeds.

    Starts from a random weight vector and, each iteration, mutates the current
    best by Gaussian noise (``sigma``); it keeps the candidate when its mean
    ticks-to-clear on ``train_seeds`` improves. All randomness comes from
    ``search_seed`` so the search is reproducible. Returns the best weights,
    their training score, and the best-so-far history.
    """
    rng = np.random.default_rng(search_seed)
    best = tuple(float(w) for w in rng.normal(0.0, 1.0, size=LEARNED_N_FEATURES))
    best_score = mean_ticks(best, train_seeds, n_items, clutter, rotate)
    history = [best_score]

    for _ in range(iters):
        cand = tuple(
            b + float(rng.normal(0.0, sigma)) for b in best
        )
        score = mean_ticks(cand, train_seeds, n_items, clutter, rotate)
        if score < best_score:
            best, best_score = cand, score
        history.append(best_score)

    return SearchResult(weights=best, train_ticks=best_score, history=history)


@dataclass
class HeldOutRow:
    """Held-out mean +/- 95% CI ticks-to-clear for one planner."""

    name: str
    mean: float
    ci: float


# Factories for the five hand-written baselines (seed -> planner).
_BASELINES = (
    ("random", lambda seed: RandomPlanner(seed=seed)),
    ("nearest", lambda seed: NearestPlanner()),
    ("leastcov", lambda seed: LeastCoveredPlanner()),
    ("nearretry", lambda seed: NearestRetryPlanner()),
    ("occlusion", lambda seed: OcclusionPlanner()),
)


def evaluate_holdout(
    weights: tuple[float, ...],
    holdout_seeds: list[int],
    n_items: int,
    clutter: float,
    rotate: bool = False,
) -> list[HeldOutRow]:
    """Evaluate the learned policy and the five baselines on HELD-OUT seeds.

    Each planner is run on a FRESH scene per held-out seed (the learned policy
    uses the FIXED ``weights`` found in training -- no further fitting on the
    held-out set). Returns mean +/- 95% CI ticks-to-clear per planner, learned
    first. Held-out seeds must be disjoint from the training seeds.
    """
    rows: list[HeldOutRow] = []

    learned_ticks = [
        float(
            run_episode(
                make_scene(seed=s, n_items=n_items, clutter=clutter, rotate=rotate),
                LearnedPlanner(weights),
                seed=s,
            ).steps
        )
        for s in holdout_seeds
    ]
    rows.append(HeldOutRow("learned", _mean(learned_ticks), _ci95(learned_ticks)))

    for name, make_planner in _BASELINES:
        ticks = [
            float(
                run_episode(
                    make_scene(seed=s, n_items=n_items, clutter=clutter, rotate=rotate),
                    make_planner(s),
                    seed=s,
                ).steps
            )
            for s in holdout_seeds
        ]
        rows.append(HeldOutRow(name, _mean(ticks), _ci95(ticks)))

    return rows


def split_seeds(
    n_train: int, n_holdout: int, base_seed: int = 0
) -> tuple[list[int], list[int]]:
    """Return disjoint TRAINING and HELD-OUT seed lists.

    Training seeds are ``base_seed .. base_seed + n_train - 1``; held-out seeds
    continue immediately after, so the two sets never overlap.
    """
    train = list(range(base_seed, base_seed + n_train))
    holdout = list(range(base_seed + n_train, base_seed + n_train + n_holdout))
    return train, holdout


def run(
    n_items: int = 15,
    clutter: float = 0.9,
    n_train: int = 15,
    n_holdout: int = 15,
    iters: int = 150,
    search_seed: int = 0,
    rotate: bool = False,
) -> tuple[SearchResult, list[HeldOutRow]]:
    """Train on TRAINING seeds, evaluate on disjoint HELD-OUT seeds.

    Fully reproducible from ``search_seed`` and the seed split. Returns the
    search result (best weights + training score) and the held-out comparison.
    """
    train_seeds, holdout_seeds = split_seeds(n_train, n_holdout)
    result = search_weights(
        train_seeds, n_items, clutter, iters=iters, search_seed=search_seed,
        rotate=rotate,
    )
    rows = evaluate_holdout(
        result.weights, holdout_seeds, n_items, clutter, rotate=rotate
    )
    return result, rows


def _format_report(
    result: SearchResult, rows: list[HeldOutRow], n_items: int, clutter: float,
    n_train: int, n_holdout: int,
) -> str:
    """Build a plain-text held-out comparison table (ticks to clear)."""
    lines = []
    lines.append(
        f"LearnedPlanner weight search @ {n_items} items, clutter {clutter}"
    )
    lines.append(
        f"train seeds={n_train}, held-out seeds={n_holdout} (disjoint); "
        f"weights={tuple(round(w, 3) for w in result.weights)}; "
        f"train ticks={result.train_ticks:.1f}"
    )
    lines.append("")
    lines.append(f"{'planner':<10} {'held-out ticks to clear (95% CI)':>34}")
    lines.append("-" * 46)
    for r in sorted(rows, key=lambda x: x.mean):
        lines.append(f"{r.name:<10} {r.mean:>22.1f} +/- {r.ci:<8.1f}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the learn CLI options."""
    p = argparse.ArgumentParser(description="LearnedPlanner weight search.")
    p.add_argument("--n-items", type=int, default=15, dest="n_items")
    p.add_argument("--clutter", type=float, default=0.9)
    p.add_argument("--train", type=int, default=15, dest="n_train")
    p.add_argument("--holdout", type=int, default=15, dest="n_holdout")
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--search-seed", type=int, default=0, dest="search_seed")
    p.add_argument("--rotate", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the search and print the held-out comparison."""
    a = _parse_args(argv)
    result, rows = run(
        n_items=a.n_items, clutter=a.clutter, n_train=a.n_train,
        n_holdout=a.n_holdout, iters=a.iters, search_seed=a.search_seed,
        rotate=a.rotate,
    )
    print(_format_report(result, rows, a.n_items, a.clutter, a.n_train, a.n_holdout))


if __name__ == "__main__":
    main()
