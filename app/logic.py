"""Headless logic layer for Messy Desk Lab.

Pure, typed, GUI-independent logic that the Streamlit app calls. This module
imports ONLY the simulation core under ``src/``, numpy, and the standard
library. It MUST NOT import ``streamlit``, ``pygame``, or ``render`` (directly
or transitively), so it stays headless and unit-testable.

The layer reuses the engine rather than duplicating behavior: scenes come from
``src.scene.make_scene``, planners from ``src.planner``, and every episode runs
through ``src.sim.run_episode``. Confidence-interval math mirrors
``src.compare`` / ``src.run_eval`` exactly.

Geometry is in pixels on a 640x480 desk; time is in ticks at 15 Hz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from src.planner import (
    LeastCoveredPlanner,
    NearestPlanner,
    NearestRetryPlanner,
    OcclusionPlanner,
    RandomPlanner,
)
from src.scene import Scene, make_scene
from src.sim import EpisodeResult, run_episode

# --------------------------------------------------------------------------- #
# Input ranges
# --------------------------------------------------------------------------- #
# Supported item-count range for the UI. Values outside are clamped, not
# rejected, so a stray control value never crashes a run.
N_ITEMS_MIN: int = 0
N_ITEMS_MAX: int = 30

# 95% normal-approximation interval half-width uses this z-score. Identical to
# ``src.compare.Z_95`` and ``src.run_eval.Z_95``.
Z_95: float = 1.96


# --------------------------------------------------------------------------- #
# Planner registry (no render.py import)
# --------------------------------------------------------------------------- #
# Name -> (seed) -> planner. Mirrors the factories in render.py WITHOUT
# importing render.py, so the logic layer stays headless. Only RandomPlanner
# consumes the seed; the others are deterministic given the scene.
_PLANNER_FACTORIES: dict[str, object] = {
    "random": lambda seed: RandomPlanner(seed=seed),
    "nearest": lambda seed: NearestPlanner(),
    "leastcov": lambda seed: LeastCoveredPlanner(),
    "nearretry": lambda seed: NearestRetryPlanner(),
    "occlusion": lambda seed: OcclusionPlanner(),
}

_PLANNER_DESCRIPTIONS: dict[str, str] = {
    "random": "Picks a random item and randomly grasps or pushes it.",
    "nearest": "Always grasps the nearest remaining item, ignoring coverage.",
    "leastcov": "Grasps the least-covered item, breaking ties by distance.",
    "nearretry": "Nearest-first, but skips an item that keeps failing.",
    "occlusion": "Grasps exposed items first; singulates when it pays off.",
}


def available_planners() -> list[str]:
    """Return the selectable planner names in a stable UI order."""
    return list(_PLANNER_FACTORIES.keys())


def make_planner(name: str, seed: int) -> object:
    """Construct a FRESH planner by name, seeded for reproducibility.

    A new instance is built every call so mutable planner state never leaks
    between trials. Raises ``ValueError`` naming the bad name and the valid
    options when ``name`` is unknown.
    """
    try:
        factory = _PLANNER_FACTORIES[name]
    except KeyError:
        valid = ", ".join(available_planners())
        raise ValueError(
            f"unknown planner {name!r}; choose from {valid}"
        ) from None
    return factory(seed)


def planner_description(name: str) -> str:
    """One-line plain-English description of a planner.

    Raises ``ValueError`` for an unknown name (same contract as make_planner).
    """
    if name not in _PLANNER_DESCRIPTIONS:
        valid = ", ".join(available_planners())
        raise ValueError(f"unknown planner {name!r}; choose from {valid}")
    return _PLANNER_DESCRIPTIONS[name]


# --------------------------------------------------------------------------- #
# Input clamping + scene construction
# --------------------------------------------------------------------------- #
def clamp_clutter(clutter: float) -> float:
    """Clamp ``clutter`` into the valid [0, 1] range."""
    return float(min(max(clutter, 0.0), 1.0))


def clamp_n_items(n_items: int) -> int:
    """Clamp ``n_items`` into the supported [N_ITEMS_MIN, N_ITEMS_MAX] range."""
    return int(min(max(n_items, N_ITEMS_MIN), N_ITEMS_MAX))


def build_scene(seed: int, n_items: int, clutter: float) -> Scene:
    """Build one Scene via ``make_scene``, clamping inputs first.

    ``clutter`` is clamped to [0, 1] and ``n_items`` to the supported range
    [0, 30] (the UI exposes 3-30). Geometry is in pixels on a 640x480 desk.
    """
    return make_scene(
        seed=seed,
        n_items=clamp_n_items(n_items),
        clutter=clamp_clutter(clutter),
    )


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
@dataclass
class TrialConfig:
    """Minimal, sufficient key to deterministically rebuild and replay one trial.

    Carries exactly the four inputs that fix a trial: the planner name, the
    seed (which seeds scene generation, planner rng, and grasp outcomes),
    ``n_items`` (items on the desk), and ``clutter`` in [0, 1].
    """

    planner_name: str
    seed: int
    n_items: int
    clutter: float


@dataclass
class ExperimentConfig:
    """What the Experiment section configures: one trial per (seed, planner)."""

    n_items: int  # items per desk (clamped on use)
    clutter: float  # in [0, 1] (clamped on use)
    seeds: list[int]
    planners: list[str]


@dataclass
class TrialRow:
    """One trial's config paired with its real ``EpisodeResult`` metrics."""

    config: TrialConfig
    result: EpisodeResult


@dataclass
class Stat:
    """Mean and 95% interval half-width for one metric over a set of trials."""

    mean: float
    ci: float


@dataclass
class PlannerStat:
    """Per-planner mean +/- 95% CI for each metric over the experiment's seeds."""

    planner: str
    n_trials: int
    ticks: Stat  # steps to clear, in ticks
    sorted: Stat  # items sorted
    failed: Stat  # failed grasps
    pushes: Stat  # push primitives used


@dataclass
class ExperimentResult:
    """All per-trial rows plus the aggregated per-planner stats."""

    rows: list[TrialRow]
    stats: list[PlannerStat]
    config: ExperimentConfig


# --------------------------------------------------------------------------- #
# Confidence-interval statistics (mirrors src.compare / src.run_eval)
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    """Arithmetic mean, 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def ci95(values: list[float]) -> float:
    """Half-width of the 95% interval: ``1.96 * std / sqrt(n)``.

    Uses the sample standard deviation (``n-1`` denominator). Returns ``0.0``
    for fewer than two samples, where the interval is undefined. Identical to
    ``src.compare._ci95`` and ``src.run_eval._ci95``.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return Z_95 * math.sqrt(var) / math.sqrt(n)


def _stat(values: list[float]) -> Stat:
    """Build a ``Stat`` (mean and 95% CI half-width) for a metric series."""
    return Stat(mean=_mean(values), ci=ci95(values))


def overlap(a: Stat, b: Stat) -> bool:
    """True when two 95% intervals overlap (a tie at this sample size).

    Matches the ``src.run_eval._sensitivity_winner`` rule: order the two by
    mean, then the intervals overlap when the lower-mean stat's upper bound
    (``mean + ci``) reaches the higher-mean stat's lower bound (``mean - ci``).
    """
    lo, hi = (a, b) if a.mean <= b.mean else (b, a)
    return lo.mean + lo.ci >= hi.mean - hi.ci


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #
def run_trial(config: TrialConfig) -> EpisodeResult:
    """Run one trial on a FRESH scene and a FRESH planner; return its result.

    Deterministic in ``config``: the same TrialConfig always yields the same
    ``EpisodeResult`` because the scene, the planner, and the grasp-outcome
    draws are all seeded from ``config.seed``.
    """
    scene = build_scene(config.seed, config.n_items, config.clutter)
    planner = make_planner(config.planner_name, config.seed)
    return run_episode(scene, planner, seed=config.seed)


def aggregate_stats(rows: list[TrialRow]) -> list[PlannerStat]:
    """Compute mean +/- 95% CI per metric per planner from trial rows.

    Groups ``rows`` by planner name (preserving first-seen order) and reduces
    each group's ``EpisodeResult`` fields to a ``PlannerStat``: ticks from
    ``steps``, sorted from ``items_sorted``, failed from ``failed_grasps``, and
    pushes from ``pushes``.
    """
    order: list[str] = []
    grouped: dict[str, list[EpisodeResult]] = {}
    for row in rows:
        name = row.config.planner_name
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(row.result)

    stats: list[PlannerStat] = []
    for name in order:
        results = grouped[name]
        stats.append(
            PlannerStat(
                planner=name,
                n_trials=len(results),
                ticks=_stat([float(r.steps) for r in results]),
                sorted=_stat([float(r.items_sorted) for r in results]),
                failed=_stat([float(r.failed_grasps) for r in results]),
                pushes=_stat([float(r.pushes) for r in results]),
            )
        )
    return stats


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run each selected planner over each seed and aggregate the results.

    For every (planner, seed) pair a FRESH scene and a FRESH planner are built
    so no mutable scene or planner state leaks between trials. Each trial runs
    through ``src.sim.run_episode``; the real ``EpisodeResult`` is kept. Planner
    names are validated up front, raising ``ValueError`` for an unknown name.
    Deterministic in ``config``.
    """
    for name in config.planners:
        if name not in _PLANNER_FACTORIES:
            valid = ", ".join(available_planners())
            raise ValueError(
                f"unknown planner {name!r}; choose from {valid}"
            )

    rows: list[TrialRow] = []
    for name in config.planners:
        for seed in config.seeds:
            trial = TrialConfig(
                planner_name=name,
                seed=seed,
                n_items=config.n_items,
                clutter=config.clutter,
            )
            rows.append(TrialRow(config=trial, result=run_trial(trial)))

    return ExperimentResult(
        rows=rows, stats=aggregate_stats(rows), config=config
    )


# --------------------------------------------------------------------------- #
# Output-location guard
# --------------------------------------------------------------------------- #
# Generated artifacts live only under this directory (gitignored). Inputs under
# data/ and robotics/ are never written.
OUT_DIR: Path = Path("out")

# Read-only top-level input trees the app must never write into. A relative
# path beginning with one of these names targets an input tree, not out/.
_READ_ONLY_ROOTS: frozenset[str] = frozenset({"data", "robotics"})


def resolve_output_path(path: str | Path) -> Path:
    """Resolve a generated-output path, requiring it to live under ``out/``.

    Returns the resolved absolute path when it lies inside the repository's
    ``out/`` tree. Raises ``ValueError`` for any path that would escape ``out/``
    or that targets a read-only input tree (``data/`` or ``robotics/``), so the
    app can never write outside ``out/``.

    A relative path is interpreted against ``out/``, except that a relative path
    whose first segment names a read-only input tree is rejected outright, since
    its intent is to reach that tree rather than a folder inside ``out/``.
    """
    out_root = OUT_DIR.resolve()
    candidate = Path(path)

    if not candidate.is_absolute():
        first = candidate.parts[0] if candidate.parts else ""
        if first in _READ_ONLY_ROOTS:
            raise ValueError(
                f"refusing to write to read-only input tree: {path}"
            )
        candidate = out_root / candidate

    resolved = candidate.resolve()
    if resolved != out_root and out_root not in resolved.parents:
        raise ValueError(
            f"refusing to write outside out/: {resolved} is not under {out_root}"
        )
    return resolved
