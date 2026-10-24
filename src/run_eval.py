"""Evaluation sweeps and plots for the Messy Desk planners.

Runs the five planners across randomized desks and writes figures to
``out/plots/`` plus a markdown results table to ``out/results.md``. Every
number comes from a real :func:`~src.sim.run_episode` call on a fresh scene;
nothing is fabricated or hand-tuned.

Matplotlib uses the headless Agg backend, so this never opens a window and
never imports pygame. Randomness always flows from the seeded scene generator
and each planner's own seeded generator; the same seed gives identical results.

Produced artifacts:

- ``out/plots/ticks_vs_items.png``   ticks-to-clear vs n_items, two clutter panels
- ``out/plots/failed_vs_items.png``  failed grasps vs n_items, two clutter panels
- ``out/plots/sensitivity.png``      ranking under varied grasp/clearance constants
- ``out/plots/ablation.png``         OcclusionPlanner pushes on vs off
- ``out/results.md``                 markdown tables of the real numbers

Usage:
    python src/run_eval.py                # 30 seeds, item counts 5/15/25
    python src/run_eval.py --seeds 50     # more seeds
    python src/run_eval.py --quick        # 5 seeds, fewer sizes (smoke run)
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Headless rendering: pick the Agg backend before importing pyplot so no display
# is needed and no GUI toolkit (or pygame) is pulled in.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

# Allow running as ``python src/run_eval.py`` by putting the repo root on path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.planner as planner_mod  # noqa: E402
import src.robot as robot_mod  # noqa: E402
import src.sim as sim_mod  # noqa: E402
from src.planner import (  # noqa: E402
    LeastCoveredPlanner,
    NearestPlanner,
    NearestRetryPlanner,
    OcclusionPlanner,
    RandomPlanner,
)
from src.robot import Grasp, Gripper, Push  # noqa: E402
from src.scene import Scene, make_scene  # noqa: E402
from src.sim import run_episode  # noqa: E402

# 95% normal-approximation interval half-width uses this z-score.
Z_95 = 1.96

# Default item counts; --quick trims to the small and large ends.
ITEM_COUNTS = (5, 15, 25)
QUICK_ITEM_COUNTS = (5, 15)
CLUTTERS = (0.3, 0.9)

# Named planner factories (seed -> planner). Shared by every sweep.
PLANNERS: tuple[tuple[str, object], ...] = (
    ("random", lambda seed: RandomPlanner(seed=seed)),
    ("nearest", lambda seed: NearestPlanner()),
    ("leastcov", lambda seed: LeastCoveredPlanner()),
    ("nearretry", lambda seed: NearestRetryPlanner()),
    ("occlusion", lambda seed: OcclusionPlanner()),
)


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
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
class Stat:
    """Mean and 95% interval half-width for one metric over a set of seeds."""

    mean: float
    ci: float

    def fmt(self, places: int = 1) -> str:
        """Render as ``mean +/- ci`` with the given decimal places."""
        return f"{self.mean:.{places}f} +/- {self.ci:.{places}f}"


@dataclass
class Metrics:
    """Per-episode metrics for one configuration, averaged over seeds."""

    ticks: Stat
    failed: Stat
    pushes: Stat


# --------------------------------------------------------------------------- #
# Constant overrides (by argument, never by editing globals)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def override_constants(
    failed_grasp_ticks: int | None = None,
    clearance_penalty: float | None = None,
):
    """Temporarily override grasp/clearance constants across the live modules.

    The simulation reads ``FAILED_GRASP_TICKS`` from :mod:`src.sim` and
    :mod:`src.planner` (both imported it by name), and ``CLEARANCE_PENALTY``
    from :mod:`src.robot` (read inside ``effective_grasp_prob``) and
    :mod:`src.planner` (read inside the singulation rule). This context manager
    rebinds those module attributes for its body and restores the originals on
    exit, so callers vary constants per run without editing any source globals.
    Nested or concurrent use is not supported (single-threaded sweeps only).
    """
    saved: list[tuple[object, str, object]] = []

    def _set(module, name: str, value) -> None:
        saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    try:
        if failed_grasp_ticks is not None:
            _set(robot_mod, "FAILED_GRASP_TICKS", failed_grasp_ticks)
            _set(sim_mod, "FAILED_GRASP_TICKS", failed_grasp_ticks)
            _set(planner_mod, "FAILED_GRASP_TICKS", failed_grasp_ticks)
        if clearance_penalty is not None:
            _set(robot_mod, "CLEARANCE_PENALTY", clearance_penalty)
            _set(planner_mod, "CLEARANCE_PENALTY", clearance_penalty)
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


# --------------------------------------------------------------------------- #
# Push-disabling wrapper for the ablation
# --------------------------------------------------------------------------- #
class NoPushPlanner:
    """Wraps a planner and suppresses pushes, grasping the target instead.

    The ablation needs OcclusionPlanner with pushing turned off. Rather than
    fork the planner, this delegates to the wrapped instance and converts any
    :class:`~src.robot.Push` it returns into a :class:`~src.robot.Grasp` of the
    same item id. Grasps and ``None`` pass through unchanged.
    """

    def __init__(self, inner) -> None:
        """Store the wrapped planner whose pushes will be converted."""
        self._inner = inner

    def next_action(self, scene: Scene, gripper: Gripper) -> Grasp | None:
        """Delegate, turning any push into a grasp of the same item."""
        action = self._inner.next_action(scene, gripper)
        if isinstance(action, Push):
            return Grasp(item_id=action.item_id)
        return action


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    make_planner, n_items: int, clutter: float, seeds: range
) -> Metrics:
    """Average one planner's per-episode metrics over ``seeds``.

    Builds a fresh scene and a fresh planner for every seed (scenes and the
    occlusion/retry planners carry mutable state, so reuse would corrupt the
    run). Metrics come from an unbudgeted run (the sim's default
    ``10 * n_items`` tick cap).
    """
    ticks, failed, pushes = [], [], []
    for seed in seeds:
        result = run_episode(
            make_scene(seed=seed, n_items=n_items, clutter=clutter),
            make_planner(seed),
            seed=seed,
        )
        ticks.append(float(result.steps))
        failed.append(float(result.failed_grasps))
        pushes.append(float(result.pushes))
    return Metrics(
        ticks=Stat(_mean(ticks), _ci95(ticks)),
        failed=Stat(_mean(failed), _ci95(failed)),
        pushes=Stat(_mean(pushes), _ci95(pushes)),
    )


def sweep_by_items(
    item_counts: tuple[int, ...], seeds: range
) -> dict[tuple[str, float, int], Metrics]:
    """Evaluate every planner at each (clutter, n_items), keyed for plotting."""
    out: dict[tuple[str, float, int], Metrics] = {}
    for clutter in CLUTTERS:
        for name, make_planner in PLANNERS:
            for n_items in item_counts:
                out[(name, clutter, n_items)] = evaluate(
                    make_planner, n_items, clutter, seeds
                )
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_metric_vs_items(
    data: dict[tuple[str, float, int], Metrics],
    item_counts: tuple[int, ...],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Two-panel (clutter 0.3 / 0.9) line plot of a metric vs n_items.

    ``metric`` is ``"ticks"`` or ``"failed"``. Error bars are the 95% interval
    half-widths. One line per planner in each panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, clutter in zip(axes, CLUTTERS):
        for name, _ in PLANNERS:
            means, cis = [], []
            for n_items in item_counts:
                stat = getattr(data[(name, clutter, n_items)], metric)
                means.append(stat.mean)
                cis.append(stat.ci)
            ax.errorbar(
                item_counts,
                means,
                yerr=cis,
                marker="o",
                capsize=3,
                label=name,
            )
        ax.set_title(f"clutter {clutter}")
        ax.set_xlabel("n_items")
        ax.set_xticks(list(item_counts))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ticks_and_failed(
    data: dict[tuple[str, float, int], Metrics],
    item_counts: tuple[int, ...],
    plots_dir: Path,
) -> None:
    """Write the ticks-vs-items and failed-vs-items two-panel figures."""
    _plot_metric_vs_items(
        data,
        item_counts,
        "ticks",
        "ticks to clear",
        "Ticks to clear vs item count (95% interval)",
        plots_dir / "ticks_vs_items.png",
    )
    _plot_metric_vs_items(
        data,
        item_counts,
        "failed",
        "failed grasps",
        "Failed grasps vs item count (95% interval)",
        plots_dir / "failed_vs_items.png",
    )


# Sensitivity sweep configuration: fixed at 15 items, clutter 0.9.
SENS_ITEMS = 15
SENS_CLUTTER = 0.9
SENS_FAILED_TICKS = (1, 3, 5)
SENS_CLEARANCE = (1.0, 0.7, 0.5)
SENS_PLANNERS: tuple[tuple[str, object], ...] = (
    ("occlusion", lambda seed: OcclusionPlanner()),
    ("leastcov", lambda seed: LeastCoveredPlanner()),
    ("nearretry", lambda seed: NearestRetryPlanner()),
)


@dataclass
class SensRow:
    """One sensitivity cell: the constants used and each planner's tick stat."""

    failed_ticks: int
    clearance: float
    ticks: dict[str, Stat]


def sweep_sensitivity(seeds: range) -> list[SensRow]:
    """Sweep FAILED_GRASP_TICKS x CLEARANCE_PENALTY at 15 items, clutter 0.9.

    For each (failed_ticks, clearance) pair the three planners are evaluated
    under :func:`override_constants`, recording ticks-to-clear. Used to check
    whether the planner ranking is stable across the constant choices.
    """
    rows: list[SensRow] = []
    for ft in SENS_FAILED_TICKS:
        for cp in SENS_CLEARANCE:
            with override_constants(failed_grasp_ticks=ft, clearance_penalty=cp):
                ticks: dict[str, Stat] = {}
                for name, make_planner in SENS_PLANNERS:
                    m = evaluate(make_planner, SENS_ITEMS, SENS_CLUTTER, seeds)
                    ticks[name] = m.ticks
            rows.append(SensRow(failed_ticks=ft, clearance=cp, ticks=ticks))
    return rows


def plot_sensitivity(rows: list[SensRow], plots_dir: Path) -> None:
    """Grouped-bar plot of ticks-to-clear per planner across constant settings.

    Each x group is one (failed_ticks, clearance) pair; bars within a group are
    the three planners. Error bars are 95% intervals. Lets a reader see whether
    the occlusion/leastcov/nearretry ordering holds as constants change.
    """
    labels = [f"ft={r.failed_ticks}\ncp={r.clearance}" for r in rows]
    names = [n for n, _ in SENS_PLANNERS]
    x = range(len(rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        means = [r.ticks[name].mean for r in rows]
        cis = [r.ticks[name].ci for r in rows]
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, means, width, yerr=cis, capsize=3, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ticks to clear")
    ax.set_title(
        f"Sensitivity at {SENS_ITEMS} items, clutter {SENS_CLUTTER} "
        "(ticks to clear, 95% interval)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "sensitivity.png", dpi=120)
    plt.close(fig)


# Ablation configuration: OcclusionPlanner pushes on vs off.
ABLATION_ITEMS = 25
ABLATION_CLUTTER = 0.9


@dataclass
class AblationResult:
    """Ticks, failed grasps, and pushes per episode for one ablation arm."""

    ticks: Stat
    failed: Stat
    pushes: Stat


def sweep_ablation(seeds: range) -> dict[str, AblationResult]:
    """Compare OcclusionPlanner with pushes enabled vs disabled.

    Both arms run at :data:`ABLATION_ITEMS` items and clutter
    :data:`ABLATION_CLUTTER` (the regime where the occlusion planner actually
    pushes). The disabled arm wraps the planner in :class:`NoPushPlanner`,
    which converts every push into a grasp of the same item.
    """
    arms = {
        "pushes_on": lambda seed: OcclusionPlanner(),
        "pushes_off": lambda seed: NoPushPlanner(OcclusionPlanner()),
    }
    out: dict[str, AblationResult] = {}
    for label, make_planner in arms.items():
        m = evaluate(make_planner, ABLATION_ITEMS, ABLATION_CLUTTER, seeds)
        out[label] = AblationResult(ticks=m.ticks, failed=m.failed, pushes=m.pushes)
    return out


def plot_ablation(data: dict[str, AblationResult], plots_dir: Path) -> None:
    """Grouped-bar plot: ticks, failed grasps, pushes for on vs off arms."""
    metrics = ("ticks", "failed", "pushes")
    arms = ("pushes_on", "pushes_off")
    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, arm in enumerate(arms):
        res = data[arm]
        means = [res.ticks.mean, res.failed.mean, res.pushes.mean]
        cis = [res.ticks.ci, res.failed.ci, res.pushes.ci]
        offsets = [xi + (i - 0.5) * width for xi in x]
        ax.bar(offsets, means, width, yerr=cis, capsize=3, label=arm)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["ticks/clear", "failed grasps", "pushes"])
    ax.set_ylabel("per episode")
    ax.set_title(
        f"OcclusionPlanner ablation at {ABLATION_ITEMS} items, "
        f"clutter {ABLATION_CLUTTER} (95% interval)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "ablation.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Results markdown
# --------------------------------------------------------------------------- #
def write_results_md(
    by_items: dict[tuple[str, float, int], Metrics],
    item_counts: tuple[int, ...],
    sens_rows: list[SensRow],
    ablation: dict[str, AblationResult],
    out_path: Path,
    n_seeds: int,
) -> None:
    """Write ``out/results.md`` with markdown tables from the real numbers.

    Every value is a mean +/- 95% interval computed from the sweep above; no
    number is hand-entered. Sections mirror the four plots.
    """
    lines: list[str] = []
    lines.append("# Messy Desk evaluation results")
    lines.append("")
    lines.append(
        f"All values are mean +/- 95% interval over {n_seeds} seeds, from real "
        "`run_episode` runs on fresh scenes. Generated by `src/run_eval.py`."
    )
    lines.append("")

    # 1 & 2: ticks and failed grasps vs item count, per clutter.
    for clutter in CLUTTERS:
        lines.append(f"## Ticks to clear and failed grasps, clutter {clutter}")
        lines.append("")
        header = "| planner | " + " | ".join(
            f"{n} items ticks" for n in item_counts
        ) + " | " + " | ".join(f"{n} items failed" for n in item_counts) + " |"
        sep = "|" + "---|" * (1 + 2 * len(item_counts))
        lines.append(header)
        lines.append(sep)
        for name, _ in PLANNERS:
            tick_cols = [
                by_items[(name, clutter, n)].ticks.fmt(1) for n in item_counts
            ]
            fail_cols = [
                by_items[(name, clutter, n)].failed.fmt(2) for n in item_counts
            ]
            lines.append(
                f"| {name} | " + " | ".join(tick_cols) + " | "
                + " | ".join(fail_cols) + " |"
            )
        lines.append("")

    # 3: sensitivity sweep.
    lines.append(
        f"## Sensitivity at {SENS_ITEMS} items, clutter {SENS_CLUTTER} "
        "(ticks to clear)"
    )
    lines.append("")
    lines.append(
        "Varying `FAILED_GRASP_TICKS` and `CLEARANCE_PENALTY` by argument "
        "(globals untouched). `winner` is the lowest mean ticks in the row."
    )
    lines.append("")
    sens_names = [n for n, _ in SENS_PLANNERS]
    lines.append(
        "| FAILED_GRASP_TICKS | CLEARANCE_PENALTY | "
        + " | ".join(sens_names)
        + " | winner |"
    )
    lines.append("|" + "---|" * (3 + len(sens_names)))
    ranking_holds = True
    baseline_winner: str | None = None
    for r in sens_rows:
        winner = min(sens_names, key=lambda nm: r.ticks[nm].mean)
        if baseline_winner is None:
            baseline_winner = winner
        elif winner != baseline_winner:
            ranking_holds = False
        cols = [r.ticks[nm].fmt(1) for nm in sens_names]
        lines.append(
            f"| {r.failed_ticks} | {r.clearance} | "
            + " | ".join(cols)
            + f" | {winner} |"
        )
    lines.append("")
    lines.append(
        f"Best planner is **{baseline_winner}** in the first setting; the "
        + (
            "same planner wins every setting (ranking holds)."
            if ranking_holds
            else "winner changes across settings (ranking does NOT hold)."
        )
    )
    lines.append("")

    # 4: ablation.
    lines.append(
        f"## OcclusionPlanner push ablation at {ABLATION_ITEMS} items, "
        f"clutter {ABLATION_CLUTTER}"
    )
    lines.append("")
    lines.append("| arm | ticks to clear | failed grasps | pushes |")
    lines.append("|---|---|---|---|")
    for arm in ("pushes_on", "pushes_off"):
        res = ablation[arm]
        lines.append(
            f"| {arm} | {res.ticks.fmt(1)} | {res.failed.fmt(2)} "
            f"| {res.pushes.fmt(2)} |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_eval(seeds: int, quick: bool, out_dir: str | Path = "out") -> Path:
    """Run every sweep, write plots and ``results.md``, return the out dir.

    ``seeds`` is the seed count; ``--quick`` passes 5 and trims item counts.
    All artifacts land under ``out_dir`` (``out/plots/`` and
    ``out/results.md``). Returns the resolved output directory.
    """
    seed_range = range(seeds)
    item_counts = QUICK_ITEM_COUNTS if quick else ITEM_COUNTS

    out_root = Path(out_dir)
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_items = sweep_by_items(item_counts, seed_range)
    plot_ticks_and_failed(by_items, item_counts, plots_dir)

    sens_rows = sweep_sensitivity(seed_range)
    plot_sensitivity(sens_rows, plots_dir)

    ablation = sweep_ablation(seed_range)
    plot_ablation(ablation, plots_dir)

    write_results_md(
        by_items,
        item_counts,
        sens_rows,
        ablation,
        out_root / "results.md",
        n_seeds=seeds,
    )
    return out_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse --seeds (default 30) and --quick (5 seeds, fewer sizes)."""
    parser = argparse.ArgumentParser(description="Messy Desk evaluation sweeps.")
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="number of seeds to average over (default 30)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke run: 5 seeds and fewer item counts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the sweeps and report where artifacts landed."""
    args = _parse_args(argv)
    seeds = 5 if args.quick else args.seeds
    out_root = run_eval(seeds=seeds, quick=args.quick)
    print(f"seeds={seeds} quick={args.quick}")
    print(f"plots -> {out_root / 'plots'}")
    print(f"results -> {out_root / 'results.md'}")


if __name__ == "__main__":
    main()
