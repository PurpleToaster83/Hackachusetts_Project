"""Headless logic layer for Messy Desk Lab.

Pure, typed, GUI-independent logic that the Streamlit app calls. This module
imports ONLY the simulation core under ``src/``, numpy, matplotlib (the Agg
backend), and the standard library. It MUST NOT import ``streamlit``,
``pygame``, or ``render`` (directly or transitively), so it stays headless and
unit-testable. Trace frames come from ``src.trace`` (the headless core), never
from ``render.py``.

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

import numpy as np

from src.planner import (
    LeastCoveredPlanner,
    NearestPlanner,
    NearestRetryPlanner,
    OcclusionPlanner,
    RandomPlanner,
)
from src.scene import DESK_H, DESK_W, Scene, make_scene
from src.sim import EpisodeResult, run_episode
from src.trace import Trace, build_trace

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


def build_scene(
    seed: int, n_items: int, clutter: float, rotate: bool = False
) -> Scene:
    """Build one Scene via ``make_scene``, clamping inputs first.

    ``clutter`` is clamped to [0, 1] and ``n_items`` to the supported range
    [0, 30] (the UI exposes 3-30). Geometry is in pixels on a 640x480 desk.
    ``rotate`` (experimental) forwards to ``make_scene`` for rotated footprints.
    """
    return make_scene(
        seed=seed,
        n_items=clamp_n_items(n_items),
        clutter=clamp_clutter(clutter),
        rotate=rotate,
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
    rotate: bool = False  # EXPERIMENTAL: rotated-rectangle footprints when True


@dataclass
class ExperimentConfig:
    """What the Experiment section configures: one trial per (seed, planner)."""

    n_items: int  # items per desk (clamped on use)
    clutter: float  # in [0, 1] (clamped on use)
    seeds: list[int]
    planners: list[str]
    rotate: bool = False  # EXPERIMENTAL: rotated footprints when True


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
    scene = build_scene(
        config.seed, config.n_items, config.clutter, rotate=config.rotate
    )
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
                rotate=config.rotate,
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


# --------------------------------------------------------------------------- #
# Live View replay: build a trace and render it to headless RGB frames
# --------------------------------------------------------------------------- #
# One letter per kind so colour is never the only cue (accessibility).
_KIND_LETTER: dict[str, str] = {"pen": "P", "key": "K", "mug": "M", "trash": "T"}

# Per-kind fill colours (0-1 RGB for matplotlib) and a distinct edge style, so
# each kind carries three redundant cues: letter, colour, and outline.
_KIND_RGB: dict[str, tuple[float, float, float]] = {
    "pen": (70 / 255, 130 / 255, 220 / 255),
    "key": (220 / 255, 200 / 255, 70 / 255),
    "mug": (90 / 255, 190 / 255, 120 / 255),
    "trash": (180 / 255, 90 / 255, 90 / 255),
}
_KIND_EDGE: dict[str, str] = {
    "pen": "solid",
    "key": "dashed",
    "mug": "solid",
    "trash": "dotted",
}
_FAIL_RGB = (235 / 255, 70 / 255, 70 / 255)


@dataclass
class ContainerBox:
    """A static drop target for drawing. Center and size in pixels."""

    kind: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class ItemBox:
    """A drawable item snapshot. Center/size in pixels; z is stack height.

    ``angle`` is the footprint rotation in radians (0.0 for axis-aligned);
    the renderer draws the true rotated footprint when it is non-zero.
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
class ReplayFrame:
    """One simulated tick of a replay (mirrors the trace-core TickFrame).

    ``gx``/``gy`` is the gripper center in pixels. ``event`` is one of ``""``,
    ``"grasp"``, ``"fail"``, ``"push"``. Counters are cumulative up to this tick.
    """

    gx: float
    gy: float
    items: list[ItemBox]
    event: str
    event_item: int  # item id acted on, or -1
    tick: int  # simulation tick at 15 Hz
    sorted_count: int
    failed_count: int
    push_count: int


@dataclass
class ReplayResult:
    """A full replay for one trial: per-tick frames plus static containers.

    ``result`` is the paired aggregate ``EpisodeResult`` so callers can confirm
    the trace matches the sim. ``items_total`` is the item count at the start.
    """

    config: TrialConfig
    frames: list[ReplayFrame]
    containers: list[ContainerBox]
    result: EpisodeResult
    items_total: int


@dataclass
class FrameImage:
    """A rendered frame as an RGB array for display, with accessible alt text."""

    rgb: "np.ndarray"  # shape (H, W, 3), dtype uint8
    alt: str


def _to_replay_frame(tf) -> ReplayFrame:
    """Convert a trace-core TickFrame into a drawable ReplayFrame."""
    return ReplayFrame(
        gx=tf.gx,
        gy=tf.gy,
        items=[
            ItemBox(
                id=s.id, kind=s.kind, x=s.x, y=s.y, w=s.w, h=s.h, z=s.z,
                angle=getattr(s, "angle", 0.0),
            )
            for s in tf.items
        ],
        event=tf.event,
        event_item=tf.event_item,
        tick=tf.tick,
        sorted_count=tf.sorted_count,
        failed_count=tf.failed_count,
        push_count=tf.push_count,
    )


def replay_trial(config: TrialConfig) -> ReplayResult:
    """Deterministically replay one trial into per-tick frames.

    Rebuilds a FRESH scene and FRESH planner from ``config`` (clamping inputs),
    builds the trace via the headless ``src.trace`` core (never ``render.py``),
    and pairs it with a direct ``run_episode`` result on an equally-fresh scene
    and planner so callers can verify trace-matches-sim. Deterministic in
    ``config``.
    """
    n_items = clamp_n_items(config.n_items)
    clutter = clamp_clutter(config.clutter)

    trace: Trace = build_trace(
        config.planner_name, seed=config.seed, n_items=n_items, clutter=clutter,
        rotate=config.rotate,
    )

    # Paired aggregate from the authoritative runner (fresh scene + planner).
    # Same seed AND same rotate as the trace, so the two agree.
    scene = make_scene(
        seed=config.seed, n_items=n_items, clutter=clutter, rotate=config.rotate
    )
    planner = make_planner(config.planner_name, config.seed)
    result = run_episode(scene, planner, seed=config.seed)

    containers = [
        ContainerBox(kind=c.kind, x=c.x, y=c.y, w=c.w, h=c.h)
        for c in trace.containers
    ]
    frames = [_to_replay_frame(tf) for tf in trace.frames]
    # items_total is the item count at the opening frame (tick 0).
    items_total = len(frames[0].items) if frames else 0

    return ReplayResult(
        config=config,
        frames=frames,
        containers=containers,
        result=result,
        items_total=items_total,
    )


# Darker per-kind accents for glyph detail (same hue family, lower value).
_KIND_DARK: dict[str, tuple[float, float, float]] = {
    "pen": (30 / 255, 70 / 255, 150 / 255),
    "key": (150 / 255, 130 / 255, 20 / 255),
    "mug": (40 / 255, 120 / 255, 70 / 255),
    "trash": (120 / 255, 50 / 255, 50 / 255),
}


def _draw_item_glyph(ax, s, fill, accent) -> None:
    """Draw a recognizable top-down glyph for item ``s`` inside its own box.

    Presentation only: the glyph is a matplotlib-drawn shape (no font/emoji)
    scaled to fit within the item's existing bounding box ``(s.w, s.h)`` so it
    never changes the collision/coverage geometry. ``fill`` is the kind colour;
    ``accent`` is a darker detail colour. A faint outline of the true bounding
    box is kept so the real footprint stays visible.
    """
    import numpy as _np
    from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle

    cx, cy, w, h = s.x, s.y, s.w, s.h
    half = min(w, h) / 2.0

    angle = getattr(s, "angle", 0.0)
    if angle:
        # EXPERIMENTAL (spike): draw the TRUE rotated footprint the sim uses --
        # a rotated rectangle -- filled with the kind colour. The detailed
        # per-kind glyph (handle arc, teeth, nib) is only drawn for axis-aligned
        # items; a rotated item shows its real footprint plus the letter label.
        hw, hh = w / 2.0, h / 2.0
        ca, sa = _np.cos(angle), _np.sin(angle)
        corners = [
            (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        ]
        ax.add_patch(Polygon(corners, facecolor=fill, edgecolor=accent, linewidth=1.0))
        return

    # Faint true-box outline (the actual sim footprint).
    ax.add_patch(
        Rectangle(
            (cx - w / 2, cy - h / 2), w, h,
            facecolor="none", edgecolor="#141418", linewidth=0.5, alpha=0.35,
        )
    )

    if s.kind == "pen":
        # A pen barrel along the longer axis: rounded body, a small pointed nib
        # at one end (kept small so it does not read as an arrowhead), and a
        # short "clip" line near the other end.
        horizontal = w >= h
        length = (w if horizontal else h) * 0.9
        thick = max(3.0, half * 0.42)
        nib = thick * 0.9  # small nib, not a big arrowhead
        if horizontal:
            x0, y0 = cx - length / 2, cy - thick / 2
            bw, bh = length - nib, thick
            tip = [(cx + length / 2 - nib, cy - thick / 2),
                   (cx + length / 2 - nib, cy + thick / 2),
                   (cx + length / 2, cy)]
            clip = ([cx - length / 2 + bw * 0.18, cx - length / 2 + bw * 0.42],
                    [cy - thick * 0.9, cy - thick * 0.9])
        else:
            x0, y0 = cx - thick / 2, cy - length / 2
            bw, bh = thick, length - nib
            tip = [(cx - thick / 2, cy + length / 2 - nib),
                   (cx + thick / 2, cy + length / 2 - nib),
                   (cx, cy + length / 2)]
            clip = ([cx - thick * 0.9, cx - thick * 0.9],
                    [cy - length / 2 + bh * 0.18, cy - length / 2 + bh * 0.42])
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0), bw, bh,
                boxstyle="round,pad=0,rounding_size=%.2f" % (thick / 2),
                facecolor=fill, edgecolor=accent, linewidth=0.8,
            )
        )
        ax.add_patch(Polygon(tip, facecolor=accent, edgecolor=accent))
        ax.plot(clip[0], clip[1], color=accent, linewidth=1.2)

    elif s.kind == "key":
        # Round bow with a hole, plus a short blade with two teeth.
        bow_r = half * 0.62
        bx = cx - w * 0.26
        ax.add_patch(Circle((bx, cy), bow_r, facecolor=fill, edgecolor=accent, linewidth=1.0))
        ax.add_patch(Circle((bx, cy), bow_r * 0.4, facecolor="#2c303a", edgecolor=accent, linewidth=0.8))
        blade_w = w * 0.6
        blade_h = max(3.0, half * 0.34)
        ax.add_patch(Rectangle((bx, cy - blade_h / 2), blade_w, blade_h,
                               facecolor=fill, edgecolor=accent, linewidth=0.6))
        tooth = blade_h
        for frac in (0.6, 0.85):
            tx = bx + blade_w * frac
            ax.add_patch(Rectangle((tx, cy + blade_h / 2), tooth * 0.5, tooth,
                                   facecolor=fill, edgecolor=accent, linewidth=0.5))

    elif s.kind == "mug":
        # Rounded square body with a handle arc on the right.
        body = min(w, h) * 0.86
        ax.add_patch(
            FancyBboxPatch(
                (cx - body / 2, cy - body / 2), body, body,
                boxstyle="round,pad=0,rounding_size=%.2f" % (body * 0.22),
                facecolor=fill, edgecolor=accent, linewidth=0.9,
            )
        )
        hr = body * 0.42
        ax.add_patch(Arc((cx + body / 2, cy), hr, hr, angle=0,
                         theta1=-80, theta2=80, edgecolor=accent, linewidth=1.6))

    else:  # trash: a crumpled scribble blob
        import numpy as _np

        rng = _np.random.default_rng(s.id)  # stable per item, no sim rng used
        r = half * 0.92
        pts = []
        for k in range(7):
            ang = 2 * _np.pi * k / 7
            rad = r * (0.6 + 0.4 * rng.random())
            pts.append((cx + rad * _np.cos(ang), cy + rad * _np.sin(ang)))
        ax.add_patch(Polygon(pts, facecolor=fill, edgecolor=accent, linewidth=0.8))
        # A couple of scribble strokes for the crumpled look.
        ax.plot([cx - r * 0.4, cx + r * 0.3], [cy - r * 0.2, cy + r * 0.3],
                color=accent, linewidth=0.7)
        ax.plot([cx - r * 0.2, cx + r * 0.4], [cy + r * 0.3, cy - r * 0.3],
                color=accent, linewidth=0.7)


# Nominal item size (pixels) used to scale the gripper so it reads at roughly
# the size of a typical item on the 640x480 desk. Items are ~24-48 px wide.
_GRIPPER_SPAN: float = 40.0


def _draw_gripper(ax, gx: float, gy: float, event: str) -> None:
    """Draw a top-down parallel-jaw gripper at ``(gx, gy)``, drawn shapes only.

    A rectangular hub with two short thick parallel fingers extending from it
    toward the target (downward in screen space), forming an open claw with a
    visible gap. The whole gripper is sized relative to a typical item
    (:data:`_GRIPPER_SPAN`) so it is clearly visible. On a ``grasp`` the fingers
    close the gap and the claw is highlighted; ``push`` and the idle move keep
    the existing colours (green grasp, amber push, red fail, pale idle).
    """
    from matplotlib.patches import Rectangle

    color = "#f0f0f5"
    if event == "grasp":
        color = "#78e68c"
    elif event == "push":
        color = "#f0c86e"
    elif event == "fail":
        color = "#eb4646"

    span = _GRIPPER_SPAN          # overall width of the claw
    finger_len = span * 0.85      # how far the fingers reach toward the target
    finger_thick = span * 0.22    # thick, clearly visible fingers
    hub_h = span * 0.28           # the base/hub the fingers hang from
    open_gap = span * 0.52        # inner gap between fingers when open
    gap = open_gap * 0.35 if event == "grasp" else open_gap  # close on grasp

    # Hub sits "behind" the target (toward -y); fingers extend toward +y so the
    # claw reaches over the item centered at (gx, gy).
    hub_top = gy - finger_len / 2 - hub_h
    hub_w = gap + 2 * finger_thick
    ax.add_patch(
        Rectangle((gx - hub_w / 2, hub_top), hub_w, hub_h,
                  facecolor=color, edgecolor="#141418", linewidth=1.2, zorder=6)
    )
    # Left and right fingers (thick parallel rectangles) extending from the hub.
    fy = hub_top + hub_h
    ax.add_patch(
        Rectangle((gx - gap / 2 - finger_thick, fy), finger_thick, finger_len,
                  facecolor=color, edgecolor="#141418", linewidth=1.2, zorder=6)
    )
    ax.add_patch(
        Rectangle((gx + gap / 2, fy), finger_thick, finger_len,
                  facecolor=color, edgecolor="#141418", linewidth=1.2, zorder=6)
    )


def draw_frame(
    frame: ReplayFrame,
    containers: list[ContainerBox],
    width_px: int = 640,
    height_px: int = 480,
) -> "np.ndarray":
    """Render one replay frame to an (H, W, 3) uint8 RGB array with Agg.

    Headless: forces the matplotlib Agg backend before importing pyplot, so no
    display and no pygame window are needed. The axes span exactly the desk
    bounds (0-640 x, 0-480 y) with equal aspect and no margin, so items and the
    gripper render at a readable size and fill the frame. Each kind is drawn
    with a letter label, a fill colour, and a drawn glyph so colour is never the
    only cue. A ``"fail"`` event highlights only ``frame.event_item``.

    ``width_px`` x ``height_px`` default to the desk's 640x480 (4:3) aspect so
    there is no empty margin shrinking the scene; they are rendered at an
    internal supersample for crispness.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    # Supersample so the frame is crisp; the output array is width_px x height_px
    # because the figure is exactly (width_px/dpi) x (height_px/dpi) inches.
    dpi = 150
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, DESK_W)
    ax.set_ylim(DESK_H, 0)  # invert y so it matches screen coordinates
    ax.set_aspect("equal")  # no distortion; desk 4:3 fills the 4:3 axes
    ax.set_facecolor("#2c303a")
    ax.set_xticks([])
    ax.set_yticks([])

    for c in containers:
        ax.add_patch(
            Rectangle(
                (c.x - c.w / 2, c.y - c.h / 2),
                c.w,
                c.h,
                facecolor="#5a6070",
                edgecolor="none",
            )
        )
        ax.text(
            c.x, c.y, _KIND_LETTER.get(c.kind, "?"),
            color="white", ha="center", va="center", fontsize=9,
        )

    for s in frame.items:  # already low-z first from the trace core
        is_fail = frame.event == "fail" and s.id == frame.event_item
        fill = _FAIL_RGB if is_fail else _KIND_RGB.get(s.kind, (0.8, 0.8, 0.8))
        accent = _KIND_DARK.get(s.kind, (0.1, 0.1, 0.1))
        # Draw a recognizable glyph inside the item's real bounding box.
        _draw_item_glyph(ax, s, fill, accent)
        # Accessible text label (also a fallback if a glyph is hard to read);
        # placed at the top-left of the box so it does not hide the glyph.
        ax.text(
            s.x - s.w / 2 + 2, s.y - s.h / 2 + 2, _KIND_LETTER.get(s.kind, "?"),
            color="#141418", ha="left", va="top", fontsize=8, fontweight="bold",
        )

    # Two-jaw gripper (drawn shapes, no font); closes/highlights on a grasp.
    _draw_gripper(ax, frame.gx, frame.gy, frame.event)

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    # Ensure the promised shape; Agg sizing can round by a pixel at some dpis.
    assert rgb.dtype == np.uint8
    return rgb


def frame_alt_text(frame: ReplayFrame, items_total: int) -> str:
    """Accessible one-line description of a replay frame's state."""
    event = frame.event or "move"
    return (
        f"Tick {frame.tick}: {event}; "
        f"sorted {frame.sorted_count}/{items_total}, "
        f"failed {frame.failed_count}, pushes {frame.push_count}."
    )


def render_frames(result: ReplayResult) -> list[FrameImage]:
    """Render a whole replay to a list of RGB frames with alt text, once.

    Produces one ``FrameImage`` per tick frame so the UI can scrub by indexing
    into the list without re-rendering. Returns an empty list for a replay with
    no frames (e.g. an empty desk).
    """
    return [
        FrameImage(
            rgb=draw_frame(fr, result.containers),
            alt=frame_alt_text(fr, result.items_total),
        )
        for fr in result.frames
    ]
