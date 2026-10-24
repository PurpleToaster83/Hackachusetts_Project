"""Demo renderer for the Messy Desk simulation.

Replays episodes and writes an animated video comparing two planners on the
same seed, side by side. This is the ONLY module allowed to import pygame; the
simulation core stays headless. When pygame is unavailable the frames are drawn
with matplotlib (Agg) instead, so the demo still renders.

The replay mirrors :func:`src.sim.run_episode` exactly: the same seeded
generator, the same grasp model (``effective_grasp_prob``), the same push
primitive and tick costs. It records a per-action trace so the gripper motion
can be interpolated tick by tick and grasp / failed-grasp / push events shown.
Because it reuses the real primitives and seeding, the animation reproduces the
sim's outcome for that seed rather than inventing one.

Output is an mp4 via imageio; if mp4 cannot be written (no ffmpeg backend) it
falls back to a gif with the same stem. Geometry is in pixels, time in ticks at
15 Hz.

Usage:
    python src/render.py --seed 3 --out out/demo.mp4
    python src/render.py --headless --seed 3 --planners random,occlusion
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Allow running as ``python src/render.py`` by putting the repo root on path.
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
from src.scene import DESK_H, DESK_W, KINDS, Scene, make_scene  # noqa: E402

# One letter per kind for the on-item label.
KIND_LABEL: dict[str, str] = {"pen": "P", "key": "K", "mug": "M", "trash": "T"}

# RGB colors for each kind (and a few UI colors). 0-255 per channel.
KIND_COLOR: dict[str, tuple[int, int, int]] = {
    "pen": (70, 130, 220),
    "key": (220, 200, 70),
    "mug": (90, 190, 120),
    "trash": (180, 90, 90),
}
BG_COLOR = (28, 30, 36)
DESK_COLOR = (44, 48, 58)
CONTAINER_COLOR = (90, 96, 112)
GRIPPER_COLOR = (240, 240, 245)
FLASH_COLOR = (235, 70, 70)
TEXT_COLOR = (235, 235, 240)

# Playback: how many animation frames per simulated tick (smooths motion).
FRAMES_PER_TICK = 2
FPS = 15 * FRAMES_PER_TICK  # match the 15 Hz sim clock scaled up

# The trace structures and the per-tick stepping logic now live in the
# headless core (src.trace). They are re-exported here so render.py's public
# API is unchanged: existing imports of PLANNER_FACTORIES, build_trace, Trace,
# TickFrame, ItemState, and ContainerState from src.render keep working.
from src.trace import (  # noqa: E402,F401
    PLANNER_FACTORIES,
    ContainerState,
    ItemState,
    TickFrame,
    Trace,
    build_trace,
)
from src.trace import _item_states  # noqa: E402,F401  (used by tests/backends)


# --------------------------------------------------------------------------- #
# Interpolation helper shared by both backends
# --------------------------------------------------------------------------- #
def _interp_gripper(a: TickFrame, b: TickFrame, t: float) -> tuple[float, float]:
    """Linear gripper position between frame ``a`` and ``b`` at fraction ``t``."""
    return (a.gx + (b.gx - a.gx) * t, a.gy + (b.gy - a.gy) * t)


def _frame_pairs(trace: Trace, max_len: int):
    """Yield (prev, cur, sub, n_sub) draw steps, padded/clamped to ``max_len``.

    Produces ``FRAMES_PER_TICK`` interpolation steps between each pair of tick
    frames. When a trace is shorter than ``max_len`` its final frame is held so
    both planners' videos stay the same length.
    """
    n = len(trace.frames)
    for i in range(1, max_len):
        prev = trace.frames[min(i - 1, n - 1)]
        cur = trace.frames[min(i, n - 1)]
        for sub in range(FRAMES_PER_TICK):
            yield prev, cur, sub, FRAMES_PER_TICK


# --------------------------------------------------------------------------- #
# pygame backend (the only pygame import in the whole project)
# --------------------------------------------------------------------------- #
def _render_pygame(
    traces: list[Trace],
    headless: bool,
    panel_w: int,
    panel_h: int,
) -> list[np.ndarray]:
    """Render all traces side by side with pygame, returning RGB frames.

    Each trace gets a ``panel_w`` x ``panel_h`` panel; panels sit in one row.
    When ``headless`` the SDL dummy video driver is used so no window opens.
    Raises ImportError if pygame is not installed (caught by the caller).
    """
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame  # local import: keeps pygame out of the headless core

    pygame.init()
    pygame.font.init()
    label_font = pygame.font.SysFont("monospace", 14, bold=True)
    status_font = pygame.font.SysFont("monospace", 15)
    title_font = pygame.font.SysFont("monospace", 17, bold=True)

    status_h = 54
    total_w = panel_w * len(traces)
    total_h = panel_h + status_h
    surface = pygame.Surface((total_w, total_h))

    # Scale desk coordinates into the drawable area of a panel.
    pad = 10
    draw_w = panel_w - 2 * pad
    draw_h = panel_h - 2 * pad - 24  # leave a title strip at the top
    sx = draw_w / DESK_W
    sy = draw_h / DESK_H

    def to_px(px: float, py: float, panel: int) -> tuple[int, int]:
        """Map desk (px, py) to screen pixels inside panel ``panel``."""
        ox = panel * panel_w + pad
        oy = pad + 24
        return int(ox + px * sx), int(oy + py * sy)

    max_len = max(len(t.frames) for t in traces)
    iters = [_frame_pairs(t, max_len) for t in traces]

    frames: list[np.ndarray] = []
    done = False
    while not done:
        surface.fill(BG_COLOR)
        any_step = False
        for panel, it in enumerate(iters):
            step = next(it, None)
            if step is None:
                # Hold the last tick frame for finished panels.
                trace = traces[panel]
                prev = cur = trace.frames[-1]
                sub, n_sub = 0, FRAMES_PER_TICK
            else:
                any_step = True
                prev, cur, sub, n_sub = step
            _draw_panel_pygame(
                surface,
                traces[panel],
                prev,
                cur,
                sub,
                n_sub,
                panel,
                panel_w,
                panel_h,
                status_h,
                to_px,
                sx,
                sy,
                label_font,
                status_font,
                title_font,
            )
        if not any_step:
            done = True
            break
        # pygame surface is (w, h, 3) with axes swapped vs numpy; transpose.
        arr = pygame.surfarray.array3d(surface)
        frames.append(np.transpose(arr, (1, 0, 2)).copy())

    pygame.quit()
    return frames


def _draw_panel_pygame(
    surface,
    trace: Trace,
    prev: TickFrame,
    cur: TickFrame,
    sub: int,
    n_sub: int,
    panel: int,
    panel_w: int,
    panel_h: int,
    status_h: int,
    to_px,
    sx: float,
    sy: float,
    label_font,
    status_font,
    title_font,
) -> None:
    """Draw one planner's panel for a single animation frame."""
    import pygame

    # Desk background.
    desk_rect = pygame.Rect(
        panel * panel_w + 10, 10 + 24, int(DESK_W * sx), int(DESK_H * sy)
    )
    pygame.draw.rect(surface, DESK_COLOR, desk_rect)

    # Title.
    title = title_font.render(trace.planner_name, True, TEXT_COLOR)
    surface.blit(title, (panel * panel_w + 12, 8))

    # Containers (static).
    for c in trace.containers:
        left, top = to_px(c.x - c.w / 2, c.y - c.h / 2, panel)
        rect = pygame.Rect(left, top, int(c.w * sx), int(c.h * sy))
        pygame.draw.rect(surface, CONTAINER_COLOR, rect, border_radius=4)
        lbl = label_font.render(KIND_LABEL[c.kind], True, TEXT_COLOR)
        surface.blit(lbl, (left + 4, top + 4))

    # Items in z-order (cur.items is already sorted low-z first).
    t_frac = (sub + 1) / n_sub
    for s in cur.items:
        color = KIND_COLOR.get(s.kind, (200, 200, 200))
        flash = cur.event == "fail" and s.id == cur.event_item
        draw_color = FLASH_COLOR if flash else color
        left, top = to_px(s.x - s.w / 2, s.y - s.h / 2, panel)
        rect = pygame.Rect(left, top, int(s.w * sx), int(s.h * sy))
        pygame.draw.rect(surface, draw_color, rect, border_radius=3)
        pygame.draw.rect(surface, (20, 20, 24), rect, width=1, border_radius=3)
        lbl = label_font.render(KIND_LABEL.get(s.kind, "?"), True, (20, 20, 24))
        surface.blit(lbl, (left + 3, top + 1))

    # Gripper, interpolated between prev and cur.
    gx, gy = _interp_gripper(prev, cur, t_frac)
    gpx, gpy = to_px(gx, gy, panel)
    radius = 9
    ring = GRIPPER_COLOR
    if cur.event == "grasp":
        ring = (120, 230, 140)
    elif cur.event == "push":
        ring = (240, 200, 110)
    pygame.draw.circle(surface, ring, (gpx, gpy), radius, width=3)
    pygame.draw.circle(surface, (40, 40, 48), (gpx, gpy), radius - 3)

    # Status line.
    status = (
        f"tick {cur.tick:4d}  sorted {cur.sorted_count:2d}  "
        f"failed {cur.failed_count:2d}  pushes {cur.push_count:2d}"
    )
    txt = status_font.render(status, True, TEXT_COLOR)
    surface.blit(txt, (panel * panel_w + 12, panel_h + 6))


# --------------------------------------------------------------------------- #
# matplotlib fallback (used only when pygame cannot be imported)
# --------------------------------------------------------------------------- #
def _render_matplotlib(
    traces: list[Trace], panel_w: int, panel_h: int
) -> list[np.ndarray]:
    """Render the same side-by-side animation with matplotlib (Agg).

    A lower-fidelity fallback for when pygame is unavailable. Produces RGB
    frames of the same logical content: desk, containers, labelled items,
    gripper, and a status line per panel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    dpi = 100
    fig_w = panel_w * len(traces) / dpi
    fig_h = (panel_h + 54) / dpi

    max_len = max(len(t.frames) for t in traces)
    iters = [_frame_pairs(t, max_len) for t in traces]

    frames: list[np.ndarray] = []
    done = False
    while not done:
        fig, axes = plt.subplots(
            1, len(traces), figsize=(fig_w, fig_h), dpi=dpi
        )
        if len(traces) == 1:
            axes = [axes]
        any_step = False
        for ax, trace, it in zip(axes, traces, iters):
            step = next(it, None)
            if step is None:
                prev = cur = trace.frames[-1]
                sub, n_sub = 0, FRAMES_PER_TICK
            else:
                any_step = True
                prev, cur, sub, n_sub = step
            _draw_panel_mpl(ax, trace, prev, cur, sub, n_sub, Rectangle, Circle)
        if not any_step:
            plt.close(fig)
            done = True
            break
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)
    return frames


def _draw_panel_mpl(ax, trace, prev, cur, sub, n_sub, Rectangle, Circle) -> None:
    """Draw one planner's panel with matplotlib primitives."""
    ax.set_xlim(0, DESK_W)
    ax.set_ylim(DESK_H, 0)  # invert y so it matches screen coordinates
    ax.set_facecolor("#2c303a")
    ax.set_xticks([])
    ax.set_yticks([])

    for c in trace.containers:
        ax.add_patch(
            Rectangle(
                (c.x - c.w / 2, c.y - c.h / 2),
                c.w,
                c.h,
                facecolor="#5a6070",
                edgecolor="none",
            )
        )
        ax.text(c.x, c.y, KIND_LABEL[c.kind], color="white", ha="center",
                va="center", fontsize=9)

    for s in cur.items:
        flash = cur.event == "fail" and s.id == cur.event_item
        color = (
            "#eb4646"
            if flash
            else "#%02x%02x%02x" % KIND_COLOR.get(s.kind, (200, 200, 200))
        )
        ax.add_patch(
            Rectangle(
                (s.x - s.w / 2, s.y - s.h / 2),
                s.w,
                s.h,
                facecolor=color,
                edgecolor="#141418",
                linewidth=0.5,
            )
        )
        ax.text(s.x, s.y, KIND_LABEL.get(s.kind, "?"), color="#141418",
                ha="center", va="center", fontsize=8)

    t_frac = (sub + 1) / n_sub
    gx, gy = _interp_gripper(prev, cur, t_frac)
    ring = "#f0f0f5"
    if cur.event == "grasp":
        ring = "#78e68c"
    elif cur.event == "push":
        ring = "#f0c86e"
    ax.add_patch(Circle((gx, gy), 9, fill=False, edgecolor=ring, linewidth=2))

    ax.set_title(
        f"{trace.planner_name}\n"
        f"tick {cur.tick}  sorted {cur.sorted_count}  "
        f"failed {cur.failed_count}  pushes {cur.push_count}",
        color="white",
        fontsize=9,
    )


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def write_video(frames: list[np.ndarray], out_path: Path) -> Path:
    """Write ``frames`` to ``out_path``, falling back from mp4 to gif.

    Tries the requested path first (typically ``.mp4``). If encoding fails
    (commonly a missing ffmpeg backend), retries with the same stem and a
    ``.gif`` extension. Returns the path actually written.
    """
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(out_path, frames, fps=FPS)
        return out_path
    except Exception as exc:  # noqa: BLE001 - any encoder failure -> gif
        gif_path = out_path.with_suffix(".gif")
        print(
            f"mp4 encode failed ({type(exc).__name__}); "
            f"falling back to gif at {gif_path}"
        )
        # The gif (pillow) writer takes per-frame duration in ms, not fps.
        imageio.mimsave(gif_path, frames, duration=1000.0 / FPS)
        return gif_path


def render(
    planners: list[str],
    seed: int,
    n_items: int,
    clutter: float,
    out: str | Path,
    headless: bool = False,
    panel_w: int = 420,
    panel_h: int = 340,
) -> Path:
    """Build traces for each planner, render side by side, and write the video.

    Runs every named planner on the same ``seed`` / ``n_items`` / ``clutter``
    and draws them in one row. Uses pygame when importable, otherwise the
    matplotlib fallback. Returns the written file path (mp4 or gif).
    """
    traces = [
        build_trace(name, seed=seed, n_items=n_items, clutter=clutter)
        for name in planners
    ]

    try:
        frames = _render_pygame(traces, headless, panel_w, panel_h)
    except ImportError:
        print("pygame unavailable; using matplotlib fallback for frames")
        frames = _render_matplotlib(traces, panel_w, panel_h)

    if not frames:
        raise RuntimeError("no frames were produced")
    return write_video(frames, Path(out))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the demo CLI options."""
    parser = argparse.ArgumentParser(description="Messy Desk demo renderer.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n-items", type=int, default=15, dest="n_items")
    parser.add_argument("--clutter", type=float, default=0.9)
    parser.add_argument(
        "--planners",
        type=str,
        default="random,occlusion",
        help="comma-separated planner names drawn side by side",
    )
    parser.add_argument("--out", type=str, default="out/demo.mp4")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="use the SDL dummy video driver (no window)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: render the demo and report the output file."""
    args = _parse_args(argv)
    names = [p.strip() for p in args.planners.split(",") if p.strip()]
    unknown = [n for n in names if n not in PLANNER_FACTORIES]
    if unknown:
        raise SystemExit(
            f"unknown planner(s): {', '.join(unknown)}. "
            f"choose from {', '.join(PLANNER_FACTORIES)}"
        )
    out_path = render(
        planners=names,
        seed=args.seed,
        n_items=args.n_items,
        clutter=args.clutter,
        out=args.out,
        headless=args.headless,
    )
    size = out_path.stat().st_size if out_path.exists() else 0
    print(f"wrote {out_path} ({size} bytes)")


if __name__ == "__main__":
    main()
