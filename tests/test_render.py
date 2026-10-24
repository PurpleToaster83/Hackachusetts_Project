"""Smoke test for the demo renderer.

Renders a short headless episode and checks the output file is written. Kept
small (few items, single planner) so the whole test stays under a second.
"""

from __future__ import annotations

from src.render import build_trace, render


def test_build_trace_matches_sim_outcome() -> None:
    """The replay trace reproduces the sim's sorted/failed/push counts.

    build_trace mirrors run_episode, so the final tick frame's cumulative
    counters must equal what the real sim reports for the same seed/scene.
    """
    from src.planner import OcclusionPlanner
    from src.scene import make_scene
    from src.sim import run_episode

    seed, n_items, clutter = 3, 8, 0.9
    result = run_episode(
        make_scene(seed=seed, n_items=n_items, clutter=clutter),
        OcclusionPlanner(),
        seed=seed,
    )
    trace = build_trace("occlusion", seed=seed, n_items=n_items, clutter=clutter)
    last = trace.frames[-1]
    assert last.sorted_count == result.items_sorted
    assert last.failed_count == result.failed_grasps
    assert last.push_count == result.pushes


def test_headless_render_writes_file(tmp_path) -> None:
    """A short headless render produces a non-empty output file.

    Requests an mp4; accepts either mp4 or the gif fallback (this environment
    may lack an ffmpeg backend). Uses a small scene so the episode is short.
    """
    out = tmp_path / "demo.mp4"
    written = render(
        planners=["random", "occlusion"],
        seed=3,
        n_items=4,
        clutter=0.9,
        out=out,
        headless=True,
    )
    assert written.exists()
    assert written.stat().st_size > 0
    assert written.suffix in (".mp4", ".gif")
