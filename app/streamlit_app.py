"""Messy Desk Lab — Streamlit app (UI only).

This module holds ONLY Streamlit UI: layout, controls, session state, spinners,
and error surfacing. All computation goes through ``app.logic``; this file
never imports the ``src/`` simulation math directly, nor ``pygame``/``render``.

Matplotlib's Agg backend is forced before any ``pyplot`` import so the app runs
headless on Streamlit Community Cloud (no display, no pygame window).

Run locally:
    .venv/bin/python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

# Make the repository root importable regardless of the working directory.
# Streamlit Community Cloud runs this script from a different cwd than a local
# `streamlit run`, so `from app import logic` (and logic.py's own `import
# src.*`) would otherwise fail with ModuleNotFoundError. Adding the repo root
# (the parent of this file's `app/` directory) to sys.path makes both `app` and
# `src` resolve in either environment.
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Force the headless matplotlib backend BEFORE anything imports pyplot. Stage 2
# draws no figures yet, but later sections will; setting this at import time
# guarantees no module pulls in a GUI backend on the Cloud container.
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import streamlit as st

from app import logic

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_RESULT_KEY = "experiment_result"  # st.session_state cache key
_REPLAY_KEY = "live_view_replay"  # cached (TrialConfig-key, frames) for Live View
_FRAME_KEY = "live_view_frame"  # current frame index
_PLAYING_KEY = "live_view_playing"  # play/pause flag

# Human-readable metric labels (text, not colour, carries meaning).
_METRIC_LABELS: dict[str, str] = {
    "ticks": "Ticks to clear",
    "sorted": "Items sorted",
    "failed": "Failed grasps",
    "pushes": "Pushes",
}


# --------------------------------------------------------------------------- #
# Sidebar: Experiment controls
# --------------------------------------------------------------------------- #
def _experiment_controls() -> logic.ExperimentConfig:
    """Render the sidebar controls and return the chosen ExperimentConfig."""
    st.sidebar.header("Experiment")

    n_items = st.sidebar.slider(
        "Items on the desk (n_items)",
        min_value=3,
        max_value=logic.N_ITEMS_MAX,
        value=15,
        help="How many items to scatter on each 640x480 desk.",
    )
    clutter = st.sidebar.slider(
        "Clutter",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Higher clutter pulls items together, causing more overlap.",
    )
    n_trials = st.sidebar.slider(
        "Trials (seeds per planner)",
        min_value=2,
        max_value=50,
        value=20,
        help="Each trial is one seeded desk; more trials tighten the CIs.",
    )
    base_seed = st.sidebar.number_input(
        "Base seed",
        min_value=0,
        value=0,
        step=1,
        help="Seeds run base_seed .. base_seed + trials - 1.",
    )
    planners = st.sidebar.multiselect(
        "Planners to compare",
        options=logic.available_planners(),
        default=["nearest", "occlusion"],
        help="Pick two or more planners to compare on identical desks.",
    )

    # Show each selected planner's one-line description as plain text.
    for name in planners:
        st.sidebar.caption(f"**{name}** — {logic.planner_description(name)}")

    seeds = list(range(int(base_seed), int(base_seed) + int(n_trials)))
    return logic.ExperimentConfig(
        n_items=int(n_items),
        clutter=float(clutter),
        seeds=seeds,
        planners=planners,
    )


# --------------------------------------------------------------------------- #
# Results rendering
# --------------------------------------------------------------------------- #
def _stats_table(result: logic.ExperimentResult) -> pd.DataFrame:
    """Build a per-planner table of 'mean +/- CI' strings for each metric."""
    rows: list[dict[str, str]] = []
    for ps in result.stats:
        row: dict[str, str] = {"Planner": ps.planner, "Trials": str(ps.n_trials)}
        for key, label in _METRIC_LABELS.items():
            stat: logic.Stat = getattr(ps, key)
            row[label] = f"{stat.mean:.1f} +/- {stat.ci:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _tie_notes(result: logic.ExperimentResult) -> list[str]:
    """For each metric, label the first two planners 'tie' when CIs overlap.

    Returns one plain-text line per metric describing whether the first two
    selected planners' 95% intervals overlap (a tie at this sample size) or
    whether one wins. Uses the Stage 1 ``overlap`` rule; real numbers only.
    """
    notes: list[str] = []
    if len(result.stats) < 2:
        return notes
    a, b = result.stats[0], result.stats[1]
    for key, label in _METRIC_LABELS.items():
        sa: logic.Stat = getattr(a, key)
        sb: logic.Stat = getattr(b, key)
        if logic.overlap(sa, sb):
            notes.append(
                f"{label}: tie ({a.planner} vs {b.planner}) — 95% intervals overlap."
            )
        else:
            winner = a.planner if sa.mean < sb.mean else b.planner
            notes.append(
                f"{label}: {winner} lower ({a.planner} vs {b.planner}) — "
                "intervals do not overlap."
            )
    return notes


def _render_results(result: logic.ExperimentResult) -> None:
    """Render the Results section from a cached ExperimentResult."""
    st.subheader("Results")
    st.caption(
        "Per-planner mean +/- 95% confidence interval over the experiment's "
        "seeds, from real episode runs. 'tie' means the two planners' intervals "
        "overlap at this sample size."
    )

    table = _stats_table(result)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    notes = _tie_notes(result)
    if notes:
        st.markdown("**Head-to-head (first two planners):**")
        for line in notes:
            st.markdown(f"- {line}")


# --------------------------------------------------------------------------- #
# Live View
# --------------------------------------------------------------------------- #
def _trial_label(row: logic.TrialRow) -> str:
    """Human-readable label for one trial in the picker."""
    c = row.config
    r = row.result
    return (
        f"{c.planner_name} | seed {c.seed} | "
        f"sorted {r.items_sorted}/{r.items_total}, {r.steps} ticks"
    )


def _get_or_build_replay(row: logic.TrialRow):
    """Return the cached FrameImages for ``row``, rendering once if needed.

    The render is keyed by the trial's config so changing the picked trial
    re-renders, but scrubbing the slider does not. Frames are cached in
    ``st.session_state`` so playback indexes into them without recomputing.
    """
    key = (
        row.config.planner_name,
        row.config.seed,
        row.config.n_items,
        row.config.clutter,
    )
    cached = st.session_state.get(_REPLAY_KEY)
    if cached is None or cached[0] != key:
        with st.spinner("Rendering replay..."):
            replay = logic.replay_trial(row.config)
            images = logic.render_frames(replay)
        st.session_state[_REPLAY_KEY] = (key, replay, images)
        st.session_state[_FRAME_KEY] = 0
        st.session_state[_PLAYING_KEY] = False
        cached = st.session_state[_REPLAY_KEY]
    return cached[1], cached[2]


def _render_live_view(result: logic.ExperimentResult) -> None:
    """Render the Live View section: pick a trial and scrub its replay."""
    st.subheader("Live View")
    st.caption(
        "Replay a single trial tick by tick from pre-rendered headless frames. "
        "Pick a trial, then play, pause, step, or scrub."
    )
    st.caption(
        "Top-down view of a kinematic sorting simulation "
        "(no 3D physics or real hardware)."
    )

    rows = result.rows
    if not rows:
        st.info("Run an experiment first to populate trials.")
        return

    idx = st.selectbox(
        "Trial to replay",
        options=list(range(len(rows))),
        format_func=lambda i: _trial_label(rows[i]),
    )
    row = rows[idx]

    try:
        replay, images = _get_or_build_replay(row)
    except Exception as exc:  # noqa: BLE001 - surface any replay/render failure
        st.error(f"Could not render this trial: {exc}")
        return

    if replay.items_total == 0 or not images:
        st.info("No items on this desk for that seed — try another trial.")
        return

    n = len(images)
    frame_i = int(st.session_state.get(_FRAME_KEY, 0))
    frame_i = max(0, min(frame_i, n - 1))

    # Transport controls.
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Play"):
        st.session_state[_PLAYING_KEY] = True
    if c2.button("Pause"):
        st.session_state[_PLAYING_KEY] = False
    if c3.button("Step back"):
        st.session_state[_PLAYING_KEY] = False
        frame_i = max(0, frame_i - 1)
    if c4.button("Step forward"):
        st.session_state[_PLAYING_KEY] = False
        frame_i = min(n - 1, frame_i + 1)

    speed = st.select_slider(
        "Speed (frames advanced per tick while playing)",
        options=[1, 2, 4, 8],
        value=2,
    )

    frame_i = st.slider("Frame", min_value=0, max_value=max(0, n - 1), value=frame_i)
    st.session_state[_FRAME_KEY] = frame_i

    img = images[frame_i]
    st.image(img.rgb, caption=img.alt, use_container_width=False)

    # Live counters for the current frame (text, not colour alone).
    fr = replay.frames[frame_i]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sorted", f"{fr.sorted_count}/{replay.items_total}")
    m2.metric("Ticks", fr.tick)
    m3.metric("Failed grasps", fr.failed_count)
    m4.metric("Pushes", fr.push_count)

    # Playback: advance and rerun until the end, then stop.
    if st.session_state.get(_PLAYING_KEY) and frame_i < n - 1:
        import time

        st.session_state[_FRAME_KEY] = min(n - 1, frame_i + int(speed))
        time.sleep(1.0 / 15.0)
        st.rerun()
    elif frame_i >= n - 1:
        st.session_state[_PLAYING_KEY] = False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the Experiment + Results app."""
    st.set_page_config(page_title="Messy Desk Lab", layout="wide")
    st.title("Messy Desk Lab")
    st.write(
        "Compare desk-sorting planners on randomized cluttered desks. "
        "Configure an experiment in the sidebar and run it to see metrics "
        "with 95% confidence intervals."
    )

    config = _experiment_controls()

    run = st.sidebar.button("Run experiment", type="primary")
    if run:
        if len(config.planners) < 1:
            st.error("Pick at least one planner to run.")
        else:
            try:
                with st.spinner("Running simulation..."):
                    result = logic.run_experiment(config)
                st.session_state[_RESULT_KEY] = result
            except Exception as exc:  # noqa: BLE001 - surface any run failure
                st.error(f"Something went wrong running the experiment: {exc}")

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info("Set up an experiment in the sidebar and click **Run experiment**.")
        return

    _render_results(result)
    st.divider()
    _render_live_view(result)


if __name__ == "__main__":
    main()
