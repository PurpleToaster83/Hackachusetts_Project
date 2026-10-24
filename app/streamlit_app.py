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


if __name__ == "__main__":
    main()
