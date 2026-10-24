"""Smoke tests for the evaluation sweeps."""

from __future__ import annotations

import src.planner as planner_mod
import src.robot as robot_mod
import src.sim as sim_mod
from src.run_eval import override_constants, run_eval


def test_quick_run_writes_all_artifacts(tmp_path) -> None:
    """A --quick run produces the four plots and results.md, all non-empty."""
    out_root = run_eval(seeds=5, quick=True, out_dir=tmp_path)

    results = out_root / "results.md"
    assert results.exists() and results.stat().st_size > 0
    text = results.read_text()
    # The markdown carries the expected section headings and a seed count.
    assert "evaluation results" in text
    assert "5 seeds" in text
    assert "Sensitivity" in text
    assert "ablation" in text

    plots = out_root / "plots"
    for name in (
        "ticks_vs_items.png",
        "failed_vs_items.png",
        "sensitivity.png",
        "ablation.png",
    ):
        plot = plots / name
        assert plot.exists() and plot.stat().st_size > 0


def test_override_constants_restores_globals() -> None:
    """Constant overrides apply inside the block and restore on exit."""
    orig_ft_robot = robot_mod.FAILED_GRASP_TICKS
    orig_ft_sim = sim_mod.FAILED_GRASP_TICKS
    orig_ft_planner = planner_mod.FAILED_GRASP_TICKS
    orig_cp_robot = robot_mod.CLEARANCE_PENALTY
    orig_cp_planner = planner_mod.CLEARANCE_PENALTY

    with override_constants(failed_grasp_ticks=99, clearance_penalty=0.1):
        assert robot_mod.FAILED_GRASP_TICKS == 99
        assert sim_mod.FAILED_GRASP_TICKS == 99
        assert planner_mod.FAILED_GRASP_TICKS == 99
        assert robot_mod.CLEARANCE_PENALTY == 0.1
        assert planner_mod.CLEARANCE_PENALTY == 0.1

    assert robot_mod.FAILED_GRASP_TICKS == orig_ft_robot
    assert sim_mod.FAILED_GRASP_TICKS == orig_ft_sim
    assert planner_mod.FAILED_GRASP_TICKS == orig_ft_planner
    assert robot_mod.CLEARANCE_PENALTY == orig_cp_robot
    assert planner_mod.CLEARANCE_PENALTY == orig_cp_planner


def test_quick_run_is_deterministic(tmp_path) -> None:
    """Two --quick runs produce byte-identical results.md."""
    a = run_eval(seeds=5, quick=True, out_dir=tmp_path / "a")
    b = run_eval(seeds=5, quick=True, out_dir=tmp_path / "b")
    assert (a / "results.md").read_text() == (b / "results.md").read_text()
