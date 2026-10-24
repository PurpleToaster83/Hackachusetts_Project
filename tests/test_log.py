"""Tests for the dataset logger and writer."""

from __future__ import annotations

import pandas as pd

from src.log import (
    EPISODE_COLUMNS,
    FRAME_COLUMNS,
    EpisodeLogger,
    write_dataset,
)
from src.planner import NearestPlanner
from src.robot import TICK_HZ
from src.scene import make_scene
from src.sim import run_episode


def _logged_episode(seed: int = 3, n_items: int = 6, clutter: float = 0.6):
    """Run one episode with a logger attached and return (result, logger)."""
    scene = make_scene(seed=seed, n_items=n_items, clutter=clutter)
    logger = EpisodeLogger(task="put the pens in the tray")
    result = run_episode(scene, NearestPlanner(), logger=logger, seed=seed)
    return result, logger


def test_frame_columns_match_schema() -> None:
    """A logged episode's data file carries exactly the schema columns, in order."""
    _, logger = _logged_episode()
    df = logger.to_frame()
    assert list(df.columns) == list(FRAME_COLUMNS)


def test_empty_logger_has_schema_columns() -> None:
    """An episode with no frames still yields the schema columns."""
    df = EpisodeLogger(task="nothing to do").to_frame()
    assert list(df.columns) == list(FRAME_COLUMNS)
    assert len(df) == 0


def test_last_frame_is_marked_done() -> None:
    """Only the final frame has done=True; earlier frames are False."""
    _, logger = _logged_episode()
    df = logger.to_frame()
    assert len(df) > 1
    assert bool(df.iloc[-1]["done"]) is True
    assert not df.iloc[:-1]["done"].any()


def test_frame_count_matches_actions() -> None:
    """One frame is logged per action (grasps plus pushes)."""
    result, logger = _logged_episode()
    expected_actions = result.items_sorted + result.failed_grasps + result.pushes
    assert len(logger.frames) == expected_actions


def test_reward_is_one_only_on_successful_grasps() -> None:
    """Total reward equals items sorted; failed grasps and pushes give 0."""
    result, logger = _logged_episode()
    df = logger.to_frame()
    assert df["reward"].sum() == float(result.items_sorted)
    pushes = df[df["action_type"] == "push"]
    assert (pushes["reward"] == 0.0).all()


def test_timestamp_tracks_tick_clock() -> None:
    """Timestamps are in seconds (tick/TICK_HZ) and never decrease."""
    _, logger = _logged_episode()
    df = logger.to_frame()
    assert (df["timestamp"].diff().dropna() >= 0).all()
    # Each timestamp is a whole number of ticks divided by the tick rate.
    for ts in df["timestamp"]:
        assert ts == round(round(ts * TICK_HZ) / TICK_HZ, 6)


def test_write_dataset_layout_and_meta(tmp_path) -> None:
    """write_dataset creates the meta index and one data file per episode."""
    loggers = []
    for seed in range(3):
        scene = make_scene(seed=seed, n_items=5, clutter=0.5)
        lg = EpisodeLogger(task=f"sort desk {seed}")
        run_episode(scene, NearestPlanner(), logger=lg, seed=seed)
        loggers.append(lg)

    root = write_dataset(loggers, out_dir=tmp_path)

    meta_path = root / "meta" / "episodes.csv"
    assert meta_path.exists()
    meta = pd.read_csv(meta_path)
    assert list(meta.columns) == list(EPISODE_COLUMNS)
    assert len(meta) == 3

    for index, lg in enumerate(loggers):
        data_path = root / "data" / f"episode_{index:06d}.csv"
        assert data_path.exists()
        data = pd.read_csv(data_path)
        assert list(data.columns) == list(FRAME_COLUMNS)
        # frames count in meta matches rows in the data file.
        assert int(meta.iloc[index]["frames"]) == len(data)
        assert meta.iloc[index]["task"] == lg.task


def test_write_dataset_is_deterministic(tmp_path) -> None:
    """Same seed produces byte-identical dataset files."""
    def build(dirname: str):
        scene = make_scene(seed=7, n_items=8, clutter=0.7)
        lg = EpisodeLogger(task="sort it")
        run_episode(scene, NearestPlanner(), logger=lg, seed=7)
        return write_dataset([lg], out_dir=tmp_path / dirname)

    root_a = build("a")
    root_b = build("b")
    a = (root_a / "data" / "episode_000000.csv").read_text()
    b = (root_b / "data" / "episode_000000.csv").read_text()
    assert a == b
