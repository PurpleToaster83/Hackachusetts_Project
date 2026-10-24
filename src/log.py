"""Dataset logging for the Messy Desk simulation.

Records each episode as a sequence of frames and writes them to disk in the
BridgeData V2 / DROID shape so runs double as a reusable robotics dataset.

Headless: no pygame, no global random state. One frame is logged per gripper
action; ``timestamp`` is derived from the tick clock at 15 Hz. Geometry is in
pixels, time in ticks/seconds.

Layout written by :func:`write_dataset` under ``out_dir`` (default ``out``):

- ``dataset/meta/episodes.csv``  columns: ``episode,task,frames``
- ``dataset/data/episode_{index:06d}.csv``  columns:
  ``frame_index,timestamp,state_x,state_y,holding,action_type,action_x,``
  ``action_y,reward,done``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.robot import TICK_HZ

# Column order for the per-episode data files. Kept as a constant so the
# dataset writer and the tests share one source of truth for the schema.
FRAME_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "timestamp",
    "state_x",
    "state_y",
    "holding",
    "action_type",
    "action_x",
    "action_y",
    "reward",
    "done",
)

# Column order for the episode index file (matches bridge-v2 / droid meta).
EPISODE_COLUMNS: tuple[str, ...] = ("episode", "task", "frames")


@dataclass
class Frame:
    """One logged timestep: the state before an action and the action taken.

    ``timestamp`` is seconds since the episode start (``tick / TICK_HZ``).
    ``state_x``/``state_y`` are the gripper position in pixels at the moment the
    action is issued. ``holding`` is the held item id or -1 when empty.
    ``action_type`` is ``"grasp"`` or ``"push"``. ``action_x``/``action_y`` are
    the action target: the item center for a grasp, the push delta for a push,
    in pixels. ``reward`` is 1.0 for a successful grasp, else 0.0. ``done`` is
    True on the final frame of the episode.
    """

    frame_index: int
    timestamp: float  # seconds since episode start
    state_x: float  # gripper x, pixels
    state_y: float  # gripper y, pixels
    holding: int  # held item id, or -1 when empty
    action_type: str  # "grasp" or "push"
    action_x: float  # grasp target x or push dx, pixels
    action_y: float  # grasp target y or push dy, pixels
    reward: float  # 1.0 on a successful grasp, else 0.0
    done: bool  # True on the last frame of the episode


@dataclass
class EpisodeLogger:
    """Collects frames for one episode and the natural-language task string.

    ``task`` is the instruction phrasing for the episode (bridge-v2 / droid
    style). Append one frame per action with :meth:`log_frame`; the writer
    marks the last frame's ``done`` True.
    """

    task: str
    frames: list[Frame] = field(default_factory=list)

    def log_frame(
        self,
        *,
        tick: int,
        state_x: float,
        state_y: float,
        holding: int | None,
        action_type: str,
        action_x: float,
        action_y: float,
        reward: float,
    ) -> None:
        """Record one action frame.

        ``tick`` is the simulation tick when the action is issued; the frame's
        ``timestamp`` is ``tick / TICK_HZ`` seconds. ``holding`` of ``None`` is
        stored as -1. ``done`` is left False here and set True on the final
        frame by :func:`write_dataset`.
        """
        self.frames.append(
            Frame(
                frame_index=len(self.frames),
                timestamp=round(tick / TICK_HZ, 6),
                state_x=float(state_x),
                state_y=float(state_y),
                holding=-1 if holding is None else int(holding),
                action_type=action_type,
                action_x=float(action_x),
                action_y=float(action_y),
                reward=float(reward),
                done=False,
            )
        )

    def to_frame(self) -> pd.DataFrame:
        """Return this episode's frames as a DataFrame in schema column order.

        The last frame's ``done`` is set True. An episode with no frames yields
        an empty DataFrame that still carries the schema columns.
        """
        if not self.frames:
            return pd.DataFrame(columns=list(FRAME_COLUMNS))
        self.frames[-1].done = True
        rows = [
            {
                "frame_index": f.frame_index,
                "timestamp": f.timestamp,
                "state_x": f.state_x,
                "state_y": f.state_y,
                "holding": f.holding,
                "action_type": f.action_type,
                "action_x": f.action_x,
                "action_y": f.action_y,
                "reward": f.reward,
                "done": f.done,
            }
            for f in self.frames
        ]
        return pd.DataFrame(rows, columns=list(FRAME_COLUMNS))


def write_dataset(
    loggers: list[EpisodeLogger],
    out_dir: str | Path = "out",
) -> Path:
    """Write ``loggers`` to disk as a dataset and return the dataset root path.

    Creates ``out_dir/dataset/meta/episodes.csv`` (one row per episode:
    ``episode,task,frames``) and one ``out_dir/dataset/data/episode_{i:06d}.csv``
    per logger in schema column order. Directories are created as needed.
    Episodes are indexed by their position in ``loggers``.
    """
    root = Path(out_dir) / "dataset"
    meta_dir = root / "meta"
    data_dir = root / "data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict[str, object]] = []
    for index, logger in enumerate(loggers):
        frame_df = logger.to_frame()
        frame_df.to_csv(data_dir / f"episode_{index:06d}.csv", index=False)
        meta_rows.append(
            {"episode": index, "task": logger.task, "frames": len(frame_df)}
        )

    meta_df = pd.DataFrame(meta_rows, columns=list(EPISODE_COLUMNS))
    meta_df.to_csv(meta_dir / "episodes.csv", index=False)
    return root
