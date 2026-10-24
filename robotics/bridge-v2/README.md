# BridgeData V2 demonstrations

Saved 2026-10-18 for Hackachusetts.

Every demonstration in BridgeData V2, the WidowX dataset from UC Berkeley: episode number, the language task, and the frame count. 53,192 episodes. The videos and the parquet trajectories are not in this zip. The live link is the episode table from the public LeRobot release.

Rows in data.csv: 53192

Columns: episode, task, frames

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot/resolve/main/meta/episodes.jsonl

Check the source license before publishing a project.
