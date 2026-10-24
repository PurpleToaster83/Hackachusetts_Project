# DROID-100 Franka demonstrations

Saved 2026-10-18 for Hackachusetts.

The public DROID-100 release: 100 real Franka Panda demonstrations at 15 Hz, with a language task and a duration. This is not the 8 TB raw stereo archive. The live link is the episode table. Cite Khazatsky et al., 2024.

Rows in data.csv: 100

Columns: episode, task, frames, seconds

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://huggingface.co/datasets/lerobot/droid_100/resolve/main/meta/episodes/chunk-000/file-000.parquet

Check the source license before publishing a project.
