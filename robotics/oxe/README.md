# Open X-Embodiment dataset index

Saved 2026-10-18 for Hackachusetts.

Index of datasets in the Open X-Embodiment release: name, robot, episode count, and size. Episode files and model weights are not in this zip. The live link is the official spreadsheet export; data.csv is the dataset table only.

Rows in data.csv: 72

Columns: version, dataset, robot, episodes, size_gb, morphology, gripper, action, language

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/export?format=csv&gid=0

Check the source license before publishing a project.
