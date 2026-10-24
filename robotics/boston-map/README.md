# Boston street grid, 256

Saved 2026-10-18 for Hackachusetts.

One 256 by 256 Boston street grid from the Moving AI lab pathfinding benchmarks. 1 is open, 0 is blocked. The live link is the lab's zip of that map. Cite Sturtevant, Benchmarks for Grid-Based Pathfinding, IEEE TCIAIG 2012.

Rows in data.csv: 65536

Columns: y, x, open

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://movingai.com/benchmarks/street/Boston_0_256.map.zip

Check the source license before publishing a project.
