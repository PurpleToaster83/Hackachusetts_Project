# Massachusetts manufacturing and warehousing employment

Saved 2026-10-18 for Hackachusetts.

BLS Quarterly Census of Employment and Wages, Massachusetts statewide, 2024 annual averages, private ownership. Manufacturing subsectors and warehousing. The live link is the full state file. data.csv keeps the summary industries.

Rows in data.csv: 24

Columns: industry_code, industry, establishments, employment, avg_annual_pay

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://data.bls.gov/cew/data/api/2024/a/area/25000.csv

Check the source license before publishing a project.
