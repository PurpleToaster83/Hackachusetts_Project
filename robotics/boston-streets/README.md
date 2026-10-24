# Boston street centerlines

Saved 2026-10-18 for Hackachusetts.

All 19,440 Boston street centerlines from the city's SAM street file. Street name, speed limit, one-way flag, neighborhood, travel cost, and the road geometry. House-number ranges are left out. The live call returns 100 segments. data.csv is the full saved network.

Rows in data.csv: 19440

Columns: street, type, speed_mph, oneway, neighborhood, zip, cost_forward, cost_back, shape_length, geometry

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://data.boston.gov/api/3/action/datastore_search?resource_id=6fa7932b-7bc8-42bc-9250-168d5f5dc1ad&fields=ST_NAME,ST_TYPE,SPEEDLIMIT,ONEWAY,NBHD_L,ZIP_L,FT_COST,TF_COST,Shape_Length,shape_wkt&limit=100

Check the source license before publishing a project.
