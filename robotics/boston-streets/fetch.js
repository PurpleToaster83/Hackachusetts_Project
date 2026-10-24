// Save a fresh copy of this Hackachusetts starter source.
// Run with: node fetch.js
const fs = require("node:fs/promises");

const url = "https://data.boston.gov/api/3/action/datastore_search?resource_id=6fa7932b-7bc8-42bc-9250-168d5f5dc1ad&fields=ST_NAME,ST_TYPE,SPEEDLIMIT,ONEWAY,NBHD_L,ZIP_L,FT_COST,TF_COST,Shape_Length,shape_wkt&limit=100";

async function main() {
  const response = await fetch(url, { headers: { "User-Agent": "HackachusettsKit" } });
  if (!response.ok) throw new Error("Request failed: " + response.status);
  const bytes = Buffer.from(await response.arrayBuffer());
  const type = response.headers.get("content-type") || "";
  let name = "fresh.json";
  if (url.includes(".csv") || type.includes("csv")) name = "fresh.csv";
  if (url.endsWith(".txt")) name = "fresh.txt";
  if (url.endsWith(".jsonl")) name = "fresh.jsonl";
  if (url.endsWith(".parquet")) name = "fresh.parquet";
  await fs.writeFile(name, bytes);
  console.log("wrote " + name + " (" + bytes.length + " bytes)");
}

main();
