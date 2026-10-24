"""Save a fresh copy of this Hackachusetts starter source."""
import pathlib
import urllib.parse
import urllib.request

URL = "https://data.boston.gov/api/3/action/datastore_search?resource_id=6fa7932b-7bc8-42bc-9250-168d5f5dc1ad&fields=ST_NAME,ST_TYPE,SPEEDLIMIT,ONEWAY,NBHD_L,ZIP_L,FT_COST,TF_COST,Shape_Length,shape_wkt&limit=100"
request = urllib.request.Request(URL, headers={"User-Agent": "HackachusettsKit"})
with urllib.request.urlopen(request, timeout=90) as response:
    payload = response.read()
suffix = pathlib.Path(urllib.parse.urlparse(URL).path).suffix
path = pathlib.Path("fresh" + suffix) if suffix in {".csv", ".txt", ".json", ".jsonl", ".parquet"} else pathlib.Path("fresh.json")
path.write_bytes(payload)
print(f"wrote {path} ({len(payload)} bytes)")
