"""Save a fresh copy of this Hackachusetts starter source."""
import pathlib
import urllib.parse
import urllib.request

URL = "https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/export?format=csv&gid=0"
request = urllib.request.Request(URL, headers={"User-Agent": "HackachusettsKit"})
with urllib.request.urlopen(request, timeout=90) as response:
    payload = response.read()
suffix = pathlib.Path(urllib.parse.urlparse(URL).path).suffix
path = pathlib.Path("fresh" + suffix) if suffix in {".csv", ".txt", ".json"} else pathlib.Path("fresh.json")
path.write_bytes(payload)
print(f"wrote {path} ({len(payload)} bytes)")
