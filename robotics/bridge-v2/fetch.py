"""Save a fresh copy of this Hackachusetts starter source."""
import pathlib
import urllib.parse
import urllib.request

URL = "https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot/resolve/main/meta/episodes.jsonl"
request = urllib.request.Request(URL, headers={"User-Agent": "HackachusettsKit"})
with urllib.request.urlopen(request, timeout=90) as response:
    payload = response.read()
suffix = pathlib.Path(urllib.parse.urlparse(URL).path).suffix
path = pathlib.Path("fresh" + suffix) if suffix in {".csv", ".txt", ".json", ".jsonl", ".parquet"} else pathlib.Path("fresh.json")
path.write_bytes(payload)
print(f"wrote {path} ({len(payload)} bytes)")
