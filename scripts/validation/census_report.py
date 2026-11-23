"""Print a census report: dispatches per step, by file and by site."""

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"{path}: not written")
    raise SystemExit(1)
c = json.loads(path.read_text())
total = c["dispatches_per_step"]
print("dispatches_per_step =", total)
print("syncs_per_step      =", c["device_to_host_syncs_per_step"])
files = {}
for row in c["by_site"]:
    name = row["site"].split(":")[0]
    files[name] = files.get(name, 0) + row["per_step"]
print("--- by file, per step ---")
for name, count in sorted(files.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {name:44s}{count:9d}{100 * count / total:6.1f}%")
print("--- top sites ---")
for row in c["by_site"][:12]:
    print(f"  {row['site']:44s}{row['per_step']:9d}  written={row['written_on_this_line']}")
print("--- top operators ---")
for row in c["by_operator"][:10]:
    print(f"  {row['operator']:44s}{row['per_step']:9d}{100 * row['per_step'] / total:6.1f}%")
