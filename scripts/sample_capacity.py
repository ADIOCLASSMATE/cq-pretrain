"""Record whole-node availability on both clusters, so the free-tier question
gets days of evidence instead of one snapshot.

The per-point comparison between the two accelerators assumes both run paid.
That assumption only holds if neither cluster's free preemptible tier is
reachable. On 2026-11-24 Ascend showed zero whole idle nodes on every check
while L40S reached six, which would make free L40S hours beat paid Ascend
hours whatever the per-card speed is. One snapshot cannot settle that, so this
samples both every few minutes and appends a line per sample.

Read-only: it calls `resources availability` and writes one JSONL file. It
never submits, stops or touches a job, and runs beside the controller rather
than inside it, so a bug here cannot disturb training.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CLI = [str(Path.home() / ".local/bin/inspire"), "--json"]
WATCH = (("昇腾卡公共空间", "910B资源", 2), ("可上网GPU资源", "L40s-cuda13.2", 4))
OUT = Path(sys.argv[1])
PERIOD = int(sys.argv[2]) if len(sys.argv) > 2 else 300


def sample(workspace, group):
    try:
        reply = subprocess.run(
            [*CLI, "resources", "availability", "--workspace", workspace],
            capture_output=True, text=True, timeout=120,
        )
        items = json.loads(reply.stdout or reply.stderr)["data"]["items"]
        row = next(r for r in items if r["compute_group"] == group)
        return dict(free_nodes=int(row.get("full_free_nodes") or 0), free_gpus=int(row.get("available_gpus") or 0))
    except Exception as error:
        return dict(error=f"{type(error).__name__}: {error}"[:200])


OUT.parent.mkdir(parents=True, exist_ok=True)
while True:
    stamp = datetime.now(timezone.utc)
    record = dict(time=stamp.isoformat(timespec="seconds"), beijing_hour=(stamp.hour + 8) % 24)
    for workspace, group, nodes_needed in WATCH:
        reading = sample(workspace, group)
        if "free_nodes" in reading:
            # The controller asks for one or two nodes of margin beyond its own.
            reading["enough_for_low"] = reading["free_nodes"] >= nodes_needed + 1
        record[group] = reading
    with OUT.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    time.sleep(PERIOD)
