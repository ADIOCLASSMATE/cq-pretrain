"""Read-only live workload estimate and batch-plan preview; not a cutover manifest."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from data.preprocess.manifest import atomic_json, atomic_jsonl
from data.preprocess.parallel_feasibility.probe import PRODUCTION, PROBE

selected = {}
with (PRODUCTION / "metadata/selected.jsonl").open() as handle:
    for line in handle:
        row = json.loads(line)
        selected[row["sample_id"]] = (row["source"]["part"], row["source"]["num_frames"])
latest = {}
for path in sorted((PRODUCTION / "_state").glob("*.jsonl")):
    with path.open() as handle:
        for line in handle:
            if not line.endswith("\n"):
                break  # A live writer may be appending this last record.
            event = json.loads(line)
            latest[event["sample_id"], event["variant"]] = (event["status"], event["updated_at"])
now = datetime.now(timezone.utc)
completed = {}
pending = defaultdict(list)
for name, (part, frames) in selected.items():
    events = [latest.get((name, v)) for v in ("960x544", "832x480")]
    if all(e and e[0] == "success" for e in events):
        completed[name] = max(datetime.fromisoformat(e[1]) for e in events)
    else:
        pending[part].append((name, frames))
remaining_frames = sum(frames for rows in pending.values() for _, frames in rows)
report = dict(checked_at=now.isoformat(), live_snapshot_not_atomic=True,
              completed_samples=len(completed), pending_samples=len(selected)-len(completed),
              completed_source_frames=sum(selected[name][1] for name in completed),
              pending_source_frames=remaining_frames,
              completed_mean_source_frames=sum(selected[name][1] for name in completed)/len(completed),
              pending_mean_source_frames=remaining_frames/(len(selected)-len(completed)),
              recent={})
for minutes in (30, 60, 120):
    names = [name for name, stamp in completed.items() if stamp >= now-timedelta(minutes=minutes)]
    frames = sum(selected[name][1] for name in names)
    fps = frames/(minutes*60)
    report["recent"][str(minutes)] = dict(samples=len(names), source_frames=frames,
        source_frames_per_second=fps, projected_remaining_hours=remaining_frames/fps/3600)
batches = []
for part, rows in sorted(pending.items()):
    names, frames = [], 0
    for name, count in sorted(rows):
        if names and (len(names) >= 128 or frames+count > 32000):
            batches.append(dict(part=part, sample_ids=names, source_frames=frames))
            names, frames = [], 0
        names.append(name); frames += count
    if names:
        batches.append(dict(part=part, sample_ids=names, source_frames=frames))
report.update(batch_count=len(batches), batch_max_source_frames=max(b["source_frames"] for b in batches),
              largest_remaining_part_frames=max(sum(n for _, n in rows) for rows in pending.values()),
              plan_status="Preview only; regenerate after a graceful production pause before cutover")
atomic_json(PROBE / "workload_snapshot.json", report)
atomic_jsonl(PROBE / "batch_plan_preview.jsonl",
             [dict(batch_id=f"batch-{i:05d}", **b) for i, b in enumerate(batches)])
print(json.dumps(report, ensure_ascii=False, indent=2))
