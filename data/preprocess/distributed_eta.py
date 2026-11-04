"""Read committed sample pairs to estimate remaining time by source-frame work.

No videos, processing journals, claims or worker state are modified. The live
scan is bounded by an explicit cutoff and ignores incomplete appended lines.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .distributed import DESTINATION, queue
from .manifest import atomic_json, read_jsonl


def main():
    root, q = DESTINATION, queue(DESTINATION)
    plan = json.loads((q / "plan.json").read_text())
    samples = {}
    with (root / "metadata/selected.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            samples[row["sample_id"]] = row["source"]["num_frames"]
    assert len(samples) == plan["candidates"]
    variants = ("960x544", "832x480")
    baseline = {(r["sample_id"], r["variant"]) for r in read_jsonl(q / "baseline/completed_files.jsonl")}
    sessions = {}
    for path in sorted((q / "sessions").glob("*/session.json")):
        row = json.loads(path.read_text())
        if row["stage"] == "full":
            sessions[row["instance"]] = (row, json.loads((path.parent / "status.json").read_text()))
    all_hosts_started = max(datetime.fromisoformat(s[0]["started_at"]) for s in sessions.values())
    steady_start = all_hosts_started + timedelta(seconds=60)
    cutoff = datetime.now(timezone.utc)

    def read_batch(task):
        batch = task["batch_id"]
        owner_path = q / "claims" / batch / "owner.json"
        owner = json.loads(owner_path.read_text())["instance"] if owner_path.exists() else None
        events = {}
        with (q / "batches" / batch / "_state" / f'part{task["part"]}.jsonl').open() as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break
                event = json.loads(line)
                stamp = datetime.fromisoformat(event["updated_at"])
                if stamp <= cutoff:
                    events[event["sample_id"], event["variant"]] = (event["status"], stamp, owner)
        return batch, events

    latest, batches_by_key = {}, {}
    tasks = read_jsonl(q / "tasks.jsonl")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for batch, events in pool.map(read_batch, tasks):
            for key, value in events.items():
                if key in batches_by_key:
                    raise ValueError("Output key occurs in two batch journals")
                batches_by_key[key] = batch
                latest[key] = value
    completed, pending = {}, []
    committed_outputs = len(baseline)
    committed_outputs += sum(value[0] == "success" and key not in baseline for key, value in latest.items())
    for name, frames in samples.items():
        events = [latest.get((name, v)) for v in variants]
        good = [bool(e and e[0] == "success") if e else (name, v) in baseline
                for v, e in zip(variants, events)]
        if all(good):
            stamps = [e for e in events if e]
            completed[name] = max(stamps, key=lambda e: e[1])[1:] if stamps else (None, "legacy")
        else:
            pending.append(name)
    remaining_frames = sum(samples[name] for name in pending)
    completed_frames = sum(samples[name] for name in completed)
    report = dict(cutoff_utc=cutoff.isoformat(), read_finished_utc=datetime.now(timezone.utc).isoformat(),
        live_snapshot=True, candidates=len(samples), completed_samples=len(completed),
        pending_samples=len(pending), completed_percent=100*len(completed)/len(samples),
        committed_outputs=committed_outputs, completed_source_frames=completed_frames,
        pending_source_frames=remaining_frames, completed_source_frame_percent=100*completed_frames/sum(samples.values()),
        mean_pending_source_frames=remaining_frames/len(pending) if pending else 0,
        steady_start_utc=steady_start.isoformat(), windows={},
        active_workers=sum(sum(p["alive"] for p in status["processes"]) for _, status in sessions.values()),
        session_heartbeats={name:status["updated_at"] for name, (_, status) in sessions.items()},
        failed_output_keys=sum(e[0] != "success" for e in latest.values()),
        methodology="A source counts once only after both resolutions commit; count original source frames, not encoded frames or completed batches. In-flight uncommitted work remains in pending; finalization and batch tail are not included in raw ETA.")
    for minutes in (5, 10, 15, 20, 25, 30):
        start = cutoff - timedelta(minutes=minutes)
        chosen = [(name, owner) for name, (stamp, owner) in completed.items() if stamp and stamp >= start]
        frames = sum(samples[name] for name, _ in chosen)
        fps = frames/(minutes*60)
        by_host = Counter()
        for name, owner in chosen:
            by_host[owner] += samples[name]
        report["windows"][str(minutes)] = dict(fully_after_all_hosts_warmup=start >= steady_start,
            samples=len(chosen), source_frames=frames, source_frames_per_second=fps,
            mean_source_frames=frames/len(chosen) if chosen else 0,
            raw_eta_hours=remaining_frames/fps/3600 if fps else None,
            raw_completion_utc=(cutoff+timedelta(seconds=remaining_frames/fps)).isoformat() if fps else None,
            host_source_frames_per_second={host:value/(minutes*60) for host,value in sorted(by_host.items())})
    atomic_json(q / "eta.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
