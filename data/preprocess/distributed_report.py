"""Record eight-instance rollout acceptance without modifying any video."""
from collections import Counter
from datetime import datetime, timezone
import json

from .distributed import DESTINATION, audit_unchanged, now, progress, queue
from .manifest import atomic_json, journal_latest, read_jsonl, require_readable


def main():
    root, q = DESTINATION, queue(DESTINATION)
    validation = json.loads((q / "validation/report.json").read_text())
    if not validation["all_passed"]:
        raise ValueError("Two-instance recovery validation did not pass")
    registry = json.loads((q / "instances.json").read_text())
    names = {row["name"] for row in registry["instances"]} | {"existing-local"}
    sessions = {}
    for path in sorted((q / "sessions").glob("*/session.json")):
        row = json.loads(path.read_text())
        if row["stage"] == "full":
            sessions[row["instance"]] = (path.parent, row)
    if set(sessions) != names:
        raise ValueError("Not all eight CPU instances and the local instance started")
    workers = []
    for instance, (path, row) in sorted(sessions.items()):
        status = json.loads((path / "status.json").read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(status["updated_at"])).total_seconds()
        live = sum(p["alive"] for p in status["processes"])
        if status["state"] != "running" or age > 30 or live != 16:
            raise ValueError(f"Expected sixteen live workers and a fresh heartbeat: {instance}")
        workers.append(dict(instance=instance, host=row["host"], live_workers=live))
    tasks = {t["batch_id"]: t for t in read_jsonl(q / "tasks.jsonl")}
    assigned_names, output_keys = set(), set()
    owners, committed_by_owner = Counter(), Counter()
    for claim in sorted((q / "claims").iterdir()):
        if not (claim / "owner.json").exists():
            continue
        owner = json.loads((claim / "owner.json").read_text())
        task = tasks[claim.name]
        directory = q / "batches" / claim.name
        rows = read_jsonl(directory / "metadata/selected.jsonl")
        names_in_batch = {r["sample_id"] for r in rows}
        if len(names_in_batch) != len(rows) or assigned_names & names_in_batch:
            raise ValueError("A sample is assigned to multiple claimed batches")
        assigned_names.update(names_in_batch)
        owners[owner["instance"]] += 1
        log = directory / "_state" / f'part{task["part"]}.jsonl'
        for key, event in journal_latest(log, per_variant=True).items():
            if event["status"] == "success":
                if key in output_keys:
                    raise ValueError("Duplicate output across batch journals")
                output_keys.add(key)
                committed_by_owner[owner["instance"]] += 1
    if any(committed_by_owner[name] == 0 for name in names):
        raise ValueError("Each instance must commit at least one output before acceptance")
    preserved = audit_unchanged(root, read_jsonl(q / "baseline/completed_files.jsonl"))
    report = dict(checked_at=now(), all_passed=True, cpu_instances=8, existing_instance_workers=16,
        total_live_workers=sum(w["live_workers"] for w in workers), workers=workers,
        claimed_batches_by_instance=dict(owners), committed_outputs_by_current_batch_owner=dict(committed_by_owner),
        duplicate_claimed_sample_ids=0, duplicate_output_keys=0,
        preserved_preexisting_files=preserved, two_instance_validation=validation,
        progress=progress(root), full_dataset_finished=False)
    require_readable(report)
    atomic_json(q / "rollout_validation.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
