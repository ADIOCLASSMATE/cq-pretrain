"""Exclusive, resumable batches around the frozen OpenVidHD v3 converter.

The original journals and completed videos are never rewritten. Each batch has
private journals/scratch and symlinks to the two existing video directories.
Atomic mkdir, rather than flock, assigns a batch to one process across hosts.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import time
import zipfile

from .archive import SplitFile
from .manifest import (atomic_json, atomic_jsonl, journal_latest, read_jsonl,
                       require_readable, sha256, sync_directory)
from .openvidhd import (DESTINATION, VARIANTS, check_pilot, event_counts,
                       load_config, now, part_work, reconcile)


def queue(root):
    return Path(root) / "_distributed"


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def process_identity(pid):
    """PID + start ticks avoids confusing a reused PID with the original owner."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(state=fields[0], start_ticks=int(fields[19]))


def owner_alive(owner):
    if owner["host"] != platform.node():
        raise ValueError("Process death must be checked on its actual host")
    actual = process_identity(owner["pid"])
    return bool(actual and actual["state"] not in ("Z", "X")
                and actual["start_ticks"] == owner["start_ticks"])


@contextmanager
def administrative_lock(root):
    path = queue(root) / "administrative.lock"
    path.mkdir()  # Cross-host atomic operation; do not use exist_ok.
    try:
        yield
    finally:
        path.rmdir()


def load_plan(root):
    root = Path(root).resolve()
    plan = json.loads((queue(root) / "plan.json").read_text())
    if plan["root"] != str(root):
        raise ValueError("Plan points to a different dataset")
    return plan


def successful_snapshots(root, events, checks):
    """Stat every committed file; hashes remain in the auxiliary integrity log."""
    def one(item):
        key, event = item
        if event["status"] != "success":
            return None
        record = event["record"]
        if key not in checks:
            raise ValueError(f"Committed output lacks integrity record: {key}")
        path = Path(root) / key[1] / record["video_path"]
        stat = path.stat()
        if stat.st_size != record["output_bytes"]:
            raise ValueError(f"Committed output size differs: {path}")
        return dict(sample_id=key[0], variant=key[1], video_path=record["video_path"],
                    bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    with ThreadPoolExecutor(max_workers=16) as pool:
        return [r for r in pool.map(one, events.items()) if r is not None]


def audit_unchanged(root, snapshot):
    def one(row):
        path = Path(root) / row["variant"] / row["video_path"]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise ValueError(f"Previously completed output changed: {path}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, snapshot))
    return len(snapshot)


def split_rows(rows, max_frames, max_samples):
    batches, current, frames = [], [], 0
    for row in rows:
        n = row["source"]["num_frames"]
        if n > max_frames:
            raise ValueError("One video exceeds the batch frame budget")
        if current and (frames + n > max_frames or len(current) >= max_samples):
            batches.append(current)
            current, frames = [], 0
        current.append(row)
        frames += n
    if current:
        batches.append(current)
    return batches


def prepare(root, pilot_root):
    root = Path(root).resolve()
    if not (root / "STOP").exists():
        raise ValueError("Legacy STOP must remain in place throughout distributed mode")
    old_run = json.loads((root / "run.json").read_text())
    if old_run["state"] == "running":
        raise ValueError("Wait for the legacy process to finish its graceful stop")
    config = load_config(root)
    check_pilot(pilot_root, config)
    q = queue(root)
    q.mkdir()  # Do not overwrite an existing plan or its completed work.
    for name in ("claims", "attempts", "sessions", "batches", "validation", "logs", "baseline"):
        (q / name).mkdir()
    selected = read_jsonl(root / "metadata/selected.jsonl")
    if len({r["sample_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate source sample ID")
    latest, checks = {}, {}
    parts = sorted({r["source"]["part"] for r in selected})
    for part in parts:
        for table, directory in ((latest, "_state"), (checks, "metadata/integrity_parts")):
            src = root / directory / f"part{part}.jsonl"
            if not src.exists():
                continue
            target = q / "baseline" / directory / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            for key, event in journal_latest(target, per_variant=True).items():
                if key in table or event["run_id"] != config["run_id"]:
                    raise ValueError("Legacy journal identity conflict")
                table[key] = event
    valid_keys = {(r["sample_id"], v) for r in selected for v in VARIANTS}
    if set(latest) - valid_keys or set(checks) - valid_keys:
        raise ValueError("Legacy journal contains samples outside selection")
    snapshots = successful_snapshots(root, latest, checks)
    atomic_jsonl(q / "baseline/completed_files.jsonl", snapshots)
    completed = {(r["sample_id"], r["variant"]) for r in snapshots}
    pending = [r for r in selected if any((r["sample_id"], v) not in completed for v in VARIANTS)]
    sources = {r["sample_id"]: r for r in read_jsonl(root / "metadata/source_integrity.jsonl")}
    grouped = defaultdict(list)
    for row in pending:
        grouped[row["source"]["part"]].append(row)
    # Sixteen four-video validation batches contain real pending data and are
    # reused in the final corpus. They do not duplicate the previous ten videos.
    eligible_parts = [p for p in grouped if len(grouped[p]) >= 4]
    chosen_parts = random.Random(20261030).sample(eligible_parts, min(16, len(eligible_parts)))
    validation_parts = set(chosen_parts)
    tasks, task_integrity, planned = [], [], set()
    for part, rows in sorted(grouped.items()):
        identities = config["archive_identities"][str(part)]
        with SplitFile([i["path"] for i in identities]) as joined, zipfile.ZipFile(joined) as archive:
            rows.sort(key=lambda r: archive.getinfo(r["source"]["member"]).header_offset)
        work = []
        if part in validation_parts:
            # Prefer ordinary lengths so the recovery rehearsal finishes quickly.
            cases = sorted(rows, key=lambda r: (abs(r["source"]["num_frames"]-400), r["sample_id"]))[:4]
            names = {r["sample_id"] for r in cases}
            work.append((True, [r for r in rows if r["sample_id"] in names]))
            rows = [r for r in rows if r["sample_id"] not in names]
        work.extend((False, b) for b in split_rows(rows, 32000, 128))
        for validation, batch_rows in work:
            batch_id = f"batch_{len(tasks):06d}"
            directory = q / "batches" / batch_id
            (directory / "metadata/integrity_parts").mkdir(parents=True)
            (directory / "_state").mkdir()
            for variant in VARIANTS:
                (directory / variant).symlink_to(root / variant, target_is_directory=True)
            (directory / "STOP").symlink_to(q / "STOP")
            for row in batch_rows:
                if row["sample_id"] in planned:
                    raise ValueError("Sample assigned to two batches")
                planned.add(row["sample_id"])
            atomic_jsonl(directory / "metadata/selected.jsonl", batch_rows)
            atomic_jsonl(directory / "metadata/source_integrity.jsonl", [sources[r["sample_id"]] for r in batch_rows])
            # Carry over only existing successful variants; they will be skipped.
            keys = [(r["sample_id"], v) for r in batch_rows for v in VARIANTS
                    if (r["sample_id"], v) in completed]
            atomic_jsonl(directory / "_state" / f"part{part}.jsonl", [latest[k] for k in keys])
            atomic_jsonl(directory / "metadata/integrity_parts" / f"part{part}.jsonl", [checks[k] for k in keys])
            tasks.append(dict(batch_id=batch_id, part=part, samples=len(batch_rows),
                source_frames=sum(r["source"]["num_frames"] for r in batch_rows), validation=validation,
                inherited_outputs=len(keys)))
            task_integrity.append(dict(batch_id=batch_id,
                selection_sha256=sha256(directory / "metadata/selected.jsonl"),
                source_integrity_sha256=sha256(directory / "metadata/source_integrity.jsonl")))
    if planned != {r["sample_id"] for r in pending}:
        raise ValueError("Plan does not exactly cover pending samples")
    atomic_jsonl(q / "tasks.jsonl", tasks)
    atomic_jsonl(q / "task_integrity.jsonl", task_integrity)
    plan = dict(version=1, created_at=now(), root=str(root), run_id=config["run_id"],
                pilot_root=str(pilot_root), candidates=len(selected),
                baseline_completed_samples=len(selected)-len(pending),
                baseline_completed_outputs=len(completed), pending_samples=len(pending),
                pending_outputs=2*len(selected)-len(completed), batches=len(tasks),
                validation_batches=[t["batch_id"] for t in tasks if t["validation"]],
                source_frames=sum(r["source"]["num_frames"] for r in pending),
                max_batch_frames=32000, max_batch_samples=128)
    require_readable(plan)
    atomic_json(q / "plan.json", plan)
    atomic_json(q / "plan_integrity.json", dict(tasks_sha256=sha256(q / "tasks.jsonl"),
                plan_sha256=sha256(q / "plan.json"), task_integrity_sha256=sha256(q / "task_integrity.jsonl")))
    print(json.dumps(plan), flush=True)


def task_state(root, task):
    directory = queue(root) / "batches" / task["batch_id"]
    path = directory / "_state" / f'part{task["part"]}.status.json'
    return json.loads(path.read_text()) if path.exists() else None


def claim_batch(root, task, instance, session_id, slot):
    claim = queue(root) / "claims" / task["batch_id"]
    try:
        claim.mkdir()
    except FileExistsError:
        return None
    owner = dict(batch_id=task["batch_id"], instance=instance, session=session_id, slot=slot,
                 host=platform.node(), pid=os.getpid(), start_ticks=process_identity(os.getpid())["start_ticks"],
                 claimed_at=now())
    atomic_json(claim / "owner.json", owner)
    sync_directory(claim.parent)
    return owner


def worker_loop(root, tasks, integrity, config, instance, session_id, slot):
    import torch
    torch.set_num_threads(1)
    root, q = Path(root), queue(root)
    for task in tasks:
        if (q / "STOP").exists():
            return
        owner = claim_batch(root, task, instance, session_id, slot)
        if owner is None:
            continue
        batch_id, part = task["batch_id"], task["part"]
        directory, claim = q / "batches" / batch_id, q / "claims" / batch_id
        print(json.dumps(dict(event="claimed", **owner)), flush=True)
        try:
            for filename, field in (("selected.jsonl", "selection_sha256"),
                                     ("source_integrity.jsonl", "source_integrity_sha256")):
                if sha256(directory / "metadata" / filename) != integrity[batch_id][field]:
                    raise ValueError("Frozen batch metadata changed")
            rows = read_jsonl(directory / "metadata/selected.jsonl")
            checks = {r["sample_id"]: r["zip_crc32"]
                      for r in read_jsonl(directory / "metadata/source_integrity.jsonl")}
            part_work(str(directory), part, rows, config, checks, 100)
            state = task_state(root, task)
            if state["success"] != len(rows):
                raise RuntimeError(f"Batch incomplete: {state}")
            atomic_json(claim / "done.json", dict(batch_id=batch_id, completed_at=now(),
                        instance=instance, session=session_id, samples=len(rows),
                        outputs=2*len(rows), source_frames=task["source_frames"]))
            print(json.dumps(dict(event="done", batch_id=batch_id, instance=instance)), flush=True)
        except Exception as error:
            atomic_json(claim / "error.json", dict(batch_id=batch_id, failed_at=now(),
                        error=f"{type(error).__name__}: {error}"))
            raise


def run_workers(root, instance, workers, stage, only_batch=None):
    root, q = Path(root).resolve(), queue(root)
    plan = load_plan(root)
    config = load_config(root)
    if config["run_id"] != plan["run_id"] or not (root / "STOP").exists():
        raise ValueError("Distributed mode requires frozen config and legacy STOP")
    checks = json.loads((q / "plan_integrity.json").read_text())
    for name in ("tasks", "plan", "task_integrity"):
        path = q / (name + (".json" if name == "plan" else ".jsonl"))
        if sha256(path) != checks[name + "_sha256"]:
            raise ValueError("Frozen plan changed")
    if stage == "full":
        validation = json.loads((q / "validation/report.json").read_text())
        if not validation["all_passed"]:
            raise ValueError("Two-host validation must pass before full processing")
    tasks = read_jsonl(q / "tasks.jsonl")
    if stage == "validation":
        tasks = [t for t in tasks if t["validation"]]
    if only_batch:
        tasks = [t for t in tasks if t["batch_id"] == only_batch]
        if len(tasks) != 1:
            raise ValueError("Unknown batch in this stage")
    integrity = {r["batch_id"]: r for r in read_jsonl(q / "task_integrity.jsonl")}
    session_id = f"{instance}-{stamp()}"
    session_dir = q / "sessions" / session_id
    session_dir.mkdir()
    descriptor = dict(session=session_id, instance=instance, host=platform.node(),
                      stage=stage, started_at=now(), pid=os.getpid(), workers=workers)
    atomic_json(session_dir / "session.json", descriptor)
    children = [multiprocessing.get_context("spawn").Process(target=worker_loop,
                args=(str(root), tasks, integrity, config, instance, session_id, slot))
                for slot in range(workers)]
    for child in children:
        child.start()
    while any(child.is_alive() for child in children):
        atomic_json(session_dir / "status.json", dict(state="running", updated_at=now(),
                    processes=[dict(pid=c.pid, alive=c.is_alive(), exitcode=c.exitcode) for c in children]))
        time.sleep(3)
    for child in children:
        child.join()
    failed = any(c.exitcode != 0 for c in children)
    atomic_json(session_dir / "status.json", dict(state="failed" if failed else "finished", updated_at=now(),
                processes=[dict(pid=c.pid, alive=False, exitcode=c.exitcode) for c in children]))
    return 1 if failed else 0


def recover(root, batch_id):
    """Run on the owner host, only after proving the original process is dead."""
    q = queue(root)
    with administrative_lock(root):
        claim = q / "claims" / batch_id
        if (claim / "done.json").exists():
            raise ValueError("Completed batches must never be requeued")
        owner = json.loads((claim / "owner.json").read_text())
        if owner_alive(owner):
            raise ValueError("Owner is alive; refusing to reassign its batch")
        destination = q / "attempts" / batch_id / stamp()
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(claim / "recovery.json", dict(checked_at=now(), owner_process_dead=True,
                    checked_on_host=platform.node()))
        os.rename(claim, destination)
        sync_directory(claim.parent)
        print(json.dumps(dict(requeued=batch_id, previous_owner=owner)), flush=True)


def recover_exited(root, instance):
    """Local-host watchdog; a timestamp or stale heartbeat is never sufficient."""
    q = queue(root)
    recovered, exhausted, active = [], [], 0
    for claim in sorted((q / "claims").iterdir()):
        if (claim / "done.json").exists() or not (claim / "owner.json").exists():
            continue
        owner = json.loads((claim / "owner.json").read_text())
        if owner["instance"] != instance or owner["host"] != platform.node():
            continue
        if owner_alive(owner):
            active += 1
            continue
        attempts = q / "attempts" / claim.name
        if attempts.exists() and len(list(attempts.iterdir())) >= 3:
            exhausted.append(claim.name)
            continue
        recover(root, claim.name)
        recovered.append(claim.name)
    return dict(instance=instance, requeued=recovered, retry_limit_reached=exhausted,
                active_owner_count=active)


def fault_test(root, batch_id):
    """Kill one validation worker, then snapshot its durably committed outputs."""
    q = queue(root)
    if (q / "validation/report.json").exists():
        raise ValueError("Fault injection is disabled after validation passes")
    plan = load_plan(root)
    if batch_id not in plan["validation_batches"]:
        raise ValueError("Fault injection is limited to designated validation data")
    task = next(t for t in read_jsonl(q / "tasks.jsonl") if t["batch_id"] == batch_id)
    claim = q / "claims" / batch_id
    owner = json.loads((claim / "owner.json").read_text())
    state = task_state(root, task)
    if not state or not 0 < state["success"] < task["samples"] or not owner_alive(owner):
        raise ValueError("Fault test requires a live worker with partly committed progress")
    if (claim / "done.json").exists():
        raise ValueError("Batch has already finished")
    # Freeze the process before checking the completion marker again, so a slow
    # control connection cannot accidentally kill its next batch.
    os.kill(owner["pid"], signal.SIGSTOP)
    if (claim / "done.json").exists():
        os.kill(owner["pid"], signal.SIGCONT)
        raise ValueError("Batch completed before injection; worker resumed")
    os.kill(owner["pid"], signal.SIGKILL)
    for _ in range(100):
        if not owner_alive(owner):
            break
        time.sleep(0.1)
    if owner_alive(owner):
        raise RuntimeError("Could not verify worker exit")
    directory = q / "batches" / batch_id
    events = journal_latest(directory / "_state" / f'part{task["part"]}.jsonl', per_variant=True)
    checks = journal_latest(directory / "metadata/integrity_parts" / f'part{task["part"]}.jsonl', per_variant=True)
    snapshots = successful_snapshots(root, events, checks)
    if not 0 < len(snapshots) < task["samples"]*2:
        raise ValueError("Fault missed the partial-completion window")
    atomic_json(q / "validation/fault.json", dict(batch_id=batch_id, killed_at=now(), owner=owner,
                committed_before_recovery=snapshots, pending_outputs=task["samples"]*2-len(snapshots)))
    print(json.dumps(dict(killed_worker=owner, committed_outputs=len(snapshots))), flush=True)


def fault_watch(root, instance):
    q = queue(root)
    tasks = [t for t in read_jsonl(q / "tasks.jsonl") if t["validation"]]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        for task in tasks:
            claim = q / "claims" / task["batch_id"]
            if not (claim / "owner.json").exists() or (claim / "done.json").exists():
                continue
            owner = json.loads((claim / "owner.json").read_text())
            state = task_state(root, task)
            if owner["instance"] == instance and state and 0 < state["success"] < task["samples"]:
                fault_test(root, task["batch_id"])
                return
        time.sleep(0.5)
    raise TimeoutError("No partially committed validation batch appeared")


def validate(root):
    from .video import verify_output
    root, q = Path(root), queue(root)
    plan = load_plan(root)
    tasks = [t for t in read_jsonl(q / "tasks.jsonl") if t["validation"]]
    fault = json.loads((q / "validation/fault.json").read_text())
    audit_unchanged(root, fault["committed_before_recovery"])
    verified, instances, names, keys = 0, set(), set(), set()
    execution_windows, instance_samples = defaultdict(list), Counter()
    for task in tasks:
        batch_id = task["batch_id"]
        done = json.loads((q / "claims" / batch_id / "done.json").read_text())
        owner = json.loads((q / "claims" / batch_id / "owner.json").read_text())
        instances.add(done["instance"])
        instance_samples[done["instance"]] += task["samples"]
        execution_windows[done["instance"]].append((owner["claimed_at"], done["completed_at"]))
        directory = q / "batches" / batch_id
        rows = read_jsonl(directory / "metadata/selected.jsonl")
        for row in rows:
            if row["sample_id"] in names:
                raise ValueError("Duplicate sample across validation batches")
            names.add(row["sample_id"])
        events_path = directory / "_state" / f'part{task["part"]}.jsonl'
        events = read_jsonl(events_path)
        success_counts = Counter((e["sample_id"], e["variant"]) for e in events if e["status"] == "success")
        if any(c != 1 for c in success_counts.values()):
            raise ValueError("An already committed output was encoded again")
        latest = journal_latest(events_path, per_variant=True)
        checks = journal_latest(directory / "metadata/integrity_parts" / f'part{task["part"]}.jsonl', per_variant=True)
        if set(latest) != {(r["sample_id"], v) for r in rows for v in VARIANTS}:
            raise ValueError("Batch journal does not exactly match its assigned samples")
        for row in rows:
            for variant in VARIANTS:
                key = row["sample_id"], variant
                if key in keys:
                    raise ValueError("Output appears in more than one batch")
                keys.add(key)
                event = latest[key]
                if event["status"] != "success":
                    raise ValueError("Validation output failed")
                record = event["record"]
                require_readable(record)
                if any(record[k] != row[k] for k in ("sample_id", "caption", "video_path")):
                    raise ValueError("Source identity/caption changed")
                if any(record["source"][k] != v for k, v in row["source"].items()):
                    raise ValueError("Original annotations/source metadata changed")
                path = root / variant / record["video_path"]
                if sha256(path) != checks[key]["output_sha256"]:
                    raise ValueError("Committed output checksum differs")
                verify_output(path, record, checks[key]["decoded_rgb_sha256"])
                verified += 1
    if len(instances) < 2:
        raise ValueError("Validation requires completed work on two CPU instances")
    windows = [(min(a for a, b in values), max(b for a, b in values))
               for values in execution_windows.values()]
    if max(a for a, b in windows) >= min(b for a, b in windows):
        raise ValueError("The CPU instance executions did not overlap in time")
    resumed = json.loads((q / "claims" / fault["batch_id"] / "done.json").read_text())
    if resumed["instance"] == fault["owner"]["instance"]:
        raise ValueError("Recovery must be demonstrated on the other instance")
    preserved = audit_unchanged(root, read_jsonl(q / "baseline/completed_files.jsonl"))
    report = dict(all_passed=True, checked_at=now(), instances=sorted(instances),
        validation_samples=len(names), verified_outputs=verified, duplicate_success_records=0,
        samples_by_final_owner=dict(instance_samples), parallel_execution_overlap=True,
        preserved_legacy_outputs=preserved, legacy_outputs_reencoded=0,
        recovered_batch=fault["batch_id"], recovery_from=fault["owner"]["instance"],
        recovery_to=resumed["instance"], preserved_precrash_outputs=len(fault["committed_before_recovery"]),
        recovery_pending_outputs=fault["pending_outputs"], run_id=plan["run_id"])
    require_readable(report)
    atomic_json(q / "validation/report.json", report)
    print(json.dumps(report), flush=True)


def progress(root):
    q = queue(root)
    plan = load_plan(root)
    tasks = read_jsonl(q / "tasks.jsonl")
    success = plan["baseline_completed_samples"]
    outputs = plan["baseline_completed_outputs"]
    done, active, errors, frames_done = 0, 0, [], 0
    for task in tasks:
        claim = q / "claims" / task["batch_id"]
        state = task_state(root, task)
        if state:
            success += state["success"]
            outputs += state["committed_video_files"] - task["inherited_outputs"]
        if (claim / "done.json").exists():
            done += 1
            frames_done += task["source_frames"]
        elif claim.exists():
            active += 1
            if (claim / "error.json").exists():
                errors.append(task["batch_id"])
    return dict(updated_at=now(), candidates=plan["candidates"], success=success,
                pending=plan["candidates"]-success, committed_video_files=outputs,
                batches=len(tasks), completed_batches=done, claimed_unfinished_batches=active,
                unclaimed_batches=len(tasks)-done-active, failed_batches=errors,
                completed_batch_source_frames=frames_done, total_planned_source_frames=plan["source_frames"])


def finalize(root):
    root, q = Path(root), queue(root)
    with administrative_lock(root):
        status = progress(root)
        if status["completed_batches"] != status["batches"] or status["failed_batches"]:
            raise ValueError("All batches must complete before finalization")
        config = load_config(root)
        tasks = read_jsonl(q / "tasks.jsonl")
        audit_unchanged(root, read_jsonl(q / "baseline/completed_files.jsonl"))
        view = q / "finalized"
        view.mkdir(exist_ok=True)
        (view / "metadata/integrity_parts").mkdir(parents=True, exist_ok=True)
        (view / "_state").mkdir(exist_ok=True)
        if not (view / "metadata/selected.jsonl").exists():
            (view / "metadata/selected.jsonl").symlink_to(root / "metadata/selected.jsonl")
        grouped = defaultdict(list)
        for task in tasks:
            grouped[task["part"]].append(task)
        selected = read_jsonl(root / "metadata/selected.jsonl")
        for part in sorted({r["source"]["part"] for r in selected}):
            for subdir in ("_state", "metadata/integrity_parts"):
                name = f"part{part}.jsonl"
                merged = journal_latest(q / "baseline" / subdir / name, per_variant=True)
                assigned = set()
                for task in grouped[part]:
                    entries = journal_latest(q / "batches" / task["batch_id"] / subdir / name, per_variant=True)
                    for key, value in entries.items():
                        if key in assigned:
                            raise ValueError("Two batches committed the same sample/variant")
                        assigned.add(key)
                        previous = merged.get(key)
                        if previous and (subdir != "_state" or previous["status"] == "success"):
                            if previous != value:
                                raise ValueError("A previously committed record changed")
                        merged[key] = value
                atomic_jsonl(view / subdir / name, merged.values())
        # Reconcile writes only manifests in this view. Videos remain in place.
        for variant in VARIANTS:
            target = view / variant
            target.mkdir(exist_ok=True)
            if not (target / "videos").exists():
                (target / "videos").symlink_to(root / variant / "videos", target_is_directory=True)
        summary = reconcile(view, config)
        if not summary["all_candidates_succeeded"]:
            raise ValueError("Final coverage differs from original selection")
        for name in ("summary.json", "commit_validation.json", "pairs.jsonl", "failures.jsonl",
                     "metadata/integrity.jsonl", *[f"{v}/samples.jsonl" for v in VARIANTS]):
            destination = root / name
            temporary = destination.with_name(destination.name + ".distributed.tmp")
            shutil.copyfile(view / name, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            sync_directory(destination.parent)
        atomic_json(q / "finished.json", dict(finished_at=now(), all_passed=True, **status))
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "worker", "recover", "recover-exited", "fault-test", "fault-watch", "validate", "progress", "finalize"])
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--pilot-root", type=Path, default=DESTINATION.with_name("openvidhd_selected_pilot10"))
    parser.add_argument("--instance")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--stage", choices=["validation", "full"], default="validation")
    parser.add_argument("--batch")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.pilot_root)
    elif args.command == "worker":
        if not args.instance or args.workers < 1:
            parser.error("worker requires --instance and a positive worker count")
        return run_workers(args.output, args.instance, args.workers, args.stage, args.batch)
    elif args.command == "recover":
        recover(args.output, args.batch)
    elif args.command == "recover-exited":
        print(json.dumps(recover_exited(args.output, args.instance)))
    elif args.command == "fault-test":
        fault_test(args.output, args.batch)
    elif args.command == "fault-watch":
        fault_watch(args.output, args.instance)
    elif args.command == "validate":
        validate(args.output)
    elif args.command == "progress":
        print(json.dumps(progress(args.output)))
    elif args.command == "finalize":
        finalize(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
