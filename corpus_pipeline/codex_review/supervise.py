"""Run, resume, and strictly export one authorized Codex review scope."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import time

from .farm import atomic_json


def counts(root):
    with sqlite3.connect(f"file:{root / 'state.sqlite'}?mode=ro", uri=True) as db:
        return dict(db.execute("SELECT status,count(*) FROM items GROUP BY status"))


def process_identity(pid):
    proc = Path(f"/proc/{pid}")
    return {"pid": pid, "host": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "start_ticks": proc.joinpath("stat").read_text().split(") ", 1)[1].split()[19],
            "argv_sha256": hashlib.sha256(proc.joinpath("cmdline").read_bytes()).hexdigest()}


class RecoveredChild:
    """Observe only a previously recorded, exactly matching child identity."""
    def __init__(self, identity):
        self.identity = identity
        self.pid = identity["pid"]

    def poll(self):
        try:
            state = Path(f"/proc/{self.pid}/stat").read_text().split(") ", 1)[1].split()[0]
            return None if state != "Z" and process_identity(self.pid) == self.identity else 0
        except (OSError, ValueError, IndexError):
            return 0

    @property
    def returncode(self):
        # We cannot waitpid a child of the prior supervisor. Durable farm state,
        # locks and strict export determine the next action after it exits.
        return self.poll()

    def send_signal(self, sig):
        if self.poll() is None:
            try:
                os.kill(self.pid, sig)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--initial-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    stop = False
    child = None

    def stopping(*_):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stopping)
    with (root / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((root / "SUPERVISOR_STATUS.json").read_text()) if (root / "SUPERVISOR_STATUS.json").exists() else {}
        adopted = None
        if previous.get("child_identity"):
            candidate = RecoveredChild(previous["child_identity"])
            if candidate.poll() is None:
                adopted = candidate
        launch = {"pid": os.getpid(), "host": socket.gethostname(), "started_at_epoch": time.time(),
                  "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split(") ", 1)[1].split()[19],
                  "argv": sys.argv, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        atomic_json(root / "SUPERVISOR.json", launch)
        restarts = 0
        while not stop or adopted is not None:
            state = counts(root)
            unfinished = sum(state.get(k, 0) for k in ("pending", "running", "retry"))
            action = previous.get("action", "run") if adopted is not None else ("run" if unfinished else "export")
            command = [sys.executable, "-m", "corpus_pipeline.codex_review.farm", action, "--root", str(root)]
            if action == "run":
                command += ["--workers", str(args.workers), "--initial-workers", str(args.initial_workers),
                            "--batch-size", str(args.batch_size)]
            elif state.get("failed", 0):
                # Strictly audit/export successful items, with explicit incomplete
                # coverage. Never turn technical failures into model uncertainty.
                command += ["--allow-incomplete"]
            with (root / "supervisor_child.log").open("ab") as log:
                if adopted is not None:
                    child, adopted = adopted, None
                    identity = child.identity
                else:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    identity = process_identity(child.pid)
                if stop and child.poll() is None:
                    try:
                        child.send_signal(signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                while child.poll() is None:
                    atomic_json(root / "SUPERVISOR_STATUS.json", {**launch, "status": "draining" if stop else action,
                                "action": action, "child_pid": child.pid, "child_identity": identity,
                                "updated_at_epoch": time.time(), "restarts": restarts})
                    time.sleep(5)
                code = child.returncode
            child = None
            if stop:
                atomic_json(root / "SUPERVISOR_STATUS.json", {**launch, "status": "interrupted_drained",
                            "updated_at_epoch": time.time(), "returncode": code, "restarts": restarts})
                return
            if code:
                restarts += 1
                if restarts >= 5:
                    atomic_json(root / "SUPERVISOR_STATUS.json", {**launch, "status": "needs_repair",
                                "updated_at_epoch": time.time(), "returncode": code, "restarts": restarts})
                    return
                deadline = time.monotonic() + min(300, 15 * 2 ** (restarts - 1))
                while not stop and time.monotonic() < deadline:
                    time.sleep(1)
                continue
            if action == "export":
                try:
                    completion_path = root / ("EXPORT.json" if state.get("failed", 0) else "COMPLETE.json")
                    complete = json.loads(completion_path.read_text())
                    from .report import generate_report
                    report = generate_report(root)
                except Exception as error:
                    atomic_json(root / "SUPERVISOR_STATUS.json", {**launch, "status": "needs_repair",
                                "updated_at_epoch": time.time(), "stage": "final_report",
                                "error": type(error).__name__ + ":" + str(error)[:1000], "restarts": restarts})
                    return
                atomic_json(root / "SUPERVISOR_STATUS.json", {**launch,
                            "status": "completed" if complete["complete"] else "needs_repair",
                            "updated_at_epoch": time.time(), "counts": counts(root), "completion": complete,
                            "report_counts": report.get("counts"), "restarts": restarts})
                return


if __name__ == "__main__":
    main()
