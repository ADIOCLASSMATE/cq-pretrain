"""Run under tmux: publish progress, recover dead workers, finalize, stop CPUs.

Only the eight task-created CPU notebooks listed in instances.json are managed.
An unreachable instance keeps its claims; no timeout-based reassignment occurs.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from .distributed import (DESTINATION, finalize, load_plan, now, process_identity,
                          progress, queue, recover_exited)
from .manifest import atomic_json

REPO = Path(__file__).resolve().parents[2]
INSPIRE_PYTHON = "/root/.local/share/uv/tools/inspire-skill/bin/python"
WORKSPACE = "昇腾卡公共空间"


def remote(name, command):
    env = dict(os.environ, NO_PROXY="*", no_proxy="*")
    result = subprocess.run([INSPIRE_PYTHON, str(REPO / "data/preprocess/distributed_remote.py"),
                             name, command, "--timeout", "30"], cwd=REPO.parent,
                            env=env, capture_output=True, text=True, timeout=60)
    if result.returncode:
        # API/transport errors can contain authenticated headers. Do not retain
        # stderr or raw CLI output in persistent monitoring logs.
        return dict(instance=name, ok=False, error="remote_execution_failed")
    data = json.loads(result.stdout)
    return dict(instance=name, ok=data["completed"] and data["returncode"] == 0,
                output=data["output"])


def launch_command(root, instance):
    payload = shlex.join(["bash", str(REPO / "data/preprocess/distributed_tmux.sh"),
                          str(root), instance, "full", "16"])
    return "tmux has-session -t openvidhd-full 2>/dev/null || " + shlex.join(
        ["tmux", "new-session", "-d", "-s", "openvidhd-full", "-c", str(REPO), payload])


def maintain(root, instances):
    actions = []
    idle = set()
    # Reclaim only after checking the actual /proc PID and process start ticks
    # on the owner host. The existing local notebook is never stopped.
    local = recover_exited(root, "existing-local")
    actions.append(local)
    if not local["active_owner_count"]:
        idle.add("existing-local")
    for name in instances:
        command = shlex.join(["bash", str(REPO / "data/preprocess/distributed.sh"),
                              "recover-exited", "--output", str(root), "--instance", name])
        try:
            result = remote(name, command)
            actions.append(result)
            if result["ok"]:
                summary = json.loads(result["output"].strip().splitlines()[-1])
                if not summary["active_owner_count"]:
                    idle.add(name)
        except (subprocess.TimeoutExpired, ValueError):
            actions.append(dict(instance=name, ok=False, error="control_request_failed"))
    if progress(root)["unclaimed_batches"]:
        if "existing-local" in idle:
            subprocess.run(["bash", "-c", launch_command(root, "existing-local")], check=True)
        for name in instances:
            if name not in idle:
                continue
            try:
                actions.append(remote(name, launch_command(root, name)))
            except (subprocess.TimeoutExpired, ValueError):
                actions.append(dict(instance=name, ok=False, error="launch_request_failed"))
    return actions


def stop_created_instances(instances):
    results = []
    for name in instances:
        result = subprocess.run(["inspire", "--json", "notebook", "stop", name,
                                 "--workspace", WORKSPACE], capture_output=True,
                                text=True, timeout=120, env=dict(os.environ, NO_PROXY="*", no_proxy="*"))
        results.append(dict(instance=name, stop_command_succeeded=result.returncode == 0))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    root, q = args.output, queue(args.output)
    load_plan(root)
    registry = json.loads((q / "instances.json").read_text())
    instances = [row["name"] for row in registry["instances"]]
    expected = {f"openvidhd-cpu-20261030-{i:02d}" for i in range(1, 9)}
    if set(instances) != expected or len(instances) != 8:
        raise ValueError("Monitor is restricted to the eight task-created instances")
    lock = q / "monitor.claim"
    lock.mkdir()
    atomic_json(lock / "owner.json", dict(pid=os.getpid(), started_at=now(),
                start_ticks=process_identity(os.getpid())["start_ticks"]))
    last_maintenance = 0
    try:
        while True:
            status = progress(root)
            atomic_json(q / "progress.json", status)
            atomic_json(root / "progress.json", status)
            print(json.dumps(status), flush=True)
            if status["completed_batches"] == status["batches"]:
                finalize(root)
                stopped = stop_created_instances(instances)
                atomic_json(q / "instance_cleanup.json", dict(checked_at=now(), instances=stopped))
                return 0
            if time.monotonic() - last_maintenance >= 300:
                actions = maintain(root, instances)
                atomic_json(q / "maintenance.json", dict(checked_at=now(), actions=actions))
                last_maintenance = time.monotonic()
            time.sleep(60)
    finally:
        (lock / "owner.json").unlink()
        lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
