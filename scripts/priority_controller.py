"""Keep the 32-GPU L40S training running, on low priority whenever possible.

Policy
- Low priority (free points, preemptible) is used whenever 4 whole L40S nodes are idle.
- Otherwise the job runs on priority 4 (HIGH, paid with public-project points, not preempted).
- While on HIGH: poll free capacity. Once enough whole nodes are idle (other than ours) for
  two consecutive polls, wait for the next committed checkpoint, then stop HIGH and submit LOW.
- While on LOW: only watch the job. The platform requeues a preempted job instead of ending it,
  so a LOW job that has run and now reads QUEUING is down at once; any other state that is not
  RUNNING is down after RESTART_GRACE (pods restarting in place), and a LOW job that never ran
  is down after LOW_START_GRACE. When down: stop it, check capacity, resubmit LOW if whole nodes
  are idle, otherwise go to HIGH.
- Never two jobs at once: a job is stopped and confirmed terminal before the next is submitted.
- A job that ends without a preemption event and without a new checkpoint counts as a crash;
  three in a row stop the controller (ATTENTION file) instead of burning points on a bug.

Files (in RUN/priority/): state.json, ledger.jsonl, controller.log, STOP (pause; checked each cycle).
The job command is `bash RUN/job.sh`, which must resume from RUN/checkpoints by itself.

    python scripts/priority_controller.py --run /path/to/run --prefix cq8k32
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN = HOME = None  # set by main()
PREFIX = "cq8k32"
CLI = [str(Path.home() / ".local/bin/inspire"), "--json"]
WS, PROJECT, GROUP = "可上网GPU资源", "公共科研项目", "L40s-cuda13.2"
IMAGE = "docker.sii.shaipower.online/base/ngc-pytorch:25.02-cuda12.8.0-py3"
DATASETS = [
    "dcvlm-baseline-200b:v2",
    "gen-und-mm-pretrain:01",
    "imgedit:v1",
    "infovis:v1",
    "textground4m:v1",
    "textatlas5m:v1",
    "poster100k:v1",
    "tide:v1",
]
NODES = 4
POINTS_PER_NODE_HOUR = 2.64  # quota row 8,210,800
POLL = 120  # seconds per cycle
LOW_START_GRACE = 15 * 60  # a LOW job that never ran and is not RUNNING by then is down
RESTART_GRACE = 5 * 60  # a LOW job that ran, in a non-QUEUING non-RUNNING state
SWITCH_STREAK = 2  # consecutive polls with room before leaving HIGH
CHECKPOINT_WAIT = 60 * 60  # longest wait for a fresh checkpoint before switching
LOW_COOLDOWN = 30 * 60  # after a LOW job is lost, stay on HIGH at least this long
MAX_CRASHES = 3
NORMAL = {"RUNNING", "PENDING", "CREATING", "SUBMITTED"}
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "STOPPED"}


def now():
    return datetime.now(timezone.utc)


def log(message, **fields):
    line = json.dumps(dict(time=now().isoformat(timespec="seconds"), message=message, **fields), ensure_ascii=False)
    with (HOME / "controller.log").open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def cli(*args, timeout=180, check=True):
    reply = {}
    for attempt in range(4):
        p = subprocess.run([*CLI, *args], capture_output=True, text=True, timeout=timeout)
        try:
            reply = json.loads(p.stdout or p.stderr)
        except ValueError:
            reply = {"success": False, "error": (p.stdout or p.stderr)[-500:]}
        if reply.get("success") or not check:
            return reply
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"inspire {' '.join(args[:3])}: {str(reply)[:500]}")


def free_nodes():
    """Whole idle L40S nodes; our own running job occupies its nodes, so they are not counted."""
    items = cli("resources", "availability", "--workspace", WS)["data"]["items"]
    row = next(r for r in items if r["compute_group"] == GROUP)
    return int(row.get("full_free_nodes") or 0)


def needed_for_switch():
    """Room demanded before leaving HIGH. Beijing afternoons are busy, so ask for more there."""
    beijing = now() + timedelta(hours=8)
    return NODES + (2 if 12 <= beijing.hour < 19 else 1)


def checkpoint_step():
    latest = RUN / "checkpoints/latest.json"
    try:
        return json.loads(latest.read_text())["step"]
    except (OSError, ValueError, KeyError):
        return -1


class Controller:
    def __init__(self):
        HOME.mkdir(parents=True, exist_ok=True)
        path = HOME / "state.json"
        self.state = (
            json.loads(path.read_text())
            if path.exists()
            else dict(job=None, counter=0, crashes=0, high_seconds=0.0, low_lost_at=None)
        )

    def persist(self):
        save(HOME / "state.json", self.state)

    def status(self, name):
        data = cli("job", "status", name, "--workspace", WS)["data"]
        return (data.get("status") or "UNKNOWN").upper(), data

    def preempted(self, name):
        p = subprocess.run(
            [*CLI[:1], "job", "events", name, "--workspace", WS, "--tail", "200"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return "preempt" in (p.stdout + p.stderr).lower()

    def submit(self, priority):
        self.state["counter"] += 1
        tag = "hi" if priority == 4 else "lo"
        name = f"{PREFIX}-{tag}-{self.state['counter']:03d}"
        args = [
            "job",
            "create",
            "-n",
            name,
            "--workspace",
            WS,
            "-p",
            PROJECT,
            "--group",
            GROUP,
            "-q",
            "8,210,800",
            "--nodes",
            str(NODES),
            "-i",
            IMAGE,
            "--priority",
            str(priority),
            "--shm-size",
            "128",
            "--exclude-node",
            "qb-prod-y40s-gpu167",
            "--auto-fault-tolerance",
            "--fault-tolerance-max-retry",
            "100",
            "--fault-tolerance-retry-interval",
            "60",
            *[x for d in DATASETS for x in ("--dataset", d)],
            "-c",
            f"bash {RUN}/job.sh",
        ]
        reply = cli(*args, check=False)
        if not reply.get("success"):
            log("submit rejected", name=name, priority=priority, error=str(reply.get("error"))[:400])
            self.persist()
            return False
        self.state["job"] = dict(
            name=name,
            priority=priority,
            submitted=now().isoformat(),
            running_since=None,
            checkpoint_at_submit=checkpoint_step(),
        )
        self.state.pop("switch", None)
        self.persist()
        log("submitted", name=name, priority=priority, checkpoint=checkpoint_step())
        return True

    def stop(self, reason):
        job = self.state["job"]
        log("stopping", name=job["name"], reason=reason)
        cli("job", "stop", job["name"], "--workspace", WS, check=False)
        for _ in range(30):
            status, _ = self.status(job["name"])
            if status in TERMINAL:
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"{job['name']} did not terminate after stop")
        self.close(status, reason)

    def close(self, status, reason):
        """Account for the finished job and decide whether it was a crash."""
        job = self.state["job"]
        self.account(job, status_running=False)
        progressed = checkpoint_step() > job["checkpoint_at_submit"]
        crash = status == "FAILED" and not progressed and reason == "down" and not self.preempted(job["name"])
        self.state["crashes"] = self.state["crashes"] + 1 if crash else 0
        if job["priority"] != 4 and reason == "down":
            self.state["low_lost_at"] = now().isoformat()
        with (HOME / "ledger.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        job=job,
                        end=now().isoformat(),
                        status=status,
                        reason=reason,
                        checkpoint=checkpoint_step(),
                        crash=crash,
                        high_points_total=self.points(),
                    )
                )
                + "\n"
            )
        self.state["job"] = None
        self.persist()

    def account(self, job, status_running):
        """Accumulate HIGH running time for the points estimate."""
        stamp = now()
        if job["priority"] == 4 and job.get("running_since"):
            last = datetime.fromisoformat(job.get("accounted") or job["running_since"])
            self.state["high_seconds"] += (stamp - last).total_seconds()
            job["accounted"] = stamp.isoformat()
        if not status_running:
            job["accounted"] = None

    def points(self):
        return round(self.state["high_seconds"] / 3600 * NODES * POINTS_PER_NODE_HOUR, 1)

    def start_job(self):
        room = free_nodes()
        lost = self.state.get("low_lost_at")
        cooling = lost and (now() - datetime.fromisoformat(lost)).total_seconds() < LOW_COOLDOWN
        if room >= NODES and not cooling:
            log("choosing LOW", free_nodes=room)
            if self.submit(3):
                return
        log("choosing HIGH", free_nodes=room, cooling=bool(cooling))
        self.submit(4)

    def cycle(self):
        if self.state["crashes"] >= MAX_CRASHES:
            save(
                HOME / "ATTENTION.json",
                dict(
                    time=now().isoformat(),
                    crashes=self.state["crashes"],
                    hint="Jobs end without progress; read RUN/logs. Delete this file and reset crashes in state.json to continue.",
                ),
            )
            log("halting after repeated crashes")
            sys.exit(2)
        job = self.state["job"]
        if job is None:
            self.start_job()
            return
        status, _ = self.status(job["name"])
        age = (now() - datetime.fromisoformat(job["submitted"])).total_seconds()
        if status == "RUNNING":
            if not job.get("running_since"):
                job["running_since"] = now().isoformat()
                log("running", name=job["name"], priority=job["priority"])
            job["last_running"] = now().isoformat()
            self.account(job, status_running=True)
            self.persist()
        elif job.get("last_running") and not job.get("left_running"):
            job["left_running"] = now().isoformat()
            log("left RUNNING", name=job["name"], status=status)
            self.persist()
        if status == "SUCCEEDED":
            self.close(status, "finished")
            save(
                HOME / "DONE.json",
                dict(time=now().isoformat(), checkpoint=checkpoint_step(), high_points=self.points()),
            )
            log("training finished", high_points=self.points())
            sys.exit(0)
        if status == "RUNNING":
            job.pop("left_running", None)
        down = status in TERMINAL or status == "UNKNOWN"
        if job["priority"] != 4 and status != "RUNNING" and not down:
            if job.get("last_running"):
                stalled = (now() - datetime.fromisoformat(job["left_running"])).total_seconds()
                down = status == "QUEUING" or stalled > RESTART_GRACE
            else:
                down = status not in NORMAL and age > LOW_START_GRACE or age > 2 * LOW_START_GRACE
        if down:
            if status not in TERMINAL:
                self.stop("down")
            else:
                self.close(status, "down")
            log("job down, rechecking capacity", status=status)
            self.start_job()
            return
        if job["priority"] == 4 and status == "RUNNING":
            self.maybe_leave_high(job)

    def maybe_leave_high(self, job):
        lost = self.state.get("low_lost_at")
        if lost and (now() - datetime.fromisoformat(lost)).total_seconds() < LOW_COOLDOWN:
            return
        room, need = free_nodes(), needed_for_switch()
        switch = self.state.get("switch")
        if room < need:
            if switch:
                log("room gone, staying on HIGH", free_nodes=room, needed=need)
            self.state.pop("switch", None)
            self.persist()
            return
        if not switch:
            switch = self.state["switch"] = dict(streak=0, checkpoint=checkpoint_step(), since=now().isoformat())
        switch["streak"] += 1
        self.persist()
        if switch["streak"] < SWITCH_STREAK:
            return
        waited = (now() - datetime.fromisoformat(switch["since"])).total_seconds()
        if checkpoint_step() == switch["checkpoint"] and waited < CHECKPOINT_WAIT:
            if switch["streak"] == SWITCH_STREAK:
                log("room for LOW; waiting for the next checkpoint", free_nodes=room, needed=need)
            return
        self.stop("switch to LOW")
        self.start_job()


def main():
    import fcntl

    global RUN, HOME, PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Run directory holding job.sh and checkpoints/")
    parser.add_argument("--prefix", default=PREFIX, help="Job name prefix")
    args = parser.parse_args()
    RUN, HOME, PREFIX = args.run, args.run / "priority", args.prefix
    HOME.mkdir(parents=True, exist_ok=True)
    lock = (HOME / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (HOME / "controller.pid").write_text(str(os.getpid()))
    controller = Controller()
    log("start", job=controller.state["job"], high_points=controller.points())
    while True:
        if not (HOME / "STOP").exists():
            try:
                controller.cycle()
            except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
                log("cycle error", error=str(error)[:500])
        time.sleep(POLL)


if __name__ == "__main__":
    main()
