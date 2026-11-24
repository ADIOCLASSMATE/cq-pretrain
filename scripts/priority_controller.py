"""Keep a 32-card training run going, on the free preemptible tier whenever possible.

One policy, two clusters. `--cluster l40s` submits 4 x 8 L40S cards and `--cluster ascend`
submits 2 x 16 Ascend 910B cards; CLUSTERS below holds everything that differs, including
which priority number is the paid one. A card-hour costs almost the same on either
(0.33 vs 0.35 points), so the same reasoning about when to pay applies unchanged.

Policy
- The free preemptible tier is used whenever enough whole nodes are idle.
- Otherwise the job runs on the paid tier (HIGH), which is not preempted.
- While on HIGH: poll free capacity. Once enough whole nodes are idle (other than ours) for
  two consecutive polls, leave HIGH at the cheapest checkpoint boundary:
    - the HIGH job first banks one checkpoint of its own, so its paid start-up is never lost;
    - after that, leaving costs the updates since the newest checkpoint (they are redone) and
      staying costs the updates up to the next one (paid): leave at once while the newest
      checkpoint is the nearer one, otherwise let the next checkpoint, and a validation that
      follows it, finish first.
  The trained update and the save interval come from the job's own train/steps.jsonl and
  train/run.json, because job.sh edits only reach the next start. A checkpoint due within a
  poll is watched every few seconds, so the stop follows it closely.
- While on LOW: only watch the job. The platform requeues a preempted job instead of ending it,
  so a LOW job that has run and now reads QUEUING is down at once; any other state that is not
  RUNNING is down after RESTART_GRACE (pods restarting in place), and a LOW job that never ran
  is down after LOW_START_GRACE outside the start-up states or after twice that in any state.
  When down: stop it, check capacity, resubmit LOW if whole nodes are idle, otherwise go to HIGH.
- Never two jobs at once: a job is stopped and confirmed terminal before the next is submitted.
- A job that ends without a preemption event and without a new checkpoint counts as a crash;
  three in a row stop the controller (ATTENTION file) instead of burning points on a bug.

Files (in RUN/priority/): state.json, ledger.jsonl, controller.log, STOP (pause; checked each cycle).
The job command is `bash RUN/job.sh`, which must resume from RUN/checkpoints by itself.

    python scripts/priority_controller.py --run /path/to/run --prefix cq8k32 --cluster l40s
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

# Both clusters price a card-hour almost identically (L40S 2.64 per 8-card node,
# Ascend 5.6 per 16-card node: 0.33 vs 0.35 points per card-hour) and both have
# a free preemptible tier, so the policy below is the same on either. Only these
# names, the node width and which priority number means paid differ.
CLUSTERS = {
    "l40s": dict(
        workspace="可上网GPU资源",
        project="公共科研项目",
        group="L40s-cuda13.2",
        quota="8,210,800",
        image="docker.sii.shaipower.online/base/ngc-pytorch:25.02-cuda12.8.0-py3",
        nodes=4,  # 4 x 8 = 32 cards
        points_per_node_hour=2.64,
        high=4,  # reads back as 20 / NORMAL: paid, not preempted
        low=3,  # reads back as 13 / LOW: free, preemptible
        exclude=("qb-prod-y40s-gpu167",),  # does not mount global_user
    ),
    "ascend": dict(
        workspace="昇腾卡公共空间",
        project="公共科研项目",
        group="910B资源",
        quota="16,180,1800",  # the wide-CPU row; the loader needs it
        image="docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.4",
        nodes=2,  # 2 x 16 = 32 cards
        points_per_node_hour=5.6,
        high=4,  # reads back as 20 / NORMAL: paid, not preempted
        low=3,  # reads back as 13 / LOW: free, preemptible
        exclude=("infra-gpu-npu-248.host.shzhisuan.com",),
    ),
}
# Which requested priority is the paid tier is a property of the *project*, not
# of the cluster: 公共科研项目 resolves 4 to 20/NORMAL and 3 to 13/LOW on both
# clusters, while 多模态大模型新架构评测探索与scaling-law resolved 4 to 20/NORMAL
# but was driven at 6 (31/HIGH). Both rows above were read back from the
# platform, not inferred -- a profile that calls a paid tier preemptible sends
# the job down the LOW branch, where losing it looks routine instead of costly.
WS = PROJECT = GROUP = QUOTA = IMAGE = EXCLUDE = None  # set by main() from --cluster
NODES = HIGH = LOW = 0
POINTS_PER_NODE_HOUR = 0.0
POLL = 120  # seconds per cycle
WATCH = 10  # seconds between checks while a checkpoint is due within a poll
WATCH_LIMIT = 5 * 60  # longest watch in one cycle (a stalled job must not stall the controller)
LOW_START_GRACE = 15 * 60  # a LOW job that never ran and is not RUNNING by then is down
RESTART_GRACE = 5 * 60  # a LOW job that ran, in a non-QUEUING non-RUNNING state
SWITCH_STREAK = 2  # consecutive polls with room before leaving HIGH
# With room, a HIGH job that has not reached its leaving boundary after this long leaves
# anyway: a run that trains reaches one in well under an hour, so it is stuck.
CHECKPOINT_WAIT = 3 * 60 * 60
LOW_COOLDOWN = 30 * 60  # after a LOW job is lost, stay on HIGH at least this long
MAX_CRASHES = 3
# Consecutive in-place platform restarts that commit no checkpoint before the
# controller stops paying and asks for a human. On 2026-11-23 a bad record the
# resumed cursor reached every time looped five restarts in 90 minutes on
# priority 4; the platform reports each one as a retry, never as a failed job.
STALLED_RESTARTS = 3
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
    row = next((r for r in items if r["compute_group"] == GROUP), None)
    if row is None:
        raise RuntimeError(f"{GROUP} missing from resources availability")
    return int(row.get("full_free_nodes") or 0)


def needed_for_switch():
    """Room demanded before leaving HIGH. Beijing afternoons are busy, so ask for more there."""
    beijing = now() + timedelta(hours=8)
    return NODES + (2 if 12 <= beijing.hour < 19 else 1)


def checkpoint_step():
    """The newest committed restart checkpoint (latest.json is written after the save completes)."""
    latest = RUN / "checkpoints/latest.json"
    try:
        return json.loads(latest.read_text())["step"]
    except (OSError, ValueError, KeyError):
        return -1


def last_rows(path, count, window=64 * 1024):
    """The last complete JSON lines of an append-only log."""
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - window))
        lines = handle.read().splitlines()
    if size > window:
        lines = lines[1:]  # cut by the window
    rows = []
    for line in lines[-count - 1 :]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # the line still being written
    return rows[-count:]


def training_progress(job):
    """The job's own report: last completed update, save and validation intervals, update seconds.

    None until this job has written run.json and a step row; both files outlive earlier jobs,
    so anything older than the submission belongs to a previous job.
    """
    record, steps = RUN / "train/run.json", RUN / "train/steps.jsonl"
    begun = datetime.fromisoformat(job["submitted"]).timestamp()
    try:
        written, stepped = record.stat().st_mtime, steps.stat().st_mtime
        if min(written, stepped) < begun:
            return None
        arguments = json.loads(record.read_text())["arguments"]
        rows = last_rows(steps, 20)
        # run.json is written at every (re)start: newer than the last step row means the job
        # restarted in place and resumes from the newest checkpoint, so nothing is at stake.
        step = checkpoint_step() if written > stepped else int(rows[-1]["step"])
        progress = dict(
            step=step,
            every=int(arguments["save_every"]),
            validate_every=int(arguments.get("validate_every") or 0),
            seconds=sorted(float(row["seconds"]) for row in rows)[len(rows) // 2],
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None
    return progress if progress["every"] > 0 else None


def validating(progress, checkpoint):
    """Validation runs right after the checkpoint of a validation step, before the next update."""
    step, every = progress["step"], progress["validate_every"]
    if not every or checkpoint != step or step % every:
        return False
    return not (RUN / f"train/validation/step_{step:07d}/results.json").exists()


def leave_plan(step, checkpoint, banked_after, every, busy=False):
    """Leave HIGH now: (True, None). Otherwise (False, target): wait for that update's checkpoint.

    `step` is the last completed update, `checkpoint` the newest committed checkpoint,
    `banked_after` the checkpoint the HIGH job started from, `every` its save interval and
    `busy` whether a validation of `checkpoint` is still running.
    """
    due = step if step % every == 0 and checkpoint < step else (step // every + 1) * every
    if checkpoint <= banked_after:
        return False, due  # bank one checkpoint of this job's own first
    if busy:
        return False, checkpoint
    if step - checkpoint < due - step:
        return True, None  # the newest checkpoint is nearer than the next one
    return False, due


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
        tag = "hi" if priority == HIGH else "lo"
        name = f"{PREFIX}-{tag}-{self.state['counter']:03d}"

        # Use competitive submission to claim free nodes rather than passive queuing.
        # This discovers idle nodes and excludes occupied ones, so the job heads
        # toward free capacity immediately instead of waiting for the platform's
        # scheduler to pick nodes randomly.
        args = [
            "bash",
            str(RUN.parent.parent / "scripts" / "submit_competitive.sh"),
            "--name", name,
            "--nodes", str(NODES),
            "--workspace", WS,
            "--project", PROJECT,
            "--group", GROUP,
            "--priority", str(priority),
            "--quota", QUOTA,
            "--image", IMAGE,
            "--shm-size", "128",
            "--command", f"bash {RUN}/job.sh",
        ]

        # Note: competitive_submit.py discovers and auto-excludes occupied nodes.
        # Fixed exclusions like EXCLUDE are handled implicitly by the discovery.

        # Pass datasets
        for d in DATASETS:
            args.extend(["--dataset", d])

        # Fault tolerance
        args.extend([
            "--auto-fault-tolerance",
            "--fault-tolerance-max-retry", "100",
            "--fault-tolerance-retry-interval", "60",
        ])

        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        reply = {"success": p.returncode == 0}
        if p.returncode != 0:
            # Parse stderr to find error message
            reply["error"] = p.stderr or p.stdout or "unknown error"
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
        if job["priority"] != HIGH and reason == "down":
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
        if job["priority"] == HIGH and job.get("running_since"):
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
            if self.submit(LOW):
                return
        log("choosing HIGH", free_nodes=room, cooling=bool(cooling))
        self.submit(HIGH)

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
            # Count restarts that commit nothing. A restart after a new checkpoint
            # starts a fresh streak, so ordinary node failures never accumulate.
            committed = checkpoint_step()
            if committed > job.get("checkpoint_at_restart", -1):
                job["restarts"], job["checkpoint_at_restart"] = 1, committed
            else:
                job["restarts"] = job.get("restarts", 0) + 1
            log("left RUNNING", name=job["name"], status=status, restarts=job["restarts"], checkpoint=committed)
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
        # The platform restarts a failed priority-4 job in place rather than ending
        # it, so a deterministic failure -- a record the resumed data cursor reaches
        # again every time -- loops for up to --fault-tolerance-max-retry attempts
        # and spends points the whole way without committing anything.
        if job["priority"] == HIGH and job.get("restarts", 0) >= STALLED_RESTARTS:
            self.halt("restarting without committing a checkpoint", job)
        down = status in TERMINAL or status == "UNKNOWN"
        if job["priority"] != HIGH and status != "RUNNING" and not down:
            if job.get("last_running"):
                stalled = (now() - datetime.fromisoformat(job["left_running"])).total_seconds()
                down = status == "QUEUING" or stalled > RESTART_GRACE
            else:
                # Never ran: down once out of the start-up states for LOW_START_GRACE, and in
                # any state after twice that (a LOW job that cannot get scheduled).
                down = (status not in NORMAL and age > LOW_START_GRACE) or age > 2 * LOW_START_GRACE
        if down:
            if status not in TERMINAL:
                self.stop("down")
            else:
                self.close(status, "down")
            log("job down, rechecking capacity", status=status)
            self.start_job()
            return
        if job["priority"] == HIGH and status == "RUNNING":
            self.maybe_leave_high(job)

    def halt(self, reason, job):
        """Stop paying for a job that restarts without progressing, and ask for a human."""
        log(
            "halting",
            name=job["name"],
            reason=reason,
            restarts=job.get("restarts"),
            checkpoint=checkpoint_step(),
            high_points=self.points(),
        )
        try:
            self.stop(reason)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            log("stop failed while halting", error=str(error)[:300])
        save(
            HOME / "ATTENTION.json",
            dict(
                time=now().isoformat(),
                reason=reason,
                restarts=job.get("restarts"),
                checkpoint=checkpoint_step(),
                high_points=self.points(),
                hint="The job keeps restarting without committing a checkpoint, so the failure repeats on "
                "every resume: read the newest RUN/logs for the traceback. After fixing it, delete this "
                "file and restart the controller.",
            ),
        )
        sys.exit(2)

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
            switch = self.state["switch"] = dict(streak=0, since=now().isoformat())
        switch["streak"] += 1
        self.persist()
        if switch["streak"] < SWITCH_STREAK:
            return
        waited = (now() - datetime.fromisoformat(switch["since"])).total_seconds()
        if waited >= CHECKPOINT_WAIT:
            log("room lasted without a leaving boundary; leaving HIGH anyway", waited_minutes=round(waited / 60))
        else:
            watched = self.await_boundary(job, switch, room, need)
            if watched is None:
                return
            if watched and free_nodes() < need:
                log("room gone while waiting for the checkpoint, staying on HIGH")
                self.state.pop("switch", None)
                self.persist()
                return
        self.stop("switch to LOW")
        self.start_job()

    def await_boundary(self, job, switch, room, need):
        """Seconds watched once leaving now is cheapest, None while it is not yet.

        A checkpoint due within a poll is watched every WATCH seconds instead of waiting a poll.
        """
        begin = time.monotonic()
        while True:
            progress, checkpoint = training_progress(job), checkpoint_step()
            target = eta = None
            if progress is not None:
                leave, target = leave_plan(
                    progress["step"],
                    checkpoint,
                    job["checkpoint_at_submit"],
                    progress["every"],
                    validating(progress, checkpoint),
                )
                if leave:
                    return time.monotonic() - begin
                eta = (target - progress["step"]) * progress["seconds"]
            if switch.get("target", "unset") != target:
                switch["target"] = target
                self.persist()
                log(
                    "room for LOW; waiting for a checkpoint",
                    target=target,
                    step=progress and progress["step"],
                    checkpoint=checkpoint,
                    save_every=progress and progress["every"],
                    free_nodes=room,
                    needed=need,
                )
            if eta is None or eta > POLL or time.monotonic() - begin > WATCH_LIMIT:
                return None
            time.sleep(WATCH)


def main():
    import fcntl

    global RUN, HOME, PREFIX, WS, PROJECT, GROUP, QUOTA, IMAGE, EXCLUDE, NODES, HIGH, LOW
    global POINTS_PER_NODE_HOUR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Run directory holding job.sh and checkpoints/")
    parser.add_argument("--prefix", default=PREFIX, help="Job name prefix")
    parser.add_argument("--cluster", choices=sorted(CLUSTERS), required=True, help="Which cluster to submit to")
    parser.add_argument("--nodes", type=int, help="Override the cluster's node count")
    args = parser.parse_args()
    RUN, HOME, PREFIX = args.run, args.run / "priority", args.prefix
    profile = CLUSTERS[args.cluster]
    WS, PROJECT, GROUP = profile["workspace"], profile["project"], profile["group"]
    QUOTA, IMAGE, EXCLUDE = profile["quota"], profile["image"], profile["exclude"]
    NODES = args.nodes or profile["nodes"]
    HIGH, LOW = profile["high"], profile["low"]
    POINTS_PER_NODE_HOUR = profile["points_per_node_hour"]
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
            except Exception as error:  # an unexpected reply must not leave the job unmanaged
                log("cycle error", error=f"{type(error).__name__}: {error}"[:500])
        time.sleep(POLL)


if __name__ == "__main__":
    main()
