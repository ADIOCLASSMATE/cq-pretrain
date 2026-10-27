"""Per-attempt records and the shell launcher's combined stdout/stderr capture."""

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

from configs.config import resolve_output_paths, save_config
from utils.checkpoint_catalog import durable_json


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def attempt_root(config, *, purpose="train", eval_id=None):
    if purpose == "train":
        return resolve_output_paths(config)["log_dir"] / "train"
    if purpose == "eval":
        from configs.evaluation import resolve_evaluation_paths

        return resolve_evaluation_paths(config, eval_id or config["evaluation"].get("eval_id"))["log_dir"]
    raise ValueError("Attempt purpose must be train or eval")


def create_attempt(config, *, purpose="train", eval_id=None):
    root = attempt_root(config, purpose=purpose, eval_id=eval_id)
    root.mkdir(parents=True, exist_ok=True)
    attempt = (
        root
        / f"attempt_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir()
    return attempt


def resolve_attempt(config, *, purpose="train", eval_id=None):
    existing = os.environ.get("UMM_LOG_ATTEMPT")
    if existing:
        attempt = Path(existing).resolve()
        if (
            attempt.parent != attempt_root(config, purpose=purpose, eval_id=eval_id)
            or not attempt.name.startswith("attempt_")
            or not attempt.is_dir()
        ):
            raise ValueError(
                "Launcher attempt directory does not match the resolved log_dir"
            )
        return attempt
    return create_attempt(config, purpose=purpose, eval_id=eval_id)


class RunLogger:
    def __init__(self, directory, *, enabled=True, purpose="train", eval_id=None):
        self.purpose = purpose
        self.eval_id = eval_id
        self.directory = Path(directory)
        self.enabled = enabled
        self.started = time.monotonic()

    def record(self, filename, payload):
        row = {"timestamp": timestamp(), "attempt_id": self.directory.name, **payload}
        if self.enabled:
            with (self.directory / filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        return row

    def start(self, config, argv, runtime):
        if not self.enabled:
            return
        project = Path(__file__).resolve().parents[1]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True
        ).stdout.strip()
        import hashlib

        source_hashes = {
            str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("training", "evaluations", "utils", "configs", "data", "models")
            for p in (project / folder).rglob("*.py")
            if "references" not in p.parts
        }
        directories = resolve_output_paths(config)
        if self.purpose == "eval":
            from configs.evaluation import resolve_evaluation_paths

            self.eval_id = self.eval_id or config["evaluation"].get("eval_id")
            directories = resolve_evaluation_paths(config, self.eval_id)
        self.run = {
            "purpose": self.purpose,
            "eval_id": self.eval_id if self.purpose == "eval" else None,
            "attempt_id": self.directory.name,
            "started_at": timestamp(),
            "status": "initializing",
            "argv": argv,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "revision": revision,
            "source_sha256": source_hashes,
            "runtime": runtime,
            "directories": {k: str(v) for k, v in directories.items()},
        }
        save_config(config, self.directory / "config.yaml")
        durable_json(self.directory / "run.json", self.run)

    def ready(self, config, *, resume, step, changes, legacy_runtime=False):
        if not self.enabled:
            return
        self.run.update(
            status="running",
            resume=resume,
            start_step=step,
            management_changes=changes,
            legacy_runtime=legacy_runtime,
        )
        save_config(config, self.directory / "config.yaml")
        durable_json(self.directory / "run.json", self.run)

    def finish(self, *, status, step, **details):
        if self.enabled:
            durable_json(
                self.directory / "summary.json",
                {
                    "attempt_id": self.directory.name,
                    "finished_at": timestamp(),
                    "status": status,
                    "step": step,
                    "seconds": time.monotonic() - self.started,
                    **details,
                },
            )


def main(argv=None):
    """Launch once per node; ranks inherit the same attempt from the parent."""
    import argparse
    from configs.config import load_config

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-backend", action="store_true", help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args(argv)
    config = load_config(args.config, args.set)
    if args.print_backend:
        print(config["distributed"]["backend"])
        return 0
    command = [sys.executable]
    if config["distributed"]["backend"] == "mindspeed" and not args.dry_run:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc-per-node={config['distributed']['num_processes']}"]
    command += ["-m", "training.train", *argv]
    if args.dry_run:
        return subprocess.call(command)
    attempt = create_attempt(config)
    env = dict(os.environ, UMM_LOG_ATTEMPT=str(attempt), PYTHONUNBUFFERED="1")
    durable_json(
        attempt / "launch.json",
        {"command": command, "started_at": timestamp(), "pid": os.getpid()},
    )
    print(f"Training logs: {attempt}", flush=True)
    with (attempt / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        def forward(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)

        previous = {
            s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    exit_code = code if code >= 0 else 128 - code
    durable_json(
        attempt / "launch_result.json",
        {"exit_code": exit_code, "finished_at": timestamp()},
    )
    if not (attempt / "summary.json").exists():
        RunLogger(attempt).finish(
            status="failed" if exit_code else "completed",
            step=None,
            exit_code=exit_code,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
