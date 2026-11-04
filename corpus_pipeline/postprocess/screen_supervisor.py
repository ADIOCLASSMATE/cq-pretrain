"""Resume full-corpus image screening after sealed reconciliation and ready Serving.

The supervisor owns one local screening child, never cluster capacity. It waits
through preemption, binds process identity before signalling/adopting, and runs
an explicitly configured completion command at most once. Credentials only enter
child environments; status and command receipts contain no credential values.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from corpus_pipeline.production.common import atomic_json, canonical, digest, file_sha, require_production_registry
from corpus_pipeline.production.client import endpoint_lease_valid
from corpus_pipeline.production.native_transport import is_native
from .api_screen import verify_input_scope

SCHEMA = "corpus.screen_supervisor.v1"


class ConfigurationError(ValueError):
    pass


def checked_config(value):
    value = dict(value)
    required = {"reconciliation_root", "pool", "output", "python", "workdir", "api_key_file"}
    if not required <= value.keys():
        raise ConfigurationError("missing_required_supervisor_configuration")
    for name in required:
        if not isinstance(value[name], str) or not Path(value[name]).is_absolute() or "\0" in value[name]:
            raise ConfigurationError("invalid_absolute_configuration_path:" + name)
    # In particular, never Path(python).resolve(): it destroys venv interpreter semantics.
    value.setdefault("mode", "primary")
    if value["mode"] not in {"primary", "annotate"}:
        raise ConfigurationError("unsupported_single_model_supervisor_mode")
    value.setdefault("concurrency", 512)
    value.setdefault("max_tokens", 3000)
    value.setdefault("timeout", 600)
    value.setdefault("poll_seconds", 15)
    value.setdefault("max_error_restarts", 3)
    value.setdefault("retry_backoff_seconds", 30)
    value.setdefault("wait_ready", 60)
    for name in ("admission_control",):
        if value.get(name) is not None and (not isinstance(value[name], str) or not Path(value[name]).is_absolute()
                                             or "\0" in value[name]):
            raise ConfigurationError("invalid_absolute_configuration_path:" + name)
    for name, high in (("image_threads", 16), ("progress_interval", 60)):
        if value.get(name) is not None and (type(value[name]) is not int or not 1 <= value[name] <= high):
            raise ConfigurationError("invalid_supervisor_limit:" + name)
    for name, low, high in (("concurrency", 1, 2048), ("max_tokens", 128, 12000),
                            ("timeout", 1, 3600), ("poll_seconds", 1, 60),
                            ("max_error_restarts", 1, 3), ("retry_backoff_seconds", 1, 300),
                            ("wait_ready", 1, 3600)):
        if type(value[name]) is not int or not low <= value[name] <= high:
            raise ConfigurationError("invalid_supervisor_limit:" + name)
    if value.get("prompt") is not None:
        prompt = value["prompt"]
        if not isinstance(prompt, str) or not Path(prompt).is_absolute() or "\0" in prompt:
            raise ConfigurationError("prompt_must_be_an_absolute_path")
        try:
            checksum = file_sha(prompt)
        except OSError as error:
            raise ConfigurationError("prompt_file_unavailable") from error
        if value.get("prompt_sha256") not in (None, checksum):
            raise ConfigurationError("bound_prompt_file_changed")
        value["prompt_sha256"] = checksum
    command = value.get("on_complete_command")
    if command is not None and (not isinstance(command, list) or not command
            or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in command)
            or not Path(command[0]).is_absolute()):
        raise ConfigurationError("completion_command_must_be_absolute_executable_argv")
    return value


def build_command(config, inputs):
    argv = [config["python"], "-m", "corpus_pipeline.postprocess.api_screen", "--backend", "serving",
            "--primary-pool", config["pool"], "--mode", config["mode"], "--output", config["output"],
            "--reconciliation-root", config["reconciliation_root"], "--concurrency", str(config["concurrency"]),
            "--max-tokens", str(config["max_tokens"]), "--timeout", str(config["timeout"]),
            "--wait-ready", str(config["wait_ready"])]
    if config.get("prompt"):
        argv.extend(["--prompt", config["prompt"]])
    for name in ("admission_control", "image_threads", "progress_interval"):
        if config.get(name) is not None:
            argv.extend(["--" + name.replace("_", "-"), str(config[name])])
    for path in inputs:
        argv.extend(["--input", str(path)])
    return argv


def ready_pool(pool_path):
    path = Path(pool_path)
    if not path.exists():
        return {"ready_replicas": 0, "configured_replicas": 0, "ready_weight": 0}
    try:
        value = json.loads(path.read_text())
        if value.get("schema_version") != "corpus.screening_pool.v1" or not isinstance(value.get("registries"), list):
            raise ConfigurationError("invalid_pool_schema")
        ready = 0
        weights = 0.0
        seen = set()
        for item in value["registries"]:
            name = item["path"]
            if not isinstance(name, str) or not Path(name).is_absolute() or name in seen:
                raise ConfigurationError("invalid_or_duplicate_pool_registry")
            seen.add(name)
            weight = item.get("weight", 1)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight < float("inf"):
                raise ConfigurationError("invalid_pool_weight")
            try:
                registry = require_production_registry(name)
                entries = [entry for entry in registry.get("endpoints", [])
                           if entry.get("role") == "verifier" and entry.get("ready") and endpoint_lease_valid(entry)]
                if len(entries) != 1:
                    continue
                endpoint = entries[0]
                if (endpoint.get("backend") != "inspire_serving" or not endpoint.get("supports_images")
                        or (not is_native(endpoint) and not endpoint.get("expected_boots")) or not endpoint.get("base_url")
                        or "qwen" not in endpoint.get("model", "").lower()):
                    continue
                ready += 1
                weights += weight
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return {"ready_replicas": ready, "configured_replicas": len(seen), "ready_weight": weights}
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ConfigurationError("malformed_pool_configuration") from error


def read_progress(output):
    path = Path(output) / "PROGRESS.json"
    try:
        if path.stat().st_size > 4 << 20:
            return {"unavailable": "progress_file_too_large"}
        value = json.loads(path.read_text())
        return {key: value[key] for key in ("counts", "requests", "elapsed_seconds", "updated_at_epoch",
                    "review_assets_per_second", "interrupted", "transport", "cpu", "phase", "imported_rows",
                    "processing_elapsed_seconds", "fresh_actual_responses", "fresh_valid_responses_per_second") if key in value}
    except (OSError, ValueError, TypeError):
        return {}


def runtime_identity():
    return {"host": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


class Runtime:
    def __init__(self):
        self.processes = {}

    def now(self):
        return time.time()

    def identity(self, pid, argv=None):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            actual = [part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
            return {**runtime_identity(), "pid": pid, "start_ticks": int(fields[19]),
                    "argv": actual if argv is None else list(argv), "argv_sha256": digest(actual if argv is None else list(argv))}
        except FileNotFoundError:
            return None

    def status(self, identity, command):
        if identity is None:
            return "unknown"
        current_runtime = runtime_identity()
        if identity.get("host") != current_runtime["host"]:
            return "unknown"
        if identity.get("boot_id") != current_runtime["boot_id"]:
            return "dead"
        if identity.get("known_exited"):
            return "dead"
        try:
            pid = identity["pid"]
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[19]) != identity["start_ticks"] or fields[0] == "Z":
                return "dead"
            actual = [part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
            if actual != command or identity.get("argv") != command or identity.get("argv_sha256") != digest(command):
                return "unknown"
            return "alive"
        except FileNotFoundError:
            return "dead"
        except (OSError, ValueError, KeyError, IndexError):
            return "unknown"

    def find(self, command, workdir):
        matches = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            path = Path("/proc") / name
            try:
                if path.stat().st_uid != os.geteuid():
                    continue
                actual = [part.decode() for part in (path / "cmdline").read_bytes().split(b"\0") if part]
                if actual != command:
                    continue
                if os.path.realpath(path / "cwd") != os.path.realpath(workdir):
                    raise ConfigurationError("matching_worker_has_unexpected_workdir")
                identity = self.identity(int(name), command)
                if identity and self.status(identity, command) == "alive":
                    matches.append(identity)
            except FileNotFoundError:
                continue
            except PermissionError as error:
                raise ConfigurationError("cannot_prove_matching_worker_absence") from error
        return matches

    def start(self, command, *, workdir, environment, log_path):
        fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=workdir, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        self.processes[child.pid] = child
        identity = self.identity(child.pid, command)
        if identity is None and child.poll() is not None:
            identity = {**runtime_identity(), "pid": child.pid, "known_exited": True,
                        "argv": command, "argv_sha256": digest(command)}
        if identity is None:
            raise RuntimeError("spawn_result_identity_unknown")
        return identity

    def exit_code(self, identity):
        process = self.processes.get(identity.get("pid")) if identity else None
        return process.poll() if process is not None else None

    def terminate(self, identity, command):
        if self.status(identity, command) != "alive":
            raise ConfigurationError("refuse_to_signal_unverified_process")
        os.kill(identity["pid"], signal.SIGTERM)


@contextmanager
def supervisor_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    owner = directory / "OWNER.json"
    host = socket.gethostname()
    try:
        fd = os.open(owner, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if json.loads(owner.read_text()).get("host") != host:
            raise ConfigurationError("supervisor_state_owned_by_another_host")
    else:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical({"host": host}))
    lock_root = Path("/tmp/corpus-screen-supervisor-locks")
    lock_root.mkdir(mode=0o700, exist_ok=True)
    with (lock_root / (digest(str(directory.resolve())) + ".lock")).open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConfigurationError("another_screening_supervisor_is_active") from None
        yield


class Supervisor:
    def __init__(self, config, *, runtime=None):
        self.config = checked_config(config)
        self.runtime = runtime or Runtime()
        self.output = Path(config["output"])
        self.directory = self.output / "supervisor"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "STATE.json"
        self.stop_path = self.directory / "STOP"
        self.config_sha = digest(self.config)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("config_sha256") != self.config_sha:
                raise ConfigurationError("immutable_supervisor_configuration_changed")
        else:
            self.state = {"schema": SCHEMA, "config_sha256": self.config_sha, "launches": 0,
                          "errors": 0, "child": None, "status": "starting", "published_training": False}
            atomic_json(self.directory / "CONFIG.json", self.config)
        if self.state.get("status") == "stopped" and not self.stop_path.exists():
            self.state.update(terminal=False, status="resuming_after_stop", stop_sent=False)
        self.state["supervisor"] = self.runtime.identity(os.getpid())
        self.save()

    def save(self, status=None, **values):
        if status is not None:
            self.state["status"] = status
        self.state.update(values, updated_at_epoch=self.runtime.now(), progress=read_progress(self.output))
        atomic_json(self.path, self.state)
        return self.state

    def reconciliation(self):
        root = Path(self.config["reconciliation_root"])
        path = root / "COMPLETE.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if (data.get("schema") != "corpus.asset_reconciliation.v1" or data.get("complete") is not True
                or data.get("scope") != "full_run"):
            raise ConfigurationError("reconciliation_is_not_a_completed_full_run")
        inputs = [root / "buckets" / f"{index:02x}" / "api_review.parquet" for index in range(256)]
        checksum = file_sha(path)
        old = self.state.get("reconciliation_complete_sha256")
        if old and old != checksum:
            raise ConfigurationError("sealed_reconciliation_changed")
        if not old:
            scope = verify_input_scope(inputs, root)
            atomic_json(self.directory / "INPUT_GATE.json", scope)
            self.save(reconciliation_complete_sha256=checksum, input_gate_sha256=file_sha(self.directory / "INPUT_GATE.json"))
        elif (not (self.directory / "INPUT_GATE.json").exists()
                or file_sha(self.directory / "INPUT_GATE.json") != self.state.get("input_gate_sha256")):
            raise ConfigurationError("supervisor_input_gate_missing_or_changed")
        return inputs

    def completion(self, *, require_fresh=False):
        path = self.output / "COMPLETE.json"
        if not path.exists():
            return None
        checksum = file_sha(path)
        if require_fresh and checksum == self.state.get("launch_intent", {}).get("previous_complete_sha256"):
            return None
        value = json.loads(path.read_text())
        if (value.get("schema_version") != "postprocess.api_screen.v1" or value.get("published_training") is not False
                or value.get("scope", {}).get("global_screening_scope_authorized") is not True
                or value.get("scope", {}).get("reconciliation_complete_sha256") != self.state.get("reconciliation_complete_sha256")):
            raise ConfigurationError("screening_completion_scope_mismatch")
        for field, name in (("input_sha256", "INPUT.json"), ("summary_sha256", "SUMMARY.json")):
            target = self.output / name
            if not target.exists() or file_sha(target) != value.get(field):
                raise ConfigurationError("screening_completion_input_or_summary_hash_mismatch")
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ConfigurationError("screening_completion_artifacts_missing")
        for item in artifacts:
            target = (self.output / item["path"]).resolve()
            if not target.is_relative_to(self.output.resolve()) or not target.exists() or file_sha(target) != item["sha256"]:
                raise ConfigurationError("screening_completion_artifact_hash_mismatch")
        summary = json.loads((self.output / "SUMMARY.json").read_text())
        if type(value.get("screening_pass_complete")) is not bool:
            raise ConfigurationError("screening_completion_flag_invalid")
        counts = summary.get("counts", {})
        if value["screening_pass_complete"] and (counts.get("state:pending", 0) or counts.get("state:retryable", 0)):
            raise ConfigurationError("screening_complete_has_unfinished_work")
        return {"receipt": value, "sha256": checksum, "summary": summary}

    def environment(self):
        try:
            key = Path(self.config["api_key_file"]).read_text().strip()
        except OSError as error:
            raise ConfigurationError("serving_key_file_unavailable") from error
        if not key or len(key) > 4096 or "\n" in key or "\r" in key:
            raise ConfigurationError("invalid_serving_key_file")
        environment = os.environ.copy()
        environment["INF_API_KEY"] = key
        environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        return environment

    def classify_failure(self):
        path = self.state.get("log_path")
        tail = ""
        if path and Path(path).exists():
            with Path(path).open("rb") as stream:
                stream.seek(max(0, Path(path).stat().st_size - 32768))
                tail = stream.read().decode(errors="replace")
        if "no_qualified_ready_serving_replica" in tail:
            return "temporary_readiness_race"
        if any(token in tail for token in ("ValueError:", "ConfigurationError:", "PermissionError:",
            "ModuleNotFoundError:", "FileNotFoundError:", "sqlite3.DatabaseError:", "sqlite3.IntegrityError:")):
            return "configuration_or_integrity_failure"
        return "unexpected_worker_exit"

    def schedule_retry(self, reason, *, charge_error):
        errors = self.state.get("errors", 0) + 1 if charge_error else 0
        self.save(child=None, launch_intent=None, errors=errors, last_failure=reason)
        if errors >= self.config["max_error_restarts"]:
            return self.save("failed_retry_budget_exhausted", terminal=True, exit_code=1)
        delay = min(300, self.config["retry_backoff_seconds"] * 2 ** max(0, errors - 1))
        return self.save("waiting_to_resume", next_retry_epoch=self.runtime.now() + delay)

    def finish(self, completed):
        self.save(completion_sha256=completed["sha256"], screening_pass_complete=True,
                  all_reviews_resolved=completed["receipt"].get("all_reviews_resolved", False), child=None, launch_intent=None)
        command = self.config.get("on_complete_command")
        if not command:
            return self.save("completed", terminal=True, exit_code=0)
        hook_path = self.directory / "ON_COMPLETE.json"
        if hook_path.exists():
            hook = json.loads(hook_path.read_text())
            if hook.get("command") != command or hook.get("completion_sha256") != completed["sha256"]:
                raise ConfigurationError("completion_hook_identity_mismatch")
            if hook.get("state") == "completed":
                success = hook.get("returncode") == 0
                return self.save("completed" if success else "completed_cleanup_failed", terminal=True,
                                 exit_code=0 if success else 2, cleanup=hook)
            identity = hook.get("process")
            if identity is None:
                matches = self.runtime.find(command, self.config["workdir"])
                if len(matches) == 1:
                    identity = matches[0]
                    hook["process"] = identity
                    atomic_json(hook_path, hook)
                else:
                    return self.save("completed_cleanup_outcome_unknown", terminal=True, exit_code=2,
                                     cleanup={"state": "unknown", "automatic_retry": False})
            status = self.runtime.status(identity, command)
            if status == "alive":
                return self.save("running_completion_cleanup", cleanup={"state": "running", "pid": identity["pid"]})
            if status == "unknown":
                return self.save("completed_cleanup_identity_unknown", terminal=True, exit_code=2)
            code = self.runtime.exit_code(identity)
            hook.update(state="completed", returncode=code, completed_at_epoch=self.runtime.now(),
                        automatic_retry=False)
            atomic_json(hook_path, hook)
            return self.finish(completed)
        hook = {"state": "intent", "command": command, "completion_sha256": completed["sha256"],
                "created_at_epoch": self.runtime.now(), "automatic_retry": False}
        atomic_json(hook_path, hook)  # At-most-once; an ambiguous send is never replayed.
        identity = self.runtime.start(command, workdir=self.config["workdir"], environment=self.environment(),
                                      log_path=str(self.directory / "on_complete.log"))
        hook.update(state="running", process=identity)
        atomic_json(hook_path, hook)
        return self.save("running_completion_cleanup", cleanup={"state": "running", "pid": identity["pid"]})

    def stop(self):
        if self.state.get("status") == "running_completion_cleanup":
            completed = self.completion()
            if completed:
                return self.finish(completed)
        identity = self.state.get("child")
        command = self.state.get("command")
        if identity:
            status = self.runtime.status(identity, command)
            if status == "unknown":
                return self.save("stop_blocked_unknown_child_identity", terminal=True, exit_code=2)
            if status == "alive":
                if not self.state.get("stop_sent"):
                    self.runtime.terminate(identity, command)
                    self.save(stop_sent=True, stop_sent_at_epoch=self.runtime.now())
                return self.save("draining_child_after_stop")
        return self.save("stopped", terminal=True, exit_code=130)

    def block_and_drain(self, error):
        failure = {"error_type": type(error).__name__,
                   "error_code": str(error) if isinstance(error, ConfigurationError) else "input_or_state_validation_failed"}
        identity, command = self.state.get("child"), self.state.get("command")
        if identity and self.runtime.status(identity, command) == "alive":
            if not self.state.get("stop_sent"):
                self.runtime.terminate(identity, command)
            return self.save("draining_after_configuration_error", fatal_failure=failure, stop_sent=True)
        return self.save("blocked_configuration_or_integrity", terminal=True, exit_code=2, **failure)

    def tick(self):
        if self.state.get("terminal"):
            return self.state
        if self.state.get("fatal_failure"):
            identity, command = self.state.get("child"), self.state.get("command")
            if identity and self.runtime.status(identity, command) == "alive":
                return self.save("draining_after_configuration_error")
            return self.save("blocked_configuration_or_integrity", terminal=True, exit_code=2, **self.state["fatal_failure"])
        if self.stop_path.exists():
            return self.stop()
        if self.config.get("prompt") and file_sha(self.config["prompt"]) != self.config["prompt_sha256"]:
            raise ConfigurationError("bound_prompt_file_changed")
        if self.state.get("child"):
            identity = self.state["child"]
            status = self.runtime.status(identity, self.state["command"])
            if status == "unknown":
                return self.save("blocked_unknown_child_identity", terminal=True, exit_code=2)
            if status == "alive":
                return self.save("screening_running", pool=ready_pool(self.config["pool"]))
            exit_code = self.runtime.exit_code(identity)
            completed = self.completion(require_fresh=True)
            if completed and completed["receipt"]["screening_pass_complete"]:
                return self.finish(completed)
            if completed and exit_code in (None, 0, 130):
                counts = completed["summary"].get("counts", {})
                if counts.get("state:retryable", 0) or counts.get("state:pending", 0) or completed["summary"].get("interrupted"):
                    return self.schedule_retry("deferred_or_interrupted_reviews", charge_error=False)
                raise ConfigurationError("incomplete_screening_without_resumable_work")
            reason = self.classify_failure()
            if reason == "configuration_or_integrity_failure":
                return self.save("blocked_configuration_or_integrity", terminal=True, exit_code=2, worker_exit_code=exit_code)
            return self.schedule_retry(reason, charge_error=reason != "temporary_readiness_race")
        inputs = self.reconciliation()
        if inputs is None:
            return self.save("waiting_for_full_reconciliation")
        command = build_command(self.config, inputs)
        existing = self.runtime.find(command, self.config["workdir"])
        if len(existing) > 1:
            raise ConfigurationError("multiple_matching_screening_workers")
        if existing:
            self.save(child=existing[0], command=command)
            return self.save("adopted_existing_screening_worker")
        if self.state.get("launch_intent"):
            # An intent without a PID can be a crash before exec. Exact argv
            # absence alone cannot prove that delayed exec will never appear.
            return self.save("blocked_spawn_outcome_unknown", terminal=True, exit_code=2,
                             automatic_retry=False)
        completed = self.completion()
        if completed and completed["receipt"]["screening_pass_complete"]:
            return self.finish(completed)
        pool = ready_pool(self.config["pool"])
        if not pool["ready_replicas"]:
            return self.save("waiting_for_ready_serving", pool=pool)
        if self.runtime.now() < self.state.get("next_retry_epoch", 0):
            return self.save("waiting_to_resume", pool=pool)
        environment = self.environment()
        number = self.state["launches"] + 1
        log_path = self.directory / f"worker_{number:05d}.log"
        complete_path = self.output / "COMPLETE.json"
        intent = {"created_at_epoch": self.runtime.now(), "command_sha256": digest(command),
                  "previous_complete_sha256": file_sha(complete_path) if complete_path.exists() else None}
        self.save("launching", launch_intent=intent, command=command, launches=number, log_path=str(log_path), stop_sent=False)
        identity = self.runtime.start(command, workdir=self.config["workdir"], environment=environment, log_path=str(log_path))
        return self.save("screening_running", child=identity, pool=pool)


def run(config):
    config = checked_config(config)
    directory = Path(config["output"]) / "supervisor"
    with supervisor_lock(directory):
        supervisor = Supervisor(config)
        def stop(signum, frame):
            supervisor.stop_path.touch(mode=0o600)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while True:
            try:
                state = supervisor.tick()
            except (ConfigurationError, ValueError, KeyError, TypeError, OSError) as error:
                state = supervisor.block_and_drain(error)
            if state.get("terminal"):
                return state.get("exit_code", 0)
            time.sleep(config["poll_seconds"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop", action="store_true", help="Request graceful stop without signalling an unverified PID")
    args = parser.parse_args(argv)
    config = checked_config(json.loads(Path(args.config).read_text()))
    if args.stop:
        path = Path(config["output"]) / "supervisor" / "STOP"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        return 0
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
