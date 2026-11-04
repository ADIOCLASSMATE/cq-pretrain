"""Local outer guardian for one frozen screening run and its owned Serving pool.

This module never calls a model or platform API. It may restore only the exact
screen-supervisor and pool-guard commands, after absence and lock checks. The
pool guard remains the sole path to its runner. A successful stop subprocess
exit is NOT resource-release evidence. Stopping this watchdog leaves children
running; an explicit STOP file or signal stops only this outer guardian.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import time

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from .screen_supervisor import (
    ConfigurationError, Runtime, Supervisor, checked_config, read_progress, supervisor_lock,
)

SCHEMA = "corpus.full_run_watchdog.v1"
TERMINAL_INSTANCES = {"INSTANCE_STOPPED", "INSTANCE_FAILED", "INSTANCE_SUCCEEDED",
                      "INSTANCE_DELETED", "INSTANCE_TERMINATED", "INSTANCE_COMPLETED"}


def require(condition, code):
    if not condition:
        raise ConfigurationError(code)


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def available_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def checked_watchdog_config(value):
    value = dict(value)
    require(value.get("schema_version") == SCHEMA, "watchdog_schema_mismatch")
    for name in ("state_dir", "workdir", "screen_config", "pool_config"):
        path = value.get(name)
        require(isinstance(path, str) and Path(path).is_absolute() and "\0" not in path,
                "watchdog_absolute_path_required:" + name)
    for name in ("supervisor_command", "guard_command"):
        command = value.get(name)
        require(isinstance(command, list) and command and Path(command[0]).is_absolute()
                and all(isinstance(arg, str) and arg and "\0" not in arg for arg in command),
                "watchdog_argv_invalid:" + name)
    require(value["supervisor_command"][1:] == ["-m", "corpus_pipeline.postprocess.screen_supervisor",
            "--config", value["screen_config"]], "watchdog_supervisor_command_mismatch")
    require(value["guard_command"][1:] == ["-m", "corpus_pipeline.serving.budget_pool", "guard",
            "--config", value["pool_config"]], "watchdog_guard_command_mismatch")
    hashes = value.get("expected_hashes")
    require(isinstance(hashes, dict) and hashes, "watchdog_hash_bindings_required")
    for path, sha in hashes.items():
        require(Path(path).is_absolute() and isinstance(sha, str) and len(sha) == 64
                and all(c in "0123456789abcdef" for c in sha), "watchdog_hash_binding_invalid")
    initial = value.get("initial_processes", {})
    for role in ("supervisor", "guard"):
        identity = initial.get(role)
        command = value[role + "_command"]
        require(isinstance(identity, dict) and identity.get("argv") == command
                and identity.get("argv_sha256") == digest(command)
                and type(identity.get("pid")) is int and identity["pid"] > 0
                and type(identity.get("start_ticks")) is int
                and bool(identity.get("host")) and bool(identity.get("boot_id")),
                "watchdog_initial_process_identity_invalid:" + role)
    for name, default, low, high in (("poll_seconds", 15, 1, 60),
            ("restart_backoff_seconds", 30, 1, 300), ("max_restarts_per_hour", 3, 1, 10)):
        value.setdefault(name, default)
        require(type(value[name]) is int and low <= value[name] <= high, "watchdog_limit_invalid:" + name)
    return value


class Watchdog:
    def __init__(self, config, *, runtime=None):
        self.config = checked_watchdog_config(config)
        self.runtime = runtime or Runtime()
        self.directory = Path(self.config["state_dir"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "STATE.json"
        self.config_sha = digest(self.config)
        if self.path.exists():
            self.state = read(self.path)
            require(self.state.get("config_sha256") == self.config_sha,
                    "watchdog_immutable_configuration_changed")
        else:
            self.state = {"schema_version": SCHEMA, "config_sha256": self.config_sha,
                          "processes": json.loads(json.dumps(self.config["initial_processes"])), "restarts": {}, "intents": {}}
        self.screen = checked_config(read(self.config["screen_config"]))
        self.pool = read(self.config["pool_config"])
        self.output = Path(self.screen["output"])
        self.pool_root = Path(self.pool["root"])
        self.supervisor_dir = self.output / "supervisor"
        self.completed = None
        self.validate_bindings()

    def save(self, status, **values):
        self.state.update(status=status, updated_at_epoch=self.runtime.now(), **values)
        self.state["progress"] = read_progress(self.output)
        atomic_json(self.path, self.state)
        return self.state

    def validate_bindings(self):
        hashes = self.config["expected_hashes"]
        root = Path(self.config["workdir"])
        required = {self.config["screen_config"], self.config["pool_config"], str(Path(__file__).absolute()),
                    str(self.output / "INPUT.json"), str(self.output / "INPUT_IMPORTED.json"),
                    str(self.supervisor_dir / "INPUT_GATE.json"), self.screen["prompt"],
                    str(Path(self.screen["reconciliation_root"]) / "COMPLETE.json"), self.pool["backend_file"]}
        required.update(str(root / "corpus_pipeline" / name) for name in (
            "postprocess/api_screen.py", "postprocess/screen_supervisor.py", "production/common.py",
            "serving/budget_pool.py"))
        identity = read(self.output / "INPUT.json")
        required.update(str(root / "corpus_pipeline" / name) for name in identity.get("dependencies", {}))
        require(required <= hashes.keys(), "watchdog_required_file_binding_missing")
        for path, sha in hashes.items():
            require(file_sha(path) == sha, "watchdog_frozen_file_changed:" + path)
        for name, sha in identity.get("dependencies", {}).items():
            require(hashes[str(root / "corpus_pipeline" / name)] == sha,
                    "watchdog_input_dependency_changed:" + name)
        require(identity.get("mode") == "annotate", "watchdog_input_mode_mismatch")
        require(self.config["supervisor_command"][0] == self.screen["python"], "watchdog_venv_argv_changed")
        require(self.screen["workdir"] == self.config["workdir"] and self.screen["mode"] == "annotate",
                "watchdog_screen_scope_invalid")
        require(self.pool.get("schema_version") == "corpus.native_budget_pool.v1"
                and self.pool_root.is_absolute() and self.screen["pool"] == str(self.pool_root / "pool.json"),
                "watchdog_pool_scope_invalid")
        stop_command = self.config["guard_command"][:3] + ["stop", "--config", self.config["pool_config"]]
        require(self.screen.get("on_complete_command") == stop_command, "watchdog_completion_command_mismatch")
        require(0 < self.pool["cards_per_replica"] * self.pool["replicas"] <= self.pool["max_cards"] <= 128,
                "watchdog_pool_budget_invalid")
        self.pool_sha = hashes[self.config["pool_config"]]
        self.recon_sha = hashes[str(Path(self.screen["reconciliation_root"]) / "COMPLETE.json")]
        reconciliation = read(Path(self.screen["reconciliation_root"]) / "COMPLETE.json")
        require(reconciliation.get("schema") == "corpus.asset_reconciliation.v1"
                and reconciliation.get("complete") is True and reconciliation.get("scope") == "full_run",
                "watchdog_reconciliation_not_full")
        scope = identity.get("scope", {})
        expected_paths = {str(Path(self.screen["reconciliation_root"]) / "buckets" / f"{n:02x}" / "api_review.parquet")
                          for n in range(256)}
        require(scope.get("global_screening_scope_authorized") is True
                and scope.get("reconciliation_complete_sha256") == self.recon_sha
                and len(scope.get("inputs", [])) == 256
                and {r["path"] for r in scope["inputs"]} == expected_paths
                and scope == read(self.supervisor_dir / "INPUT_GATE.json"), "watchdog_input_scope_mismatch")
        imported = read(self.output / "INPUT_IMPORTED.json")
        require(imported.get("input_identity_sha256") == digest(identity)
                and type(imported.get("eligible_assets")) is int and imported["eligible_assets"] > 0
                and imported.get("eligible_assets") == reconciliation["counts"]["api_review_assets"],
                "watchdog_imported_count_or_identity_mismatch")
        self.expected_assets = imported["eligible_assets"]

    def process_lock(self, role):
        if role == "guard":
            return self.pool_root / "guard.lock"
        return Path("/tmp/corpus-screen-supervisor-locks") / (digest(str(self.supervisor_dir.resolve())) + ".lock")

    def ensure_process(self, role):
        command = self.config[role + "_command"]
        identity = self.state["processes"].get(role)
        status = self.runtime.status(identity, command)
        require(status != "unknown", "watchdog_process_identity_unknown:" + role)
        matches = self.runtime.find(command, self.config["workdir"])
        require(len(matches) <= 1, "watchdog_duplicate_processes:" + role)
        if status == "alive":
            require(matches == [identity], "watchdog_process_discovery_mismatch:" + role)
            return "alive"
        if matches:
            self.state["processes"][role] = matches[0]
            self.state["intents"].pop(role, None)
            self.save("adopted_" + role)
            return "adopted"
        require(role not in self.state["intents"], "watchdog_spawn_outcome_unknown:" + role)
        with available_lock(self.process_lock(role)) as free:
            if not free:
                return "lock_busy_pending"
        now = self.runtime.now()
        history = [x for x in self.state["restarts"].get(role, []) if x > now - 3600]
        require(len(history) < self.config["max_restarts_per_hour"], "watchdog_restart_budget_exhausted:" + role)
        if history and now < history[-1] + self.config["restart_backoff_seconds"] * 2 ** (len(history) - 1):
            return "restart_backoff"
        # An external launch racing the check still meets the child's own flock.
        # An unknown launch outcome never causes an automatic second spawn.
        self.validate_bindings()
        self.state["intents"][role] = {"created_at_epoch": now, "command_sha256": digest(command)}
        self.state["restarts"][role] = history + [now]
        self.save("restarting_" + role)
        identity = self.runtime.start(command, workdir=self.config["workdir"], environment=os.environ.copy(),
                                      log_path=str(self.directory / (role + ".log")))
        self.state["processes"][role] = identity
        self.state["intents"].pop(role, None)
        self.save("restarted_" + role)
        return "restarted"

    def screening_complete(self):
        path = self.output / "COMPLETE.json"
        if not path.exists() or read(path).get("screening_pass_complete") is not True:
            return None
        sha = file_sha(path)
        if self.completed and self.completed["sha256"] == sha:
            return self.completed
        proxy = SimpleNamespace(output=self.output, state={"reconciliation_complete_sha256": self.recon_sha})
        result = Supervisor.completion(proxy)
        counts = result["summary"].get("counts", {})
        require(result["receipt"]["input_sha256"] == self.config["expected_hashes"][str(self.output / "INPUT.json")]
                and result["receipt"]["scope"] == read(self.output / "INPUT.json")["scope"]
                and counts.get("eligible_assets") == self.expected_assets
                and counts.get("state:done") == self.expected_assets,
                "watchdog_completion_does_not_cover_all_assets")
        self.completed = result
        return result

    def released(self):
        receipt_path = self.pool_root / "STOPPED.json"
        if not (self.pool_root / "DISABLED").exists() or not receipt_path.exists():
            return None
        receipt = read(receipt_path)
        if receipt.get("complete") is not True:
            return None
        ledger_path = self.pool_root / "RESERVATIONS.json"
        ledger = read(ledger_path)
        require(ledger.get("root") == str(self.pool_root) and ledger.get("config_sha256") == self.pool_sha,
                "watchdog_pool_ledger_binding_mismatch")
        rows = receipt.get("slots", [])
        # Stale/partial STOPPED is not sufficient to end an active pool.
        if (len(rows) != len(ledger["slots"]) or len({str(r["slot"]) for r in rows}) != len(rows)
                or {str(r["slot"]) for r in rows} != set(ledger["slots"]) or not all(r.get("released") is True for r in rows)):
            return None
        proofs = []
        for number, reserved in ledger["slots"].items():
            directory = self.pool_root / "slots" / f"{int(number):02}"
            state_path = directory / "state.json"
            if not state_path.exists():
                return None
            state = read(state_path)
            require(state.get("name") == reserved["name"] and state.get("cards") == reserved["cards"]
                    and state.get("pool_config_sha256") == self.pool_sha
                    and reserved["intent"].get("pool_config_sha256") == self.pool_sha
                    and state.get("binding") == reserved["intent"].get("binding"),
                    "watchdog_released_slot_binding_mismatch")
            if state.get("release_verified") is not True:
                return None
            if state.get("cancelled_never_sent") is True:
                path = directory / "CANCELLED_NEVER_SENT.json"
                if not path.exists():
                    return None
                proof = read(path)
                original_intent = read(directory / "CREATE_INTENT.json")
                immutable = ("name", "cards", "priority", "operation_id", "binding", "pool_scope", "pool_slot")
                require(proof.get("schema_version") == "corpus.cancelled_never_sent.v1"
                        and proof.get("pool_config_sha256") == self.pool_sha
                        and proof.get("pool_root") == str(self.pool_root)
                        and proof.get("slot") == int(number) and proof.get("name") == state["name"]
                        and proof.get("operation_id") == state.get("operation_id")
                        and proof.get("binding") == state.get("binding")
                        and proof.get("cancelled_never_sent") is True
                        and proof.get("transmission_marker_absent") is True
                        and proof.get("live_exact_name_matches") == 0
                        and proof.get("platform_stop_claimed") is False
                        and state.get("phase") == "CANCELLED_UNSENT"
                        and proof.get("original_intent_pool_config_sha256") == original_intent.get("pool_config_sha256")
                        and all(original_intent.get(key) == state.get(key) == reserved["intent"].get(key)
                                for key in immutable)
                        and not original_intent.get("service_id") and not reserved["intent"].get("service_id")
                        and not state.get("service_id") and not (directory / "CREATE_TRANSMISSION.json").exists()
                        and proof.get("create_intent_sha256") == file_sha(directory / "CREATE_INTENT.json"),
                        "watchdog_unsent_release_proof_invalid")
            else:
                path = directory / "STOPPED.json"
                if not path.exists():
                    return None
                proof = read(path)
                if not (proof.get("complete") is True and state.get("service_id")
                        and proof.get("service_id") == state["service_id"] and proof.get("status") == "STOPPED"
                        and state.get("phase") == "RELEASED" and state.get("platform_status") == "STOPPED"
                        and isinstance(proof.get("instance_statuses"), list)
                        and state.get("instance_statuses") == proof["instance_statuses"]
                        and all(s in TERMINAL_INSTANCES for s in proof["instance_statuses"])):
                    return None
            proofs.append({"slot": number, "state_sha256": file_sha(state_path),
                           "proof_path": str(path), "proof_sha256": file_sha(path)})
        # External HIGH is not owned by this guard; require its already-published
        # release evidence, never attempt to stop it from this outer watchdog.
        external_proofs = []
        for item in self.pool.get("external_reservations", []):
            state_path = Path(item["state"])
            proof_path = state_path.parent / "STOPPED.json"
            if not proof_path.exists() or not (state_path.parent / "DISABLED").exists():
                return None
            state, proof = read(state_path), read(proof_path)
            if not (state.get("service_id") == item["service_id"] and state.get("release_verified") is True
                    and state.get("phase") == "RELEASED" and state.get("platform_status") == "STOPPED"
                    and proof.get("service_id") == item["service_id"] and proof.get("complete") is True
                    and proof.get("status") == "STOPPED" and isinstance(proof.get("instance_statuses"), list)
                    and state.get("instance_statuses") == proof["instance_statuses"]
                    and all(s in TERMINAL_INSTANCES for s in proof["instance_statuses"])):
                return None
            external_proofs.append({"service_id": item["service_id"], "state_sha256": file_sha(state_path),
                                    "proof_path": str(proof_path), "proof_sha256": file_sha(proof_path)})
        return {"stopped_sha256": file_sha(receipt_path), "ledger_sha256": file_sha(ledger_path),
                "slot_proofs": proofs, "external_release_proofs": external_proofs}

    def tick(self):
        if self.state.get("terminal"):
            return self.state
        try:
            self.validate_bindings()
            completed = self.screening_complete()
            supervisor_state = read(self.supervisor_dir / "STATE.json")
            if completed:
                child = supervisor_state.get("child")
                if child:
                    child_status = self.runtime.status(child, supervisor_state["command"])
                    require(child_status != "unknown", "watchdog_completed_child_identity_unknown")
                    if child_status == "alive":
                        return self.save("waiting_completed_worker_exit")
                # This idempotent local marker prevents new capacity before the
                # guard converges. Never launch an extra stop subprocess here.
                marker = self.pool_root / "DISABLED"
                if not marker.exists():
                    atomic_json(marker, {"reason": "full_screening_complete", "completion_sha256": completed["sha256"],
                                         "watchdog_config_sha256": self.config_sha, "time": self.runtime.now()})
                proof = self.released()
                if proof:
                    # Rehash full artifacts once at finalization, not each poll.
                    self.completed = None
                    completed = self.screening_complete()
                    result = {"schema_version": SCHEMA, "complete": True, "screening_pass_complete": True,
                              "all_reviews_resolved": completed["receipt"].get("all_reviews_resolved", False),
                              "screening_complete_sha256": completed["sha256"], "config_sha256": self.config_sha,
                              "pool_config_sha256": self.pool_sha, "release": proof, "time": self.runtime.now(),
                              "published_training": False, "detector_alarm_cleared": False}
                    atomic_json(self.directory / "COMPLETE.json", result)
                    return self.save("completed_and_resources_released", terminal=True, exit_code=0)
                guard = self.ensure_process("guard")
                return self.save("cleanup_pending", guard=guard, screening_pass_complete=True)
            if (self.pool_root / "DISABLED").exists():
                if not self.released():
                    self.ensure_process("guard")
                return self.save("pool_disabled_before_screening_complete")
            require(not (self.supervisor_dir / "STOP").exists(), "watchdog_explicit_supervisor_stop")
            require(not supervisor_state.get("terminal"), "watchdog_supervisor_terminal_requires_review")
            guard = self.ensure_process("guard")
            supervisor = self.ensure_process("supervisor")
            return self.save("running", guard=guard, supervisor=supervisor)
        except (ConfigurationError, OSError, ValueError, KeyError, TypeError) as error:
            # Error messages from SDK/HTTP can contain secrets. This module only
            # publishes its own typed validation codes, never exception repr.
            code = str(error) if isinstance(error, ConfigurationError) else "watchdog_local_state_validation_failed"
            blocked = {"schema_version": SCHEMA, "code": code, "error_type": type(error).__name__,
                       "config_sha256": self.config_sha, "time": self.runtime.now(), "automatic_retry": False}
            atomic_json(self.directory / "BLOCKED.json", blocked)
            return self.save("blocked", terminal=True, exit_code=2, failure=blocked)


def run(config):
    directory = Path(config["state_dir"])
    with supervisor_lock(directory):
        try:
            watchdog = Watchdog(config)
        except (ConfigurationError, OSError, ValueError, KeyError, TypeError) as error:
            atomic_json(directory / "BLOCKED.json", {"schema_version": SCHEMA,
                "code": str(error) if isinstance(error, ConfigurationError) else "watchdog_initialization_validation_failed",
                "error_type": type(error).__name__, "config_sha256": digest(config), "time": time.time(),
                "automatic_retry": False})
            return 2
        stop = [False]
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
        while not stop[0] and not (directory / "STOP").exists():
            state = watchdog.tick()
            if state.get("terminal"):
                return state["exit_code"]
            time.sleep(watchdog.config["poll_seconds"])
        watchdog.save("watchdog_stopped_children_unchanged")
        return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    return run(checked_watchdog_config(read(args.config)))


if __name__ == "__main__":
    raise SystemExit(main())
