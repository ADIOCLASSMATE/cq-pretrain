from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from .screen_supervisor import (
    Supervisor, Runtime, ConfigurationError, build_command, checked_config, ready_pool, supervisor_lock,
)


class FakeRuntime:
    def __init__(self):
        self.clock = 1000
        self.started = []
        self.identities = {}
        self.statuses = {}
        self.codes = {}
        self.signals = []

    def now(self):
        return self.clock

    def identity(self, pid, argv=None):
        return {"pid": pid, "start_ticks": 1, "host": "fixture", "boot_id": "fixture",
                "argv": argv or ["fixture-supervisor"], "argv_sha256": digest(argv or ["fixture-supervisor"])}

    def start(self, command, *, workdir, environment, log_path):
        pid = 100 + len(self.started)
        identity = self.identity(pid, command)
        self.identities[pid] = identity
        self.statuses[pid] = "alive"
        self.started.append({"command": list(command), "workdir": workdir, "environment": dict(environment), "log_path": log_path})
        Path(log_path).write_text("")
        return identity

    def status(self, identity, command):
        if identity["argv"] != command:
            return "unknown"
        return self.statuses.get(identity["pid"], "dead")

    def find(self, command, workdir):
        return [value for pid, value in self.identities.items() if value["argv"] == command and self.statuses[pid] == "alive"]

    def exit_code(self, identity):
        return self.codes.get(identity["pid"])

    def terminate(self, identity, command):
        assert self.status(identity, command) == "alive"
        self.signals.append(identity["pid"])

    def end(self, pid, code=0):
        self.statuses[pid], self.codes[pid] = "dead", code


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recon = self.root / "reconciliation"
        self.output = self.root / "screen"
        self.pool = self.root / "pool.json"
        self.registry = self.root / "registry.json"
        self.key = self.root / "private.key"
        self.key.write_text("fixture-private-key-never-logged")
        self.python = str(self.root / ".venv" / "bin" / "python")
        self.config = checked_config({"reconciliation_root": str(self.recon), "output": str(self.output),
            "pool": str(self.pool), "workdir": str(self.root), "python": self.python,
            "api_key_file": str(self.key), "retry_backoff_seconds": 1})
        self.runtime = FakeRuntime()

    def tearDown(self):
        self.tmp.cleanup()

    def seal_reconciliation(self, scope="full_run"):
        buckets = []
        for number in range(256):
            bucket = f"{number:02x}"
            path = self.recon / "buckets" / bucket / "api_review.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture-artifact")
            receipt = path.parent / "COMPLETE.json"
            atomic_json(receipt, {"schema": "corpus.asset_reconciliation.bucket.v1", "bucket": bucket,
                "identity_sha256": "a" * 64, "artifacts": [{"path": str(path.relative_to(self.recon)),
                "sha256": file_sha(path), "size_bytes": path.stat().st_size}]})
            buckets.append({"bucket": bucket, "receipt_path": str(receipt.relative_to(self.recon)), "receipt_sha256": file_sha(receipt)})
        atomic_json(self.recon / "COMPLETE.json", {"schema": "corpus.asset_reconciliation.v1", "complete": True,
            "scope": scope, "identity_sha256": "a" * 64, "input_index_sha256": "b" * 64, "buckets": buckets})

    def make_ready(self, ready=True):
        atomic_json(self.registry, {"endpoints": [{"role": "verifier", "backend": "inspire_serving",
            "ready": ready, "ready_expires_at": 9999999999, "expected_boots": ["boot"],
            "base_url": "http://fixture.invalid/v1", "model": "Qwen3.5-397B-A17B", "supports_images": True}]})
        atomic_json(self.pool, {"schema_version": "corpus.screening_pool.v1", "registries": [{"path": str(self.registry), "weight": 1}]})

    def supervisor(self):
        return Supervisor(self.config, runtime=self.runtime)

    def complete(self, supervisor, *, passed=True, retryable=0, resolved=False):
        atomic_json(self.output / "INPUT.json", {"fixture": True})
        atomic_json(self.output / "SUMMARY.json", {"counts": {"eligible_assets": 3, "state:done": 3 - retryable,
                                                            "state:retryable": retryable}})
        artifact = self.output / "decisions.parquet"
        artifact.write_bytes(b"fixture-decisions")
        atomic_json(self.output / "COMPLETE.json", {"schema_version": "postprocess.api_screen.v1", "published_training": False,
            "scope": {"global_screening_scope_authorized": True, "reconciliation_complete_sha256": supervisor.state["reconciliation_complete_sha256"]},
            "screening_pass_complete": passed, "all_reviews_resolved": resolved,
            "input_sha256": file_sha(self.output / "INPUT.json"), "summary_sha256": file_sha(self.output / "SUMMARY.json"),
            "artifacts": [{"path": artifact.name, "sha256": file_sha(artifact)}]})

    def start(self):
        self.seal_reconciliation()
        self.make_ready()
        supervisor = self.supervisor()
        self.assertEqual(supervisor.tick()["status"], "screening_running")
        return supervisor

    def test_waits_for_full_reconciliation_and_then_ready_pool(self):
        supervisor = self.supervisor()
        self.assertEqual(supervisor.tick()["status"], "waiting_for_full_reconciliation")
        self.seal_reconciliation()
        self.assertEqual(supervisor.tick()["status"], "waiting_for_ready_serving")
        self.make_ready()
        self.assertEqual(supervisor.tick()["status"], "screening_running")
        self.assertEqual(len(self.runtime.started), 1)
        call = self.runtime.started[0]
        self.assertEqual(call["command"][0], self.python)
        self.assertEqual(call["command"].count("--input"), 256)
        self.assertEqual(call["environment"]["INF_API_KEY"], "fixture-private-key-never-logged")
        self.assertNotIn("fixture-private-key-never-logged", (self.output / "supervisor" / "STATE.json").read_text())

    def test_annotation_mode_is_explicit_and_forwarded(self):
        self.config["mode"] = "annotate"
        supervisor = self.start()
        command = self.runtime.started[0]["command"]
        self.assertEqual(command[command.index("--mode") + 1], "annotate")
        self.complete(supervisor, resolved=False)
        self.runtime.end(supervisor.state["child"]["pid"])
        self.assertEqual(supervisor.tick()["status"], "completed")
        self.assertFalse(supervisor.state["all_reviews_resolved"])

    def test_bounded_reconciliation_cannot_launch_full_review(self):
        self.seal_reconciliation("bounded_validation")
        self.make_ready()
        with self.assertRaisesRegex(ConfigurationError, "completed_full_run"):
            self.supervisor().tick()
        self.assertFalse(self.runtime.started)

    def test_sealed_artifact_hash_failure_cannot_launch(self):
        self.seal_reconciliation()
        self.make_ready()
        (self.recon / "buckets" / "ff" / "api_review.parquet").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "not_bound_by_global_complete"):
            self.supervisor().tick()
        self.assertFalse(self.runtime.started)

    def test_restart_adopts_live_child_without_new_process(self):
        original = self.start()
        resumed = self.supervisor()
        self.assertEqual(resumed.tick()["status"], "screening_running")
        self.assertEqual(len(self.runtime.started), 1)
        self.assertEqual(resumed.state["child"], original.state["child"])

    def test_uncertain_spawn_recovers_exact_matching_process(self):
        supervisor = self.start()
        supervisor.save(child=None)
        resumed = self.supervisor()
        self.assertEqual(resumed.tick()["status"], "adopted_existing_screening_worker")
        self.assertEqual(len(self.runtime.started), 1)

    def test_ambiguous_spawn_without_pid_never_blindly_replays(self):
        supervisor = self.start()
        pid = supervisor.state["child"]["pid"]
        supervisor.save(child=None)
        self.runtime.end(pid, -9)
        resumed = self.supervisor()
        self.assertEqual(resumed.tick()["status"], "blocked_spawn_outcome_unknown")
        self.assertEqual(len(self.runtime.started), 1)

    def test_unknown_process_identity_never_restarts_or_signals(self):
        supervisor = self.start()
        self.runtime.statuses[supervisor.state["child"]["pid"]] = "unknown"
        self.assertEqual(supervisor.tick()["status"], "blocked_unknown_child_identity")
        self.assertEqual(len(self.runtime.started), 1)
        self.assertFalse(self.runtime.signals)

    def test_retryable_completion_resumes_after_proven_exit(self):
        supervisor = self.start()
        self.complete(supervisor, passed=False, retryable=2)
        self.runtime.end(supervisor.state["child"]["pid"])
        self.assertEqual(supervisor.tick()["status"], "waiting_to_resume")
        self.assertEqual(supervisor.state["errors"], 0)
        self.runtime.clock += 2
        self.assertEqual(supervisor.tick()["status"], "screening_running")
        self.assertEqual(len(self.runtime.started), 2)
        self.assertEqual(self.runtime.started[0]["command"], self.runtime.started[1]["command"])

    def test_unexpected_errors_stop_after_three_failures(self):
        supervisor = self.start()
        for number in range(3):
            self.runtime.end(supervisor.state["child"]["pid"], code=-9)
            state = supervisor.tick()
            if number < 2:
                self.assertEqual(state["status"], "waiting_to_resume")
                self.runtime.clock += 300
                supervisor.tick()
        self.assertEqual(state["status"], "failed_retry_budget_exhausted")
        self.assertEqual(len(self.runtime.started), 3)

    def test_configuration_failure_is_not_retried(self):
        supervisor = self.start()
        Path(supervisor.state["log_path"]).write_text("ValueError: immutable_screening_input_or_policy_changed")
        self.runtime.end(supervisor.state["child"]["pid"], code=1)
        self.assertEqual(supervisor.tick()["status"], "blocked_configuration_or_integrity")
        self.assertEqual(len(self.runtime.started), 1)

    def test_stop_sends_one_graceful_signal_and_waits(self):
        supervisor = self.start()
        pid = supervisor.state["child"]["pid"]
        supervisor.stop_path.touch()
        self.assertEqual(supervisor.tick()["status"], "draining_child_after_stop")
        self.assertEqual(supervisor.tick()["status"], "draining_child_after_stop")
        self.assertEqual(self.runtime.signals, [pid])
        self.runtime.end(pid, 130)
        self.assertEqual(supervisor.tick()["status"], "stopped")

    def test_configuration_error_drains_verified_child_before_terminal(self):
        supervisor = self.start()
        pid = supervisor.state["child"]["pid"]
        self.assertEqual(supervisor.block_and_drain(ConfigurationError("fixture_bad_config"))["status"], "draining_after_configuration_error")
        self.assertEqual(self.runtime.signals, [pid])
        self.assertEqual(supervisor.tick()["status"], "draining_after_configuration_error")
        self.runtime.end(pid, 130)
        self.assertEqual(supervisor.tick()["status"], "blocked_configuration_or_integrity")
        self.assertEqual(len(self.runtime.started), 1)

    def test_completed_flag_with_retryable_rows_is_rejected(self):
        supervisor = self.start()
        self.complete(supervisor, passed=True, retryable=1)
        self.runtime.end(supervisor.state["child"]["pid"])
        with self.assertRaisesRegex(ConfigurationError, "unfinished_work"):
            supervisor.tick()

    def test_user_can_resume_stopped_run_after_clearing_stop_marker(self):
        supervisor = self.start()
        pid = supervisor.state["child"]["pid"]
        supervisor.stop_path.touch()
        supervisor.tick()
        self.runtime.end(pid, 130)
        self.assertEqual(supervisor.tick()["status"], "stopped")
        supervisor.stop_path.unlink()
        resumed = self.supervisor()
        self.assertFalse(resumed.state["terminal"])
        self.assertEqual(resumed.tick()["status"], "waiting_to_resume")

    def test_completed_with_pending_second_reviews_runs_cleanup_once(self):
        self.config["on_complete_command"] = ["/fixture/sdk/python", "-m", "fixture.pool", "stop"]
        supervisor = self.start()
        self.complete(supervisor, resolved=False)
        self.runtime.end(supervisor.state["child"]["pid"])
        self.assertEqual(supervisor.tick()["status"], "running_completion_cleanup")
        self.assertEqual(len(self.runtime.started), 2)
        cleanup_pid = max(self.runtime.identities)
        self.runtime.end(cleanup_pid)
        self.assertEqual(supervisor.tick()["status"], "completed")
        self.assertFalse(supervisor.state["all_reviews_resolved"])
        self.assertEqual(supervisor.state["cleanup"]["returncode"], 0)
        supervisor.tick()
        self.assertEqual(len(self.runtime.started), 2)

    def test_ambiguous_cleanup_intent_never_replays_stop(self):
        self.config["on_complete_command"] = ["/fixture/sdk/python", "stop"]
        supervisor = self.start()
        self.complete(supervisor)
        self.runtime.end(supervisor.state["child"]["pid"])
        complete = supervisor.completion()
        atomic_json(supervisor.directory / "ON_COMPLETE.json", {"state": "intent", "command": self.config["on_complete_command"],
                                                                 "completion_sha256": complete["sha256"]})
        self.assertEqual(supervisor.tick()["status"], "completed_cleanup_outcome_unknown")
        self.assertEqual(len(self.runtime.started), 1)

    def test_cleanup_failure_is_recorded_without_repeat(self):
        self.config["on_complete_command"] = ["/fixture/sdk/python", "stop"]
        supervisor = self.start()
        self.complete(supervisor)
        self.runtime.end(supervisor.state["child"]["pid"])
        supervisor.tick()
        self.runtime.end(max(self.runtime.identities), 1)
        self.assertEqual(supervisor.tick()["status"], "completed_cleanup_failed")
        self.assertEqual(supervisor.state["cleanup"]["returncode"], 1)
        supervisor.tick()
        self.assertEqual(len(self.runtime.started), 2)

    def test_input_scope_changes_block_resume(self):
        supervisor = self.start()
        self.runtime.end(supervisor.state["child"]["pid"], -9)
        supervisor.tick()
        atomic_json(self.recon / "COMPLETE.json", {"schema": "corpus.asset_reconciliation.v1", "complete": True, "scope": "full_run", "changed": True})
        with self.assertRaisesRegex(ConfigurationError, "sealed_reconciliation_changed"):
            supervisor.tick()

    def test_local_supervisor_lock_excludes_another_writer(self):
        with supervisor_lock(self.root / "locked"):
            with self.assertRaisesRegex(ConfigurationError, "another_screening_supervisor"):
                with supervisor_lock(self.root / "locked"):
                    pass

    def test_actual_process_identity_detects_argv_tampering(self):
        runtime = Runtime()
        identity = runtime.identity(os.getpid())
        self.assertEqual(runtime.status(identity, identity["argv"]), "alive")
        bad = deepcopy(identity)
        bad["argv"] = ["unexpected"]
        self.assertEqual(runtime.status(bad, identity["argv"]), "unknown")

    def test_custom_prompt_is_forwarded_and_hash_bound_on_resume(self):
        prompt = self.root / "prompt-v4.txt"
        prompt.write_text("fixture revised v3-compatible screening instructions")
        self.config["prompt"] = str(prompt)
        supervisor = self.start()
        command = self.runtime.started[0]["command"]
        self.assertEqual(command[command.index("--prompt") + 1], str(prompt))
        self.assertEqual(supervisor.config["prompt_sha256"], file_sha(prompt))
        prompt.write_text("changed prompt must not silently replace existing instructions")
        with self.assertRaisesRegex(ConfigurationError, "bound_prompt_file_changed"):
            supervisor.tick()
        with self.assertRaisesRegex(ConfigurationError, "immutable_supervisor_configuration_changed"):
            self.supervisor()

    def test_python_venv_symlink_is_preserved(self):
        executable = self.root / "real-python"
        executable.touch()
        target = Path(self.python)
        target.parent.mkdir(parents=True)
        target.symlink_to(executable)
        self.assertEqual(build_command(self.config, ["/fixture/input.parquet"])[0], self.python)


if __name__ == "__main__":
    unittest.main()
