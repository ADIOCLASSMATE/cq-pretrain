from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from . import full_run_watchdog as module
from .full_run_watchdog import Watchdog, ConfigurationError, checked_watchdog_config, available_lock
from .test_screen_supervisor import FakeRuntime


class FixtureRuntime(FakeRuntime):
    def start(self, command, *, workdir, environment, log_path):
        pid = max(self.identities, default=99) + 1
        identity = self.identity(pid, command)
        self.identities[pid], self.statuses[pid] = identity, "alive"
        self.started.append({"command": list(command), "workdir": workdir,
                             "environment": dict(environment), "log_path": log_path})
        Path(log_path).write_text("")
        return identity


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output, self.pool = self.root / "output", self.root / "pool"
        self.recon = self.root / "recon"
        for directory in (self.output / "supervisor", self.pool, self.recon):
            directory.mkdir(parents=True)
        self.screen_path, self.pool_path = self.root / "screen.json", self.pool / "config.json"
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("fixture prompt")
        self.supervisor_command = [str(self.root / ".venv/bin/python"), "-m",
            "corpus_pipeline.postprocess.screen_supervisor", "--config", str(self.screen_path)]
        self.guard_command = [str(self.root / "sdk/bin/python"), "-m", "corpus_pipeline.serving.budget_pool",
                              "guard", "--config", str(self.pool_path)]
        self.runtime = FixtureRuntime()
        self.runtime.start(self.supervisor_command, workdir=str(self.root), environment={}, log_path=str(self.root / "s.log"))
        self.runtime.start(self.guard_command, workdir=str(self.root), environment={}, log_path=str(self.root / "g.log"))
        self.runtime.started.clear()
        self.screen = {"reconciliation_root": str(self.recon), "pool": str(self.pool / "pool.json"),
            "output": str(self.output), "python": self.supervisor_command[0], "workdir": str(self.root),
            "api_key_file": str(self.root / "private.key"), "mode": "annotate", "prompt": str(self.prompt),
            "on_complete_command": self.guard_command[:3] + ["stop", "--config", str(self.pool_path)]}
        atomic_json(self.screen_path, self.screen)
        backend = self.root / "backend.py"
        backend.write_text("# fixture backend\n")
        self.pool_config = {"schema_version": "corpus.native_budget_pool.v1", "root": str(self.pool),
            "backend_file": str(backend), "cards_per_replica": 2, "replicas": 64, "max_cards": 128}
        atomic_json(self.pool_path, self.pool_config)
        self.pool_sha = file_sha(self.pool_path)
        self.recon_path = self.recon / "COMPLETE.json"
        atomic_json(self.recon_path, {"schema": "corpus.asset_reconciliation.v1", "complete": True,
            "scope": "full_run", "counts": {"api_review_assets": 2}})
        self.scope = {"global_screening_scope_authorized": True,
            "reconciliation_complete_sha256": file_sha(self.recon_path), "scope": "selected_globally_reconciled_buckets",
            "inputs": [{"path": str(self.recon / "buckets" / f"{n:02x}" / "api_review.parquet"), "sha256": "a" * 64}
                       for n in range(256)]}
        self.input = {"scope": self.scope, "mode": "annotate", "dependencies": {}}
        atomic_json(self.output / "INPUT.json", self.input)
        atomic_json(self.output / "INPUT_IMPORTED.json", {"input_identity_sha256": digest(self.input), "eligible_assets": 2})
        atomic_json(self.output / "supervisor/INPUT_GATE.json", self.scope)
        atomic_json(self.output / "supervisor/STATE.json", {"status": "screening_running", "child": None})
        self.code_paths = []
        for name in ("postprocess/api_screen.py", "postprocess/screen_supervisor.py", "production/common.py", "serving/budget_pool.py"):
            path = self.root / "corpus_pipeline" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen fixture\n")
            self.code_paths.append(path)
        paths = self.code_paths + [self.screen_path, self.pool_path, Path(module.__file__).absolute(), backend,
            self.prompt, self.recon_path, self.output / "INPUT.json", self.output / "INPUT_IMPORTED.json",
            self.output / "supervisor/INPUT_GATE.json"]
        self.config = {"schema_version": module.SCHEMA, "state_dir": str(self.root / "watchdog"),
            "workdir": str(self.root), "screen_config": str(self.screen_path), "pool_config": str(self.pool_path),
            "supervisor_command": self.supervisor_command, "guard_command": self.guard_command,
            "initial_processes": {"supervisor": self.runtime.identities[100], "guard": self.runtime.identities[101]},
            "expected_hashes": {str(path): file_sha(path) for path in paths}}
        self.watchdog = Watchdog(self.config, runtime=self.runtime)

    def tearDown(self):
        self.tmp.cleanup()

    def complete_screen(self, **counts):
        artifact = self.output / "decisions.parquet"
        artifact.write_bytes(b"fixture parquet bytes")
        summary = {"counts": {"eligible_assets": 2, "state:done": 2, "decision:pending": 2, **counts}}
        atomic_json(self.output / "SUMMARY.json", summary)
        complete = {"schema_version": "postprocess.api_screen.v1", "screening_pass_complete": True,
            "all_reviews_resolved": False, "published_training": False, "scope": self.scope,
            "input_sha256": file_sha(self.output / "INPUT.json"), "summary_sha256": file_sha(self.output / "SUMMARY.json"),
            "artifacts": [{"path": "decisions.parquet", "sha256": file_sha(artifact)}]}
        atomic_json(self.output / "COMPLETE.json", complete)

    def release_slot(self, *, sent=True):
        directory = self.pool / "slots/01"
        directory.mkdir(parents=True, exist_ok=True)
        state = {"name": "owned-01", "cards": 2, "pool_config_sha256": self.pool_sha,
                 "binding": {"runtime_sha256": "a" * 64}, "operation_id": "owned-01", "release_verified": True,
                 "priority": 3, "pool_scope": str(self.pool), "pool_slot": 1}
        if sent:
            state["service_id"] = "owned-platform-id"
            state.update(phase="RELEASED", platform_status="STOPPED", instance_statuses=["INSTANCE_SUCCEEDED"])
            atomic_json(directory / "STOPPED.json", {"complete": True, "service_id": state["service_id"],
                       "status": "STOPPED", "instance_statuses": ["INSTANCE_SUCCEEDED"]})
        else:
            state["cancelled_never_sent"] = True
            state["phase"] = "CANCELLED_UNSENT"
            atomic_json(directory / "CREATE_INTENT.json", state)
            atomic_json(directory / "CANCELLED_NEVER_SENT.json", {
                "schema_version": "corpus.cancelled_never_sent.v1", "pool_root": str(self.pool),
                "pool_config_sha256": self.pool_sha, "slot": 1, "name": state["name"],
                "operation_id": state["operation_id"], "binding": state["binding"],
                "cancelled_never_sent": True, "transmission_marker_absent": True,
                "live_exact_name_matches": 0, "platform_stop_claimed": False,
                "original_intent_pool_config_sha256": self.pool_sha,
                "create_intent_sha256": file_sha(directory / "CREATE_INTENT.json")})
        atomic_json(directory / "state.json", state)
        atomic_json(self.pool / "RESERVATIONS.json", {"root": str(self.pool), "config_sha256": self.pool_sha,
            "slots": {"1": {"name": state["name"], "cards": 2, "intent": state}}})
        atomic_json(self.pool / "STOPPED.json", {"complete": True, "slots": [{"slot": "1", "released": True}]})
        return directory

    def test_adopt_existing_does_not_spawn_or_disable(self):
        result = self.watchdog.tick()
        self.assertEqual(result["status"], "running")
        self.assertEqual(self.runtime.started, [])
        self.assertFalse((self.pool / "DISABLED").exists())

    def test_genuinely_dead_supervisor_restarts_exact_venv_command(self):
        self.runtime.end(100)
        self.assertEqual(self.watchdog.tick()["status"], "running")
        self.assertEqual(len(self.runtime.started), 1)
        self.assertEqual(self.runtime.started[0]["command"], self.supervisor_command)
        self.assertEqual(self.runtime.started[0]["command"][0], str(self.root / ".venv/bin/python"))

    def test_dead_guard_only_guard_command_restored(self):
        self.runtime.end(101)
        self.watchdog.tick()
        self.assertEqual([r["command"] for r in self.runtime.started], [self.guard_command])

    def test_unknown_identity_never_spawns(self):
        self.runtime.statuses[100] = "unknown"
        self.assertEqual(self.watchdog.tick()["status"], "blocked")
        self.assertEqual(self.runtime.started, [])
        self.assertTrue((self.watchdog.directory / "BLOCKED.json").exists())

    def test_busy_target_lock_defers_restart(self):
        self.runtime.end(100)
        with available_lock(self.watchdog.process_lock("supervisor")):
            result = self.watchdog.tick()
        self.assertEqual(result["supervisor"], "lock_busy_pending")
        self.assertEqual(self.runtime.started, [])

    def test_duplicate_same_argv_blocks(self):
        identity = self.runtime.identity(102, self.supervisor_command)
        self.runtime.identities[102], self.runtime.statuses[102] = identity, "alive"
        self.assertEqual(self.watchdog.tick()["status"], "blocked")
        self.assertEqual(self.runtime.started, [])

    def test_existing_replacement_is_adopted_without_spawn(self):
        self.runtime.end(100)
        identity = self.runtime.identity(102, self.supervisor_command)
        self.runtime.identities[102], self.runtime.statuses[102] = identity, "alive"
        self.assertEqual(self.watchdog.tick()["supervisor"], "adopted")
        self.assertEqual(self.runtime.started, [])

    def test_ambiguous_spawn_intent_is_not_repeated(self):
        self.runtime.end(100)
        self.watchdog.state["intents"]["supervisor"] = {"created_at_epoch": 900}
        self.assertEqual(self.watchdog.tick()["status"], "blocked")
        self.assertEqual(self.runtime.started, [])

    def test_restart_budget_is_finite(self):
        self.runtime.end(100)
        self.watchdog.state["restarts"]["supervisor"] = [800, 850, 900]
        self.assertEqual(self.watchdog.tick()["status"], "blocked")
        self.assertEqual(self.runtime.started, [])

    def test_code_change_blocks_restart(self):
        self.runtime.end(100)
        self.code_paths[0].write_text("# edited\n")
        self.assertEqual(self.watchdog.tick()["status"], "blocked")
        self.assertEqual(self.runtime.started, [])

    def test_operator_stop_marker_is_respected(self):
        (self.output / "supervisor/STOP").touch()
        self.runtime.end(100)
        self.assertEqual(self.watchdog.tick()["status"], "blocked")
        self.assertEqual(self.runtime.started, [])

    def test_supervisor_terminal_error_not_blindly_restarted(self):
        atomic_json(self.output / "supervisor/STATE.json", {"status": "blocked_configuration_or_integrity", "terminal": True})
        self.runtime.end(100)
        self.assertEqual(self.watchdog.tick()["status"], "blocked")
        self.assertEqual(self.runtime.started, [])

    def test_stop_exit_zero_and_supervisor_completed_are_not_release(self):
        self.complete_screen()
        atomic_json(self.output / "supervisor/STATE.json", {"status": "completed", "terminal": True})
        atomic_json(self.output / "supervisor/ON_COMPLETE.json", {"state": "completed", "returncode": 0})
        atomic_json(self.pool / "STOPPED.json", {"complete": False, "slots": [{"slot": "1", "released": False}]})
        self.assertEqual(self.watchdog.tick()["status"], "cleanup_pending")
        self.assertFalse((self.watchdog.directory / "COMPLETE.json").exists())
        self.assertTrue((self.pool / "DISABLED").exists())
        self.assertEqual(self.runtime.started, [])

    def test_cleanup_restores_dead_guard_but_never_spawns_stop_or_run(self):
        self.complete_screen()
        self.runtime.end(101)
        self.assertEqual(self.watchdog.tick()["status"], "cleanup_pending")
        self.assertEqual([r["command"] for r in self.runtime.started], [self.guard_command])
        self.watchdog.tick()
        self.assertEqual(len(self.runtime.started), 1)

    def test_full_screen_and_all_owned_release_finish_even_annotations_pending(self):
        self.complete_screen()
        self.release_slot()
        result = self.watchdog.tick()
        self.assertEqual(result["status"], "completed_and_resources_released")
        complete = json.loads((self.watchdog.directory / "COMPLETE.json").read_text())
        self.assertFalse(complete["all_reviews_resolved"])
        self.assertFalse(complete["detector_alarm_cleared"])
        self.assertEqual(self.runtime.started, [])

    def test_cancelled_never_sent_is_distinct_valid_proof(self):
        self.complete_screen()
        self.release_slot(sent=False)
        self.assertEqual(self.watchdog.tick()["status"], "completed_and_resources_released")

    def test_cancelled_never_sent_with_transmission_marker_blocks(self):
        self.complete_screen()
        directory = self.release_slot(sent=False)
        (directory / "CREATE_TRANSMISSION.json").write_text("{}")
        self.assertEqual(self.watchdog.tick()["status"], "blocked")

    def test_cancelled_never_sent_foreign_pool_binding_blocks(self):
        self.complete_screen()
        directory = self.release_slot(sent=False)
        proof = json.loads((directory / "CANCELLED_NEVER_SENT.json").read_text())
        proof["pool_config_sha256"] = "b" * 64
        atomic_json(directory / "CANCELLED_NEVER_SENT.json", proof)
        self.assertEqual(self.watchdog.tick()["status"], "blocked")

    def test_stopped_receipt_missing_reserved_slot_remains_pending(self):
        self.complete_screen()
        self.release_slot()
        atomic_json(self.pool / "STOPPED.json", {"complete": True, "slots": []})
        self.assertEqual(self.watchdog.tick()["status"], "cleanup_pending")

    def test_running_instance_in_stop_proof_remains_pending(self):
        self.complete_screen()
        directory = self.release_slot()
        proof = json.loads((directory / "STOPPED.json").read_text())
        proof["instance_statuses"] = ["INSTANCE_RUNNING"]
        atomic_json(directory / "STOPPED.json", proof)
        self.assertEqual(self.watchdog.tick()["status"], "cleanup_pending")

    def test_stale_stop_proof_with_running_current_state_is_not_done(self):
        self.complete_screen()
        directory = self.release_slot()
        state = json.loads((directory / "state.json").read_text())
        state.update(phase="RUNNING", platform_status="RUNNING", instance_statuses=["INSTANCE_RUNNING"])
        atomic_json(directory / "state.json", state)
        self.assertEqual(self.watchdog.tick()["status"], "cleanup_pending")

    def test_stop_state_and_instance_receipt_disagreement_is_pending(self):
        self.complete_screen()
        directory = self.release_slot()
        state = json.loads((directory / "state.json").read_text())
        state["instance_statuses"] = ["INSTANCE_FAILED"]
        atomic_json(directory / "state.json", state)
        self.assertEqual(self.watchdog.tick()["status"], "cleanup_pending")

    def test_unknown_resource_cannot_finish(self):
        self.complete_screen()
        directory = self.release_slot()
        (directory / "STOPPED.json").unlink()
        self.assertEqual(self.watchdog.tick()["status"], "cleanup_pending")
        self.assertFalse((self.watchdog.directory / "COMPLETE.json").exists())

    def test_external_release_requires_current_state_and_matching_proof(self):
        self.release_slot()
        (self.pool / "DISABLED").touch()
        external = self.root / "external"
        external.mkdir()
        state_path = external / "state.json"
        state = {"service_id": "external-id", "release_verified": True, "phase": "RUNNING",
                 "platform_status": "RUNNING", "instance_statuses": ["INSTANCE_RUNNING"]}
        atomic_json(state_path, state)
        (external / "DISABLED").touch()
        atomic_json(external / "STOPPED.json", {"complete": True, "service_id": "external-id",
                   "status": "STOPPED", "instance_statuses": ["INSTANCE_SUCCEEDED"]})
        self.watchdog.pool["external_reservations"] = [{"state": str(state_path), "service_id": "external-id"}]
        self.assertIsNone(self.watchdog.released())
        state.update(phase="RELEASED", platform_status="STOPPED", instance_statuses=["INSTANCE_SUCCEEDED"])
        atomic_json(state_path, state)
        self.assertEqual(len(self.watchdog.released()["external_release_proofs"]), 1)

    def test_external_missing_stop_proof_is_not_release(self):
        self.release_slot()
        (self.pool / "DISABLED").touch()
        external = self.root / "external"
        external.mkdir()
        state_path = external / "state.json"
        atomic_json(state_path, {"service_id": "external-id", "release_verified": True})
        self.watchdog.pool["external_reservations"] = [{"state": str(state_path), "service_id": "external-id"}]
        self.assertIsNone(self.watchdog.released())

    def test_partial_asset_count_cannot_complete(self):
        self.complete_screen(**{"state:done": 1})
        self.release_slot()
        self.assertEqual(self.watchdog.tick()["status"], "blocked")
        self.assertFalse((self.pool / "DISABLED").exists())

    def test_retryable_count_cannot_complete(self):
        self.complete_screen(**{"state:retryable": 1})
        self.assertEqual(self.watchdog.tick()["status"], "blocked")

    def test_artifact_tamper_blocks_full_completion(self):
        self.complete_screen()
        (self.output / "decisions.parquet").write_bytes(b"changed")
        self.assertEqual(self.watchdog.tick()["status"], "blocked")

    def test_incomplete_prior_pass_is_not_completion(self):
        self.complete_screen()
        value = json.loads((self.output / "COMPLETE.json").read_text())
        value["screening_pass_complete"] = False
        atomic_json(self.output / "COMPLETE.json", value)
        self.assertEqual(self.watchdog.tick()["status"], "running")

    def test_existing_outer_state_resumes_without_relaunch(self):
        self.watchdog.tick()
        resumed = Watchdog(self.config, runtime=self.runtime)
        self.assertEqual(resumed.tick()["status"], "running")
        self.assertEqual(self.runtime.started, [])

    def test_required_code_pin_cannot_be_omitted(self):
        config = deepcopy(self.config)
        config["expected_hashes"].pop(str(self.code_paths[0]))
        with self.assertRaises(ConfigurationError):
            Watchdog(config, runtime=self.runtime)

    def test_initial_validation_failure_writes_blocked_without_start(self):
        self.code_paths[0].write_text("# changed\n")
        self.assertEqual(module.run(self.config), 2)
        self.assertTrue((self.watchdog.directory / "BLOCKED.json").exists())
        self.assertEqual(self.runtime.started, [])

    def test_cancelled_original_config_migration_is_preserved(self):
        self.complete_screen()
        directory = self.release_slot(sent=False)
        intent = json.loads((directory / "CREATE_INTENT.json").read_text())
        intent["pool_config_sha256"] = "a" * 64
        atomic_json(directory / "CREATE_INTENT.json", intent)
        proof = json.loads((directory / "CANCELLED_NEVER_SENT.json").read_text())
        proof["original_intent_pool_config_sha256"] = "a" * 64
        proof["create_intent_sha256"] = file_sha(directory / "CREATE_INTENT.json")
        atomic_json(directory / "CANCELLED_NEVER_SENT.json", proof)
        self.assertEqual(self.watchdog.tick()["status"], "completed_and_resources_released")

    def test_guard_argv_cannot_be_direct_run(self):
        config = deepcopy(self.config)
        config["guard_command"][3] = "run"
        with self.assertRaises(ConfigurationError):
            checked_watchdog_config(config)

    def test_outer_exclusive_lock(self):
        with module.supervisor_lock(self.watchdog.directory):
            with self.assertRaises(ConfigurationError):
                with module.supervisor_lock(self.watchdog.directory):
                    pass


if __name__ == "__main__":
    unittest.main()
