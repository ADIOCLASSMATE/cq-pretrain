"""Small fake-worker tests; these never invoke Codex or a model endpoint."""
import gzip
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
import uuid

from .farm import EFFORT, MODEL, Farm, atomic_json, canonical, exclusive, file_sha


class FakeWorker:
    def __init__(self, behavior=None, stop=None):
        self.behavior = behavior or (lambda number, items: "success")
        self.stop = stop
        self.calls = []
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    @staticmethod
    def validate_result(result, expected_id):
        if set(result) != {"item_id", "decision"} or result["item_id"] != expected_id or result["decision"] not in {"pending", "admit"}:
            raise ValueError("fake validation rejected result")

    def run_batch(self, items, directory, config):
        with self.lock:
            self.calls.append([item["item_id"] for item in items])
            number = len(self.calls)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.01)
            mode = self.behavior(number, items)
            if mode == "exception":
                raise RuntimeError("fake local exception")
            if mode == "transport":
                return {"results": [], "errors": {}, "transport_ok": False,
                        "error_type": "transport_error", "metadata": {}}
            if mode == "decode":
                return {"results": [], "errors": {item["item_id"]: "decode_error:unsupported" for item in items},
                        "transport_ok": False, "error_type": "decode_error", "metadata": {}}
            values = [{"item_id": item["item_id"], "decision": "pending"} for item in items]
            if mode == "partial":
                values = values[:1]
            raw = {"items": values}
            paths = {name: directory / filename for name, filename in
                     {"command": "command.json", "prompt": "prompt.txt", "schema": "schema.json",
                      "last_message": "last_message.json", "events": "events.jsonl.gz",
                      "stderr": "stderr.log", "decode_evidence": "decode_evidence.json"}.items()}
            attachments = []
            for index, item in enumerate(items):
                path = directory / f"{index:02}.fake-image"
                path.write_bytes(b"FAKE TEST ATTACHMENT: no real inference")
                attachments.append({"item_id": item["item_id"], "path": str(path), "sha256": file_sha(path),
                                    "width": 32, "height": 32, "decoded": True})
            command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--cd", str(directory),
                       "--model", MODEL, "-c", 'model_reasoning_effort="low"', "--sandbox", "read-only",
                       "--output-schema", str(paths["schema"]), "--output-last-message", str(paths["last_message"])]
            for attachment in attachments:
                command.extend(["--image", attachment["path"]])
            schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
                "type": "object", "properties": {"item_id": {"type": "string"}, "decision": {"enum": ["admit", "pending"]}},
                "required": ["item_id", "decision"], "additionalProperties": False}}}, "required": ["items"]}
            atomic_json(paths["command"], command)
            paths["prompt"].write_text("Fake test prompt. This is never sent to a model.")
            atomic_json(paths["schema"], schema)
            atomic_json(paths["last_message"], raw)
            atomic_json(paths["decode_evidence"], attachments)
            paths["stderr"].write_text("")
            thread_id = uuid.uuid4().hex
            usage = {"input_tokens": 100, "output_tokens": 20}
            events = [{"type": "thread.started", "thread_id": thread_id},
                      {"type": "item.completed", "item": {"type": "agent_message", "text": canonical(raw)}},
                      {"type": "turn.completed", "usage": usage}]
            with gzip.open(paths["events"], "wt") as stream:
                for event in events:
                    stream.write(canonical(event) + "\n")
            metadata = {"artifacts": {key: {"path": str(path), "sha256": file_sha(path)} for key, path in paths.items()},
                        "model": MODEL, "reasoning_effort": EFFORT, "thread_id": thread_id, "usage": usage,
                        "attachments": attachments, "claimed_item_ids": [item["item_id"] for item in items],
                        "attached_item_ids": [item["item_id"] for item in items]}
            if mode == "forged":
                values = [{"item_id": value["item_id"], "decision": "admit"} for value in values]
            if self.stop is not None:
                self.stop.set()
            return {"results": values, "errors": {}, "transport_ok": True,
                    "error_type": "partial_response" if mode == "partial" else None, "metadata": metadata}
        finally:
            with self.lock:
                self.active -= 1


class FarmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "farm"
        self.root.mkdir()
        atomic_json(self.root / "config.json", {"retry_base_seconds": 0, "retry_max_seconds": 0,
                                               "circuit_base_seconds": 0.01, "circuit_max_seconds": 0.01})
        self.farms = []

    def tearDown(self):
        for farm in self.farms:
            farm.close()
        self.tmp.cleanup()

    def make(self, count=5, worker=None):
        manifest = Path(self.tmp.name) / "input.jsonl"
        if not manifest.exists():
            manifest.write_text("".join(canonical({"item_id": f"item-{i}", "source_ref": f"source-{i}",
                                                    "queue_order": i, "provenance": {}}) + "\n" for i in range(count)))
        farm = Farm(self.root, worker=worker or FakeWorker())
        self.farms.append(farm)
        farm.initialize(manifest)
        return farm

    def test_end_to_end_export_is_exact_and_sharded(self):
        farm = self.make(5)
        result = farm.run(workers=3, initial_workers=2, batch_size=2)
        self.assertEqual(result["counts"], {"succeeded": 5})
        self.assertFalse(result["complete"])
        published = farm.export(shard_size=2)
        self.assertTrue(published["complete"])
        manifest = json.loads(Path(published["manifest"]).read_text())
        self.assertEqual([row["records"] for row in manifest["shards"]], [2, 2, 1])
        self.assertTrue(farm.audit(export_manifest=published["manifest"])["passed"])
        self.assertEqual(farm.worker.peak, 2)

    def test_partial_commits_success_and_only_retries_missing_item(self):
        worker = FakeWorker(lambda n, items: "partial" if n == 1 else "success")
        farm = self.make(2, worker)
        farm.run(workers=1, initial_workers=1, batch_size=2)
        self.assertEqual(worker.calls, [["item-0", "item-1"], ["item-1"]])
        self.assertEqual(dict(farm.db.execute("SELECT item_id,attempts FROM items")), {"item-0": 1, "item-1": 2})
        self.assertTrue(farm.audit()["passed"])

    def test_success_never_recomputed_after_restart(self):
        worker = FakeWorker()
        farm = self.make(3, worker)
        farm.run(workers=1, initial_workers=1, batch_size=1, max_batches=1)
        other = self.make(3, worker)
        other.run(workers=1, initial_workers=1, batch_size=1)
        self.assertEqual(worker.calls, [["item-0"], ["item-1"], ["item-2"]])

    def test_claimed_running_work_recovers_without_resetting_attempt(self):
        farm = self.make(1)
        farm.max_attempts = 5
        batch = farm.claim(1)
        self.assertIsNotNone(batch)
        farm.run(workers=1, initial_workers=1, batch_size=1)
        row = farm.db.execute("SELECT status,attempts FROM items").fetchone()
        self.assertEqual(tuple(row), ("succeeded", 2))
        self.assertEqual(farm.db.execute("SELECT status FROM batches WHERE batch_id=?", (batch[0],)).fetchone()[0], "interrupted")

    def test_stop_refill_drains_existing_successes(self):
        stop = threading.Event()
        worker = FakeWorker(stop=stop)
        farm = self.make(8, worker)
        result = farm.run(workers=4, initial_workers=2, batch_size=1, stop_event=stop)
        self.assertEqual(result["counts"], {"pending": 6, "succeeded": 2})
        self.assertEqual(result["run"]["interrupted"], True)
        self.assertEqual(worker.active, 0)

    def test_transport_circuit_persists_and_uses_one_probe(self):
        config = json.loads((self.root / "config.json").read_text())
        config["circuit_threshold"] = 2
        atomic_json(self.root / "config.json", config)
        farm = self.make(6, FakeWorker(lambda *_: "transport"))
        farm.run(workers=4, initial_workers=2, batch_size=1, max_batches=2)
        state = farm.metadata("adaptive")
        self.assertEqual(state["circuit"], "open")
        self.assertEqual(state["effective_workers"], 1)
        healthy = FakeWorker()
        resumed = self.make(6, healthy)
        resumed.run(workers=4, initial_workers=2, batch_size=1, max_batches=1)
        probes = resumed.db.execute("SELECT count(*) FROM batches WHERE is_probe=1").fetchone()[0]
        self.assertEqual(probes, 1)
        self.assertEqual(healthy.peak, 1)
        self.assertEqual(resumed.metadata("adaptive")["circuit"], "closed")

    def test_sigterm_drains_success_and_restores_handler(self):
        before = signal.getsignal(signal.SIGTERM)
        def interrupt(number, items):
            if number == 1:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.02)
            return "success"
        worker = FakeWorker(interrupt)
        farm = self.make(5, worker)
        result = farm.run(workers=2, initial_workers=2, batch_size=1)
        self.assertEqual(result["counts"], {"pending": 3, "succeeded": 2})
        self.assertTrue(result["run"]["interrupted"])
        self.assertIs(signal.getsignal(signal.SIGTERM), before)
        self.assertEqual(worker.active, 0)

    def test_max_attempts_failure_does_not_publish_completion(self):
        farm = self.make(1, FakeWorker(lambda *_: "exception"))
        result = farm.run(workers=1, initial_workers=1, batch_size=1)
        self.assertEqual(result["counts"], {"failed": 1})
        self.assertEqual(farm.db.execute("SELECT attempts FROM items").fetchone()[0], 5)
        with self.assertRaisesRegex(ValueError, "incomplete coverage"):
            farm.export()
        self.assertFalse((self.root / "COMPLETE.json").exists())

    def test_decode_failure_is_not_retried_without_alternative_decoder(self):
        farm = self.make(1, FakeWorker(lambda *_: "decode"))
        farm.run(workers=1, initial_workers=1, batch_size=1)
        self.assertEqual(tuple(farm.db.execute("SELECT status,attempts,error_type FROM items").fetchone()),
                         ("failed", 1, "decode_error"))

    def test_forged_derived_result_cannot_commit(self):
        farm = self.make(1, FakeWorker(lambda *_: "forged"))
        result = farm.run(workers=1, initial_workers=1, batch_size=1, max_batches=1)
        self.assertEqual(result["counts"], {"retry": 1})
        self.assertIsNone(farm.db.execute("SELECT result_json FROM items").fetchone()[0])

    def test_tampered_raw_artifact_fails_strict_audit(self):
        farm = self.make(1)
        farm.run(workers=1, initial_workers=1, batch_size=1)
        batch_dir = Path(farm.db.execute("SELECT directory FROM batches").fetchone()[0])
        atomic_json(batch_dir / "last_message.json", {"items": []})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            farm.audit()

    def test_tampered_export_fails_audit(self):
        farm = self.make(1)
        farm.run(workers=1, initial_workers=1, batch_size=1)
        result = farm.export()
        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text())
        shard = manifest_path.parent / manifest["shards"][0]["path"]
        with shard.open("ab") as stream:
            stream.write(b"garbage")
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            farm.audit(export_manifest=manifest_path)

    def test_source_duplicate_ids_roll_back_all_import(self):
        manifest = Path(self.tmp.name) / "duplicate.jsonl"
        manifest.write_text(canonical({"item_id": "same", "source_ref": "a"}) + "\n"
                            + canonical({"item_id": "same", "source_ref": "b"}) + "\n")
        farm = Farm(self.root, worker=FakeWorker())
        self.farms.append(farm)
        with self.assertRaises(Exception):
            farm.initialize(manifest)
        self.assertEqual(farm.db.execute("SELECT count(*) FROM items").fetchone()[0], 0)
        self.assertIsNone(farm.metadata("source"))

    def test_changed_config_and_manifest_are_rejected(self):
        farm = self.make(2)
        farm.run(workers=1, initial_workers=1, batch_size=1, max_batches=1)
        farm.config["timeout_seconds"] = 7
        with self.assertRaisesRegex(ValueError, "runtime code or config changed"):
            farm.run(workers=1, initial_workers=1)
        path = Path(farm.metadata("source")["path"])
        path.write_text(path.read_text().replace("source-0", "changed-0"))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            farm.initialize(path)

    def test_single_writer_lock_rejects_second_owner(self):
        with exclusive(self.root):
            with self.assertRaisesRegex(RuntimeError, "already has a writer"):
                with exclusive(self.root):
                    pass

    def test_limits_reject_unbounded_execution(self):
        farm = self.make(1)
        for kwargs in ({"workers": 65}, {"batch_size": 17}, {"workers": 2, "initial_workers": 3}):
            with self.assertRaises(ValueError):
                farm.run(**kwargs)
        self.assertEqual(farm.worker.calls, [])


if __name__ == "__main__":
    unittest.main()
