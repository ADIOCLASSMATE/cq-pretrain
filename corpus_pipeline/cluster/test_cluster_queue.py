"""Filesystem queue regressions: real process locks, leases and output bytes."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import queue


def task(index=0):
    return {"task_id": f"shard-{index:04d}", "source": "fixture_source",
            "parquet_path": f"/read-only-fixture/shard-{index:04d}.parquet",
            "row_start": 5, "row_end": 8}


def claim_all(arguments):
    root, worker_id = arguments
    claimed = []
    while True:
        lease = queue.claim(root, worker_id, now=101)
        if lease is None:
            return claimed
        claimed.append((lease["task_id"], lease["token"]))


class ClusterQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "queue"

    def initialize(self, count=1, **kwargs):
        return queue.init_queue(self.root, (task(i) for i in range(count)), now=100,
                                lease_seconds=10, **kwargs)

    def output(self, lease, body=b"fixture completed output\n"):
        path = Path(lease["artifact_dir"]) / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return {"path": str(path), "sha256": hashlib.sha256(body).hexdigest()}

    def finish(self, lease, **kwargs):
        artifact = kwargs.pop("artifacts") if "artifacts" in kwargs else [self.output(lease)]
        return queue.complete(self.root, lease["task_id"], lease["token"],
                              artifacts=artifact, processed_records=3,
                              processed_images=2, now=kwargs.pop("now", 102), **kwargs)

    def test_preemption_reclaims_and_fences_old_token_from_every_transition(self):
        self.initialize()
        first = queue.claim(self.root, "preempted", now=101)
        first_output = self.output(first)
        second = queue.claim(self.root, "replacement", now=112)
        self.assertEqual(first["task_id"], second["task_id"])
        self.assertNotEqual(first["token"], second["token"])
        self.assertEqual(second["attempt"], 2)
        for callback in (
            lambda: queue.heartbeat(self.root, first["task_id"], first["token"], now=113),
            lambda: queue.fail(self.root, first["task_id"], first["token"], "late", now=113),
            lambda: self.finish(first, artifacts=[first_output], now=113),
        ):
            with self.assertRaises(queue.LeaseLost):
                callback()
        receipt = self.finish(second, now=113)
        self.assertEqual(receipt["attempt"], 2)
        result = queue.status(self.root, now=114)
        self.assertEqual(result["counts"], {"pending": 0, "running": 0,
                                          "failed": 0, "completed": 1})
        self.assertEqual(result["failed_attempts"], 1)

    def test_concurrent_processes_claim_each_task_exactly_once(self):
        self.initialize(32)
        with ProcessPoolExecutor(max_workers=6) as pool:
            batches = list(pool.map(claim_all, [(str(self.root), f"worker-{i}")
                                               for i in range(6)]))
        claims = [item for batch in batches for item in batch]
        self.assertEqual(len(claims), 32)
        self.assertEqual(len({item[0] for item in claims}), 32)
        self.assertEqual(len({item[1] for item in claims}), 32)
        self.assertEqual(queue.status(self.root, now=102)["counts"]["running"], 32)

    def test_completion_is_idempotent_immutable_and_not_counted_twice(self):
        self.initialize()
        lease = queue.claim(self.root, "worker", now=101)
        output = self.output(lease)
        first = self.finish(lease, artifacts=[output])
        self.assertEqual(first, self.finish(lease, artifacts=[output], now=900))
        with self.assertRaisesRegex(queue.QueueError, "immutable"):
            self.finish(lease, artifacts=[output], metadata={"changed": True}, now=900)
        status = queue.status(self.root, now=901)
        self.assertEqual(status["processed_records"], 3)
        self.assertEqual(status["processed_images"], 2)
        self.assertTrue(status["all_tasks_completed"])

    def test_corrupt_or_missing_output_never_completes(self):
        self.initialize()
        lease = queue.claim(self.root, "worker", now=101)
        output = self.output(lease)
        Path(output["path"]).write_bytes(b"damaged output")
        with self.assertRaises(queue.ArtifactError):
            self.finish(lease, artifacts=[output])
        Path(output["path"]).unlink()
        with self.assertRaises(queue.ArtifactError):
            self.finish(lease, artifacts=[output])
        with self.assertRaises(queue.ArtifactError):
            self.finish(lease, artifacts=[])
        status = queue.status(self.root, now=103)
        self.assertEqual(status["counts"]["completed"], 0)
        self.assertEqual(status["counts"]["running"], 1)
        self.assertEqual(status["processed_records"], 0)

    def test_partial_row_range_is_not_completion(self):
        self.initialize()
        lease = queue.claim(self.root, "worker", now=101)
        with self.assertRaisesRegex(queue.QueueError, "entire exclusive task row range"):
            queue.complete(self.root, lease["task_id"], lease["token"],
                           artifacts=[self.output(lease)], processed_records=2,
                           processed_images=2, now=102)
        self.assertEqual(queue.status(self.root, now=103)["counts"]["completed"], 0)

    def test_attempt_isolation_blocks_old_output_and_symlink_escape(self):
        self.initialize()
        first = queue.claim(self.root, "first", now=101)
        old = self.output(first)
        second = queue.claim(self.root, "second", now=112)
        with self.assertRaises(queue.ArtifactError):
            self.finish(second, artifacts=[old], now=113)
        directory = Path(second["artifact_dir"])
        directory.mkdir(parents=True)
        link = directory / "borrowed.jsonl"
        link.symlink_to(old["path"])
        with self.assertRaises(queue.ArtifactError):
            self.finish(second, artifacts=[{**old, "path": str(link)}], now=113)

    def test_lease_rechecked_after_hashing_before_commit(self):
        self.initialize()
        first = queue.claim(self.root, "preempted", now=101)
        output = self.output(first)
        actual = queue._artifacts
        successor = []

        def reclaim_during_hash(*args):
            checked = actual(*args)
            successor.append(queue.claim(self.root, "replacement", now=112))
            return checked

        with mock.patch.object(queue, "_artifacts", side_effect=reclaim_during_hash):
            with self.assertRaises(queue.LeaseLost):
                self.finish(first, artifacts=[output], now=102)
        self.assertEqual(queue.status(self.root, now=113)["counts"]["completed"], 0)
        self.finish(successor[0], now=113)

    def test_artifact_rechecked_after_hashing_before_commit(self):
        self.initialize()
        lease = queue.claim(self.root, "worker", now=101)
        output = self.output(lease)
        actual = queue._artifacts

        def mutate_after_hash(*args):
            checked = actual(*args)
            Path(output["path"]).write_bytes(b"changed just before commit")
            return checked

        with mock.patch.object(queue, "_artifacts", side_effect=mutate_after_hash):
            with self.assertRaises(queue.ArtifactError):
                self.finish(lease, artifacts=[output])
        self.assertEqual(queue.status(self.root, now=103)["counts"]["completed"], 0)

    def test_heartbeat_extends_lease_but_cannot_resurrect_expired_lease(self):
        self.initialize()
        lease = queue.claim(self.root, "worker", now=101)
        renewed = queue.heartbeat(self.root, lease["task_id"], lease["token"], now=109)
        self.assertEqual(renewed["lease_expires_at"], 119)
        self.assertIsNone(queue.claim(self.root, "other", now=112))
        with self.assertRaises(queue.LeaseLost):
            queue.heartbeat(self.root, lease["task_id"], lease["token"], now=119)

    def test_retry_limit_counts_expiry_and_explicit_failure(self):
        self.initialize(max_attempts=2)
        first = queue.claim(self.root, "first", now=101)
        state = queue.fail(self.root, first["task_id"], first["token"], "node shutdown", now=102)
        self.assertEqual(state["status"], "pending")
        second = queue.claim(self.root, "second", now=103)
        self.assertEqual(second["attempt"], 2)
        self.assertIsNone(queue.claim(self.root, "third", now=114))
        status = queue.status(self.root, now=115)
        self.assertEqual(status["counts"]["failed"], 1)
        self.assertEqual(status["failed_attempts"], 2)
        self.assertTrue(status["all_tasks_terminal"])
        self.assertFalse(status["all_tasks_completed"])
        self.assertEqual(status["processed_records"], 0)

    def test_status_tracks_work_and_deadline_independently(self):
        self.initialize(2, deadline_at=105)
        first = queue.claim(self.root, "worker", now=101)
        self.finish(first)
        current = queue.status(self.root, now=106)
        self.assertEqual(current["elapsed_seconds"], 6)
        self.assertEqual(current["seconds_to_deadline"], -1)
        self.assertTrue(current["deadline_exceeded"])
        self.assertEqual(current["counts"]["pending"], 1)
        self.assertEqual(current["processed_records"], 3)
        # Deadline is a reported target, not silent abandonment of remaining work.
        self.assertIsNotNone(queue.claim(self.root, "worker", now=106))

    def test_init_is_idempotent_and_rejects_changed_identity(self):
        first = self.initialize(3)
        self.assertEqual(first, self.initialize(3))
        with self.assertRaisesRegex(queue.QueueError, "identity differs"):
            self.initialize(4)
        with self.assertRaisesRegex(queue.QueueError, "identity differs"):
            self.initialize(3, max_attempts=4)
        self.assertEqual(queue.status(self.root, now=101)["task_count"], 3)

    def test_duplicate_or_invalid_manifest_does_not_initialize_queue(self):
        with self.assertRaisesRegex(ValueError, "duplicate task_id"):
            queue.init_queue(self.root, [task(), task()])
        self.assertFalse((self.root / "queue.json").exists())
        for value in ({**task(), "row_end": 4}, {**task(), "records": [{}]},
                      {**task(), "task_id": "../escape"}):
            with self.assertRaises(ValueError):
                queue.init_queue(self.root, [value])
        self.initialize()
        self.assertEqual(queue.status(self.root, now=101)["counts"]["pending"], 1)

    def test_orphan_claim_and_interrupted_retry_transition_recover(self):
        self.initialize()
        directory = self.root / "claims" / task()["task_id"]
        directory.mkdir()  # Simulate death between mkdir and state commit.
        lease = queue.claim(self.root, "first", now=101)
        state_path = self.root / "states" / (lease["task_id"] + ".json")
        state = json.loads(state_path.read_text())
        state["status"] = "pending"
        queue._atomic(state_path, state)  # Simulate death before retry marker creation.
        resumed = queue.claim(self.root, "second", now=102)
        self.assertEqual(resumed["attempt"], 2)
        self.assertNotEqual(lease["token"], resumed["token"])

    def test_mutated_task_manifest_is_not_silently_processed(self):
        config = self.initialize()
        path = self.root / config["catalog"] / "000000000000.json"
        value = json.loads(path.read_text())
        value["task"]["row_end"] = 999
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(queue.QueueError, "checksum changed"):
            queue.claim(self.root, "worker", now=101)

    def test_claim_cursor_does_not_reread_entire_pending_catalog(self):
        self.initialize(40)
        actual = queue._manifest
        with mock.patch.object(queue, "_manifest", wraps=actual) as reads:
            for _ in range(6):
                queue.claim(self.root, "worker", now=101)
            self.assertEqual(reads.call_count, 6)


if __name__ == "__main__":
    unittest.main()
