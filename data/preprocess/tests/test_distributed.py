"""Concurrency, recovery and final publication regression checks."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

from data.preprocess import distributed as d
from data.preprocess.manifest import atomic_json, atomic_jsonl


class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.q = d.queue(self.root)
        for name in ("claims", "attempts", "batches", "baseline", "validation"):
            (self.q / name).mkdir(parents=True)

    def test_atomic_claim_has_one_winner(self):
        task = dict(batch_id="batch_000000")
        with ThreadPoolExecutor(max_workers=32) as pool:
            owners = list(pool.map(lambda slot: d.claim_batch(self.root, task, "cpu", "test", slot), range(32)))
        self.assertEqual(sum(owner is not None for owner in owners), 1)

    def test_live_owner_and_completed_batch_cannot_be_requeued(self):
        task = dict(batch_id="batch_000000")
        d.claim_batch(self.root, task, "cpu", "test", 0)
        with self.assertRaisesRegex(ValueError, "alive"):
            d.recover(self.root, task["batch_id"])
        atomic_json(self.q / "claims" / task["batch_id"] / "done.json", {})
        with self.assertRaisesRegex(ValueError, "Completed"):
            d.recover(self.root, task["batch_id"])

    def test_dead_owner_recovery_keeps_outputs_and_previous_attempt(self):
        task = dict(batch_id="batch_000000")
        claim = self.q / "claims" / task["batch_id"]
        claim.mkdir()
        atomic_json(claim / "owner.json", dict(host=platform.node(), pid=99999999, start_ticks=1))
        output = self.root / "committed.mp4"
        output.write_bytes(b"keep")
        d.recover(self.root, task["batch_id"])
        self.assertFalse(claim.exists())
        self.assertEqual(output.read_bytes(), b"keep")
        self.assertEqual(len(list((self.q / "attempts" / task["batch_id"]).glob("*/owner.json"))), 1)

    def test_partition_covers_every_sample_once_with_bounded_work(self):
        rows = [dict(sample_id=str(i), source=dict(num_frames=161+i*37)) for i in range(50)]
        batches = d.split_rows(rows, 4000, 5)
        self.assertEqual([r for b in batches for r in b], rows)
        for batch in batches:
            self.assertLessEqual(len(batch), 5)
            self.assertLessEqual(sum(r["source"]["num_frames"] for r in batch), 4000)

    def test_finalize_preserves_existing_files_and_combines_disjoint_journals(self):
        selected, old, new, checks_old, checks_new = [], [], [], [], []
        for name, events, checks in (("old.mp4", old, checks_old), ("new.mp4", new, checks_new)):
            row = dict(sample_id=name, video=name, video_path=f"videos/part1/{name}", caption="caption", source=dict(part=1))
            selected.append(row)
            for variant, size in d.VARIANTS.items():
                path = self.root / variant / row["video_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"mp4")
                record = dict(row, variant=variant, **size, num_frames=81, fps_num=15, fps_den=1, output_bytes=3)
                events.append(dict(sample_id=name, variant=variant, run_id="test", status="success", record=record))
                checks.append(dict(sample_id=name, variant=variant, run_id="test", output_sha256=d.sha256(path)))
        atomic_jsonl(self.root / "metadata/selected.jsonl", selected)
        atomic_jsonl(self.q / "baseline/_state/part1.jsonl", old)
        atomic_jsonl(self.q / "baseline/metadata/integrity_parts/part1.jsonl", checks_old)
        baseline = d.successful_snapshots(self.root, {(e["sample_id"],e["variant"]):e for e in old},
                                         {(e["sample_id"],e["variant"]):e for e in checks_old})
        atomic_jsonl(self.q / "baseline/completed_files.jsonl", baseline)
        task = dict(batch_id="batch_000000", part=1, samples=1, source_frames=161, inherited_outputs=0)
        atomic_jsonl(self.q / "tasks.jsonl", [task])
        atomic_json(self.q / "plan.json", dict(root=str(self.root), candidates=2, baseline_completed_samples=1,
                    baseline_completed_outputs=2, source_frames=161))
        atomic_jsonl(self.q / "batches/batch_000000/_state/part1.jsonl", new)
        atomic_jsonl(self.q / "batches/batch_000000/metadata/integrity_parts/part1.jsonl", checks_new)
        atomic_json(self.q / "batches/batch_000000/_state/part1.status.json",
                    dict(success=1, committed_video_files=2))
        atomic_json(self.q / "claims/batch_000000/done.json", {})
        before = {p: p.stat().st_mtime_ns for p in self.root.glob("*/videos/*/*.mp4")}
        with patch.object(d, "load_config", return_value=dict(run_id="test")):
            summary = d.finalize(self.root)
        self.assertTrue(summary["all_candidates_succeeded"])
        self.assertEqual(summary["success"], 2)
        self.assertEqual(before, {p:p.stat().st_mtime_ns for p in before})
        self.assertEqual(len(d.read_jsonl(self.root / "pairs.jsonl")), 2)


if __name__ == "__main__":
    unittest.main()
