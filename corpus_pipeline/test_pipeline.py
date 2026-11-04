"""End-to-end filesystem tests of publication, resume, failure and exports."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from corpus_pipeline.pipeline import main, configure_adapter, file_digest


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "inputs"
        self.source = self.root / "photos"
        self.source.mkdir(parents=True)
        # Deliberately not a decodable image: this stage checks references only.
        (self.source / "one.jpg").write_bytes(b"present-not-decoded")
        self.metadata = self.source / "records.jsonl"
        records = [
            {"id": "natural_ai_caption", "image": "one.jpg", "caption": "A bird.", "caption_origin": "ai"},
            {"id": "generated_image", "image": "one.jpg", "caption": "A bird.", "image_origin": "ai_generated"},
            {"id": "text", "text": "Standalone natural language text."},
            {"id": "missing", "image": "missing.jpg", "caption": "A bird."},
        ]
        self.metadata.write_text("".join(json.dumps(x) + "\n" for x in records))
        self.registry = self.base / "sources.json"
        self.registry.write_text(json.dumps({"version": "fixture", "sources": [{
            "id": "fixture:photos", "match": {"collection": "genund", "source": "photos"},
            "image_origin": "natural", "caption_origin": "unknown", "evidence_level": "fixture"}]}))
        self.output = self.base / "run"
        self.args = ["run", "--collection", "genund", "--root", str(self.root),
                     "--output", str(self.output), "--registry", str(self.registry), "--mode", "full"]

    def indexed(self):
        rows = []
        for line in (self.output / "manifest.jsonl").read_text().splitlines():
            unit = json.loads(line)
            with gzip.open(self.output / unit["index_path"], "rt") as stream:
                rows.extend(json.loads(line) for line in stream)
        return rows

    def test_end_to_end_resume_and_export_views(self):
        self.assertEqual(main(self.args), 0)
        by_id = {row["source_sample_id"]: row for row in self.indexed() if row.get("source_sample_id")}
        self.assertEqual(by_id["natural_ai_caption"]["decision"], "accepted")
        self.assertTrue(by_id["natural_ai_caption"]["provenance"]["text"]["ai_generated"])
        self.assertEqual(by_id["generated_image"]["decision"], "rejected")
        self.assertEqual(by_id["missing"]["decision"], "quarantine")
        self.assertEqual(by_id["text"]["category"], "pure_text")
        files = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.output.glob("index/*/*.gz")}
        self.assertEqual(main(self.args), 0)
        for path, value in files.items():
            self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), value)
        export = self.base / "t2i.jsonl.gz"
        self.assertEqual(main(["export", "--run", str(self.output), "--output", str(export), "--view", "single_t2i"]), 0)
        with gzip.open(export, "rt") as stream:
            exported = [json.loads(line) for line in stream]
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["source_sample_id"], "natural_ai_caption")
        self.assertEqual(exported[0]["selected_view"], "single_t2i")
        self.assertFalse(exported[0]["view_materialized"])
        self.assertNotIn("text", exported[0]["segments"][-1])
        self.assertEqual(main(self.args + ["--include-text"]), 2)

    def test_partial_unit_is_not_published(self):
        self.metadata.write_text('{"text":"valid before failure"}\n{invalid\n')
        self.assertEqual(main(self.args), 1)
        self.assertFalse(json.loads((self.output / "COMPLETE.json").read_text())["success"])
        self.assertFalse(any(row.get("category") == "pure_text" for row in self.indexed()))
        self.assertEqual(list(self.output.rglob("*.partial")), [])

    def test_changed_inputs_and_corrupt_indexes_refused(self):
        self.assertEqual(main(self.args), 0)
        first = next(self.output.glob("index/*/*.gz"))
        original = first.read_bytes()
        first.write_bytes(b"corrupted")
        self.assertEqual(main(self.args), 2)
        first.write_bytes(original)
        self.metadata.write_text(self.metadata.read_text() + '{"text":"changed"}\n')
        self.assertEqual(main(self.args), 2)
        self.assertFalse(json.loads((self.output / "COMPLETE.json").read_text())["success"])

    def test_removed_input_invalidates_completion(self):
        self.assertEqual(main(self.args), 0)
        self.metadata.unlink()
        self.assertEqual(main(self.args), 2)
        self.assertFalse(json.loads((self.output / "COMPLETE.json").read_text())["success"])
        self.assertEqual(main(["export", "--run", str(self.output), "--output", str(self.base / "bad.gz")]), 2)

    def test_source_output_guard(self):
        args = list(self.args)
        args[args.index("--output") + 1] = str(self.root / "results")
        self.assertEqual(main(args), 2)

    def test_explicit_path_mapping_keeps_suffix(self):
        registry = {"sources": [{"match": {"collection": "genund", "source": "Infinity-MM"},
                                 "config": {"path_prefix_map": {"/old/Infinity-MM": "{collection_root}/Infinity-MM"}}}]}
        unit = {"collection": "genund", "source": "Infinity-MM", "path": "/cpu/Infinity-MM/train.json",
                "config": {"collection_root": "/cpu", "source_root": "/cpu/Infinity-MM"}}
        configured = configure_adapter(unit, registry)
        self.assertEqual(configured["config"]["path_prefix_map"], {"/old/Infinity-MM": "/cpu/Infinity-MM"})
        self.assertNotIn("path_prefix_map", unit["config"])

    def test_worker_partitions_are_disjoint(self):
        self.assertEqual(main(self.args), 0)
        expected = {row["sample_id"] for row in self.indexed()}
        partitions = []
        for i in range(2):
            self.output = self.base / f"worker{i}"
            args = list(self.args)
            args[args.index("--output") + 1] = str(self.output)
            self.assertEqual(main(args + ["--worker-count", "2", "--worker-index", str(i)]), 0)
            partitions.append({row["sample_id"] for row in self.indexed()})
        self.assertFalse(partitions[0] & partitions[1])
        self.assertEqual(partitions[0] | partitions[1], expected)

    def test_default_policy_preserves_real_source_edit_provenance(self):
        rows = [{"id": "rendered", "image": "one.jpg", "caption": "A chart", "image_origin": "programmatic"},
                {"id": "edited", "image": {"path": "one.jpg", "image_origin": "ai_edited",
                    "edit_source_provenance": {"origin": "natural", "source_id": "photo-real-1"}},
                    "caption": "An edited real photo", "caption_origin": "ai"},
                {"id": "unknown-edit", "image": "one.jpg", "caption": "An edit", "image_origin": "ai_edited"}]
        self.metadata.write_text("".join(json.dumps(x) + "\n" for x in rows))
        self.assertEqual(main(self.args), 0)
        by_id = {row["source_sample_id"]: row for row in self.indexed() if row.get("source_sample_id")}
        self.assertEqual(by_id["rendered"]["decision"], "accepted")
        self.assertEqual(by_id["edited"]["decision"], "accepted")
        self.assertEqual(by_id["edited"]["provenance"]["images"][0]["policy_exception"], "ai_edit_from_natural_source")
        self.assertTrue(by_id["edited"]["provenance"]["text"]["ai_generated"])
        self.assertEqual(by_id["unknown-edit"]["decision"], "quarantine")

    def historical_index(self, rows, collection="dcvlm"):
        """A checksum-valid old run may still advertise obsolete reverse views."""
        self.output.mkdir()
        path = self.output / "index.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            for row in rows:
                stream.write(json.dumps({"collection": collection, **row}) + "\n")
        (self.output / "manifest.jsonl").write_text(json.dumps({
            "index_path": path.name, "index_sha256": file_digest(path)}) + "\n")
        (self.output / "run.json").write_text(json.dumps({"run_id": "historical", "collection": collection,
            "mode": "sample", "worker_count": 1}))
        (self.output / "COMPLETE.json").write_text(json.dumps({"success": True, "full_corpus_processed": False}))

    def test_historical_dcvlm_t2i_export_cannot_bypass_collection_policy(self):
        self.historical_index([{"decision": "accepted", "category": "single_i2t",
                                "eligible_views": ["single_t2i", "single_i2t"]}])
        for flag in ("--view", "--category"):
            for category in ("single_t2i", "multi_t2i"):
                with self.subTest(flag=flag, category=category):
                    output = self.base / f"forbidden-{flag[2:]}-{category}.gz"
                    self.assertEqual(main(["export", "--run", str(self.output), "--output", str(output),
                                           flag, category]), 2)
                    self.assertFalse(output.exists())
                    self.assertFalse(Path(str(output) + ".manifest.json").exists())

    def test_default_dcvlm_training_export_skips_t2i_and_preserves_audits(self):
        rows = [{"decision": "accepted", "category": "single_i2t",
                 "eligible_views": ["single_t2i", "single_i2t"]}]
        rows.extend({"decision": decision, "category": "single_t2i", "eligible_views": []}
                    for decision in ("accepted", "rejected", "quarantine"))
        self.historical_index(rows)
        export = self.base / "training.gz"
        self.assertEqual(main(["export", "--run", str(self.output), "--output", str(export)]), 0)
        with gzip.open(export, "rt") as stream:
            training = [json.loads(line) for line in stream]
        self.assertEqual(len(training), 1)
        self.assertEqual(training[0]["selected_view"], "single_i2t")
        self.assertEqual(training[0]["eligible_views"], ["single_i2t"])
        self.assertEqual(training[0]["indexed_eligible_views"], ["single_t2i", "single_i2t"])
        manifest = json.loads(Path(str(export) + ".manifest.json").read_text())
        self.assertEqual(manifest["skipped_reason_counts"], {"t2i_training_requires_genund": 1})
        for decision in ("rejected", "quarantine"):
            for flags in ([], ["--category", "single_t2i"], ["--view", "single_t2i"]):
                with self.subTest(decision=decision, flags=flags):
                    output = self.base / f"audit-{decision}-{len(flags)}-{flags[0][2:] if flags else 'all'}.gz"
                    self.assertEqual(main(["export", "--run", str(self.output), "--output", str(output),
                                           "--decision", decision, *flags]), 0)
                    with gzip.open(output, "rt") as stream:
                        audit = [json.loads(line) for line in stream]
                    self.assertEqual(len(audit), 1)
                    self.assertEqual(audit[0]["category"], "single_t2i")
                    self.assertEqual(audit[0]["export_purpose"], "audit")

    def test_genund_export_does_not_accept_dcvlm_row_labeled_as_t2i_view(self):
        self.historical_index([{"collection": "dcvlm", "decision": "accepted", "category": "single_i2t",
                                "eligible_views": ["single_t2i", "single_i2t"]}], collection="genund")
        export = self.base / "mixed.gz"
        self.assertEqual(main(["export", "--run", str(self.output), "--output", str(export),
                               "--view", "single_t2i"]), 0)
        with gzip.open(export, "rt") as stream:
            self.assertEqual(list(stream), [])
        manifest = json.loads(Path(str(export) + ".manifest.json").read_text())
        self.assertEqual(manifest["skipped_reason_counts"], {"t2i_training_requires_genund": 1})


if __name__ == "__main__":
    unittest.main()
