"""Boundary checks for offline benchmark denominator and event accounting."""
import json
from pathlib import Path
import tempfile
import unittest

from corpus_pipeline.api_benchmark import analyze_runs as a


class AnalysisTest(unittest.TestCase):
    def test_half_open_concurrency_and_invalid_interval(self):
        rows = [{"start_unix": 0, "end_unix": 10},
                {"start_unix": 10, "end_unix": 20},
                {"start_unix": 5, "end_unix": 15},
                {"start_unix": 11, "end_unix": 10},
                {"start_unix": 12, "end_unix": 12}]
        result = a.concurrency_stats(rows, 20)
        self.assertEqual(result["observed_peak_client_inflight"], 2)
        self.assertEqual(result["invalid_or_missing_intervals"], 1)
        self.assertEqual(result["mean_client_inflight_over_api_batch"], 1.5)

    def test_bins_keep_fixed_denominator_and_final_boundary(self):
        rows = [{"end_unix": 10, "http_status": 200, "effective_success": True},
                {"end_unix": 12, "http_status": None, "effective_success": False}]
        bins = a.completion_bins(rows, 0, 12)
        self.assertEqual(bins[1]["observed_seconds"], 2)
        self.assertEqual(bins[1]["schema_effective_per_second_fixed_bin"], .1)
        self.assertEqual(bins[1]["attempt_completions_per_second_fixed_bin"], .2)
        self.assertTrue(bins[1]["partial_final_bin"])
        self.assertEqual(a.completion_bins(rows[:1], 0, 10)[0]["completed_attempts"], 1)

    def test_success_latency_does_not_hide_failed_tail(self):
        self.assertEqual(a.distribution([1, 2, 100])["p95"], 90.19999999999999)
        self.assertEqual(a.distribution([1, 2])["p95"], 1.95)

    def test_missing_output_usage_is_not_zero_or_input_text(self):
        row = {"usage": {"input_tokens": 999, "prompt_tokens_details": {"text_tokens": 777}}}
        self.assertIsNone(a.usage_value(row, "reported_text_tokens"))
        self.assertIsNone(a.usage_value(row, "billed_output_tokens"))
        self.assertIsNone(a.distribution([None])["mean"])

    def test_extrapolation_is_caption_only_and_not_quality_pass(self):
        self.assertFalse(a.extrapolation("review", 2)["applicable"])
        result = a.extrapolation("caption", 2)
        self.assertAlmostEqual(result["days_per_million_schema_success_outputs"], 1_000_000 / 172800)
        self.assertFalse(result["quality_qualified"])
        self.assertFalse(result["full_pipeline_estimate"])
        self.assertIsNone(a.extrapolation("caption", 0)["days_for_dcvlm_caption_count_at_this_rate"])

    def make_run(self, root):
        directory = root / "complete"
        directory.mkdir()
        run = {"model": "qwen-test", "task": "caption", "logical_requests": 2, "concurrency": 2}
        caps = {"short": "one word", "medium": "one two three", "long": "one two three four"}
        rows = [
            {"request_id": "r0a0", "logical_request_index": 0, "sample_id": "s0", "start_unix": 0, "end_unix": 10,
             "latency_seconds": 10, "http_status": 429, "status": "api_error", "effective_success": False,
             "image_inputs": [{"original_sha256": "image0"}]},
            {"request_id": "r0a1", "logical_request_index": 0, "sample_id": "s0", "start_unix": 10, "end_unix": 20,
             "latency_seconds": 10, "http_status": 200, "status": "ok", "effective_success": True, "schema_valid": True,
             "parsed": {"captions": caps}, "image_inputs": [{"original_sha256": "image0"}]},
            {"request_id": "r1a0", "logical_request_index": 1, "sample_id": "s1", "start_unix": 0, "end_unix": 100,
             "latency_seconds": 100, "http_status": None, "status": "api_error", "error_type": "ReadTimeout",
             "effective_success": False, "image_inputs": [{"original_sha256": "image0"}]},
        ]
        run_bytes = json.dumps(run).encode()
        request_bytes = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
        summary = {"schema_version": "api_benchmark.summary.v1", "wall_seconds": 100, "start_unix": 0, "end_unix": 100,
                   "preparation_seconds": 1, "http_attempts": 3, "effective_successful_requests": 1,
                   "requests_sha256": a.sha(request_bytes), "run_sha256": a.sha(run_bytes)}
        (directory / "run.json").write_bytes(run_bytes)
        (directory / "requests.jsonl").write_bytes(request_bytes)
        (directory / "summary.json").write_text(json.dumps(summary))
        return directory

    def test_retries_and_repeated_images_keep_correct_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = a.analyze_run(self.make_run(Path(temporary)))
        self.assertEqual(result["counts"]["observed_logical_requests"], 2)
        self.assertEqual(result["counts"]["attempts"], 3)
        self.assertEqual(result["rates"]["schema_effective_per_logical_request"], .5)
        self.assertEqual(result["full_batch_throughput"]["primary_schema_effective_logical_requests_per_second"], .01)
        self.assertEqual(result["sampling"]["retry_attempts_not_additional_unique_inputs"], 1)
        self.assertEqual(result["sampling"]["repeated_image_set_logical_requests"], 1)
        self.assertEqual(result["sampling"]["unique_original_image_sha256"], 1)
        self.assertEqual(result["counts"]["http_429"], 1)
        self.assertEqual(result["counts"]["timeout_attempts"], 1)
        self.assertEqual(result["caption_length_among_schema_effective_logical_requests"]["medium"]["whitespace_words"]["mean"], 3)
        self.assertEqual(result["caption_length_among_schema_effective_logical_requests"]["three_variants_total_per_response"]["whitespace_words"]["mean"], 9)

    def test_changed_artifact_is_excluded_and_unfinished_not_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.make_run(root)
            (directory / "requests.jsonl").write_text("{}\n")
            pending = root / "pending"
            pending.mkdir()
            (pending / "run.json").write_text("{}")
            (pending / "requests.jsonl").write_text("")
            result = a.analyze_root(root)
        self.assertEqual(result["completed_runs_analyzed"], 0)
        self.assertEqual(len(result["excluded_completed_runs"]), 1)
        self.assertEqual(len(result["unfinished_runs_not_scored"]), 1)


if __name__ == "__main__":
    unittest.main()
