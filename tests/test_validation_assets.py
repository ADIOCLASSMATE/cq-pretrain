"""A validation suite resolves its own assets, and a broken one cannot stop training.

The in-training suite named its images relatively, so they resolved against the
working directory, and `git archive` left them out of frozen snapshots because
.gitignore excludes them. Validation then sampled its T2I cases, raised
FileNotFoundError on the first I2T case and aborted the update; the platform
restarted the job, which replayed from the last checkpoint. That cost every step
since the checkpoint, once per validation interval, and wrote no report at all.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluations.sampling import anchor_case_paths, suite_assets


class AnchorPathTests(unittest.TestCase):
    def test_a_relative_image_resolves_against_the_suite_directory(self):
        case = anchor_case_paths({"image": "validation_assets/a.jpg"}, Path("/run/src"))
        self.assertEqual(case["image"], "/run/src/validation_assets/a.jpg")

    def test_an_absolute_image_is_left_alone(self):
        case = anchor_case_paths({"image": "/data/a.jpg"}, Path("/run/src"))
        self.assertEqual(case["image"], "/data/a.jpg")

    def test_the_original_case_is_not_mutated(self):
        original = {"image": "rel.jpg"}
        anchor_case_paths(original, Path("/run/src"))
        self.assertEqual(original["image"], "rel.jpg")

    def test_history_images_video_and_video_path_are_anchored(self):
        case = anchor_case_paths(
            {
                "images": [{"path": "a.jpg", "sha256": "x"}, {"path": "/abs/b.jpg"}],
                "video": {"path": "clip.mp4", "frames_path": "frames/"},
                "video_path": "legacy.mp4",
            },
            Path("/run/src"),
        )
        self.assertEqual(case["images"][0]["path"], "/run/src/a.jpg")
        self.assertEqual(case["images"][0]["sha256"], "x")
        self.assertEqual(case["images"][1]["path"], "/abs/b.jpg")
        self.assertEqual(case["video"]["path"], "/run/src/clip.mp4")
        self.assertEqual(case["video"]["frames_path"], "/run/src/frames/")
        self.assertEqual(case["video_path"], "/run/src/legacy.mp4")

    def test_a_text_only_case_is_unchanged(self):
        case = anchor_case_paths({"id": "t", "task": "T2I", "prompt": "a cat"}, Path("/run/src"))
        self.assertEqual(case, {"id": "t", "task": "T2I", "prompt": "a cat"})


class SuiteAssetTests(unittest.TestCase):
    def suite(self, root, cases):
        path = Path(root) / "suite.json"
        path.write_text(json.dumps({"cases": cases}))
        return path

    def test_it_reports_every_referenced_file_against_the_suite_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.suite(temporary, [
                {"id": "a", "task": "T2I", "prompt": "a cat"},
                {"id": "b", "task": "I2T", "image": "assets/one.jpg"},
            ])
            self.assertEqual(suite_assets(path), [Path(temporary) / "assets/one.jpg"])

    def test_the_real_shipped_suite_declares_its_images(self):
        """The regression this file exists for: suite.json referenced four COCO images."""
        root = Path(__file__).resolve().parents[1] / "validation_assets"
        if not (root / "suite.json").is_file():
            self.skipTest("no local validation suite")
        assets = suite_assets(root / "suite.json")
        self.assertTrue(assets, "the shipped suite references no images")
        self.assertTrue(all(p.is_absolute() for p in assets))


class SkipRatherThanAbortTests(unittest.TestCase):
    """Missing assets must skip validation, not take the training job down."""

    def persistence(self, suite_path):
        from training.persistence import Persistence

        instance = Persistence.__new__(Persistence)
        instance.args = type("Args", (), {"validation_suite": suite_path})()
        instance.run = type("Run", (), {"rank": 0})()
        instance._warned_validation_assets = False
        return instance

    def test_a_complete_suite_reports_nothing_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "one.jpg").write_bytes(b"x")
            path = Path(temporary) / "suite.json"
            path.write_text(json.dumps({"cases": [{"id": "b", "task": "I2T", "image": "one.jpg"}]}))
            with patch("builtins.print"):
                self.assertIsNone(self.persistence(path).missing_validation_assets())

    def test_an_absent_image_is_named_and_does_not_raise(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "suite.json"
            path.write_text(json.dumps({"cases": [{"id": "b", "task": "I2T", "image": "gone.jpg"}]}))
            with patch("builtins.print") as printed:
                missing = self.persistence(path).missing_validation_assets()
            self.assertEqual(len(missing), 1)
            self.assertIn("gone.jpg", missing[0])
            self.assertIn("validation_skipped", printed.call_args[0][0])

    def test_an_unreadable_suite_is_reported_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "suite.json"
            path.write_text("{ not json")
            with patch("builtins.print"):
                missing = self.persistence(path).missing_validation_assets()
            self.assertEqual(len(missing), 1)
            self.assertIn("unreadable suite", missing[0])

    def test_the_warning_is_printed_once_not_every_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "suite.json"
            path.write_text(json.dumps({"cases": [{"id": "b", "task": "I2T", "image": "gone.jpg"}]}))
            instance = self.persistence(path)
            with patch("builtins.print") as printed:
                for _ in range(4):
                    self.assertIsNotNone(instance.missing_validation_assets())
            self.assertEqual(printed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
