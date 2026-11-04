import copy
import unittest

from corpus_pipeline.detector_calibration.analyze import analyze, empirical_threshold, review_candidates, wilson


class CalibrationTest(unittest.TestCase):
    def fixture(self):
        samples = [
            {"image_sha256": "a", "calibration_split": "calibration", "source_image_origin": "natural", "domain": "photo"},
            {"image_sha256": "b", "calibration_split": "calibration", "source_image_origin": "unknown", "domain": "mixed_web"},
            {"image_sha256": "c", "calibration_split": "holdout", "source_image_origin": "programmatic", "domain": "chart"},
        ]
        scores = [{"image_sha256": s["image_sha256"], "scores": {
            m: {"prob": p} for m in ("cf224", "cf384", "unifd")}}
            for s, p in zip(samples, (.2, .99, .8))]
        return samples, scores

    def test_holdout_never_changes_calibration_threshold(self):
        samples, scores = self.fixture()
        before, _ = analyze(samples, scores)
        modified = copy.deepcopy(scores)
        for score in modified[2]["scores"].values():
            score["prob"] = .01
        after, _ = analyze(samples, modified)
        a = before["models"]["cf384"]["negative_control_calibration"]["0.01"]
        b = after["models"]["cf384"]["negative_control_calibration"]["0.01"]
        self.assertEqual(a["threshold"], b["threshold"])
        self.assertEqual(a["calibration_source_supported_keep_n"], 1)
        self.assertEqual(a["holdout_flagged"], 1)
        self.assertEqual(b["holdout_flagged"], 0)

    def test_unknown_is_not_assumed_real_and_no_invented_positive_metrics(self):
        samples, scores = self.fixture()
        result, _ = analyze(samples, scores)
        self.assertIsNone(result["precision_recall_auc"])
        self.assertLess(result["models"]["cf384"]["negative_control_calibration"]["0.01"]["threshold"], .3)

    def test_saturated_negative_can_disable_threshold(self):
        self.assertGreater(empirical_threshold([1.0], 0), 1)
        self.assertIsNone(empirical_threshold([], .01))
        self.assertGreater(wilson(0, 100)[1], .03)

    def test_invalid_scores_are_absent_from_pairwise_and_review(self):
        samples, scores = self.fixture()
        scores[0]["scores"]["cf224"]["prob"] = float("nan")
        result, merged = analyze(samples, scores)
        self.assertEqual(result["models"]["cf224"]["all"]["n"], 2)
        self.assertEqual(result["pairwise"]["cf224/cf384"]["n"], 2)
        review_candidates(merged)


if __name__ == "__main__":
    unittest.main()
