import unittest
import hashlib
from pathlib import Path
import tempfile
from PIL import Image
from corpus_pipeline.api_review.finalize import decide, finalize, apply_chartx_mapping


def decision(status, rule='unresolved_source', strong_ai=False):
    return {'image_sha256': 'a' * 64, 'source_rule_decision': rule, 'review_status': status,
            'pilot_group': 'detector_alert', 'reviews': {'one': {'parsed': {
                'assessment': 'likely_ai_generated' if strong_ai else 'uncertain',
                'confidence': 'high' if strong_ai else 'low'}}}}


class FinalPolicyTests(unittest.TestCase):
    def test_source_precedence_and_no_deletion(self):
        r = decide(decision('unanimous_high_ai_candidate', 'protected_source', True))
        self.assertEqual(r['final_disposition'], 'protected_source')
        self.assertFalse(r['automatic_deletion'])

    def test_single_round_ai_not_excluded(self):
        self.assertEqual(decide(decision('unanimous_high_ai_candidate'))['final_disposition'], 'cannot_determine')

    def test_second_round_does_not_erase_strong_conflict(self):
        r = decide(decision('needs_adjudication', strong_ai=True), decision('unanimous_non_ai_candidate'))
        self.assertEqual(r['final_disposition'], 'cannot_determine')

    def test_failure_is_retry_not_unknown(self):
        r = decide(decision('api_incomplete'))
        self.assertTrue(r['operational_retry_required'])
        self.assertFalse(r['manual_review_required'])
        protected = decide(decision('api_incomplete', 'protected_source'))
        self.assertEqual(protected['final_disposition'], 'protected_source')
        self.assertTrue(protected['operational_retry_required'])

    def test_duplicate_or_unmatched_rejected(self):
        d = decision('needs_adjudication')
        with self.assertRaises(ValueError):
            finalize([d], [d, d])

    def test_source_claim_cannot_override_different_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            local, upstream = Path(directory) / 'local.png', Path(directory) / 'upstream.png'
            Image.new('RGB', (16, 16), 'white').save(local)
            Image.new('RGB', (16, 16), 'black').save(upstream)
            sha = hashlib.sha256(local.read_bytes()).hexdigest()
            d = {**decision('needs_adjudication'), 'source': 'chartx', 'image_path': str(local), 'image_sha256': sha}
            m = {'image_sha256': sha, 'upstream_repository': 'InternScience/ChartX',
                 'metadata_match_count': 1, 'matches': [{'rgba_pixel_equal': True,
                 'upstream_image_path': str(upstream), 'upstream_image_sha256': hashlib.sha256(upstream.read_bytes()).hexdigest()}]}
            with self.assertRaisesRegex(ValueError, 'pixels differ'):
                apply_chartx_mapping(d, m)


if __name__ == '__main__':
    unittest.main()
