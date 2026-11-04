import gzip
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from corpus_pipeline.dedup.decisions import file_sha, normalize_quality, normalize_qwen, prepare


def qrow(n, recommendation='admit', assessment='likely_non_ai', confidence='medium', reason='ai_visual_review_passed_screening'):
    evidence = {'model_recommendation': recommendation, 'input_sha256': 'input-' + str(n),
                'reviews': {'primary': {'decision': recommendation, 'assessment': assessment,
                                        'confidence': confidence, 'reason': reason}}}
    return {'asset_id': f'{n:064x}', 'input_sha256': evidence['input_sha256'],
            'decision': 'pending', 'reason': 'visual_annotation_only_requires_confirmation',
            'state': 'done', 'decision_json': json.dumps(evidence)}


def orow(n, category='admitted', reason='none', **kwargs):
    return {'item_id': f'{n:064x}', 'decision': {'admitted': 'admit', 'quality_excluded': 'reject',
            'ai_excluded': 'reject', 'visual_pending': 'pending'}[category],
            'report_category': category, 'rejection_reason': reason,
            'decision_authority': 'codex_model', 'human_review': None,
            'model_screening': {'result': {'assessment': 'likely_non_ai', 'quality': 'good'}}, **kwargs}


class DecisionTests(unittest.TestCase):
    def test_outer_pending_is_not_recommendation_and_no_quality_inference(self):
        result = normalize_qwen(qrow(1))
        self.assertEqual(result['screen_action'], 'retain_annotated')
        self.assertFalse(result['quality_reviewed'])
        self.assertFalse(result['strict_blur_reviewed'])

    def test_ai_requires_existing_high_confidence_generated_rule(self):
        strong = qrow(1, 'reject', 'likely_ai_generated', 'high', 'ai_visual_review_high_confidence_generated')
        self.assertEqual(normalize_qwen(strong)['screen_action'], 'exclude')
        weak = qrow(2, 'reject', 'possible_ai_edit', 'high', 'ai_visual_review_high_confidence_generated')
        self.assertEqual(normalize_qwen(weak)['screen_action'], 'pending')
        weak = qrow(3, 'reject', 'likely_ai_generated', 'medium', 'ai_visual_review_high_confidence_generated')
        self.assertEqual(normalize_qwen(weak)['screen_action'], 'pending')

    def test_historical_human_ai_exclusion_cannot_be_cleared_by_model(self):
        row = orow(1, baseline_screening={'human_review': {'decision': 'exclude_ai'}})
        normalized = normalize_quality(row)
        self.assertEqual(normalized['screen_action'], 'exclude')
        self.assertEqual(normalized['ai_decision'], 'human_confirmed_ai_generated')
        self.assertEqual(normalized['decision_authority'], 'human_user')

    def test_quality_coverage_and_pending_are_explicit(self):
        old = normalize_quality(orow(1))
        strict = normalize_quality(orow(2, 'visual_pending', 'blur_review_uncertain', blur_status='uncertain'))
        self.assertTrue(old['quality_reviewed'])
        self.assertFalse(old['strict_blur_reviewed'])
        self.assertTrue(strict['strict_blur_reviewed'])
        self.assertEqual(strict['screen_action'], 'pending')

    def fixture(self, root, quality=None, qwen=None):
        q = root / 'qwen'; q.mkdir()
        qwen = qwen if qwen is not None else [qrow(1), qrow(2), qrow(3)]
        pq.write_table(pa.Table.from_pylist(qwen), q / 'decisions.parquet')
        (q / 'COMPLETE.json').write_text(json.dumps({'screening_pass_complete': True,
            'artifacts': [{'path': 'decisions.parquet', 'sha256': file_sha(q / 'decisions.parquet')}],
            'scope': {'reconciliation_complete_sha256': 'fixture'}}))
        overlay = root / 'overlay.jsonl.gz'
        rows = quality if quality is not None else [orow(2, 'quality_excluded', 'visible_blur_excluded', blur_status='blur_present'),
                                                   orow(3, 'visual_pending', 'blur_review_uncertain', blur_status='uncertain')]
        with gzip.open(overlay, 'wt') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        c = root / 'quality_complete.json'
        c.write_text(json.dumps({'complete': True, 'revision_sha256': 'revision'}))
        pointer = root / 'QUALITY_POLICY_CURRENT.json'
        pointer.write_text(json.dumps({'revision_sha256': 'revision',
            'completion': {'path': str(c), 'sha256': file_sha(c)},
            'resolved_screening_overlay': {'path': str(overlay), 'sha256': file_sha(overlay),
                                           'records': len(rows), 'bytes': overlay.stat().st_size}}))
        return q, pointer, root / 'out'

    def test_end_to_end_overlay_precedence_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            result = prepare(*args)
            self.assertEqual(result['counts'], {'retain_annotated': 1, 'exclude': 1, 'pending': 1})
            self.assertEqual(result['records'], 3)
            self.assertEqual(result['strict_blur_reviewed_records'], 2)
            self.assertEqual(prepare(*args), result)
            output = args[2] / result['files'][0]['path']
            with output.open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                prepare(*args)

    def test_source_hash_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with (args[0] / 'decisions.parquet').open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                prepare(*args)
            self.assertFalse(args[2].exists())

    def test_unknown_overlay_asset_and_duplicate_input_block_publication(self):
        for quality, qwen, message in [([orow(9)], None, 'outside Qwen'),
                                       ([], [qrow(1), qrow(1)], 'duplicate Qwen')]:
            with tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory), quality, qwen)
                with self.assertRaisesRegex(ValueError, message):
                    prepare(*args)
                self.assertFalse(args[2].exists())


if __name__ == '__main__':
    unittest.main()
