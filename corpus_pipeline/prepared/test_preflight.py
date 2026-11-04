"""Behavioral checks for integrity, task preservation and non-admission."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image

from corpus_pipeline.prepared import preflight


FROZEN = {
    'review_thresholds': {'cf384': 0.9, 'unifd': 0.95},
    'priority_both_thresholds': {'cf384': 0.9, 'unifd': 0.95},
}


def score(digest, value=0.01):
    return {'image_sha256': digest,
            'scores': {name: {'status': 'ok', 'prob': value}
                       for name in ('cf384', 'cf224', 'unifd')}}


def fixture(folder, size=(64, 128)):
    stream = io.BytesIO()
    Image.new('RGB', size, (120, 45, 200)).save(stream, format='PNG')
    blob = stream.getvalue()
    tar_path = folder / 'original.tar'
    with tarfile.open(tar_path, 'w') as tar:
        info = tarfile.TarInfo('source.png')
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    with tarfile.open(tar_path) as tar:
        info = tar.getmember('source.png')
    sample = {
        'image_sha256': hashlib.sha256(blob).hexdigest(), 'source': 'fixture',
        'domain': 'photo', 'source_image_origin': 'unknown',
        'calibration_split': 'calibration', 'sample_id': 'record-1',
        'image_path': str(folder / 'intentionally_missing_copy.png'),
        'tar_path': str(tar_path), 'tar_member': 'source.png',
        'tar_offset': info.offset_data, 'tar_length': info.size,
        'decoded_width': size[0], 'decoded_height': size[1],
    }
    association = {
        'image_sha256': sample['image_sha256'], 'source': 'fixture',
        'sample_id': 'record-1', 'image_index': 0, 'source_image_count': 1,
        'format': 'conversation', 'segments': [
            {'kind': 'image', 'image_index': 0, 'role': 'user', 'turn': 0},
            {'kind': 'text', 'text': 'Which color?', 'role': 'user', 'turn': 0},
            {'kind': 'text', 'text': 'Purple.', 'role': 'assistant', 'turn': 1},
        ],
    }
    return sample, association


class PreflightTests(unittest.TestCase):
    def test_reads_original_tar_and_never_accepts_low_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample, _ = fixture(Path(temporary))
            row = preflight.probe_asset(sample, score(sample['image_sha256']), FROZEN)
            self.assertTrue(row['hash_verified'])
            self.assertEqual(row['width'], 64)
            self.assertEqual(row['detector_route'], 'no_model_alert')
            self.assertIsNone(row['training_eligibility'])
            self.assertFalse(row['published_training'])
            self.assertFalse(row['automatic_deletion'])

    def test_hash_mismatch_blocks_score_reuse_and_dimension_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample, _ = fixture(Path(temporary))
            sample['image_sha256'] = '0' * 64
            row = preflight.probe_asset(sample, score(sample['image_sha256']), FROZEN)
            self.assertEqual(row['integrity_status'], 'hash_mismatch')
            self.assertEqual(row['detector_route'], 'input_integrity_error')
            self.assertEqual(row['decode_status'], 'not_attempted')
            self.assertTrue(all(v is None for v in row['short_edge_rejected_by_candidate'].values()))

    def test_threshold_is_strictly_below_not_equal(self):
        values = preflight.small_image_flags(64, 400, (32, 64, 128))
        self.assertEqual(values, {'32': False, '64': False, '128': True})

    def test_multimage_record_is_incomplete_and_direction_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample, association = fixture(Path(temporary))
            association['source_image_count'] = 3
            asset = preflight.probe_asset(sample, score(sample['image_sha256']), FROZEN)
            row = preflight.record_association(association, asset)
            self.assertEqual(row['native_task_type'], 'single_i2t')
            self.assertEqual(row['task_views_preserving_native_direction'], ['single_i2t'])
            self.assertEqual(row['segments'], association['segments'])
            self.assertFalse(row['all_record_images_observed'])
            self.assertFalse(row['all_record_images_integrity_checked'])
            self.assertFalse(row['record_admission_complete'])
            self.assertEqual(row['remaining_image_count'], 2)

    def test_truncated_range_is_not_a_small_image_or_model_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample, _ = fixture(Path(temporary))
            sample['tar_offset'] = 100_000
            row = preflight.probe_asset(sample, score(sample['image_sha256']), FROZEN)
            self.assertEqual(row['integrity_status'], 'source_read_error')
            self.assertIsNone(row['short_edge_rejected_by_candidate']['64'])
            self.assertEqual(row['detector_route'], 'input_integrity_error')

    def test_missing_model_score_is_inference_error_not_non_ai(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample, _ = fixture(Path(temporary))
            row = preflight.probe_asset(sample, None, FROZEN)
            self.assertEqual(row['integrity_status'], 'ok')
            self.assertEqual(row['detector_route'], 'inference_error')
            self.assertIsNone(row['training_eligibility'])

    def test_run_retains_same_image_multiple_supervision_and_no_api_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            sample, association = fixture(folder)
            second = {**association, 'sample_id': 'record-2', 'segments': [
                {'kind': 'image', 'image_index': 0, 'role': 'user', 'turn': 0},
                {'kind': 'text', 'text': 'What aspect ratio?', 'role': 'user', 'turn': 0},
                {'kind': 'text', 'text': 'One to two.', 'role': 'assistant', 'turn': 1},
            ]}
            files = {
                'samples': [sample], 'associations': [association, second],
                'scores': [score(sample['image_sha256'])],
                'api': [{'image_sha256': sample['image_sha256'],
                         'final_disposition': 'likely_non_ai_candidate'}],
            }
            for name, values in files.items():
                (folder / f'{name}.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in values))
            (folder / 'thresholds.json').write_text(json.dumps(FROZEN))
            arguments = preflight.parser().parse_args([
                '--samples', str(folder / 'samples.jsonl'),
                '--associations', str(folder / 'associations.jsonl'),
                '--scores', str(folder / 'scores.jsonl'),
                '--thresholds', str(folder / 'thresholds.json'),
                '--api-decisions', str(folder / 'api.jsonl'),
                '--output', str(folder / 'output'), '--workers', '2',
            ])
            summary = preflight.run(arguments)
            self.assertEqual(summary['unique_sampled_assets'], 1)
            self.assertEqual(summary['record_associations'], 2)
            self.assertEqual(summary['additional_supervision_associations_preserved'], 1)
            records = list(preflight.read_jsonl(folder / 'output/record_index.jsonl'))
            self.assertEqual({r['sample_id'] for r in records}, {'record-1', 'record-2'})
            self.assertNotEqual(records[0]['segments'], records[1]['segments'])
            assets = list(preflight.read_jsonl(folder / 'output/asset_index.jsonl'))
            self.assertFalse(assets[0]['prior_api_evidence_reference']['used_as_training_approval'])
            complete = json.loads((folder / 'output/COMPLETE.json').read_text())
            self.assertTrue(complete['sample_only'])
            self.assertFalse(complete['published_training'])
            self.assertIsNone(complete['training_eligibility'])
            with self.assertRaises(FileExistsError):
                preflight.run(arguments)


if __name__ == '__main__':
    unittest.main()
