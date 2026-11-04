"""Finalize against real queue receipts, without NPU or corpus data reads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from corpus_pipeline.cluster import monitor, queue
from corpus_pipeline.production.common import atomic_json, file_sha


class MonitorFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'
        self.tasks = [{'task_id': f'shard-{index}', 'parquet_path': f'/fixture/{index}.parquet',
                       'row_start': 0, 'row_end': 3, 'source': 'fixture'} for index in range(2)]
        queue.init_queue(self.root, self.tasks, lease_seconds=120)
        self.plan = {'record_count': 6, 'task_count': 2, 'snapshot_root': '/fixture/snapshot',
                     'snapshot_fingerprint': 'fixture-fingerprint'}
        atomic_json(self.root / 'plan.json', self.plan)

    def finish_one(self, *, summary_records=3, summary_images=2, change_summary=None):
        lease = queue.claim(self.root, 'fixture-worker')
        out = Path(lease['artifact_dir'])
        out.mkdir(parents=True)
        summary = {'policy': 'dcvlm.offline_frontpass.v1', 'task': copy.deepcopy(lease['task']),
                   'token': lease['token'], 'counts': {'records': summary_records,
                   'image_occurrences': summary_images}, 'categories': {'single_turn_i2t': summary_records},
                   'model_identity_sha256': 'a' * 64}
        if change_summary:
            change_summary(summary)
        atomic_json(out / 'summary.json', summary)
        receipt = queue.complete(self.root, lease['task_id'], lease['token'],
            artifacts=[{'path': str(out / 'summary.json'), 'sha256': file_sha(out / 'summary.json')}],
            processed_records=3, processed_images=2)
        return out, receipt

    def assert_not_published(self):
        self.assertFalse((self.root / 'frontpass_index.json').exists())
        self.assertFalse((self.root / 'FRONTPASS_COMPLETE.json').exists())

    def test_complete_real_receipts_and_relative_paths_publish_offline_only(self):
        for _ in range(2):
            _, receipt = self.finish_one()
            self.assertFalse(Path(receipt['artifacts'][0]['path']).is_absolute())
        progress = queue.status(self.root)
        result = monitor.finalize(self.root, progress)
        self.assertTrue(result['complete'])
        self.assertFalse(result['training_release'])
        self.assertEqual(result['counts']['records'], 6)
        self.assertEqual(result['counts']['image_occurrences'], 4)
        self.assertEqual(len(result['parts']), 2)
        complete = json.loads((self.root / 'FRONTPASS_COMPLETE.json').read_text())
        self.assertEqual(complete['index_sha256'], file_sha(self.root / 'frontpass_index.json'))

    def test_partial_queue_never_finalizes(self):
        self.finish_one()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            monitor.finalize(self.root, queue.status(self.root))
        self.assert_not_published()

    def test_completed_queue_requires_frozen_plan_record_count(self):
        for _ in range(2):
            self.finish_one()
        atomic_json(self.root / 'plan.json', {**self.plan, 'record_count': 7})
        with self.assertRaisesRegex(ValueError, 'record count'):
            monitor.finalize(self.root, queue.status(self.root))
        self.assert_not_published()

    def test_changed_committed_summary_rejected(self):
        out, _ = self.finish_one()
        self.finish_one()
        summary = json.loads((out / 'summary.json').read_text())
        summary['categories'] = {'unexpected': 3}
        atomic_json(out / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'summary has changed'):
            monitor.finalize(self.root, queue.status(self.root))
        self.assert_not_published()

    def test_summary_aggregate_record_count_must_match(self):
        self.finish_one(summary_records=2)
        self.finish_one()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            monitor.finalize(self.root, queue.status(self.root))
        self.assert_not_published()

    def test_per_task_record_counts_cannot_cancel_each_others_errors(self):
        self.finish_one(summary_records=2)
        self.finish_one(summary_records=4)
        with self.assertRaises(ValueError):
            monitor.finalize(self.root, queue.status(self.root))
        self.assert_not_published()

    def test_summary_task_and_lease_must_match_committed_receipt(self):
        self.finish_one()
        self.finish_one(change_summary=lambda summary: summary.update(
            task=copy.deepcopy(self.tasks[0]), token='wrong-lease-token'))
        with self.assertRaises(ValueError):
            monitor.finalize(self.root, queue.status(self.root))
        self.assert_not_published()

    def test_summary_image_count_must_match_committed_receipt(self):
        self.finish_one(summary_images=200)
        self.finish_one()
        with self.assertRaises(ValueError):
            monitor.finalize(self.root, queue.status(self.root))
        self.assert_not_published()


class MonitorETATests(unittest.TestCase):
    def estimate(self, *, records=100, images=100, record_rate=100, image_rate=10,
                 total_records=1000, total_images=1000, completed=False, deadline=120):
        return monitor.estimate_eta(
            {'processed_records': records, 'processed_images': images,
             'all_tasks_completed': completed, 'seconds_to_deadline': deadline},
            {'record_count': total_records}, records_per_s=record_rate,
            images_per_s=image_rate, image_basis={'total': total_images, 'scope': 'fixture'})

    def test_fast_text_only_progress_cannot_predict_full_completion(self):
        result = self.estimate(records=1_000_000, images=0, record_rate=1000,
                               image_rate=0, total_records=119_010_208,
                               total_images=108_026_384, deadline=86400)
        self.assertGreater(result['estimated_records_seconds_remaining'], 0)
        self.assertEqual(result['remaining_image_references'], 108_026_384)
        self.assertIsNone(result['estimated_image_seconds_remaining'])
        self.assertIsNone(result['estimated_seconds_remaining'])
        self.assertIsNone(result['deadline_projected_feasible'])
        self.assertIn('image_channel_has_no_positive_observed_rate', result['eta_unavailable_reasons'])

    def test_both_channels_progress_eta_covers_slower_channel(self):
        result = self.estimate(deadline=60)
        self.assertEqual(result['estimated_records_seconds_remaining'], 9)
        self.assertEqual(result['estimated_image_seconds_remaining'], 90)
        self.assertEqual(result['estimated_seconds_remaining'], 90)
        self.assertFalse(result['deadline_projected_feasible'])
        self.assertIn('conditional', result['eta_scope'])

    def test_record_channel_stall_also_prevents_full_eta(self):
        result = self.estimate(record_rate=0)
        self.assertIsNone(result['estimated_seconds_remaining'])
        self.assertIn('record_channel_has_no_positive_observed_rate', result['eta_unavailable_reasons'])

    def test_completed_image_channel_does_not_require_positive_image_rate(self):
        result = self.estimate(images=1000, image_rate=0, deadline=None)
        self.assertEqual(result['estimated_image_seconds_remaining'], 0)
        self.assertEqual(result['estimated_seconds_remaining'], 9)
        self.assertIsNone(result['deadline_projected_feasible'])

    def test_fully_completed_queue_has_zero_eta_without_rate_history(self):
        result = self.estimate(records=1000, images=1000, completed=True,
                               record_rate=0, image_rate=0)
        self.assertEqual(result['estimated_seconds_remaining'], 0)
        self.assertTrue(result['deadline_projected_feasible'])
        self.assertEqual(result['eta_reason'], 'all_queue_tasks_completed')

    def test_unknown_bounded_image_denominator_cannot_extrapolate(self):
        basis = monitor.image_reference_basis({'scope': 'bounded_real_validation'})
        self.assertIsNone(basis['total'])
        result = self.estimate(total_images=None)
        self.assertIsNone(result['estimated_seconds_remaining'])
        self.assertIn('image_reference_denominator_unknown', result['eta_unavailable_reasons'])
        completed = self.estimate(records=1000, completed=True, total_images=None,
                                  record_rate=0, image_rate=0)
        self.assertEqual(completed['estimated_seconds_remaining'], 0)

    def test_manifest_denominator_is_hash_bound_without_plan_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'phase': 'complete', 'schema_version': 'dcvlm.pretrain.v1',
                        'statistics': {'samples': 119_010_208, 'image_occurrences': 108_026_384}}
            atomic_json(root / 'manifest.json', manifest)
            plan = {'scope': 'full_native_dcvlm', 'snapshot_root': str(root),
                    'record_count': 119_010_208,
                    'snapshot_fingerprint': {'manifest.json': file_sha(root / 'manifest.json')}}
            before = copy.deepcopy(plan)
            basis = monitor.image_reference_basis(plan)
            self.assertEqual(basis['total'], 108_026_384)
            self.assertEqual(plan, before)
            manifest['statistics']['image_occurrences'] = 1
            atomic_json(root / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                monitor.image_reference_basis(plan)


if __name__ == '__main__':
    unittest.main()
