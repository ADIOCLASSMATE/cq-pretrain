import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from . import reconcile
from . import sscd_index, sscd_parallel_catalog
from .test_reconcile import Fixture, observation, sha
from corpus_pipeline.unified.screening import build_decisions
from corpus_pipeline.dedup import asset_release
from corpus_pipeline.dedup import decisions as old
from corpus_pipeline.unified.test_screening import history, qrow


def i2t(key, **changes):
    row = observation(key, status='i2t_dedup_only', cf384_prob=None, unifd_prob=None, **changes)
    provenance = json.loads(row['provenance_json'])
    provenance.update(processing_policy='corpus.task_policy.v1', task_route='i2t', image_loss_allowed=False)
    row['provenance_json'] = json.dumps(provenance)
    return row


class TaskRoutingReconcileTests(unittest.TestCase):
    def test_i2t_only_skips_api_and_shared_usage_retains_t2i_review_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; root.mkdir(); fixture = Fixture(root)
            fixture.part('understanding', [
                i2t('only-ai', origin='ai_generated'),
                i2t('only-small', original_width=64, original_height=64),
                i2t('shared-ai', origin='ai_generated'),
                i2t('shared-small', original_width=64, original_height=64),
                i2t('shared-review'),
                i2t('shared-detector-pending'),
                i2t('broken', pixel_sha256=None, sscd_byte_offset=None, error_json='{"error_stage":"decode"}'),
                observation('forged', status='i2t_dedup_only', cf384_prob=None, unifd_prob=None)])
            fixture.part('generation', [
                observation('shared-ai', status='rejected_known_ai_source', origin='ai_generated'),
                observation('shared-small', status='rejected_too_small'),
                observation('shared-review', status='ai_suspect_pending_api'),
                observation('shared-detector-pending', status='pending_inference_or_integrity_error',
                            cf384_prob=None, error_json='{"models":{"cf384":{"error":"unavailable"}}}')])
            fixture.finish(); output = Path(tmp)/'recon'
            complete = reconcile.run(root, output, threads=1, memory_limit='256MB')
            rows = {row['asset_id']: row for path in output.glob('buckets/*/assets.parquet')
                    for row in pq.ParquetFile(path).read().to_pylist()}
            self.assertEqual(complete['counts']['assets'], 8)
            self.assertEqual(complete['counts']['image_occurrences'], 12)
            for name in ('only-ai', 'only-small'):
                self.assertTrue(rows[sha(name)]['i2t_only'])
                self.assertEqual(rows[sha(name)]['global_status'], 'i2t_dedup_only')
                self.assertTrue(rows[sha(name)]['sscd_usable'])
            small = rows[sha('shared-small')]
            self.assertFalse(small['i2t_only'])
            self.assertNotEqual(small['global_status'], 'pending_integrity')
            self.assertNotIn('size_status_conflict', small['exception_reasons'])
            self.assertTrue(rows[sha('shared-ai')]['sscd_usable'])
            self.assertEqual(rows[sha('shared-ai')]['global_status'], 'rejected_source')
            for name in ('broken', 'forged'):
                self.assertEqual(rows[sha(name)]['global_status'], 'pending_integrity')
            shared = rows[sha('shared-detector-pending')]
            self.assertEqual(shared['global_status'], 'pending_integrity')
            self.assertIs(shared['i2t_integrity_pending'], False)
            self.assertIs(shared['sscd_usable'], True)
            self.assertIs(rows[sha('broken')]['i2t_integrity_pending'], True)
            self.assertEqual(complete['counts']['api_review_assets'], 1)
            pending = build_decisions(output, Path(tmp)/'screen')
            self.assertFalse(pending['complete'])
            selection = pending['api_selection']['files'][0]['path']
            self.assertEqual(pq.ParquetFile(selection).read()['asset_id'].to_pylist(), [sha('shared-review')])
            admitted = old.normalize_qwen(qrow(1)); admitted['asset_id'] = sha('shared-review')
            historical = history(Path(tmp)/'history', [admitted])
            build_decisions(output, Path(tmp)/'screen', history_decision_manifests=[historical])
            mapped = asset_release.build(output, Path(tmp)/'screen/COMPLETE.json', Path(tmp)/'assets',
                                         threads=1, memory='256MB')
            mapping = {row['asset_id']: row for path in mapped['files']
                       for row in pq.ParquetFile(path['path']).read().to_pylist()}
            self.assertIs(mapping[sha('shared-detector-pending')]['i2t_integrity_pending'], False)
            self.assertIs(mapping[sha('broken')]['i2t_integrity_pending'], True)
            serial = Path(tmp)/'serial'; parallel = Path(tmp)/'parallel'
            serial_result = sscd_index.catalog(output, serial, block_rows=2, allow_bounded=True)
            sscd_parallel_catalog.prepare(output, parallel, block_rows=2, allow_bounded=True)
            sscd_parallel_catalog.worker(parallel, workers=1, install_signals=False)
            parallel_result = sscd_parallel_catalog.finalize(parallel)
            self.assertEqual(serial_result['vectors'], 6)
            self.assertEqual(parallel_result['vectors'], 6)
            for directory in (serial, parallel):
                catalog = sscd_index.Catalog(directory, allow_bounded=True)
                import numpy as np
                _, ids = catalog.fetch(np.arange(6), asset_ids=True)
                self.assertIn(sha('shared-ai'), ids)
                self.assertIn(sha('shared-detector-pending'), ids)
                self.assertNotIn(sha('broken'), ids)


if __name__ == '__main__':
    unittest.main()
