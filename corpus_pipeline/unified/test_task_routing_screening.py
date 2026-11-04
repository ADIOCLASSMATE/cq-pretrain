from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from corpus_pipeline.dedup import decisions as old
from .screening import build_decisions
from .test_screening import history, qrow, write_json


def reconciliation(root):
    root.mkdir()
    rows = [{'asset_id': f'{1:064x}', 'i2t_only': True},
            {'asset_id': f'{2:064x}', 'i2t_only': False}]
    artifacts = []
    for name in ('assets', 'api_review'):
        path = root/(name + '.parquet')
        pq.write_table(pa.Table.from_pylist(rows), path)
        artifacts.append({**old.bound(path), 'path': path.name})
    write_json(root/'bucket.json', {'bucket': '00', 'artifacts': artifacts})
    write_json(root/'COMPLETE.json', {'complete': True, 'counts': {'api_review_assets': 2},
                                    'buckets': [{'receipt_path': 'bucket.json',
                                                 'receipt_sha256': old.file_sha(root/'bucket.json')}]})


class TaskRoutingScreeningTests(unittest.TestCase):
    def test_i2t_only_cannot_enter_api_selection_even_with_old_conflicting_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); reconciliation(root/'recon')
            first = history(root/'h1', [old.normalize_qwen(qrow(1))])
            second = history(root/'h2', [old.normalize_qwen(qrow(1, 'reject', 'likely_ai_generated'))])
            result = build_decisions(root/'recon', root/'screen', history_decision_manifests=[first, second])
            self.assertFalse(result['complete'])
            self.assertEqual(result['pending_infra'], 1)
            self.assertEqual(result['conflicts'], 0)
            self.assertEqual(result['review_candidates'], 1)
            self.assertEqual(result['i2t_only_screening_bypassed'], 1)
            path = result['api_selection']['files'][0]['path']
            self.assertEqual(pq.ParquetFile(path).read()['asset_id'].to_pylist(), [f'{2:064x}'])

    def test_i2t_only_history_exclusion_does_not_create_a_quality_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); reconciliation(root/'recon')
            historical = history(root/'history', [old.normalize_qwen(qrow(1, 'reject', 'likely_ai_generated')),
                                                   old.normalize_qwen(qrow(2))])
            result = build_decisions(root/'recon', root/'screen', history_decision_manifests=[historical])
            self.assertTrue(result['complete'])
            self.assertEqual(result['records'], 1)
            self.assertEqual(pq.ParquetFile(root/'screen/screening.parquet').read()['asset_id'].to_pylist(), [f'{2:064x}'])
            self.assertEqual(result['out_of_scope_assets'], 0)

    def test_i2t_only_still_preserves_confirmed_corruption_without_api(self):
        for historical_corruption in (False, True):
            with self.subTest(historical=historical_corruption), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); reconciliation(root/'recon')
                records = [old.normalize_qwen(qrow(2))]
                quality = []
                if historical_corruption:
                    corrupt = old.normalize_qwen(qrow(1, 'reject', 'likely_ai_generated'))
                    corrupt.update(screen_reason='confirmed_corrupt_image', ai_decision='not_inferred_from_quality')
                    records.append(corrupt)
                else:
                    quality = [{'scope': 'asset', 'asset_id': f'{1:064x}', 'reason': 'corrupt_image',
                                'confirmed': True, 'evidence_ids': ['verified-original-corruption']}]
                historical = history(root/'history', records)
                result = build_decisions(root/'recon', root/'screen', quality_decisions=quality,
                                         history_decision_manifests=[historical])
                self.assertTrue(result['complete'])
                self.assertEqual(result['review_candidates'], 1)
                rows = {row['asset_id']: row for row in pq.ParquetFile(root/'screen/screening.parquet').read().to_pylist()}
                self.assertEqual(rows[f'{1:064x}']['screen_reason'], 'confirmed_corrupt_image')


if __name__ == '__main__':
    unittest.main()
