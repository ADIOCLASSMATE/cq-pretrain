from pathlib import Path
import hashlib
import io
import tempfile
import unittest

import numpy as np
from PIL import Image

from corpus_pipeline.dedup.sscd import Catalog, load_image_asset


class DedupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'catalog.sqlite'
        self.identity = {'dimensions': 4, 'model': 'fixture'}
        self.catalog = Catalog(self.path, self.identity, threads=1)

    def tearDown(self):
        self.catalog.close()
        self.tmp.cleanup()

    def meta(self, identifier, sha=None, pixel=None):
        return {'asset_id': identifier, 'sha256': sha or identifier,
                'pixel_sha256': pixel or identifier, 'source_refs': [identifier]}

    def test_same_bytes_and_pixels_keep_all_observations(self):
        v = np.array([1, 0, 0, 0], np.float32)
        result = self.catalog.integrate([(self.meta('a'), v), (self.meta('b', sha='a'), v),
                                         (self.meta('c', pixel='a'), v)])
        self.assertEqual([r['canonical_asset_id'] for r in result['assets']], ['a', 'a', 'a'])
        self.assertEqual([r['exact_duplicate_kind'] for r in result['assets']], [None, 'byte', 'pixel'])
        self.assertEqual(self.catalog.db.execute('SELECT count(*) FROM assets').fetchone()[0], 1)
        self.assertEqual(self.catalog.db.execute('SELECT count(*) FROM observations').fetchone()[0], 3)
        self.assertTrue(all(r['supervision_records_dropped'] == 0 for r in result['assets']))

    def test_near_match_is_not_automatic_merge(self):
        result = self.catalog.integrate([(self.meta('a'), [1, 0, 0, 0]),
                                         (self.meta('different_chart_text'), [1, .001, 0, 0])])
        self.assertEqual(result['assets'][1]['canonical_asset_id'], 'different_chart_text')
        self.assertEqual(len(result['relations']), 1)
        self.assertFalse(result['relations'][0]['automatic_merge'])

    def test_second_worker_rechecks_new_committed_assets(self):
        other = Catalog(self.path, self.identity, threads=1)
        try:
            self.catalog.integrate([(self.meta('a'), [1, 0, 0, 0])])
            result = other.integrate([(self.meta('b'), [1, .001, 0, 0])])
            self.assertEqual(result['relations'][0]['other_asset_id'], 'a')
        finally:
            other.close()

    def test_mismatched_model_cannot_reuse_vectors(self):
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            Catalog(self.path, {**self.identity, 'model': 'different'}, threads=1)

    def test_rollback_does_not_leave_phantom_ann_rows(self):
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.catalog.integrate([(self.meta('a'), [1, 0, 0, 0]),
                                    (self.meta('a', sha='different'), [1, 0, 0, 0])])
        self.assertEqual(self.catalog.index.ntotal, 0)
        self.assertEqual(self.catalog.last_rowid, 0)
        result = self.catalog.integrate([(self.meta('b'), [1, 0, 0, 0])])
        self.assertFalse(result['relations'])

    def image_asset(self, size=(128, 128), mode='RGB', color='red'):
        data = io.BytesIO()
        Image.new(mode, size, color).save(data, format='PNG')
        blob = data.getvalue()
        return {'asset_id': 'test', 'sha256': hashlib.sha256(blob).hexdigest(),
                'blob_loader': lambda: blob}

    def test_128_is_hard_original_size_gate(self):
        self.assertEqual(load_image_asset(self.image_asset())[0]['width'], 128)
        with self.assertRaisesRegex(ValueError, 'below 128'):
            load_image_asset(self.image_asset((127, 4096)))

    def test_hash_and_alpha_are_not_ignored(self):
        bad = self.image_asset()
        bad['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            load_image_asset(bad)
        a, _ = load_image_asset(self.image_asset(mode='RGBA', color=(255, 0, 0, 255)))
        b, _ = load_image_asset(self.image_asset(mode='RGBA', color=(255, 0, 0, 0)))
        self.assertNotEqual(a['pixel_sha256'], b['pixel_sha256'])


if __name__ == '__main__':
    unittest.main()
