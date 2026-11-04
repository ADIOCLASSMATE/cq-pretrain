import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import sscd_index as s


class OriginalVectorCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.rows = []
        for i in range(4):
            path = self.root / f'original-{i}.f32'
            vector = np.zeros((1, 512), np.float32); vector[0, i] = 1
            vector.tofile(path)
            self.rows.append({'sscd_dimension': 512, 'sscd_dtype': 'float32', 'sscd_normalized': True,
                              'sscd_file': str(path), 'sscd_file_sha256': s.file_sha(path),
                              'sscd_file_size_bytes': path.stat().st_size, 'sscd_byte_offset': 0})

    def tearDown(self):
        self.tmp.cleanup()

    def test_retained_maps_are_reused_without_rehash_or_reopen(self):
        reader = s.OriginalVectors(self.root / 'evidence', max_open=4, verify_workers=4)
        with patch.object(s, 'file_sha', wraps=s.file_sha) as hashes, patch.object(s.np, 'memmap', wraps=s.np.memmap) as maps:
            np.testing.assert_array_equal(reader.read(self.rows), np.eye(512, dtype=np.float32)[:4])
            self.assertEqual(hashes.call_count, 4); self.assertEqual(maps.call_count, 4)
            np.testing.assert_array_equal(reader.read(self.rows[::-1]), np.eye(512, dtype=np.float32)[:4][::-1])
            self.assertEqual(hashes.call_count, 4); self.assertEqual(maps.call_count, 4)
        self.assertEqual(len(reader.maps), 4)

    def test_explicit_cache_bound_evicts_and_reopens_without_rehash(self):
        reader = s.OriginalVectors(self.root / 'evidence', max_open=2, verify_workers=2)
        with patch.object(s, 'file_sha', wraps=s.file_sha) as hashes, patch.object(s.np, 'memmap', wraps=s.np.memmap) as maps:
            reader.read(self.rows[:3]); self.assertEqual(len(reader.maps), 2)
            self.assertNotIn(self.rows[0]['sscd_file'], reader.maps)
            reader.read(self.rows[:1])
            self.assertEqual(hashes.call_count, 3); self.assertEqual(maps.call_count, 4)
            self.assertEqual(len(reader.maps), 2)

    def test_parallel_hash_validation_and_corruption_rejection(self):
        reader = s.OriginalVectors(self.root / 'evidence', max_open=4, verify_workers=4)
        barrier = threading.Barrier(4)
        original_hash = s.file_sha
        threads = set()
        def coordinated_hash(path):
            threads.add(threading.get_ident())
            barrier.wait(timeout=10)
            return original_hash(path)
        with patch.object(s, 'file_sha', side_effect=coordinated_hash):
            reader.read(self.rows)
        self.assertEqual(len(threads), 4)
        bad = Path(self.rows[1]['sscd_file'])
        content = bytearray(bad.read_bytes()); content[0] ^= 1; bad.write_bytes(content)
        fresh = s.OriginalVectors(self.root / 'fresh-evidence', max_open=4, verify_workers=4)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            fresh.read(self.rows)
        with self.assertRaisesRegex(ValueError, 'changed within catalog'):
            reader.read(self.rows[1:2])

    def test_stamp_rechecked_after_parallel_verification_before_mapping(self):
        reader = s.OriginalVectors(self.root / 'evidence', max_open=4, verify_workers=4)
        original_verify = reader._verify_file
        def mutate_after_verify(value):
            result = original_verify(value)
            if value[0] == self.rows[0]['sscd_file']:
                path = Path(value[0]); data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
            return result
        with patch.object(reader, '_verify_file', side_effect=mutate_after_verify):
            with self.assertRaisesRegex(ValueError, 'changed after verification'):
                reader.read(self.rows)
        self.assertEqual(len(reader.maps), 0)

    def test_descriptor_budget_caps_default_and_invalid_bounds_fail(self):
        with patch('resource.getrlimit', return_value=(1024, 1024)):
            reader = s.OriginalVectors(self.root / 'evidence')
        self.assertGreaterEqual(reader.max_open, 1)
        self.assertLessEqual(reader.max_open, (1024 - 256) // 2)
        for kwargs in ({'max_open': 0}, {'verify_workers': 0}):
            with self.assertRaises(ValueError):
                s.OriginalVectors(self.root / 'invalid', **kwargs)


def make_input(root, vectors, *, statuses=None, scope='bounded_validation'):
    root.mkdir(parents=True)
    original = root / 'original.f32'; np.asarray(vectors, dtype='<f4').tofile(original)
    sha = s.file_sha(original); buckets = []; total = len(vectors)
    rows = []
    for i in range(total):
        # The two identical halves live in DIFFERENT SHA buckets.
        asset_id = ('00' if i < total // 2 else 'ff') + f'{i:062x}'
        rows.append({'asset_id': asset_id, 'pixel_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
                     'pixel_sha256_conflict': False, 'global_status': statuses[i] if statuses else 'no_detector_alert',
                     'sscd_usable': True, 'sscd_file': str(original), 'sscd_byte_offset': i * s.VECTOR_BYTES,
                     'sscd_file_sha256': sha, 'sscd_file_size_bytes': original.stat().st_size,
                     'sscd_dimension': 512, 'sscd_dtype': 'float32', 'sscd_normalized': True,
                     'source': 'photo' if i < total // 2 else 'document', 'occurrence_count': 2, 'source_count': 1})
    schema = pa.Table.from_pylist(rows).schema
    for number in range(256):
        bucket = f'{number:02x}'; directory = root / 'buckets' / bucket; directory.mkdir(parents=True)
        path = directory / 'assets.parquet'
        pq.write_table(pa.Table.from_pylist([r for r in rows if r['asset_id'].startswith(bucket)], schema=schema), path)
        receipt = {'schema': 'corpus.asset_reconciliation.bucket.v1', 'bucket': bucket, 'artifacts': [s.artifact(path, root)]}
        s.atomic_json(directory / 'COMPLETE.json', receipt)
        buckets.append({'bucket': bucket, 'receipt_path': str((directory / 'COMPLETE.json').relative_to(root)),
                        'receipt_sha256': s.file_sha(directory / 'COMPLETE.json')})
    s.atomic_json(root / 'COMPLETE.json', {'complete': True, 'schema': 'corpus.asset_reconciliation.v1',
                                        'scope': scope, 'counts': {'assets': total}, 'buckets': buckets})
    return rows


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.input = self.root / 'input'; self.output = self.root / 'output'

    def tearDown(self):
        self.tmp.cleanup()

    def test_scope_policy_exact_values_and_resume(self):
        vectors = np.eye(512, dtype=np.float32)[:8]
        rows = make_input(self.input, vectors, statuses=['no_detector_alert', 'rejected_source',
                          'pending_integrity', 'ai_suspect_pending_api', *(['valid_protected'] * 4)])
        with self.assertRaisesRegex(ValueError, 'allow-bounded'):
            s.catalog(self.input, self.output)
        partial = s.catalog(self.input, self.output, block_rows=2, allow_bounded=True, max_new_blocks=1)
        self.assertFalse(partial['complete'])
        self.assertFalse((self.output / 'catalog' / 'COMPLETE.json').exists())
        result = s.catalog(self.input, self.output, block_rows=2, allow_bounded=True)
        self.assertEqual(result['vectors'], 6); self.assertEqual(result['skipped_assets'], 2)
        data = s.Catalog(self.output, allow_bounded=True)
        actual, asset_ids = data.fetch(np.arange(6), asset_ids=True)
        np.testing.assert_array_equal(actual, vectors[[0, 3, 4, 5, 6, 7]])
        self.assertEqual(asset_ids, [rows[i]['asset_id'] for i in [0, 3, 4, 5, 6, 7]])
        self.assertEqual(s.catalog(self.input, self.output, block_rows=2, allow_bounded=True), result)
        with self.assertRaisesRegex(ValueError, 'changed'):
            s.catalog(self.input, self.output, block_rows=3, allow_bounded=True)

    def test_original_corruption_and_packed_corruption_fail_closed(self):
        vectors = np.eye(512, dtype=np.float32)[:2]; make_input(self.input, vectors)
        path = self.input / 'original.f32'; contents = bytearray(path.read_bytes()); contents[0] ^= 1; path.write_bytes(contents)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            s.catalog(self.input, self.output, allow_bounded=True)
        contents[0] ^= 1; path.write_bytes(contents)
        s.catalog(self.input, self.output, allow_bounded=True)
        data = s.Catalog(self.output, allow_bounded=True)
        packed = self.output / data.blocks[0]['artifacts'][-1]['path']
        self.assertEqual(packed.name, 'vectors.f32')
        changed = bytearray(packed.read_bytes()); changed[10] ^= 1; packed.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            data.fetch([0])

    def test_bad_vectors_are_explicit_exceptions(self):
        vectors = np.eye(512, dtype=np.float32)[:4]; vectors[1] = 0; vectors[2, 0] = np.nan
        make_input(self.input, vectors)
        result = s.catalog(self.input, self.output, allow_bounded=True)
        self.assertEqual(result['vectors'], 2); self.assertEqual(result['invalid_vectors'], 2)
        reasons = []
        for path in (self.output / 'catalog' / 'blocks').glob('*/exceptions.parquet'):
            reasons.extend(pq.read_table(path).to_pylist())
        self.assertEqual(len(reasons), 2)

    def test_unsorted_or_duplicate_asset_ids_rejected(self):
        make_input(self.input, np.eye(512, dtype=np.float32)[:4])
        path = self.input / 'buckets' / '00' / 'assets.parquet'
        table = pq.read_table(path); pq.write_table(table.take(pa.array([1, 0])), path)
        receipt_path = path.parent / 'COMPLETE.json'; receipt = s.read_json(receipt_path)
        receipt['artifacts'] = [s.artifact(path, self.input)]; s.atomic_json(receipt_path, receipt)
        complete = s.read_json(self.input / 'COMPLETE.json'); complete['buckets'][0]['receipt_sha256'] = s.file_sha(receipt_path)
        s.atomic_json(self.input / 'COMPLETE.json', complete)
        with self.assertRaisesRegex(ValueError, 'sorted'):
            s.catalog(self.input, self.output, allow_bounded=True)


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.input = cls.root / 'input'; cls.output = cls.root / 'output'
        vectors = s.normalized(np.random.default_rng(7).normal(size=(400, 512)).astype(np.float32))
        make_input(cls.input, np.concatenate((vectors, vectors)))
        s.catalog(cls.input, cls.output, block_rows=200, allow_bounded=True)
        cls.config = s.IndexConfig(nlist=4, pq_m=8, pq_bits=4, coarse_hnsw=0, niter=2, threads=1)
        s.train(cls.output, config=cls.config, train_count=640, allow_bounded=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_build_merge_resume_and_global_cross_bucket_exact_rerank(self):
        partial = s.build(self.output, allow_bounded=True, max_new_shards=1)
        self.assertFalse(partial['complete'])
        full = s.build(self.output, allow_bounded=True)
        self.assertTrue(full['complete'])
        partial = s.merge(self.output, allow_bounded=True, checkpoint_shards=2, max_new_shards=1)
        self.assertFalse(partial['complete']); self.assertEqual(partial['count'], 200)
        merged = s.merge(self.output, allow_bounded=True, checkpoint_shards=2)
        self.assertEqual(merged['count'], 800)
        data = s.Catalog(self.output, allow_bounded=True)
        index, _ = s.global_index(self.output, data, 4, 4)
        found = s.candidates(data, index, [0, 401], 799)
        self.assertEqual(found[0][0][0], 400); self.assertEqual(found[1][0][0], 1)
        self.assertAlmostEqual(found[0][1][0], 1., places=6)
        qualification = s.calibrate(self.output, threshold=.99, query_count=8, top_k=3, candidate_k=799,
                                    nprobe=4, ef_search=4, min_recall=1., min_positive_recall=1., threads=1, allow_bounded=True)
        self.assertTrue(qualification['passed']); self.assertFalse(qualification['qualified_for_full_run'])
        self.assertEqual(qualification['threshold_positive_count'], 8)
        evidence = np.load(self.output / qualification['artifacts'][0]['path'])
        training_ids = np.fromfile(self.output / 'train' / 'sample_ids.i64', dtype=np.int64)
        self.assertFalse(set(evidence['query_ids']) & set(training_ids))
        latest = s.read_json(self.output / 'calibration' / 'LATEST.json')
        calibration = self.output / latest['receipt_path']
        partial = s.search(self.output, calibration, query_end=4, batch_size=2, allow_bounded=True, max_new_batches=1)
        self.assertFalse(partial['complete'])
        result = s.search(self.output, calibration, query_end=4, batch_size=2, allow_bounded=True)
        self.assertEqual(result['edge_rows'], 4); self.assertEqual(result['training_records_deleted'], 0)
        self.assertFalse(result['automatic_merge'])
        for entry in result['batches']:
            receipt = s.read_json(self.output / entry['path'])
            rows = pq.read_table(self.output / receipt['artifacts'][0]['path']).to_pylist()
            self.assertTrue(all(r['asset_id_a'].startswith('00') and r['asset_id_b'].startswith('ff') for r in rows))
            self.assertTrue(all(r['decision'] == 'requires_pair_review' and not r['automatic_merge'] for r in rows))

    def test_02_failed_recall_gate_and_bounded_scope(self):
        s.build(self.output, allow_bounded=True); s.merge(self.output, allow_bounded=True)
        with patch.object(s, 'candidates', return_value=[(np.array([], np.int64), np.array([], np.float32)) for _ in range(4)]):
            failed = s.calibrate(self.output, threshold=.99, query_count=4, top_k=2, candidate_k=4,
                                 nprobe=1, ef_search=4, threads=1, allow_bounded=True)
        self.assertFalse(failed['passed']); self.assertEqual(failed['threshold_positive_recall'], 0.)
        latest = s.read_json(self.output / 'calibration' / 'LATEST.json')
        with self.assertRaisesRegex(ValueError, 'did not pass'):
            s.search(self.output, self.output / latest['receipt_path'], query_end=4, allow_bounded=True)
        with self.assertRaisesRegex(ValueError, 'Bounded catalog'):
            s.Catalog(self.output)

    def test_03_reverse_only_neighbor_not_lost(self):
        s.build(self.output, allow_bounded=True); s.merge(self.output, allow_bounded=True)
        qualified = s.calibrate(self.output, threshold=.98, query_count=4, top_k=2, candidate_k=799,
                               nprobe=4, ef_search=4, threads=1, allow_bounded=True)
        latest = s.read_json(self.output / 'calibration' / 'LATEST.json')
        with patch.object(s, 'candidates', return_value=[(np.array([0]), np.array([1.], np.float32))]):
            result = s.search(self.output, self.output / latest['receipt_path'], query_start=400, query_end=401, allow_bounded=True)
        self.assertTrue(qualified['passed']); self.assertEqual(result['edge_rows'], 1)
        receipt = s.read_json(self.output / result['batches'][0]['path'])
        row = pq.read_table(self.output / receipt['artifacts'][0]['path']).to_pylist()[0]
        self.assertEqual((row['vector_id_a'], row['vector_id_b']), (0, 400))


class EstimateTests(unittest.TestCase):
    def test_scale_and_no_corpus_hnsw(self):
        result = s.estimate()
        self.assertEqual(result['packed_fp32_bytes'], 184_320_000_000)
        self.assertEqual(result['ivfpq_codes_and_ids_bytes'], 6_480_000_000)
        self.assertEqual(s.IndexConfig().factory, 'IVF16384_HNSW32,PQ64x8')
        with self.assertRaises(ValueError):
            s.IndexConfig(pq_m=15).validate()


if __name__ == '__main__':
    unittest.main()
