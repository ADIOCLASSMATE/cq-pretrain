import multiprocessing
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from . import sscd_index as s
from . import sscd_parallel_search as p
from .test_sscd_index import make_input


def hold_lock(path, entered, release):
    with p.claim(path) as acquired:
        entered.put(acquired); release.get(timeout=20)


class ParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_tmp = tempfile.TemporaryDirectory(); cls.base = Path(cls.base_tmp.name)
        rng = np.random.default_rng(124)
        half = s.normalized(rng.normal(size=(400, 512)).astype(np.float32))
        make_input(cls.base / 'input', np.concatenate([half, half]))
        cls.index = cls.base / 'index'
        s.catalog(cls.base / 'input', cls.index, block_rows=200, allow_bounded=True)
        s.train(cls.index, config=s.IndexConfig(nlist=4, pq_m=8, pq_bits=4, coarse_hnsw=0,
                                               niter=2, threads=1), train_count=640, allow_bounded=True)
        p.prepare_build(cls.index, threads=1, allow_bounded=True)
        p.build_worker(cls.index, worker_id='seed'); p.finalize_build(cls.index)
        s.merge(cls.index, allow_bounded=True)
        result = p.calibrate(cls.index, threshold=.75, query_count=16, top_k=10, candidate_k=799,
                             nprobe=4, ef_search=4, threads=1, exact_batch_size=400, allow_bounded=True)
        assert result['passed'] and not result['qualified_for_full_run']
        cls.calibration = s.read_json(cls.index / 'calibration' / 'LATEST.json')['receipt_path']

    @classmethod
    def tearDownClass(cls):
        cls.base_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name) / 'index'
        shutil.copytree(self.index, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare_search(self):
        return p.prepare_search(self.root, self.root / self.calibration, queries_per_unit=200,
                                batch_size=64, threads=1, allow_bounded=True)

    def test_build_compatible_merge_and_partial_resume_without_global_lock(self):
        shutil.rmtree(self.root / 'shards'); (self.root / 'parallel_build' / 'COMPLETE.json').unlink()
        with s.locked(self.root):
            first = p.build_worker(self.root, worker_id='first', max_units=1)
        self.assertEqual(first['new_units'], 1); self.assertEqual(first['completed_vectors'], 200)
        with self.assertRaises(FileNotFoundError): p.finalize_build(self.root)
        second = p.build_worker(self.root, worker_id='second')
        self.assertEqual(second['new_units'], 3)
        self.assertEqual(p.build_worker(self.root, worker_id='second')['new_units'], 0)
        complete = p.finalize_build(self.root); self.assertEqual(complete['count'], 800)
        shutil.rmtree(self.root / 'merged')
        self.assertEqual(s.merge(self.root, allow_bounded=True)['count'], 800)

    def test_cross_process_lock_and_preemption_release(self):
        ctx = multiprocessing.get_context('spawn'); entered = ctx.Queue(); release = ctx.Queue()
        lock = self.root / 'lock'; proc = ctx.Process(target=hold_lock, args=(lock, entered, release))
        proc.start()
        try:
            self.assertTrue(entered.get(timeout=10))
            with p.claim(lock) as acquired: self.assertFalse(acquired)
            proc.terminate(); proc.join(timeout=10); self.assertFalse(proc.is_alive())
            with p.claim(lock) as acquired: self.assertTrue(acquired)
        finally:
            if proc.is_alive(): proc.kill(); proc.join(timeout=10)

    def test_lockf_requires_thread_exclusion_without_closing_locked_fd(self):
        lock = self.root / 'thread-lock'; results = []
        def contender():
            with p.claim(lock) as acquired: results.append(acquired)
        with p.claim(lock) as acquired:
            self.assertTrue(acquired)
            thread = threading.Thread(target=contender); thread.start(); thread.join(timeout=5)
            self.assertEqual(results, [False])
            with p.claim(lock) as nested: self.assertFalse(nested)
        contender(); self.assertEqual(results, [False, True])

    def test_grouped_fetch_preserves_duplicates_order_and_empty(self):
        data = p.SharedCatalog(self.root, allow_bounded=True)
        original = s.Catalog(self.root, allow_bounded=True)
        ids = [799, 0, 420, 2, 799, 200, 199]
        a, names_a = data.fetch(ids, asset_ids=True); b, names_b = original.fetch(ids, asset_ids=True)
        np.testing.assert_array_equal(a, b); self.assertEqual(names_a, names_b)
        self.assertEqual(data.fetch([]).shape, (0, 512))

    def test_plan_binding_rejects_code_and_parameter_drift(self):
        with self.assertRaisesRegex(ValueError, 'changed'):
            p.prepare_build(self.root, threads=2, allow_bounded=True)
        path = p.plan_path(self.root, 'build'); value = s.read_json(path)
        value['identity']['code_sha256'] = '0' * 64; s.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, 'code changed'):
            p.build_worker(self.root, worker_id='no')

    def test_missing_or_corrupt_completed_build_is_not_reusable(self):
        plan = p.load_plan(self.root, 'build'); path = self.root / 'shards' / '000000.faiss'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            p.finalize_build(self.root)
        self.assertEqual(p.status(self.root, 'build')['complete_units'], 4)
        self.assertTrue(p.status(self.root, 'build')['counts_are_unverified_observation'])

    def test_shared_checksum_cache_and_live_drift(self):
        shutil.rmtree(self.root / 'parallel_verified')
        a = p.SharedCatalog(self.root, allow_bounded=True); b = p.SharedCatalog(self.root, allow_bounded=True)
        with patch.object(s, 'file_sha', wraps=s.file_sha) as hashes:
            a.block(0); b.block(0)
            self.assertEqual(hashes.call_count, 2)  # vectors and ids once, not once per worker
        path = self.root / next(x['path'] for x in a.blocks[0]['artifacts'] if x['path'].endswith('vectors.f32'))
        changed = bytearray(path.read_bytes()); changed[0] ^= 1; path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, 'changed after verification'): a.block(0)
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            p.SharedCatalog(self.root, allow_bounded=True).block(0)

    def test_calibration_hard_gates_and_evidence(self):
        with self.assertRaisesRegex(ValueError, 'may not be lowered'):
            p.calibrate(self.root, threshold=.75, min_recall=.94)
        path = self.root / self.calibration; value = s.read_json(path)
        self.assertEqual(value['exact_reference_vectors'], 800)
        self.assertGreater(value['threshold_positive_count'], 0)
        self.assertFalse(value['qualified_for_full_run'])
        for field, wrong, message in [('exact_reference_vectors', 799, 'entire reference'),
                                       ('threshold_positive_count', 0, 'metrics'), ('passed', False, 'did not pass')]:
            modified = dict(value); modified[field] = wrong; s.atomic_json(path, modified)
            with self.assertRaisesRegex(ValueError, message): self.prepare_search()
        s.atomic_json(path, value)
        evidence = self.root / value['artifacts'][0]['path']; evidence.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'): self.prepare_search()

    def test_search_full_global_across_buckets_streaming_resume(self):
        plan = self.prepare_search()
        # An orphan temporary file is uncommitted; a replacement worker must redo it.
        path = p.receipt_path(self.root, 'search', plan['units'][0]); path.parent.mkdir(parents=True)
        path.with_suffix('.tmp').write_text('interrupted')
        with s.locked(self.root):
            first = p.search_worker(self.root, worker_id='one', max_units=1)
        self.assertEqual(first['new_units'], 1)
        with self.assertRaises(FileNotFoundError): p.finalize_search(self.root)
        p.search_worker(self.root, worker_id='two'); result = p.finalize_search(self.root)
        self.assertEqual(result['query_count'], 800); self.assertFalse(result['automatic_merge'])
        self.assertEqual(result['training_records_deleted'], 0); self.assertTrue(result['candidate_edges_only'])
        self.assertFalse(result['near_duplicates_resolved'])
        self.assertEqual(p.search_worker(self.root, worker_id='three')['new_units'], 0)
        rows = []
        for path in (self.root / 'parallel_search' / 'units').glob('*.parquet'):
            rows.extend(pq.read_table(path).to_pylist())
            self.assertGreaterEqual(pq.ParquetFile(path).metadata.num_row_groups, 3)
        pairs = {(r['vector_id_a'], r['vector_id_b']) for r in rows}
        self.assertEqual(pairs, {(i, i + 400) for i in range(400)})
        self.assertTrue(all(r['asset_id_a'].startswith('00') and r['asset_id_b'].startswith('ff') for r in rows))
        self.assertTrue(all(r['exact_cosine'] > .99999 for r in rows))
        self.assertEqual(p.finalize_search(self.root), result)

    def test_search_final_revalidates_missing_and_corrupt_artifact(self):
        self.prepare_search(); p.search_worker(self.root, worker_id='all'); p.finalize_search(self.root)
        path = next((self.root / 'parallel_search' / 'units').glob('*.parquet'))
        original = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'): p.finalize_search(self.root)
        damaged = bytearray(original); damaged[0] ^= 1; path.write_bytes(damaged)
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'): p.finalize_search(self.root)

    def test_coverage_and_calibration_binding(self):
        plan = self.prepare_search(); malformed = {**plan, 'units': [dict(u) for u in plan['units']]}
        malformed['units'][1]['start'] -= 1
        with self.assertRaisesRegex(ValueError, 'gap/overlap'): p.check_ranges(malformed)
        with self.assertRaisesRegex(ValueError, 'changed'):
            p.prepare_search(self.root, self.root / self.calibration, queries_per_unit=400,
                             batch_size=64, threads=1, allow_bounded=True)
        path = self.root / self.calibration; value = s.read_json(path); value['additional'] = 'drift'; s.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, 'Bound calibration changed'):
            p.search_worker(self.root, worker_id='no')


if __name__ == '__main__': unittest.main()
