import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import sscd_index as s
from . import sscd_parallel_catalog as p
from .test_sscd_index import make_input


class ParallelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.input = self.root / 'input'; self.old = self.root / 'old'; self.output = self.root / 'parallel'
        self.vectors = np.eye(512, dtype=np.float32)[:16]
        self.rows = make_input(self.input, self.vectors)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, **kwargs):
        return p.prepare(self.input, self.output, block_rows=3, allow_bounded=True, **kwargs)

    def test_concurrent_work_stealing_adopts_immutable_old_and_matches_serial(self):
        s.catalog(self.input, self.old, block_rows=3, allow_bounded=True, max_new_blocks=2)
        before = {str(path.relative_to(self.old)): (s.file_sha(path), s.stamp(path)) for path in self.old.rglob('*') if path.is_file()}
        plan = self.prepare(adopt=self.old)
        self.assertEqual(plan['adopted_task_count'], 2)
        self.assertEqual(plan['task_count'], 6)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(p.worker, self.output, workers=2, install_signals=False) for _ in range(2)]
            results = [f.result() for f in futures]
        self.assertEqual(sum(r['completed_tasks'] for r in results), 6)
        self.assertTrue(p.status(self.output)['ready_to_finalize'])
        completed = p.finalize(self.output)
        self.assertEqual(completed['vectors'], 16)
        self.assertFalse(p.status(self.output)['ready_to_finalize'])
        catalog = s.Catalog(self.output, allow_bounded=True)
        values, ids = catalog.fetch(np.arange(16), asset_ids=True)
        np.testing.assert_array_equal(values, self.vectors)
        self.assertEqual(ids, [r['asset_id'] for r in self.rows])
        self.assertEqual(before, {str(path.relative_to(self.old)): (s.file_sha(path), s.stamp(path))
                                 for path in self.old.rglob('*') if path.is_file()})
        for block in catalog.blocks:
            mapping = next(a for a in block['artifacts'] if a['path'].endswith('/mapping.parquet'))
            table = pq.read_table(self.output / mapping['path'])
            np.testing.assert_array_equal(table['vector_id'].to_numpy(), np.arange(block['start_id'], block['start_id'] + block['count']))
        self.assertEqual(p.finalize(self.output), completed)
        self.assertEqual(p.worker(self.output, workers=2, install_signals=False)['completed_tasks'], 0)

    def test_partial_resume_busy_lock_and_no_early_finalization(self):
        plan = self.prepare()
        first = plan['tasks'][0]['task_id']
        with p.file_lock(self.output / 'catalog_parallel/task_locks' / (first + '.lock')):
            result = p.worker(self.output, workers=2, max_tasks=1, install_signals=False)
            self.assertEqual(result['completed_tasks'], 1)
            self.assertFalse((self.output / 'catalog_parallel/tasks' / first / 'COMPLETE.json').exists())
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            p.finalize(self.output)
        p.worker(self.output, workers=2, install_signals=False)
        self.assertTrue(p.finalize(self.output)['complete'])
        self.assertEqual(p.prepare(self.input, self.output, block_rows=3, allow_bounded=True), plan)
        with self.assertRaisesRegex(ValueError, 'changed'):
                p.prepare(self.input, self.output, block_rows=4, allow_bounded=True)

    def test_posix_lock_blocks_other_process_and_releases(self):
        path = self.root / 'shared.lock'
        script = ('import sys; from corpus_pipeline.postprocess.sscd_parallel_catalog import file_lock; '
                  'ctx=file_lock(sys.argv[1],blocking=False); acquired=ctx.__enter__(); '
                  'print(acquired); ctx.__exit__(None,None,None)')
        with p.file_lock(path):
            out = subprocess.check_output([sys.executable, '-c', script, str(path)], text=True)
            self.assertEqual(out.strip(), 'False')
        out = subprocess.check_output([sys.executable, '-c', script, str(path)], text=True)
        self.assertEqual(out.strip(), 'True')

    def test_shared_legacy_sha_receipt_avoids_rehash_and_changed_source_rejected(self):
        s.catalog(self.input, self.old, block_rows=3, allow_bounded=True, max_new_blocks=1)
        plan = self.prepare(adopt=self.old)
        task = next(t for t in plan['tasks'] if t['adoption'] is None)
        original = self.input / 'original.f32'; original_sha = s.file_sha
        def no_vector_rehash(path):
            if Path(path) == original:
                raise AssertionError('Valid legacy verification should be reused')
            return original_sha(path)
        with patch.object(s, 'file_sha', side_effect=no_vector_rehash):
            p.worker(self.output, workers=2, task_id=task['task_id'], install_signals=False)
        evidence = [s.read_json(path) for path in (self.output / 'catalog_parallel/verification_evidence').glob('*.json')]
        self.assertTrue(any(e['kind'] == 'reused_exact_legacy_verification' for e in evidence))
        data = bytearray(original.read_bytes()); data[0] ^= 1; original.write_bytes(data)
        untouched = next(t for t in plan['tasks'] if t['adoption'] is None and t != task)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            p.worker(self.output, workers=2, task_id=untouched['task_id'], install_signals=False)
        self.assertFalse((self.output / 'catalog_parallel/tasks' / untouched['task_id'] / 'COMPLETE.json').exists())

    def test_pread_and_mmap_exact_results_offset_reordering_and_group_ranges(self):
        plan = self.prepare(); task = plan['tasks'][0]
        table, _ = p.read_slice(task)
        table = pa.concat_tables([table.take(pa.array([2, 0, 1])), table.slice(0, 1)])
        for method in ['pread', 'mmap']:
            reader = p.ParallelVectors(self.output / ('reader-' + method), workers=2, method=method)
            try:
                got = reader.read(table)
                np.testing.assert_array_equal(got, self.vectors[[2, 0, 1, 0]])
                if method == 'pread':
                    self.assertEqual(reader.stats['pread_calls'], 1)
                    self.assertEqual(reader.stats['pread_bytes'], 3 * s.VECTOR_BYTES)
            finally:
                reader.close()

    def test_corrupt_adopted_block_is_not_published(self):
        s.catalog(self.input, self.old, block_rows=3, allow_bounded=True, max_new_blocks=1)
        plan = self.prepare(adopt=self.old); task = next(t for t in plan['tasks'] if t['adoption'])
        original = self.old / 'catalog/blocks' / task['task_id'] / 'vectors.f32'
        data = bytearray(original.read_bytes()); data[0] ^= 1; original.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            p.worker(self.output, workers=2, task_id=task['task_id'], install_signals=False)
        self.assertFalse((self.output / 'catalog_parallel/tasks' / task['task_id'] / 'COMPLETE.json').exists())

    def test_frozen_parquet_change_rejected_before_work(self):
        plan = self.prepare(); task = plan['tasks'][0]; path = Path(task['source']['path'])
        data = bytearray(path.read_bytes()); data[-20] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Frozen source parquet changed'):
            p.worker(self.output, workers=2, task_id=task['task_id'], install_signals=False)

    def test_invalid_vectors_accounted_and_source_only_projection(self):
        original = self.input / 'original.f32'
        vectors = self.vectors.copy(); vectors[0] = 0; vectors.tofile(original)
        # Rebuild fixture receipts to bind the deliberately invalid input.
        import shutil
        shutil.rmtree(self.input)
        make_input(self.input, vectors)
        self.prepare(); p.worker(self.output, workers=2, install_signals=False)
        result = p.finalize(self.output)
        self.assertEqual((result['vectors'], result['invalid_vectors'], result['skipped_assets']), (15, 1, 0))
        values = s.Catalog(self.output, allow_bounded=True).fetch(np.arange(15))
        np.testing.assert_array_equal(values, self.vectors[1:])


if __name__ == '__main__':
    unittest.main()
