import concurrent.futures
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import sscd_exact_cache as c
from . import sscd_index as s


def fixture(root, vectors, block_rows=11):
    blocks = []
    for number, start in enumerate(range(0, len(vectors), block_rows)):
        path = root / 'catalog/blocks' / str(number)
        path.mkdir(parents=True)
        values = vectors[start:start + block_rows]
        values.astype('<f4').tofile(path / 'vectors.f32')
        receipt = {'complete': True, 'start_id': start, 'count': len(values),
                   'artifacts': [s.artifact(path / 'vectors.f32', root)]}
        s.atomic_json(path / 'COMPLETE.json', receipt)
        blocks.append({'start_id': start, 'count': len(values),
                       'receipt_path': str((path / 'COMPLETE.json').relative_to(root)),
                       'receipt_sha256': s.file_sha(path / 'COMPLETE.json')})
    s.atomic_json(root / 'catalog/COMPLETE.json', {'complete': True, 'stage': 'catalog',
                  'scope': 'bounded_validation', 'vectors': len(vectors), 'blocks': blocks})


class ExactCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.source, self.output = self.root / 'source', self.root / 'cache'
        self.original = np.random.default_rng(124).normal(size=(37, 512)).astype(np.float32)
        fixture(self.source, self.original)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self):
        return c.prepare(self.source, self.output, chunk_rows=7, allow_bounded=True)

    def test_full_coverage_normalized_bytes_bounds_and_resume(self):
        self.prepare()
        with self.assertRaises(FileNotFoundError):
            c.finalize(self.output)
        a = c.worker(self.output, max_blocks=1, threads=1)
        self.assertEqual(a['new_blocks'], 1)
        partial = self.output / 'blocks/.000001.partial'; partial.mkdir()
        (partial / 'vectors.f16').write_bytes(b'uncommitted')
        b = c.worker(self.output, threads=1)
        self.assertEqual(b['new_blocks'], 3)
        self.assertEqual(c.worker(self.output, threads=1)['new_blocks'], 0)
        result = c.finalize(self.output)
        self.assertEqual(result, c.finalize(self.output))
        self.assertEqual(result['vectors'], 37)
        self.assertEqual([x['start_id'] for x in result['blocks']], [0, 11, 22, 33])
        cached = np.concatenate([np.fromfile(self.output / x['path'], dtype='<f2').reshape(-1, 512)
                                 for x in result['blocks']])
        original = s.normalized(self.original)
        np.testing.assert_array_equal(cached, original.astype('<f2'))
        a, b = original.astype(np.float64), cached.astype(np.float64)
        self.assertLessEqual(float(np.linalg.norm(a-b, axis=1).max()), result['max_quantization_l2_error'])
        self.assertLessEqual(float(np.linalg.norm(a, axis=1).max()), result['max_original_norm'])
        self.assertLessEqual(float(np.linalg.norm(b, axis=1).max()), result['max_half_norm'])
        self.assertLessEqual(float(np.abs(a @ a.T-b @ b.T).max()), result['quantization_dot_error_bound'])
        self.assertTrue(result['bounds_cover_all_vectors'])

    def test_original_hash_catches_same_stamp_corruption(self):
        plan = self.prepare(); item = plan['blocks'][0]
        path = self.source / item['source_artifact']['path']; before = path.stat()
        data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(s.stamp(path), item['source_stamp'])
        with self.assertRaisesRegex(ValueError, 'Original vector SHA mismatch'):
            c.worker(self.output, threads=1)
        self.assertFalse((self.output / 'blocks/000000/COMPLETE.json').exists())

    def test_source_stamp_drift_rejected(self):
        plan = self.prepare(); path = self.source / plan['blocks'][0]['source_artifact']['path']
        path.write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'stamp changed'):
            c.worker(self.output, threads=1)

    def test_finalization_rehashes_and_rejects_corrupt_cached_file(self):
        self.prepare(); c.worker(self.output, threads=1); c.finalize(self.output)
        path = self.output / 'blocks/000000/vectors.f16'; before = path.stat()
        data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            c.finalize(self.output)

    def test_runtime_plan_binding_and_coverage(self):
        plan = self.prepare()
        with self.assertRaisesRegex(ValueError, 'changed'):
            c.prepare(self.source, self.output, chunk_rows=8, allow_bounded=True)
        bad = s.read_json(self.output / 'PLAN.json'); bad['blocks'][1]['start_id'] += 1
        s.atomic_json(self.output / 'PLAN.json', bad)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            c.worker(self.output, threads=1)
        plan['identity']['runtime']['code_sha256'] = '0' * 64
        s.atomic_json(self.output / 'PLAN.json', plan)
        with self.assertRaisesRegex(ValueError, 'runtime changed'):
            c.worker(self.output, threads=1)

    def test_nonfinite_and_zero_inputs_never_publish(self):
        for name, value in [('nan', np.nan), ('zero', 0.)]:
            with self.subTest(name=name):
                source, output = self.root / name, self.root / (name + '_cache')
                fixture(source, np.full((1, 512), value, np.float32))
                c.prepare(source, output, allow_bounded=True)
                with self.assertRaisesRegex(ValueError, 'Invalid vector'):
                    c.worker(output, threads=1)
                self.assertFalse((output / 'blocks/000000/COMPLETE.json').exists())

    def test_concurrent_workers_claim_disjoint_blocks(self):
        self.prepare()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: c.worker(self.output, threads=1), range(2)))
        self.assertEqual(sum(x['new_blocks'] for x in results), 4)
        self.assertEqual(c.finalize(self.output)['vectors'], 37)


if __name__ == '__main__':
    unittest.main()
