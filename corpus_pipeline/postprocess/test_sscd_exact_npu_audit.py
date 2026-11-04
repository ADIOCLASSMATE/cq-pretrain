"""Independent boundary/receipt audit of the bounded exact-search backend."""
from pathlib import Path
import unittest

import numpy as np

from . import sscd_exact_npu as exact
from . import sscd_index as s
from . import test_sscd_exact_npu as fixture


class TriangleBoundaryTests(unittest.TestCase):
    def test_every_pair_once_for_unequal_blocks_and_partial_tiles(self):
        rng = np.random.default_rng(791)
        for compute_dtype in ('float32', 'float16'):
            for _ in range(12):
                sizes = rng.integers(1, 8, size=int(rng.integers(1, 6))).tolist()
                starts = np.r_[0, np.cumsum(sizes)[:-1]]
                blocks = [{'start_id': int(a), 'count': int(b)} for a, b in zip(starts, sizes)]
                ident = {'compute_dtype': compute_dtype, 'threshold': .75, 'candidate_threshold': .749,
                         'query_tile': int(rng.integers(1, 9)), 'database_tile': int(rng.integers(1, 9)),
                         'candidate_batch': int(rng.integers(1, 7))}
                plan = {'identity': ident, 'blocks': blocks}
                vectors = np.zeros((sum(sizes), 512), np.float32); vectors[:, 0] = 1
                class Store:
                    def __init__(self): self.plan = plan
                    def originals(self, number):
                        block = blocks[number]
                        return vectors[block['start_id']:block['start_id'] + block['count']]
                    input = originals
                store = Store(); backend = exact.NumpyBackend(plan); pairs = []
                for task in exact.pair_tasks(blocks):
                    for left, right, scores in exact.iter_pairs(store, backend, task):
                        self.assertTrue(np.all(scores == 1))
                        pairs.extend(zip(left.tolist(), right.tolist()))
                expected = {(i, j) for i in range(sum(sizes)) for j in range(i + 1, sum(sizes))}
                self.assertEqual(len(pairs), len(set(pairs)))
                self.assertEqual(set(pairs), expected, (compute_dtype, sizes, ident))


class ReceiptAuditTests(unittest.TestCase):
    setUp = fixture.ExactNpuTests.setUp
    tearDown = fixture.ExactNpuTests.tearDown
    prepare = fixture.ExactNpuTests.prepare
    qualify = fixture.ExactNpuTests.qualify
    all_pairs = fixture.ExactNpuTests.all_pairs

    def test_no_parameter_or_receipt_can_weaken_recall_gates(self):
        self.prepare()
        for kwargs in ({'min_recall': .949}, {'min_positive_recall': .989}):
            with self.assertRaisesRegex(ValueError, 'weakened'):
                exact.calibrate(self.output, self.reference, device='cpu', **kwargs)
        result = exact.calibrate(self.output, self.reference, device='cpu')
        result['recall_at_k'] = .949
        s.atomic_json(self.output / 'calibration/COMPLETE.json', result)
        with self.assertRaises(ValueError):
            exact.worker(self.output, worker_id='must-not-run', device='cpu')

    def test_orphan_task_after_hard_preemption_is_discarded(self):
        self.qualify(); plan = s.read_json(self.output / 'PLAN.json')
        task = next(exact.pair_tasks(plan['blocks']))
        target = exact.task_directory(self.output, task)
        stale = target.parent / ('.' + target.name + '.killed.tmp')
        stale.mkdir(parents=True); (stale / 'edges.parquet').write_bytes(b'partial-uncommitted')
        exact.worker(self.output, worker_id='recovery', device='cpu')
        self.assertFalse(stale.exists())
        self.assertEqual(self.all_pairs(), self.expected)
        self.assertTrue(exact.finalize(self.output)['complete'])

    def test_committed_task_cannot_claim_incorrect_pair_coverage(self):
        self.qualify(); exact.worker(self.output, worker_id='all', device='cpu')
        path = next(self.output.glob('tasks/*/*/COMPLETE.json'))
        receipt = s.read_json(path); receipt['pairs_covered'] += 1
        s.atomic_json(path, receipt)
        with self.assertRaisesRegex(ValueError, 'pair coverage'):
            exact.worker(self.output, worker_id='resume', device='cpu')
        with self.assertRaisesRegex(ValueError, 'pair coverage'):
            exact.finalize(self.output)

    def test_finalization_detects_missing_original_after_all_tasks_finish(self):
        self.qualify(); exact.worker(self.output, worker_id='all', device='cpu')
        original = next(self.index.glob('catalog/blocks/*/vectors.f32'))
        original.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            exact.finalize(self.output)


if __name__ == '__main__':
    unittest.main()
