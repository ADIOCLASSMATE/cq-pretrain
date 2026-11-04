import concurrent.futures
from collections import namedtuple
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from . import sscd_exact_cache as cache
from . import sscd_exact_npu as exact
from . import sscd_index as s
from .test_sscd_index import make_input


class ExactNpuTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.index, self.cache, self.output = [self.root / x for x in ('index', 'cache', 'output')]
        base = s.normalized(np.random.default_rng(919).normal(size=(4,512)).astype(np.float32))
        self.raw = np.concatenate([base, base, base]) * np.float32(.9999)
        make_input(self.root / 'input', self.raw)
        s.catalog(self.root/'input', self.index, block_rows=3, allow_bounded=True)
        cache.prepare(self.index, self.cache, chunk_rows=2, allow_bounded=True)
        cache.worker(self.cache, threads=1); self.bounds = cache.finalize(self.cache)
        self.reference = self.index / 'calibration/test/evidence.npz'
        self.reference.parent.mkdir(parents=True)
        data = s.Catalog(self.index, allow_bounded=True)
        values = s.normalized(data.fetch(np.arange(data.count)))
        scores = (values.astype(np.float64) @ values.astype(np.float64).T).astype(np.float32)
        np.fill_diagonal(scores, -np.inf)
        top = np.asarray([np.lexsort((np.arange(len(scores)), -row))[:3] for row in scores])
        top_scores = np.take_along_axis(scores, top, axis=1)
        np.savez(self.reference, query_ids=np.arange(len(scores)), exact_ids=top, exact_scores=top_scores,
                 threshold_positive_count=(scores >= .75).sum(1))
        s.atomic_json(self.reference.parent/'COMPLETE.json', {
            'identity': {'catalog_sha256': data.sha256, 'parameters': {'threshold': .75}},
            'exact_reference_vectors': data.count, 'artifacts': [s.artifact(self.reference, self.index)]})
        self.expected = {(i,j) for i in range(len(scores)) for j in range(i+1,len(scores)) if scores[i,j] >= .75}

    def tearDown(self): self.tmp.cleanup()

    def prepare(self, **kwargs):
        return exact.prepare(self.index, self.cache, self.output, query_tile=2, database_tile=2,
                             candidate_batch=2, allow_bounded=True, min_free_gib=0, **kwargs)

    def qualify(self, **kwargs):
        self.prepare(**kwargs)
        result = exact.calibrate(self.output, self.reference, device='cpu')
        self.assertTrue(result['passed']); self.assertFalse(result['qualified_for_full_run'])
        return result

    def all_pairs(self):
        found = []
        for path in self.output.glob('tasks/*/*/edges.parquet'):
            rows = pq.read_table(path).to_pylist()
            for row in rows:
                self.assertGreaterEqual(row['exact_cosine'], .75)
                self.assertFalse(row['automatic_merge'])
                self.assertEqual(row['decision'], 'requires_pair_review')
                found.append((row['vector_id_a'], row['vector_id_b']))
        self.assertEqual(len(found), len(set(found)))
        return set(found)

    def test_full_upper_triangle_cross_blocks_same_blocks_and_resume(self):
        self.qualify()
        first = exact.worker(self.output, worker_id='first', device='cpu', max_tasks=2)
        self.assertEqual(first['new_tasks'],2)
        with self.assertRaises(FileNotFoundError): exact.finalize(self.output)
        exact.worker(self.output, worker_id='second', device='cpu')
        self.assertEqual(self.all_pairs(), self.expected)
        result = exact.finalize(self.output)
        self.assertEqual(result['completed_pairs'], len(self.raw)*(len(self.raw)-1)//2)
        self.assertTrue(result['all_pairs_covered']); self.assertFalse(result['near_duplicates_resolved'])
        self.assertEqual(result, exact.finalize(self.output))
        self.assertEqual(exact.worker(self.output, worker_id='again', device='cpu')['new_tasks'],0)

    def test_no_worker_before_reference_gate_or_incomplete_gate(self):
        self.prepare()
        with self.assertRaises(FileNotFoundError): exact.worker(self.output, worker_id='no', device='cpu')
        result = exact.calibrate(self.output,self.reference,device='cpu')
        result['passed']=False; s.atomic_json(self.output/'calibration/COMPLETE.json',result)
        with self.assertRaisesRegex(ValueError,'not qualified'): exact.worker(self.output,worker_id='no',device='cpu')

    def test_interrupted_output_never_publishes_and_retries(self):
        self.qualify()
        with patch.object(exact,'edge_table',side_effect=RuntimeError('preempted')):
            with self.assertRaisesRegex(RuntimeError,'preempted'):
                exact.worker(self.output,worker_id='preempted',device='cpu')
        exact.worker(self.output,worker_id='retry',device='cpu')
        self.assertEqual(self.all_pairs(),self.expected)
        self.assertTrue(exact.finalize(self.output)['complete'])

    def test_two_workers_never_commit_duplicate_pairs(self):
        self.qualify()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(exact.worker,self.output,worker_id=str(i),device='cpu') for i in range(2)]
            rows=[job.result() for job in jobs]
        plan=s.read_json(self.output/'PLAN.json')
        self.assertEqual(sum(row['new_tasks'] for row in rows),plan['total_tasks'])
        self.assertEqual(self.all_pairs(),self.expected)
        exact.finalize(self.output)

    def test_dense_positions_bounded_and_complete(self):
        plan=self.prepare(); backend=exact.NumpyBackend(plan)
        pieces=list(backend.positions(np.ones((5,9),np.float32),.75,7))
        self.assertTrue(all(len(rows)<=7 for rows,_ in pieces))
        pairs=[(int(r),int(c)) for rows,cols in pieces for r,c in zip(rows,cols)]
        self.assertEqual(set(pairs),{(r,c) for r in range(5) for c in range(9)})
        self.assertEqual(len(pairs),45)

    def test_half_input_bound_preserves_complete_true_pairs(self):
        self.qualify(compute_dtype='float16')
        exact.worker(self.output,worker_id='half',device='cpu')
        self.assertEqual(self.all_pairs(),self.expected)
        margin=exact.conservative_margin(self.bounds,'float16')
        vectors=s.normalized(self.raw); half=vectors.astype(np.float16)
        calculated=(half.astype(np.float32)@half.astype(np.float32).T).astype(np.float16).astype(np.float64)
        truth=vectors.astype(np.float64)@vectors.astype(np.float64).T
        self.assertLessEqual(float(np.abs(calculated-truth).max()),margin['total'])
        self.assertGreater(margin['cpu_comparison'],0)

    def test_receipt_corruption_missing_parquet_and_runtime_drift_fail(self):
        self.qualify(); exact.worker(self.output,worker_id='all',device='cpu')
        path=next(self.output.glob('tasks/*/*/edges.parquet'))
        value=bytearray(path.read_bytes());value[-1]^=1;path.write_bytes(value)
        with self.assertRaisesRegex(ValueError,'SHA'): exact.finalize(self.output)
        path.unlink()
        with self.assertRaisesRegex(ValueError,'missing'): exact.finalize(self.output)
        plan=s.read_json(self.output/'PLAN.json');plan['identity']['runtime']['code_sha256']='0'*64
        s.atomic_json(self.output/'PLAN.json',plan)
        with self.assertRaisesRegex(ValueError,'runtime changed'): exact.status(self.output)

    def test_disk_reserve_writes_stop_before_output(self):
        self.qualify();plan=s.read_json(self.output/'PLAN.json');plan['identity']['min_free_bytes']=1000
        Disk=namedtuple('Disk','total used free')
        with patch.object(exact.shutil,'disk_usage',return_value=Disk(1000,990,10)):
            with self.assertRaisesRegex(RuntimeError,'disk'):
                exact.disk_guard(self.output,plan)
        self.assertEqual(s.read_json(self.output/'STOP')['reason'],'disk_reserve')

    def test_upper_triangle_total_all_sizes(self):
        for sizes in ([1],[1,1],[2,5,3],[0,2,1]):
            tasks=list(exact.pair_tasks([{'count':v} for v in sizes])); n=sum(sizes)
            self.assertEqual(sum(t['pairs'] for t in tasks),n*(n-1)//2)
            self.assertEqual(len({t['number'] for t in tasks}),len(tasks))


if __name__=='__main__': unittest.main()
