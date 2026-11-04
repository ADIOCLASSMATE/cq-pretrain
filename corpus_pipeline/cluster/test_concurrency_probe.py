"""Real subprocess lifecycle tests with stand-in work, never numerical NPU evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from .concurrency_probe import compare_predictions, load_predictions, run_scenario, save_predictions, split_indices


STUB = r'''
import json,os,sys,time
from pathlib import Path
spec=json.loads(Path(sys.argv[1]).read_text()); out=Path(spec['output'])
behavior=sys.argv[2]
def save(name,value):
 p=out/name; temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value));temp.replace(p)
if behavior=='fail' and spec['rank']==1: raise RuntimeError('expected child failure')
if behavior=='hang':
 import signal
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 while True:time.sleep(.05)
time.sleep(.1)
save('INPUT_AUDIT.json',[{'sha256':'same','tensor_sha256':'same'}])
save('READY.json',{'model_identity_sha256':'same','input_audit_sha256':'same','pid':os.getpid()})
barrier=Path(spec['barrier'])
while not barrier.exists():time.sleep(.005)
start=json.loads(barrier.read_text())['start_monotonic']
while time.monotonic()<start:time.sleep(.001)
actual=time.monotonic();time.sleep(.05+spec['rank']*.01);end=time.monotonic()
save('DONE.json',{'success':True,'images':len(spec['indices']),'model_inferred_images':len(spec['indices']),
 'started_monotonic':actual,'ended_monotonic':end,'barrier_start_lag_seconds':actual-start,
 'elapsed_seconds':end-actual,'process_cpu_seconds':.01})
'''


def predictions(count=4):
    vectors = np.tile(np.arange(1, 513, dtype=np.float32), (count, 1))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return {'indices': np.arange(count), 'sha256': np.asarray([str(i) for i in range(count)]),
            'status': np.asarray(['ok'] * count), 'pixel_sha256': np.asarray(['pixel'] * count),
            'width': np.full(count, 512), 'height': np.full(count, 512),
            'probabilities': np.full((count, 2), .2, dtype=np.float32),
            'logits': np.full((count, 2), -1.386, dtype=np.float32), 'sscd': vectors}


class ConcurrencyProbeTests(unittest.TestCase):
    def test_disjoint_round_robin_has_fixed_total_work_and_balanced_assignments(self):
        for workers in (1, 2, 3):
            indices = split_indices(1024, workers)
            self.assertEqual(sorted(i for group in indices for i in group), list(range(1024)))
            self.assertLessEqual(max(map(len, indices)) - min(map(len, indices)), 1)
        for count, workers in ((1, 2), (10, 0), (True, 1)):
            with self.assertRaises(ValueError):
                split_indices(count, workers)

    def test_numerical_qualification_rejects_metadata_scores_vectors_and_near_threshold_flip(self):
        reference = predictions()
        self.assertEqual(compare_predictions(reference, deepcopy(reference))['threshold_decision_disagreements'], 0)
        for kind in ('metadata', 'score', 'vector', 'nonfinite', 'flip'):
            expected, actual = deepcopy(reference), deepcopy(reference)
            if kind == 'metadata':
                actual['width'][0] += 1
            elif kind == 'score':
                actual['probabilities'][0, 0] += .01
            elif kind == 'vector':
                actual['sscd'][0] *= -1
            elif kind == 'nonfinite':
                actual['logits'][0, 0] = np.nan
            else:
                expected['probabilities'][0, 0] = .8999
                actual['probabilities'][0, 0] = .9001
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                compare_predictions(expected, actual)

    def test_prediction_evidence_restores_original_order_and_rejects_duplicate_occurrences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = predictions(5)
            for rank, indices in enumerate(split_indices(5, 2)):
                worker = root / f'worker-{rank:02d}';worker.mkdir()
                np.savez(worker / 'PREDICTIONS.npz', **{key: value[indices] for key, value in source.items()})
            loaded = load_predictions(root, 5)
            self.assertEqual(compare_predictions(source, loaded)['images'], 5)
            np.savez(root / 'worker-01/PREDICTIONS.npz', **{key: value[[1, 1]] for key, value in source.items()})
            with self.assertRaisesRegex(AssertionError, 'duplicate'):
                load_predictions(root, 5)

    def test_saved_evidence_preserves_small_images_and_rejects_nonfinite_model_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'worker-00';output.mkdir()
            assets = [{'sha256': '0'}, {'sha256': '1'}]
            vector = np.ones(512, dtype=np.float32) / np.sqrt(512)
            rows = [{'image_sha256': '0', 'status': 'ok', 'original_width': 512, 'original_height': 512,
                     'pixel_sha256': 'pixel', 'sscd': {'embedding': vector},
                     'scores': {'cf384': {'prob': .2, 'logit': -1.386}, 'unifd': {'prob': .2, 'logit': -1.386}}},
                    {'image_sha256': '1', 'status': 'too_small', 'original_width': 64, 'original_height': 100}]
            save_predictions(output / 'PREDICTIONS.npz', [0, 1], assets, rows)
            value = load_predictions(Path(directory), 2)
            self.assertEqual(compare_predictions(value, value)['model_inferred_images'], 1)
            rows[0]['scores']['cf384']['prob'] = float('nan')
            with self.assertRaisesRegex(AssertionError, 'nonfinite'):
                save_predictions(output / 'bad.npz', [0, 1], assets, rows)

    def test_linux_parent_death_guard_signals_only_the_spawned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_code = r'''
import os,signal,sys,time
from pathlib import Path
from corpus_pipeline.cluster.concurrency_probe import child_parent_guard
root=Path(sys.argv[1]); expected=int(sys.argv[2])
def died(*args):
 (root/'DIED').write_text('parent death signal received');sys.exit(0)
signal.signal(signal.SIGTERM,died)
child_parent_guard(expected)
(root/'READY').write_text(str(os.getpid()))
while True:time.sleep(.05)
'''
            parent_code = r'''
import os,subprocess,sys,time
subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],str(os.getpid())],start_new_session=True)
while True:time.sleep(.05)
'''
            parent = subprocess.Popen([sys.executable, '-c', parent_code, child_code, str(root)])
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not (root / 'READY').is_file() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((root / 'READY').is_file())
                child_pid = int((root / 'READY').read_text())
                parent.kill();parent.wait(timeout=3)
                while not (root / 'DIED').is_file() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((root / 'DIED').is_file())
            finally:
                if parent.poll() is None:
                    parent.kill();parent.wait(timeout=3)
                if child_pid is not None and not (root / 'DIED').exists():
                    try:
                        os.kill(child_pid, 15)
                    except ProcessLookupError:
                        pass

    def test_real_subprocess_barrier_excludes_startup_and_completes_all_children(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scenario'
            result = run_scenario(output, Path(directory) / 'unused.json', 11, 3, {},
                startup_timeout=5, run_timeout=5, sample_resources=False,
                command_builder=lambda spec: [sys.executable, '-c', STUB, str(spec), 'ok'])
            self.assertTrue(result['success'])
            self.assertEqual(result['total_images'], 11)
            self.assertEqual(sum(r['images'] for r in result['worker_results']), 11)
            self.assertLess(result['steady_seconds'], .5)
            self.assertGreater(result['steady_seconds'], .05)
            cleanup = json.loads((output / 'CLEANUP.json').read_text())
            self.assertTrue(all(row['returncode'] == 0 for row in cleanup['children']))

    def test_child_failure_fails_closed_and_terminates_waiting_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scenario'
            with self.assertRaisesRegex(RuntimeError, 'exited'):
                run_scenario(output, Path(directory) / 'unused.json', 11, 3, {},
                    startup_timeout=5, run_timeout=5, sample_resources=False,
                    command_builder=lambda spec: [sys.executable, '-c', STUB, str(spec), 'fail'])
            self.assertFalse((output / 'SCENARIO.json').exists())
            cleanup = json.loads((output / 'CLEANUP.json').read_text())
            self.assertTrue(all(row['returncode'] is not None for row in cleanup['children']))
            for row in cleanup['children']:
                self.assertFalse(Path('/proc', str(row['pid'])).exists())

    def test_startup_timeout_kills_child_even_if_sigterm_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scenario'
            with self.assertRaisesRegex(TimeoutError, 'READY'):
                run_scenario(output, Path(directory) / 'unused.json', 4, 1, {},
                    startup_timeout=.2, run_timeout=1, sample_resources=False,
                    command_builder=lambda spec: [sys.executable, '-c', STUB, str(spec), 'hang'])
            cleanup = json.loads((output / 'CLEANUP.json').read_text())
            self.assertEqual(cleanup['children'][0]['returncode'], -9)
            self.assertFalse((output / 'SCENARIO.json').exists())


if __name__ == '__main__':
    unittest.main()
