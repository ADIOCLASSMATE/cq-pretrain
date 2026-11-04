"""CPU-only fixtures: no production queue and no detector weights are touched."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from . import queue, queue_rpc, text_worker
from .frontpass import load_rules, prepare, process_task as full_process_task
from .test_frontpass import native_fixture
from corpus_pipeline.production.common import file_sha
from corpus_pipeline.production.records import snapshot


class TargetedClaimTests(unittest.TestCase):
    def test_targeted_and_sequential_claims_never_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue.init_queue(root, [{'task_id': f't{i:02d}'} for i in range(32)])
            def get(i):
                return (queue.claim_task(root, f'cpu{i}', f't{i:02d}') if i % 2
                        else queue.claim(root, f'gpu{i}'))
            with ThreadPoolExecutor(16) as pool:
                claims = [claim for claim in pool.map(get, range(32)) if claim]
            while claim := queue.claim(root, 'tail'):
                claims.append(claim)
            self.assertEqual(len(claims), 32)
            self.assertEqual(len({claim['task_id'] for claim in claims}), 32)
            self.assertIsNone(queue.claim_task(root, 'late', 't31'))

    def test_targeted_retry_limit_and_stale_token_fencing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue.init_queue(root, [{'task_id': 'one'}], max_attempts=2, lease_seconds=5, now=0)
            first = queue.claim_task(root, 'old', 'one', now=0)
            second = queue.claim_task(root, 'new', 'one', now=6)
            self.assertEqual(second['attempt'], 2)
            with self.assertRaises(queue.LeaseLost):
                queue.heartbeat(root, 'one', first['token'], now=6)
            queue.fail(root, 'one', second['token'], 'fixture', now=7)
            self.assertIsNone(queue.claim_task(root, 'third', 'one', now=8))
            self.assertEqual(queue.status(root, now=8)['counts']['failed'], 1)

    def test_unknown_id_and_substituted_catalog_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue.init_queue(root, [{'task_id': 'one', 'parquet_path': '/original'}])
            for value in ('absent', '../one'):
                with self.assertRaises(ValueError):
                    queue.claim_task(root, 'worker', value)
            config = queue._config(root)
            queue.catalog_index(root)
            path = root / config['catalog'] / '000000000000.json'
            item = queue._read(path)
            item['task']['parquet_path'] = '/replacement'
            item['task_sha256'] = queue._digest(item['task'])
            queue._atomic(path, item)
            with self.assertRaises(queue.QueueError):
                queue.claim_task(root, 'worker', 'one')
            self.assertEqual(list((root / 'states').iterdir()), [])


def text_fixture(directory, *, unexpected_image=False):
    root = Path(directory) / 'dataset'
    raw, _, task = native_fixture(root)
    rows = [deepcopy(raw[3]) for _ in range(4)]
    for index, row in enumerate(rows):
        row['sample_id'] = str(index).zfill(64)
        row['source_key'] = str(index)
    if unexpected_image:
        rows[-1] = raw[0]
    path = Path(task['parquet_path'])
    schema = pq.ParquetFile(path).schema_arrow
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=2)
    catalog = json.loads((root / 'catalog.json').read_text())
    catalog[0].update(rows=len(rows), bytes=path.stat().st_size)
    (root / 'catalog.json').write_text(json.dumps(catalog))
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['statistics'] = {'sources': [['unit_native', len(rows), 100, 0]]}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'COMPLETE.json').write_text(json.dumps({'validation_passed': True,
        'catalog_sha256': file_sha(root / 'catalog.json')}))
    run = Path(directory) / 'run'
    plan = prepare(run, dataset=root, rows_per_task=2)
    snap = snapshot(root)
    source_plan = {'schema_version': 'dcvlm_text_only_source_plan.v1',
        'dataset_root': str(root), 'source_count': 1, 'source_names': ['unit_native'],
        'image_occurrences': 0, 'record_count': 4, 'task_count': 2,
        'sources': [{'source': 'unit_native', 'manifest_image_occurrences': 0,
                     'manifest_record_count': 4}],
        'evidence_sha256': {**{key: snap['fingerprint'][key]
                              for key in ('manifest.json', 'catalog.json')},
                           'plan.json': file_sha(run / 'plan.json')}}
    (run / 'text_only_sources.json').write_text(json.dumps(source_plan))
    return run, snap, plan, source_plan, rows


class TextWorkerTests(unittest.TestCase):
    def test_text_outputs_match_normal_frontpass_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root, snap, plan, source_plan, rows = text_fixture(directory)
            selected = text_worker.select_tasks(root, source_plan, snap, plan)
            claim = queue.claim_task(root, 'text', selected[0])
            contracts = SimpleNamespace(entries={})
            summary, artifacts = text_worker.process_task(root, claim, snap, load_rules(), contracts)
            out = Path(claim['artifact_dir'])
            actual = pq.read_table(out / 'records.parquet').to_pylist()
            self.assertEqual(summary['counts']['records'], 2)
            self.assertEqual(summary['models_loaded'], [])
            self.assertEqual(summary['api_calls'], 0)
            self.assertEqual(pq.read_table(out / 'images.parquet').num_rows, 0)
            self.assertEqual(pq.read_table(out / 'suspects.parquet').num_rows, 0)
            self.assertEqual((out / 'sscd.f32').stat().st_size, 0)
            normal_root = Path(directory) / 'normal'
            normal_claim = {**claim, 'token': 'normal-no-models'}
            class NoModel:
                identity_sha256 = 'fixture-no-models'
                def iter_results(self, inputs, *, models):
                    self.assert_no_inputs(inputs)
                    return iter(())
                def assert_no_inputs(self, inputs):
                    if list(inputs):
                        raise AssertionError('detector must never receive an image')
            expected_summary, _ = full_process_task(normal_root, normal_claim, NoModel(), snap,
                load_rules(), contracts, {'cf384': .9, 'unifd': .95})
            expected = pq.read_table(normal_root / 'artifacts' / claim['task_id'] /
                                     normal_claim['token'] / 'records.parquet').to_pylist()
            self.assertEqual(actual, expected)
            self.assertEqual(summary['counts'], expected_summary['counts'])
            queue.complete(root, claim['task_id'], claim['token'], artifacts=artifacts,
                           processed_records=2, processed_images=0)
            self.assertIsNone(queue.claim_task(root, 'late', claim['task_id']))
            self.assertNotEqual(queue.claim(root, 'normal')['task_id'], claim['task_id'])

    def test_unexpected_image_fails_entire_task_and_returns_it_to_normal_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _, _, _ = text_fixture(directory, unexpected_image=True)
            server = queue_rpc.QueueServer(root, bind='127.0.0.1', port=0,
                                            advertise_host='127.0.0.1')
            try:
                with self.assertRaises(text_worker.UnexpectedImage):
                    text_worker.run_worker(root)
                status = queue.status(root)
                self.assertEqual(status['counts']['completed'], 1)
                self.assertEqual(status['counts']['pending'], 1)
                self.assertEqual(status['processed_records'], 2)
                self.assertEqual(status['processed_images'], 0)
                markers = list((root / 'text_cpu_exclusions').glob('*.json'))
                self.assertEqual(len(markers), 1)
                again = text_worker.run_worker(root)
                self.assertEqual(again['completed_tasks'], 0)
                claim = queue.claim(root, 'normal-gpu')
                self.assertEqual(claim['task_id'], markers[0].stem)
                self.assertEqual(claim['attempt'], 2)
            finally:
                server.server_close()

    def test_control_host_rejects_remote_and_dead_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _, _, _ = text_fixture(directory)
            server = queue_rpc.QueueServer(root, bind='127.0.0.1', port=0,
                                            advertise_host='127.0.0.1')
            try:
                self.assertEqual(text_worker.verify_control_host(root), server.server_id)
                with patch.object(socket, 'gethostname', return_value='another-host'):
                    with self.assertRaises(RuntimeError):
                        text_worker.ControlHostQueue(root)
                with patch.object(os, 'kill', side_effect=ProcessLookupError):
                    with self.assertRaises(RuntimeError):
                        text_worker.ControlHostQueue(root)
                backend = text_worker.ControlHostQueue(root)
                with patch.object(socket, 'gethostname', return_value='another-host'):
                    with self.assertRaises(RuntimeError):
                        backend.claim_task(root, 'worker', 's00000-r00000000')
                self.assertEqual(queue.status(root)['counts']['pending'], 2)
            finally:
                server.server_close()

    def test_malformed_image_slots_raise_image_exception_before_slot_validation(self):
        row = {'raw': {'image_count': 0, 'images': [], 'segments': []},
               'source_image_refs': [],
               'normalized': {'provenance': {'images': [{'image_index': 9}]}}}
        with self.assertRaises(text_worker.UnexpectedImage):
            text_worker.assert_text_only(row)

    def test_processing_failure_is_excluded_from_later_cpu_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _, _, _ = text_fixture(directory)
            server = queue_rpc.QueueServer(root, bind='127.0.0.1', port=0,
                                            advertise_host='127.0.0.1')
            try:
                with patch.object(text_worker, 'process_task', side_effect=ValueError('native image slot invalid')):
                    with self.assertRaises(ValueError):
                        text_worker.run_worker(root)
                marker = next((root / 'text_cpu_exclusions').glob('*.json'))
                self.assertEqual(json.loads(marker.read_text())['error_type'], 'ValueError')
                self.assertEqual(text_worker.run_worker(root)['completed_tasks'], 1)
                self.assertEqual(queue._read(queue._state_path(root, marker.stem))['attempt'], 1)
                self.assertEqual(queue.claim(root, 'normal-gpu')['task_id'], marker.stem)
            finally:
                server.server_close()

    def test_exclusion_published_during_claim_prevents_any_text_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _, _, _ = text_fixture(directory)
            server = queue_rpc.QueueServer(root, bind='127.0.0.1', port=0,
                                            advertise_host='127.0.0.1')
            original = queue.claim_task
            def race(*args, **kwargs):
                claim = original(*args, **kwargs)
                directory = root / 'text_cpu_exclusions'
                directory.mkdir(exist_ok=True)
                (directory / (claim['task_id'] + '.json')).write_text('{}')
                return claim
            try:
                with patch.object(queue, 'claim_task', side_effect=race), \
                        patch.object(text_worker, 'process_task') as process:
                    with self.assertRaises(text_worker.UnexpectedImage):
                        text_worker.run_worker(root)
                    process.assert_not_called()
                self.assertEqual(queue.status(root)['processed_records'], 0)
                self.assertEqual(queue.status(root)['counts']['running'], 0)
            finally:
                server.server_close()

    def test_selection_does_not_trust_unbound_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root, snap, plan, source_plan, _ = text_fixture(directory)
            cases = []
            bad = deepcopy(source_plan); bad['sources'][0]['manifest_image_occurrences'] = 1; cases.append(bad)
            bad = deepcopy(source_plan); bad['evidence_sha256']['plan.json'] = '0' * 64; cases.append(bad)
            bad = deepcopy(source_plan); bad['record_count'] -= 1; cases.append(bad)
            for bad in cases:
                with self.assertRaises(ValueError):
                    text_worker.select_tasks(root, bad, snap, plan)
            bad_snap = deepcopy(snap)
            bad_snap['manifest']['statistics']['sources'][0][3] = 1
            with self.assertRaises(ValueError):
                text_worker.select_tasks(root, source_plan, bad_snap, plan)

    def test_import_never_loads_torch_or_inference(self):
        subprocess.run([sys.executable, '-c',
            'import sys; import corpus_pipeline.cluster.text_worker; '
            'assert "torch" not in sys.modules; '
            'assert "corpus_pipeline.cluster.inference" not in sys.modules'], check=True)


if __name__ == '__main__':
    unittest.main()
