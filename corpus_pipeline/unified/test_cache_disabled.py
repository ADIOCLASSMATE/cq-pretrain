"""The optional cache never changes frontpass models, evidence, or failures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq

from corpus_pipeline.cluster import queue
from . import workflow
from .config import checked
from .test_cache import FakeWorker
from . import test_workflow as workflow_fixtures


class ModelFailure(FakeWorker):
    def iter_results(self, assets, *, models=None):
        for result in super().iter_results(assets, models=models):
            result['scores']['cf384'] = {'status': 'error', 'error_type': 'SyntheticModelFailure'}
            yield result


class CacheDisabledTests(unittest.TestCase):
    def setup_runs(self, root):
        original = workflow_fixtures.WorkflowTests().config(root)
        value = json.loads(original.read_text())
        paths = {}
        for enabled in (True, False):
            name = 'on' if enabled else 'off'
            selected = deepcopy(value)
            selected['output'] = str(root/name)
            selected['cache'] = {'enabled': enabled, 'directory': str(root/(name+'_cache')), 'histories': []}
            path = root/(name+'.json')
            path.write_text(json.dumps(selected))
            paths[enabled] = path
        return paths

    def test_configuration_defaults_and_rejects_ambiguous_disabled_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = json.loads(self.setup_runs(root)[True].read_text())
            value['cache'].pop('enabled')
            self.assertIs(checked(value)['cache']['enabled'], True)
            for invalid in (0, 1, None, 'false', []):
                with self.subTest(enabled=invalid), self.assertRaisesRegex(ValueError, 'cache.enabled'):
                    checked({**value, 'cache': {**value['cache'], 'enabled': invalid}})
            for histories in ([str(root/'history')], 'not-a-list'):
                with self.subTest(histories=histories), self.assertRaises(ValueError):
                    checked({**value, 'cache': {**value['cache'], 'enabled': False, 'histories': histories}})

    @staticmethod
    def evidence(run):
        records, images, suspects, vectors = [], [], [], []
        for state in sorted((run/'frontpass/states').glob('*.json')):
            value = json.loads(state.read_text())
            if value['status'] != 'completed':
                raise AssertionError('Uncommitted fixture task')
            directory = run/'frontpass/artifacts'/value['task_id']/value['token']
            records.extend(pq.read_table(directory/'records.parquet').to_pylist())
            images.extend(pq.read_table(directory/'images.parquet').to_pylist())
            suspects.extend(pq.read_table(directory/'suspects.parquet').to_pylist())
            vectors.append((directory/'sscd.f32').read_bytes())
        return records, images, suspects, vectors

    def test_real_frontpass_cache_on_off_evidence_equal_and_off_never_opens_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.setup_runs(root)
            delegates = {}
            for enabled in (True, False):
                delegate = FakeWorker()
                delegate.execution_identity = {'fixture': 'same-qualified-delegate'}
                delegates[enabled] = delegate
                if enabled:
                    result = workflow.run(paths[enabled], local=True, stop_after='frontpass', delegate=delegate)
                else:
                    with patch('corpus_pipeline.unified.cache.EvidenceCache', side_effect=AssertionError('Cache opened')):
                        result = workflow.run(paths[enabled], local=True, stop_after='frontpass', delegate=delegate)
                self.assertEqual(result['phase'], 'stage_complete')
                self.assertTrue(delegate.closed)
                self.assertTrue(any(set(models) == {'cf384', 'unifd', 'sscd'} for _, models in delegate.calls))
            self.assertEqual(self.evidence(root/'on'), self.evidence(root/'off'))
            self.assertFalse((root/'off_cache').exists())
            self.assertTrue((root/'on_cache').is_dir())
            for enabled in (True, False):
                run = root/('on' if enabled else 'off')
                plan = json.loads((run/'frontpass/plan.json').read_text())
                self.assertIs(plan['old_result_reuse'], enabled)
                workers = list((run/'frontpass/workers').iterdir())
                self.assertEqual(len(workers), 1)
                worker = workers[0]
                self.assertEqual(json.loads((worker/'model_identity.json').read_text()), delegates[enabled].identity)
                finished = json.loads((worker/'FINISHED.json').read_text())
                self.assertIs(finished['cache']['enabled'], enabled)
                if not enabled:
                    self.assertEqual(finished['cache'], {'enabled': False, 'status': 'disabled', 'reads': 0, 'writes': 0, 'seals': 0})
                    self.assertEqual(json.loads((worker/'execution_identity.json').read_text()), delegates[enabled].execution_identity)

    def test_disabled_cache_model_failure_still_fails_task_and_closes_delegate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.setup_runs(root)
            delegate = ModelFailure()
            with self.assertRaisesRegex(RuntimeError, 'resident model forward failed'):
                workflow.run(paths[False], local=True, stop_after='frontpass', delegate=delegate)
            self.assertTrue(delegate.closed)
            self.assertFalse((root/'off_cache').exists())
            self.assertFalse((root/'off/receipts/frontpass.json').exists())
            state = queue.status(root/'off/frontpass')
            self.assertGreater(state['failed_attempts'], 0)
            self.assertFalse(state['all_tasks_completed'])
            self.assertFalse(list((root/'off/frontpass/workers').glob('*/FINISHED.json')))


if __name__ == '__main__':
    unittest.main()
