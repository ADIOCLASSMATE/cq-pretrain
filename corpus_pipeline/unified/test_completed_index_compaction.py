"""A published index delegates its repeated checksum audit to compaction."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import baseline_index, native_compact, workflow
from .test_baseline_index import fixture, mark_all, source_complete


class CompletedIndexCompactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw, self.control, self.manifest, _ = fixture(self.root)
        self.run = self.root/'run'
        self.index = self.run/'native_index'
        for name, value in [('EXPECTED_TRAINING_FILES', 2), ('_space', lambda *args: None)]:
            selected = patch.object(baseline_index, name, value)
            selected.start(); self.addCleanup(selected.stop)
        selected = patch.object(native_compact, '_space')
        selected.start(); self.addCleanup(selected.stop)
        self.config = self.root/'pipeline.json'
        self.config.write_text(json.dumps({'schema': 'corpus.unified.pipeline.v1', 'scope': 'full',
            'required_collections': ['dcvlm'], 'output': str(self.run),
            'inputs': [{'collection': 'dcvlm', 'adapter': 'dcvlm_baseline_wds', 'root': str(self.raw),
                        'download_control': str(self.control), 'index_output': str(self.index)}],
            'compute': {'rows_per_task': 8192, 'io_workers': 2}, 'cache': {'enabled': False}}))
        self.cfg, _ = workflow.initialize(self.config)
        mark_all(self.control, self.manifest)
        baseline_index.prepare(self.control, self.index)
        baseline_index.worker(self.control, self.index, 'fixture')
        source_complete(self.control, self.manifest)

    def test_completed_index_skips_redundant_finalize_but_real_compaction_verifies_all(self):
        baseline_index.finalize(self.control, self.index)
        self.assertEqual(baseline_index.status(self.control, self.index)['phase'], 'complete')
        with patch.object(baseline_index, 'finalize', side_effect=AssertionError('Repeated index hash')):
            workflow.execute_stage(self.cfg, self.run, 'normalize')
        complete = json.loads((self.run/'native/COMPLETE.json').read_text())
        self.assertTrue(complete['validation_passed'])
        self.assertEqual(complete['statistics']['samples'], 4)
        self.assertTrue((self.run/'receipts/normalize.json').exists())

    def test_completed_index_bad_parquet_is_rejected_by_real_compaction(self):
        baseline_index.finalize(self.control, self.index)
        catalog = json.loads((self.index/'catalog.json').read_text())
        with (self.index/catalog[0]['path']).open('ab') as stream:
            stream.write(b'corrupted-after-publication')
        # Status checks publication/source bindings, while the next consumer
        # remains responsible for checking each metadata file's actual bytes.
        self.assertEqual(baseline_index.status(self.control, self.index)['phase'], 'complete')
        with patch.object(baseline_index, 'finalize', side_effect=AssertionError('Repeated index hash')):
            with self.assertRaisesRegex(ValueError, 'checksum changed'):
                workflow.execute_stage(self.cfg, self.run, 'normalize')
        self.assertFalse((self.run/'native/COMPLETE.json').exists())
        self.assertFalse((self.run/'receipts/normalize.json').exists())

    def test_unpublished_ready_index_still_runs_full_finalize(self):
        self.assertEqual(baseline_index.status(self.control, self.index)['phase'], 'ready_to_finalize')
        with patch.object(baseline_index, 'finalize', wraps=baseline_index.finalize) as finalize:
            workflow.execute_stage(self.cfg, self.run, 'normalize')
        finalize.assert_called_once_with(str(self.control), self.index)
        self.assertTrue((self.run/'native/COMPLETE.json').exists())


if __name__ == '__main__':
    unittest.main()
