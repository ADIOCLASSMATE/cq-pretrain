"""Real baseline mismatch shapes retain evidence and never become training rows."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from corpus_pipeline.production.common import canonical, file_sha
from . import ingest, baseline_compat
from .test_ingest import balanced


EXAMPLES = [
    ('duplicate', 'instruction', 'cambrian_gpt4o',
     'human:<image> What is in this picture?\n<image><EOCL>\ngpt:Original answer.\n', 1),
    ('literal', 'mmdoc_mt', 'mint_html_mehdi',
     'human:<image><EOCL>\ngpt:docker tag <image> demo.ctfd.io/project\n'
     '<EOCL>\nhuman:<image><EOCL>\ngpt:Then deploy.', 2),
]


class BaselineCompatibilityTests(unittest.TestCase):
    def test_successful_strict_rows_are_exactly_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, path = balanced(root)
            unit = dict(path=str(path), relative_path='shards/000/000000.tar', source=None)
            strict = list(ingest._balanced_rows(unit))
            self.assertEqual(canonical(strict), canonical(list(baseline_compat.rows(unit))))

    def test_real_mismatch_shapes_keep_all_text_roles_and_image_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, path = balanced(root, EXAMPLES)
            before = file_sha(path)
            rows = list(baseline_compat.rows(dict(path=str(path), relative_path='shards/000/000000.tar', source=None)))
            for row, example in zip(rows, EXAMPLES):
                original = json.loads(row['original_supervision_json'])
                metadata = json.loads(row['adapter_metadata_json'])
                self.assertEqual(original['source_annotations']['conversations.txt'], example[3])
                self.assertEqual([part['text'] for part in row['segments']],
                                 [turn.partition(':')[2] for turn in example[3].split('<EOCL>\n')])
                self.assertTrue(all(part['kind'] == 'text' and part['image_index'] is None for part in row['segments']))
                self.assertEqual(len(row['images']), example[4])
                self.assertEqual(len(original['member_inventory']), 6 + example[4])
                self.assertEqual(metadata['prepared_adapter_status'], 'quarantine')
                self.assertEqual(metadata['pairing_state'], 'unresolved')
                self.assertFalse(metadata['binding_repair_applied'])
                self.assertEqual(metadata['issues'], [baseline_compat.ISSUE])
                unsafe = deepcopy(metadata); unsafe['prepared_adapter_status'] = 'ready'
                with self.assertRaisesRegex(ValueError, 'invalid unresolved'):
                    ingest._row('dcvlm', original, row['images'], {}, unsafe)
            self.assertEqual(file_sha(path), before)

    def test_unknown_roles_remain_fatal_even_after_a_binding_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path = balanced(Path(tmp), [('a', 'instruction', 'x',
                'human:<image><image><EOCL>\nrobot:Do not guess.', 1)])
            with self.assertRaisesRegex(ValueError, 'conversation role'):
                list(baseline_compat.rows(dict(path=str(path), relative_path='a.tar', source=None)))


if __name__ == '__main__':
    unittest.main()
