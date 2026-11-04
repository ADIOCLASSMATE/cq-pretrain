"""Second-pass scope/provenance safeguards; no paid inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from corpus_pipeline.context_review import lenient_run, lenient_worker, worker
from corpus_pipeline.context_review.test_worker import item, result


class LenientTests(unittest.TestCase):
    def test_new_version_same_fixed_runtime_and_output_contract(self):
        self.assertNotEqual(lenient_worker.PROMPT_VERSION, worker.PROMPT_VERSION)
        self.assertEqual(lenient_worker.MODEL, 'gpt-5.6-sol')
        self.assertEqual(lenient_worker.EFFORT, 'low')
        self.assertEqual(lenient_worker.SCHEMA, worker.SCHEMA)

    def test_source_context_only_not_prior_verdict(self):
        source = item()
        source['initial_context_result'] = 'SECRET_EXCLUSION'
        self.assertNotIn('SECRET_EXCLUSION', lenient_worker.build_prompt([source]))
        self.assertNotIn('SECRET_PRIOR_VERDICT', lenient_worker.build_prompt([source]))

    def test_visible_scene_and_unverified_identity_can_pass_as_low_information(self):
        value = result(image_quality='degraded_usable', alignment='weakly_related',
                       training_quality='usable_low_information', recommendation='flag_low_information',
                       reason='脸部模糊使个人身份无法验证，但足球比赛、球员和球场场景仍可用于一般知识学习。')
        lenient_worker.validate_result(value, 'opaque-a', item())
        value['recommendation'] = 'exclude_pair'
        value['training_quality'] = 'unusable'
        with self.assertRaisesRegex(ValueError, 'damage_or_mismatch'):
            lenient_worker.validate_result(value, 'opaque-a', item())

    def test_complete_subject_destruction_remains_excludable(self):
        value = result(image_quality='unusable', alignment='unassessable',
                       training_quality='unusable', recommendation='exclude_pair')
        lenient_worker.validate_result(value, 'opaque-a', item())

    def test_selection_preserves_exact_inputs_and_excludes_initial_retained_items(self):
        with tempfile.TemporaryDirectory() as name:
            parent = Path(name)
            (parent/'config.json').write_text(json.dumps({'model':'gpt-5.6-sol','reasoning_effort':'low'}))
            rows=[]
            for index in range(209):
                source=item(f'id-{index}')
                source.update(queue_order=index, review_number=index+1)
                model=result(f'id-{index}',recommendation='exclude_pair' if index<208 else 'keep')
                rows.append({'candidate':source,'result':model})
            before=deepcopy(rows)
            root=parent/'second'
            with mock.patch('corpus_pipeline.context_review.report.load_completed',return_value=(rows,{'immutable':'binding'})):
                selection=lenient_run.prepare(parent,root)
            actual=[json.loads(line) for line in (root/'candidates.jsonl').read_text().splitlines()]
            self.assertEqual(actual,[r['candidate'] for r in rows[:208]])
            self.assertEqual(selection['records'],208)
            self.assertEqual(rows,before)
            self.assertFalse(selection['initial_results_in_prompt'])
            self.assertTrue(all('initial_context_result' not in candidate for candidate in actual))
            with self.assertRaisesRegex(ValueError,'overwrite'):
                lenient_run.prepare(parent,root)


if __name__=='__main__':
    unittest.main()
