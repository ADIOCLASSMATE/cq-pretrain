"""Submission boundary tests; no platform calls."""
import json
import importlib.util
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from corpus_pipeline.cluster import submit_jobs as submit

sys.path.insert(0, str(submit.SKILL))


def payload(command):
    return {'command': shlex.join(['bash', '-c', command]), 'task_priority': 3,
            'is_publicpath_readonly': True, 'auto_fault_tolerance': True,
            'fault_tolerance_max_retry': 100, 'fault_tolerance_retry_interval_sec': 60,
            'exclude_nodes': [submit.INSPIRE_EXCLUSION], 'framework_config': [{
                'gpu_count': 16, 'instance_count': 1, 'cpu': 128,
                'mem_gi': 1024, 'shm_gi': 128, 'image': submit.IMAGE_URL}]}


@unittest.skipUnless(importlib.util.find_spec('inspire'), 'run submission tests with the Inspire SDK interpreter')
class SubmissionBoundaries(unittest.TestCase):
    def test_qualification_requires_true_and_exact_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(submit.qualification(root, False)['success'])
            with self.assertRaises(RuntimeError):
                submit.qualification(root, True)
            for receipt in ({'success': 'true', 'dtype': 'float32'},
                            {'success': True, 'dtype': 'float16'}):
                (root / 'QUALIFIED.json').write_text(json.dumps(receipt))
                with self.assertRaises(RuntimeError):
                    submit.qualification(root, True)

    def test_spaces_in_shared_path_preserve_command_arguments(self):
        root = Path('/inspire/a path with spaces/run')
        command = submit.worker_command(root)
        args = shlex.split(command)
        self.assertEqual(args[args.index('--root') + 1], str(root))
        for flag, value in (('--devices', '16'), ('--dtype', 'float32'),
                            ('--batch-size', '32'), ('--threads', '4')):
            self.assertEqual(args[args.index(flag) + 1], value)
        submit.validate_payload(payload(command), command)

    def test_reject_priority_or_retry_or_public_path_drift(self):
        command = submit.worker_command(Path('/inspire/run'))
        for key, changed in [('task_priority', 6), ('fault_tolerance_max_retry', 10),
                             ('is_publicpath_readonly', False),
                             ('fault_tolerance_retry_interval_sec', 1)]:
            changed_payload = payload(command)
            changed_payload[key] = changed
            with self.assertRaises(ValueError):
                submit.validate_payload(changed_payload, command)

    def test_ambiguous_create_sends_once_and_preserves_intent(self):
        command = submit.worker_command(Path('/inspire/run'))
        plans = [SimpleNamespace(create_kwargs=payload(command), name='fixture')]
        with tempfile.TemporaryDirectory() as tmp, \
             patch('node_policy.require_fixed_exclusions'), \
             patch('corpus_pipeline.serving.inspire_serving.downsize.observable_project_budget',
                   return_value={'observable_requested_cards': 112}), \
             patch.object(submit, 'send_exact', side_effect=TimeoutError) as send:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'Create not confirmed'):
                submit.dispatch_batch(None, plans, out, {'observable_requested_cards': 112}, {})
            self.assertEqual(send.call_count, 1)
            self.assertTrue((out / 'job-01/submitted-payload.json').exists())
            self.assertTrue((out / 'job-01/CREATE_INTENT.json').exists())
            self.assertTrue((out / 'job-01/CREATE_UNCONFIRMED.json').exists())
            self.assertFalse((out / 'job-01/ACCEPTED.json').exists())
            with self.assertRaises(FileExistsError):
                submit.dispatch_batch(None, plans, out, {'observable_requested_cards': 112}, {})
            self.assertEqual(send.call_count, 1)

    def test_budget_growth_blocks_before_create(self):
        plans = [SimpleNamespace(create_kwargs=payload('true'), name='fixture')]
        with tempfile.TemporaryDirectory() as tmp, \
             patch('node_policy.require_fixed_exclusions'), \
             patch('corpus_pipeline.serving.inspire_serving.downsize.observable_project_budget',
                   return_value={'observable_requested_cards': 208}), \
             patch.object(submit, 'send_exact') as send:
            with self.assertRaisesRegex(RuntimeError, 'budget changed'):
                submit.dispatch_batch(None, plans, Path(tmp), {'observable_requested_cards': 112}, {})
            send.assert_not_called()

    def test_four_independent_small_hooks_and_visibility_lag(self):
        plans = [SimpleNamespace(create_kwargs=payload('true'), name=f'fixture-{i}')
                 for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp, \
             patch('node_policy.require_fixed_exclusions'), \
             patch('corpus_pipeline.serving.inspire_serving.downsize.observable_project_budget',
                   return_value={'observable_requested_cards': 112}), \
             patch.object(submit, 'send_exact', side_effect=['a', 'b', 'c', 'd']) as send, \
             patch('job_controller.bind_after_submit',
                   return_value={'enabled': False, 'reason': 'below_64_gpus'}) as hook:
            out = Path(tmp)
            accepted = submit.dispatch_batch(None, plans, out, {'observable_requested_cards': 112}, {})
            self.assertEqual(len(accepted), 4)
            self.assertEqual(send.call_count, 4)
            self.assertEqual(hook.call_count, 4)
            for index in range(4):
                before = json.loads((out / f'job-{index + 1:02d}/budget-before-create.json').read_text())
                self.assertEqual(before['conservative_total_with_remaining'], 176)
                self.assertEqual(before['conservative_current_cards'], 112 + index * 16)
            self.assertTrue((out / 'SUBMITTED.json').exists())


if __name__ == '__main__':
    unittest.main()
