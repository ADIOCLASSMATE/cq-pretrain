"""Rollout fixtures never access the platform or production task queue."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from . import tuned_rollout as rollout
from . import queue
from .fragment_controller import WORKSPACE, PROJECT, GROUP


class TunedRolloutTests(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name) / 'run'; root.mkdir()
        (root / 'states').mkdir()
        (root / 'TUNED_QUALIFIED.json').write_text('{}')
        lease = {'task_id': 'task1', 'token': 'token1', 'worker_id': 'owned-worker'}
        state = {**lease, 'status': 'running', 'lease_expires_at': 10000}
        (root / 'states/task1.json').write_text(json.dumps(state))
        item = {'name': 'accepted-low', 'notebook_id': 'nb1', 'workspace_id': 'ws1',
                'devices': 1, 'node': 'infra-gpu-npu-001.host.shzhisuan.com'}
        live = {'notebook_id': item['notebook_id'], 'name': item['name'], 'workspace': WORKSPACE,
                'project': PROJECT, 'compute_group': GROUP, 'priority': 3, 'priority_level': 'LOW',
                'resource': {'gpu_count': 1}, 'node': {'name': item['node']}, 'status': 'RUNNING'}
        stopped = {'name': item['name'], 'root': str(root), 'all_original_processes_exited': True,
                   'retry_candidates': [lease]}
        return root, item, live, stopped

    def exercise(self, *, maintenance_ok=True, stop_verified=True, high=False, fail_race=False):
        root, item, live, stopped = self.fixture()
        output = root / 'rollout'
        if high: live.update(priority=2, priority_level='HIGH')
        stopped['all_original_processes_exited'] = stop_verified
        platform = MagicMock()
        platform.observe.return_value = live
        # Controller has already started the replacement: idempotent launch
        # reports the same running worker and does not require a second spawn.
        platform.worker.side_effect = [{'status': 'running', 'hostname': 'accepted-host'},
                                      {'status': 'running', 'launched': False}]
        def execute(*args, **kwargs):
            self.assertEqual(kwargs['notebook_id'], item['notebook_id'])
            directory = output / item['name']; directory.mkdir()
            (directory / 'STOPPED.json').write_text(json.dumps(stopped))
            return {'completed': maintenance_ok, 'returncode': 0 if maintenance_ok else 125}
        fake_client = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'inspire': SimpleNamespace(),
                'inspire.sdk': SimpleNamespace(InspireClient=lambda **_: fake_client)}))
            stack.enter_context(patch.object(rollout, 'configuration', return_value={'qualified': True}))
            stack.enter_context(patch.object(rollout, 'verify_control_host', return_value='server'))
            stack.enter_context(patch.object(rollout, 'load_submissions', return_value=[item]))
            allocator = stack.enter_context(patch.object(rollout, 'Allocator'))
            allocator.return_value.load_managed.return_value = []
            stack.enter_context(patch.object(rollout, 'Platform', return_value=platform))
            remote = stack.enter_context(patch.object(rollout, 'execute', side_effect=execute))
            fail = stack.enter_context(patch.object(queue, 'fail', return_value={'status': 'pending'},
                side_effect=queue.LeaseLost('reclaimed') if fail_race else None))
            stack.enter_context(patch.object(rollout.time, 'time', return_value=100))
            if high or not maintenance_ok or not stop_verified:
                with self.assertRaises((ValueError, RuntimeError)):
                    rollout.run(root, output)
                fail.assert_not_called()
                if high: remote.assert_not_called()
                self.assertNotIn('launch', [call.args[2] for call in platform.worker.call_args_list])
            else:
                rollout.run(root, output)
                fail.assert_called_once_with(root, 'task1', 'token1',
                    'planned_tuned_rollout_after_verified_worker_exit', retryable=True)
                self.assertEqual(platform.worker.call_args_list[-1].args, (root, item, 'launch'))
                complete = json.loads((output / 'COMPLETE.json').read_text())
                self.assertTrue(complete['success'])
                self.assertFalse(complete['rows'][0]['replacement']['launched'])
                self.assertEqual(complete['rows'][0]['requeued_tasks'], [] if fail_race else ['task1'])

    def test_verified_low_rollout_and_idempotent_controller_launch(self):
        self.exercise()

    def test_high_priority_benchmark_never_gets_maintenance(self):
        self.exercise(high=True)

    def test_transport_or_pid_stop_failure_never_requeues(self):
        self.exercise(maintenance_ok=False)
        self.exercise(stop_verified=False)

    def test_lease_reclaimed_between_read_and_fail_does_not_abort_launch(self):
        self.exercise(fail_race=True)

    def test_expired_completed_and_other_owner_tokens_are_skipped(self):
        for change in ({'lease_expires_at': 99}, {'status': 'completed'}, {'token': 'replacement'},
                        {'worker_id': 'different-worker'}):
            root, _, _, stopped = self.fixture()
            state = json.loads((root / 'states/task1.json').read_text()); state.update(change)
            (root / 'states/task1.json').write_text(json.dumps(state))
            with patch.object(rollout.time, 'time', return_value=100), patch.object(queue, 'fail') as fail:
                result = rollout.requeue_stopped_leases(root, stopped, 'server')
                self.assertEqual(result[0]['status'], 'skipped_not_current_live_lease')
                fail.assert_not_called()

    def test_control_host_replacement_blocks_queue_mutation(self):
        root, _, _, stopped = self.fixture()
        with patch.object(rollout.time, 'time', return_value=100), \
             patch.object(rollout, 'verify_control_host', return_value='changed'), \
             patch.object(queue, 'fail') as fail:
            with self.assertRaisesRegex(RuntimeError, 'control ownership changed'):
                rollout.requeue_stopped_leases(root, stopped, 'server')
            fail.assert_not_called()

    def test_requeue_helper_requires_verified_stop_itself(self):
        root, _, _, stopped = self.fixture(); stopped['all_original_processes_exited'] = False
        with patch.object(queue, 'fail') as fail:
            with self.assertRaisesRegex(RuntimeError, 'Verified worker stop'):
                rollout.requeue_stopped_leases(root, stopped, 'server')
            fail.assert_not_called()

    def exercise_multiple_workers(self, failure=None):
        root, first, live, stopped = self.fixture()
        items = [first, {**first, 'name': 'accepted-low-2', 'notebook_id': 'nb2'}]
        output = root / 'rollout'
        events = []
        platform = MagicMock()
        platform.observe.side_effect = lambda item: {**live, 'name': item['name'],
                                                     'notebook_id': item['notebook_id']}
        def worker(_root, item, action):
            if action == 'status':
                return {'status': 'running'}
            events.append(('launch', item['name']))
            self.assertTrue((output / 'STOPS_VERIFIED.json').is_file())
            return {'status': 'running', 'launched': False}
        platform.worker.side_effect = worker
        def execute(*args, **kwargs):
            item = next(x for x in items if x['notebook_id'] == kwargs['notebook_id'])
            events.append(('stop', item['name']))
            directory = output / item['name']; directory.mkdir()
            evidence = {**stopped, 'name': item['name']}
            second = item is items[1]
            if second and failure == 'unverified':
                evidence['all_original_processes_exited'] = False
            (directory / 'STOPPED.json').write_text(json.dumps(evidence))
            if second and failure == 'qualification_changed':
                (root / 'TUNED_QUALIFIED.json').write_text('{"changed":true}')
            return {'completed': not (second and failure == 'transport'),
                    'returncode': 125 if second and failure == 'transport' else 0}
        def requeue(_root, evidence, control_server_id):
            self.assertEqual(events[:2], [('stop', first['name']), ('stop', items[1]['name'])])
            barrier = json.loads((output / 'STOPS_VERIFIED.json').read_text())
            self.assertEqual(len(barrier['rows']), 2)
            self.assertTrue(all(row['stopped']['all_original_processes_exited'] is True
                                for row in barrier['rows']))
            events.append(('requeue', evidence['name']))
            return []
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'inspire': SimpleNamespace(),
                'inspire.sdk': SimpleNamespace(InspireClient=lambda **_: MagicMock())}))
            stack.enter_context(patch.object(rollout, 'configuration', return_value={'qualified': True}))
            stack.enter_context(patch.object(rollout, 'verify_control_host', return_value='server'))
            stack.enter_context(patch.object(rollout, 'load_submissions', return_value=items))
            allocator = stack.enter_context(patch.object(rollout, 'Allocator'))
            allocator.return_value.load_managed.return_value = []
            stack.enter_context(patch.object(rollout, 'Platform', return_value=platform))
            stack.enter_context(patch.object(rollout, 'execute', side_effect=execute))
            retry = stack.enter_context(patch.object(rollout, 'requeue_stopped_leases', side_effect=requeue))
            if failure:
                with self.assertRaises(RuntimeError):
                    rollout.run(root, output)
                retry.assert_not_called()
                self.assertFalse(any(event[0] == 'launch' for event in events))
                self.assertFalse((output / 'STOPS_VERIFIED.json').exists())
                self.assertFalse((output / 'COMPLETE.json').exists())
                saved = json.loads((output / 'ROLLOUT.json').read_text())
                self.assertEqual(saved['rows'][0]['phase'], 'old_workers_exit_verified')
                self.assertEqual(saved['rows'][0]['stopped']['name'], first['name'])
                self.assertEqual(json.loads((output / 'FAILURE.json').read_text())['stage'],
                                 'stopping_all_selected_workers')
                # A failed attempt cannot be blindly rerun over its evidence.
                with self.assertRaises(FileExistsError):
                    rollout.run(root, output)
            else:
                rollout.run(root, output)
                self.assertEqual(events, [('stop', first['name']), ('stop', items[1]['name']),
                    ('requeue', first['name']), ('launch', first['name']),
                    ('requeue', items[1]['name']), ('launch', items[1]['name'])])
                self.assertTrue(json.loads((output / 'COMPLETE.json').read_text())['success'])

    def test_all_old_workers_exit_before_any_requeue_or_launch(self):
        self.exercise_multiple_workers()

    def test_later_stop_failure_preserves_evidence_without_requeue(self):
        for failure in ('transport', 'unverified', 'qualification_changed'):
            with self.subTest(failure=failure):
                self.exercise_multiple_workers(failure=failure)


if __name__ == '__main__': unittest.main()
