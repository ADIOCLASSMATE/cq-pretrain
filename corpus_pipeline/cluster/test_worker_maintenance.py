"""Local fixtures only: no production processes, platform or queue mutations."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from . import worker_maintenance as maintenance
from .notebook_worker import _node_command
from corpus_pipeline.production.common import digest


class MaintenanceTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / 'run'; root.mkdir()
        attempt = root / 'attempt'; attempt.mkdir()
        proc = root / 'proc'; proc.mkdir()
        name, host = 'accepted', 'host-999-component'
        supervisor = ['/python', '-m', 'corpus_pipeline.cluster.notebook_worker', '_supervise',
                      '--root', str(root), '--devices', '1', '--name', name,
                      '--attempt', 'a' * 32, '--lock-fd', '4']
        node = _node_command(root, 1)
        intent = {'root': str(root), 'name': name, 'hostname': host, 'attempt': 'a' * 32,
            'devices': 1, 'supervisor_command': supervisor, 'supervisor_command_sha256': digest(supervisor),
            'node_command': node, 'node_command_sha256': digest(node)}
        commands = {101: supervisor, 202: ['/python', *node[2:]],
            303: ['/python', '-m', 'corpus_pipeline.cluster.frontpass', 'worker', '--root', str(root),
                  '--device', 'npu:0', '--dtype', 'float32', '--batch-size', '32', '--threads', '4']}
        identities = {pid: {'pid': pid, 'start_ticks': pid * 10, 'boot_id': 'boot',
                           'command_sha256': digest(argv), 'argv': argv} for pid, argv in commands.items()}
        owner = {k: v for k, v in identities[101].items() if k != 'argv'}
        owner.update(attempt=intent['attempt'], hostname=host, node_pid=202)
        (attempt / 'OWNER.json').write_text(json.dumps(owner))
        for pid, ppid in ((101, 1), (202, 101), (303, 202)):
            path = proc / str(pid); path.mkdir()
            (path / 'stat').write_text(f'{pid} (process) S {ppid} 0 0')
        leases = root / 'claims'; leases.mkdir()
        for task, worker in [('owned', f'{host}-npu_0-303-abcdef01'),
                             ('other', f'{host}-npu_0-999-abcdef02'),
                             ('misleading', f'{host}-303-component-npu_0-999-abcdef03')]:
            path = leases / task; path.mkdir()
            (path / 'lease.json').write_text(json.dumps({'task_id': task, 'token': task + '-token', 'worker_id': worker}))
        return root, attempt, proc, name, host, intent, identities

    def test_exact_lease_owner_excludes_pid_substrings_and_other_devices(self):
        child = {'pid': 303, 'argv': ['--device', 'npu:0']}
        self.assertTrue(maintenance.lease_belongs({'worker_id': 'host-303-a-npu_0-303-abcdef01'}, child, 'host-303-a'))
        for worker in ('host-303-a-npu_0-999-abcdef01', 'host-303-a-npu_1-303-abcdef01',
                       'host-303-a-npu_0-303-abcdef01-extra'):
            self.assertFalse(maintenance.lease_belongs({'worker_id': worker}, child, 'host-303-a'))

    def test_same_rejects_numeric_pid_reuse(self):
        identity = {'pid': 10, 'start_ticks': 20, 'boot_id': 'boot', 'command_sha256': 'same'}
        with patch.object(maintenance, '_process', return_value=identity | {'start_ticks': 21}):
            self.assertFalse(maintenance.same(identity))

    def run_fixture(self, *, mutate=None, reused=False, still_live=False, tuned_receipt=False, tuned_child=False):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root, attempt, proc, name, host, intent, identities = self.fixture(temp.name)
        if tuned_receipt:
            (root / 'TUNED_QUALIFIED.json').write_text('{}')
        if tuned_child:
            identities[303]['argv'][2] = 'corpus_pipeline.cluster.tuned_frontpass'
            identities[303]['command_sha256'] = digest(identities[303]['argv'])
        if mutate: mutate(intent, identities)
        live = True
        calls = []
        def process(pid): return deepcopy(identities.get(pid)) if live else None
        def send(fd, sig):
            nonlocal live
            self.assertEqual((fd, sig), (456, signal.SIGTERM))
            self.assertTrue((root/'maintenance/STOP_INTENT.json').is_file())
            calls.append('pidfd_signal')
            if not still_live: live = False
        def pidfd(pid, flags):
            self.assertEqual((pid, flags), (101, 0))
            if reused: identities[101]['start_ticks'] += 1
            return 456
        with ExitStack() as stack:
            for target, value in [('PROC', proc), ('_location', lambda *_: (root, root)),
                ('_read_attempt', lambda *_: (attempt, intent)), ('_identity_matches', lambda *_: True),
                ('_process', process)]:
                stack.enter_context(patch.object(maintenance, target, value))
            stack.enter_context(patch.object(maintenance.socket, 'gethostname', return_value=host))
            stack.enter_context(patch.object(maintenance.os, 'pidfd_open', side_effect=pidfd))
            stack.enter_context(patch.object(maintenance.os, 'close'))
            stack.enter_context(patch.object(maintenance.os, 'kill', side_effect=AssertionError('numeric kill forbidden')))
            stack.enter_context(patch.object(maintenance.signal, 'pidfd_send_signal', side_effect=send))
            optimized = stack.enter_context(patch('corpus_pipeline.cluster.frontpass.optimized_configuration', return_value={'qualified': True}))
            tuned = stack.enter_context(patch('corpus_pipeline.cluster.tuned_configuration.configuration', return_value={'qualified': True}))
            if still_live:
                stack.enter_context(patch.object(maintenance.time, 'monotonic', side_effect=[0, 111]))
            result = maintenance.stop(root, name, root/'maintenance'), calls
            if tuned_receipt:
                optimized.assert_not_called()
                self.assertEqual(tuned.call_count, 2)
            else:
                tuned.assert_not_called()
                self.assertEqual(optimized.call_count, 2)
            return result

    def test_stop_uses_pidfd_and_only_returns_owned_leases_after_all_exit(self):
        receipt, calls = self.run_fixture()
        self.assertEqual(calls, ['pidfd_signal'])
        self.assertTrue(receipt['all_original_processes_exited'])
        self.assertEqual([row['task_id'] for row in receipt['retry_candidates']], ['owned'])
        self.assertEqual(receipt['replacement_mode'], 'optimized')

    def test_tuned_qualification_allows_old_to_new_rollout_and_tuned_to_tuned_maintenance(self):
        for tuned_child in (False, True):
            with self.subTest(tuned_child=tuned_child):
                receipt, calls = self.run_fixture(tuned_receipt=True, tuned_child=tuned_child)
                self.assertEqual(calls, ['pidfd_signal'])
                self.assertEqual(receipt['replacement_mode'], 'tuned')
                self.assertEqual(receipt['current_worker_module'], 'corpus_pipeline.cluster.' +
                                 ('tuned_frontpass' if tuned_child else 'frontpass'))
                self.assertEqual([row['task_id'] for row in receipt['retry_candidates']], ['owned'])

    def test_removing_tuned_receipt_requires_qualified_original_replacement_for_rollback(self):
        receipt, _ = self.run_fixture(tuned_child=True)
        self.assertEqual(receipt['replacement_mode'], 'optimized')

    def test_bad_tuned_receipt_cannot_fall_back_to_optimized_or_signal_any_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'TUNED_QUALIFIED.json').write_text('{}')
            for error in (RuntimeError('tuned qualified code changed'), None):
                with patch('corpus_pipeline.cluster.tuned_configuration.configuration',
                           side_effect=error, return_value=None), \
                     patch('corpus_pipeline.cluster.frontpass.optimized_configuration') as optimized, \
                     patch.object(maintenance.os, 'pidfd_open') as pidfd:
                    with self.assertRaisesRegex(RuntimeError, 'tuned'):
                        maintenance.stop(root, 'accepted', root / 'never-created')
                    optimized.assert_not_called();pidfd.assert_not_called()
                    self.assertFalse((root / 'never-created').exists())

    def test_dangling_tuned_receipt_never_stops_without_valid_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'TUNED_QUALIFIED.json').symlink_to(root / 'missing-receipt.json')
            with patch('corpus_pipeline.cluster.frontpass.optimized_configuration') as optimized, \
                 patch.object(maintenance.os, 'pidfd_open') as pidfd:
                with self.assertRaisesRegex(RuntimeError, 'unreadable or malformed tuned qualification'):
                    maintenance.stop(root, 'accepted', root / 'never-created')
                optimized.assert_not_called()
                pidfd.assert_not_called()
                self.assertFalse((root / 'never-created').exists())

    def test_tuned_entrypoint_never_broadens_root_device_dtype_or_launcher_flags(self):
        for flag, value in (('--root', '/other-run'), ('--device', 'npu:1'), ('--dtype', 'bfloat16'),
                            ('--batch-size', '64'), ('--threads', '8')):
            def mutate(_intent, identities):
                argv = identities[303]['argv']
                argv[argv.index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaisesRegex(RuntimeError, 'Unexpected child'):
                self.run_fixture(tuned_receipt=True, tuned_child=True, mutate=mutate)

    def test_extra_process_on_same_device_is_not_accepted(self):
        def mutate(intent, identities):
            identities[404] = deepcopy(identities[303])
            identities[404].update(pid=404, start_ticks=4040)
            path = Path(intent['root']) / 'proc/404';path.mkdir()
            (path / 'stat').write_text('404 (process) S 202 0 0')
        with self.assertRaisesRegex(RuntimeError, 'Unexpected child'):
            self.run_fixture(tuned_receipt=True, tuned_child=True, mutate=mutate)

    def test_pid_reuse_between_capture_and_pidfd_open_prevents_signal(self):
        with self.assertRaisesRegex(RuntimeError, 'changed before stop'):
            self.run_fixture(reused=True)

    def test_node_argv_must_match_accepted_queue_and_flags(self):
        def mutate(intent, identities):
            identities[202]['argv'][-1] = '99'
        with self.assertRaisesRegex(RuntimeError, 'Node worker identity'):
            self.run_fixture(mutate=mutate)

    def test_live_old_child_never_authorizes_queue_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'remain; queue retry not authorized'):
            self.run_fixture(still_live=True)

    def test_changed_root_or_host_in_intent_prevents_stop(self):
        with self.assertRaisesRegex(RuntimeError, 'attempt identity'):
            self.run_fixture(mutate=lambda intent, _: intent.update(hostname='other-host'))


if __name__ == '__main__': unittest.main()
