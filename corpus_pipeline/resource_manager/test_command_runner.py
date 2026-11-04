from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

from . import command_runner as runner
from .__main__ import main as manager_main


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = 'test-resource'

    def wait(self, predicate, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            result = runner.status(self.root, self.key)
            if predicate(result):
                return result
            time.sleep(.02)
        self.fail('Runner did not reach expected state')

    def test_real_launch_is_idempotent_and_new_completed_operation_can_run(self):
        marker = self.root / 'calls'
        command = [sys.executable, '-c', 'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.open("a").write("once\\n")', str(marker)]
        runner.launch(self.root, self.key, 'operation1', command)
        runner.launch(self.root, self.key, 'operation1', command)
        result = self.wait(lambda value: value['completed'])
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(marker.read_text(), 'once\n')
        runner.launch(self.root, self.key, 'operation2', command)
        self.wait(lambda value: value['completed'] and value['launch_operation_id'] == 'operation2')
        self.assertEqual(marker.read_text(), 'once\nonce\n')

    def test_running_identity_and_readiness_are_independent(self):
        command = [sys.executable, '-c', 'import time; time.sleep(.6)']
        runner.launch(self.root, self.key, 'operation1', command)
        self.wait(lambda value: value['running'])
        result = runner.status(self.root, self.key, [sys.executable, '-c', 'raise SystemExit(1)'])
        self.assertTrue(result['running']); self.assertFalse(result['ready'])
        with self.assertRaisesRegex(RuntimeError, 'Previous operation'):
            runner.launch(self.root, self.key, 'operation2', command)
        self.wait(lambda value: value['completed'])

    def test_same_operation_cannot_change_command_and_argv_is_not_a_shell(self):
        marker = self.root / 'literal'
        literal = '$(touch forbidden); echo injected'
        command = [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])', str(marker), literal]
        runner.launch(self.root, self.key, 'operation1', command)
        self.wait(lambda value: value['completed'])
        self.assertEqual(marker.read_text(), literal)
        with self.assertRaisesRegex(ValueError, 'different command'):
            runner.launch(self.root, self.key, 'operation1', [sys.executable, '-V'])

    def test_nonzero_exit_is_reported_and_drain_has_its_own_operation(self):
        runner.launch(self.root, self.key, 'operation1', [sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(self.wait(lambda value: value['completed'])['exit_code'], 7)
        marker = self.root / 'drained'
        command = [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("ok")', str(marker)]
        runner.launch(self.root, self.key, 'drain1', command, kind='drain')
        result = self.wait(lambda value: value['drained'])
        self.assertEqual(result['drain_operation_id'], 'drain1')
        self.assertEqual(result['launch_operation_id'], 'operation1')
        self.assertEqual(marker.read_text(), 'ok')

    def test_probe_and_runner_cli_schema_match_adapter(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.main(['--root', str(self.root), '--resource-key', self.key,
                         'probe', '--', sys.executable, '-c', 'pass'])
        result = json.loads(output.getvalue())
        self.assertIn('running', result)
        self.assertTrue(result['probe_success'])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            runner.main(['--root', str(self.root), '--resource-key', self.key,
                         'status', '--readiness', json.dumps([sys.executable, '-c', 'pass'])])
        self.assertFalse(json.loads(output.getvalue())['ready'])

    def test_recycled_pid_identity_is_not_running(self):
        identity = runner._identity(__import__('os').getpid())
        self.assertTrue(runner._alive(identity))
        self.assertFalse(runner._alive(identity | {'start_ticks': identity['start_ticks'] + 1}))

    def test_proven_dead_supervisor_and_child_allow_new_id_without_rewriting_history(self):
        runner.launch(self.root, self.key, 'killed', [sys.executable, '-c', 'import time; time.sleep(60)'])
        self.wait(lambda value: value['running'])
        directory = runner.resource_directory(self.root, self.key)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            previous = runner._current(directory, 'launch')
            if previous.get('child'):
                break
            time.sleep(.02)
        self.assertTrue(previous.get('child'))
        os.kill(previous['supervisor']['pid'], signal.SIGKILL)
        os.kill(previous['child']['pid'], signal.SIGKILL)
        self.wait(lambda value: value['process_gone'])
        original = (directory / 'operations/killed.json').read_bytes()
        marker = self.root / 'recovered'
        command = [sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).open("a").write("once")', str(marker)]
        runner.launch(self.root, self.key, 'recovery', command)
        runner.launch(self.root, self.key, 'recovery', command)
        result = self.wait(lambda value: value['completed'] and value['launch_operation_id'] == 'recovery')
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(marker.read_text(), 'once')
        self.assertEqual((directory / 'operations/killed.json').read_bytes(), original)
        self.assertEqual(runner._current(directory, 'launch')['replaces_proven_dead_operation_id'], 'killed')

    def test_incomplete_same_runtime_intent_is_still_unknown(self):
        directory = runner.resource_directory(self.root, self.key)
        runner._save(directory / 'operations/unknown.json', {
            'operation_id': 'unknown', 'kind': 'launch', 'phase': 'intent',
            'runtime_id': runner._runtime_id(), 'command': [sys.executable, '-V']})
        runner._save(directory / 'CURRENT_LAUNCH.json', {'operation_id': 'unknown'})
        with self.assertRaisesRegex(RuntimeError, 'unknown result'):
            runner.launch(self.root, self.key, 'recovery', [sys.executable, '-V'])


class ManagerCliTests(unittest.TestCase):
    def test_plan_status_and_stop_are_real_configured_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({
                'adapter': 'corpus_pipeline.resource_manager.test_engine:FakeAdapter',
                'adapter_config': {}, 'state_dir': 'state',
                'workload': {'scope': 'test', 'profile': 'batch', 'max_cards': 7,
                             'command': [sys.executable, '-V'], 'allowed_cards': [4, 2, 1]}}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                manager_main(['plan', '--config', str(config)])
            self.assertEqual([row['quota']['cards'] for row in json.loads(output.getvalue())['placements']], [4, 2, 1])
            self.assertFalse((root / 'state').exists())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                manager_main(['run', '--once', '--config', str(config)])
            self.assertEqual(json.loads(output.getvalue())['reserved_cards'], 7)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                manager_main(['stop', '--config', str(config)])
            self.assertTrue(json.loads(output.getvalue())['stop_requested'])
            self.assertTrue((root / 'state' / 'STOP_REQUEST.json').exists())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                manager_main(['status', '--config', str(config)])
            self.assertEqual(json.loads(output.getvalue())['scope'], 'test')


if __name__ == '__main__':
    unittest.main()
