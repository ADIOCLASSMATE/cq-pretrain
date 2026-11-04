"""Real local detachment/locking tests with a copied fixture-only CPU child.

No Notebook platform access, model import, production run or NPU is involved.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from . import notebook_worker as worker


class NotebookWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.run = self.base / 'run'
        self.run.mkdir()
        for name in ('QUALIFIED.json', 'queue_rpc.json'):
            (self.run / name).write_text('{}')
        # Copy the exact production supervisor into an isolated import root;
        # replace only its npu_env.sh sibling with a bounded CPU fixture script.
        package = self.base / 'corpus_pipeline'
        for directory in (package, package / 'cluster', package / 'production'):
            directory.mkdir(exist_ok=True)
            (directory / '__init__.py').write_text('')
        shutil.copy2(worker.__file__, package / 'cluster' / 'notebook_worker.py')
        source_common = Path(worker.__file__).parents[1] / 'production' / 'common.py'
        shutil.copy2(source_common, package / 'production' / 'common.py')
        self.fixture = package / 'cluster' / 'npu_env.sh'
        self.fixture.write_text('#!/bin/bash\nexec "' + sys.executable + '" -c '
            + "'import time; print(\"fixture child\", flush=True); time.sleep(3)'\n")
        self.name = 'fixture-notebook'
        self.supervisors = []

    def tearDown(self):
        for pid in self.supervisors:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(worker._process(pid) for pid in self.supervisors):
            time.sleep(.05)
        self.temp.cleanup()
        # The production lock must never be unlinked during use. Fixture roots
        # are unique and all children have ended before this test-only cleanup.
        if not any(worker._process(pid) for pid in self.supervisors):
            worker._lock_path(self.run).unlink(missing_ok=True)

    def command(self, operation, *, devices=1):
        command = [sys.executable, '-m', worker.MODULE, operation,
                   '--root', str(self.run), '--name', self.name]
        if operation == 'launch':
            command += ['--devices', str(devices)]
        result = subprocess.run(command, cwd=self.base, capture_output=True, text=True,
                                timeout=10, check=True)
        value = json.loads(result.stdout)
        if value.get('pid') and value['pid'] not in self.supervisors:
            self.supervisors.append(value['pid'])
        return value

    def wait_exit(self, timeout=6):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.command('status')
            if value['status'] == 'exited' and not value['lock_busy']:
                return value
            time.sleep(.05)
        self.fail('fixture supervisor failed to publish exit and release lock')

    def test_detached_launch_survives_terminal_exit_is_idempotent_and_records_exit(self):
        first = self.command('launch')
        self.assertEqual(first['status'], 'running')
        self.assertTrue(first['launched'])
        # subprocess.run has already exited: the detached process survives it.
        self.assertEqual(os.getsid(first['pid']), first['pid'])
        second = self.command('launch')
        self.assertFalse(second['launched'])
        self.assertEqual(second['pid'], first['pid'])
        self.assertEqual(second['attempt'], first['attempt'])
        exited = self.wait_exit()
        self.assertEqual(exited['returncode'], 0)
        self.assertNotIn('supervisor_command', exited)
        directory = self.run / 'notebooks' / self.name / socket.gethostname() / first['attempt']
        self.assertIn('fixture child', (directory / 'node_worker.log').read_text())
        third = self.command('launch')
        self.assertTrue(third['launched'])
        self.assertNotEqual(third['attempt'], first['attempt'])

    def test_concurrent_launches_create_only_one_attempt(self):
        with ThreadPoolExecutor(4) as executor:
            values = list(executor.map(lambda _: self.command('launch'), range(4)))
        self.assertEqual(sum(value['launched'] for value in values), 1)
        base = self.run / 'notebooks' / self.name / socket.gethostname()
        attempts = [path for path in base.iterdir() if path.is_dir()]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(self.command('status')['status'], 'running')

    def test_pid_reuse_or_wrong_command_is_unknown_and_busy_lock_prevents_duplicate(self):
        first = self.command('launch')
        directory = self.run / 'notebooks' / self.name / socket.gethostname() / first['attempt']
        path = directory / 'OWNER.json'
        owner = json.loads(path.read_text())
        owner['start_ticks'] += 1
        path.write_text(json.dumps(owner))
        value = self.command('launch')
        self.assertEqual(value['status'], 'unknown')
        self.assertTrue(value['lock_busy'])
        self.assertFalse(value['launched'])
        self.assertEqual(value['attempt'], first['attempt'])

    def test_new_hostname_does_not_consume_previous_hostname_receipt(self):
        self.command('launch')
        with patch.object(socket, 'gethostname', return_value='replacement-container'):
            value = worker.status(self.run, self.name)
        self.assertEqual(value['status'], 'unknown')
        self.assertTrue(value['lock_busy'])  # Same test container; actual replacement has its own /tmp.
        self.assertNotIn('pid', value)

    def test_signal_stops_child_and_publishes_nonzero_exit(self):
        first = self.command('launch')
        os.kill(first['pid'], signal.SIGTERM)
        exited = self.wait_exit()
        self.assertNotEqual(exited['returncode'], 0)
        directory = self.run / 'notebooks' / self.name / socket.gethostname() / first['attempt']
        self.assertTrue(json.loads((directory / 'EXIT.json').read_text())['termination_requested'])

    def test_failed_node_exit_is_relaunchable(self):
        self.fixture.write_text('#!/bin/bash\nexit 23\n')
        first = self.command('launch')
        exited = self.wait_exit()
        self.assertEqual(exited['returncode'], 23)
        second = self.command('launch')
        self.assertTrue(second['launched'])
        self.assertNotEqual(second['attempt'], first['attempt'])

    def test_killed_supervisor_cannot_duplicate_a_surviving_node(self):
        first = self.command('launch')
        directory = self.run / 'notebooks' / self.name / socket.gethostname() / first['attempt']
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            owner = json.loads((directory / 'OWNER.json').read_text())
            if owner.get('node_pid'):
                break
            time.sleep(.02)
        self.assertIn('node_pid', owner)
        os.kill(first['pid'], signal.SIGKILL)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and worker._process(first['pid']):
            time.sleep(.02)
        second = self.command('launch')
        self.assertEqual(second['status'], 'unknown')
        self.assertTrue(second['lock_busy'])
        self.assertFalse(second['launched'])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = self.command('status')
            if not current['lock_busy']:
                break
            time.sleep(.05)
        self.assertFalse(current['lock_busy'])
        self.assertEqual(current['status'], 'unknown')  # SIGKILL cannot publish EXIT.
        self.assertTrue(self.command('launch')['launched'])

    def test_process_identity_rejects_unrelated_pid_without_exposing_argv(self):
        actual = worker._process(os.getpid())
        self.assertIsNotNone(actual)
        owner = {key: value for key, value in actual.items() if key != 'argv'}
        self.assertFalse(worker._identity_matches(owner, actual['argv']))
        self.assertIsNone(worker._process(-1))

    def test_invalid_launch_arguments_never_spawn(self):
        with patch.object(subprocess, 'Popen') as popen:
            for name, devices in (('../escape', 1), ('safe', 0), ('safe', True), ('safe', 17)):
                with self.assertRaises(ValueError):
                    worker.launch(self.run, devices, name)
            popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
