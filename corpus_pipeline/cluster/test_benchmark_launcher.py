"""Detached runner lifecycle tests; no models or platform resources used."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from . import benchmark_launcher as launcher


class BenchmarkLauncherTests(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name); directory = root / 'benchmark'; directory.mkdir()
        spec = {'command': [sys.executable, '-m', 'corpus_pipeline.cluster.tuning_probe', '--help'],
                'timeout': 1, 'hostname': socket.gethostname()}
        (directory / 'INTENT.json').write_text(json.dumps(spec))
        return root, directory

    def fake_child(self, waits):
        child = SimpleNamespace(pid=999991, returncode=None)
        events = iter(waits)
        def wait(**_):
            result = next(events)
            if isinstance(result, BaseException): raise result
            child.returncode = result
            return result
        child.wait = MagicMock(side_effect=wait)
        child.poll = lambda: child.returncode
        return child

    def exercise(self, waits):
        root, directory = self.fixture()
        child = self.fake_child(waits)
        with patch.object(launcher, 'LOCK_PATH', root / 'host.lock'), \
             patch.object(launcher.subprocess, 'Popen', return_value=child) as popen, \
             patch.object(launcher.os, 'killpg') as kill:
            code = launcher.run(directory)
            kwargs = popen.call_args.kwargs
            self.assertEqual(len(kwargs['pass_fds']), 1)
            self.assertTrue(kwargs['start_new_session'])
            self.assertIn('--child-spec', popen.call_args.args[0])
            receipt = json.loads((directory / 'EXIT.json').read_text())
            self.assertEqual(receipt['returncode'], code)
            return code, receipt, kill.call_args_list

    def test_success_and_child_failure_always_emit_exit(self):
        for code in (0, 2):
            actual, receipt, signals = self.exercise([code])
            self.assertEqual(actual, code)
            self.assertEqual(receipt['success'], code == 0)
            self.assertEqual(signals, [])

    def test_timeout_term_cleanup_and_exit_sentinel(self):
        code, receipt, signals = self.exercise([subprocess.TimeoutExpired('probe', 1), -15])
        self.assertEqual(code, 124)
        self.assertFalse(receipt['success'])
        self.assertEqual(receipt['child_returncode'], -15)
        self.assertEqual([call.args[1] for call in signals], [signal.SIGTERM])

    def test_timeout_escalates_only_owned_child_group(self):
        code, receipt, signals = self.exercise([subprocess.TimeoutExpired('probe', 1),
                                                subprocess.TimeoutExpired('probe', 10), -9])
        self.assertEqual(code, 124)
        self.assertEqual([call.args for call in signals], [(999991, signal.SIGTERM), (999991, signal.SIGKILL)])
        self.assertEqual(receipt['child_returncode'], -9)

    def test_interrupted_owner_still_cleans_and_emits_exit(self):
        code, receipt, _ = self.exercise([InterruptedError('TERM'), -15])
        self.assertEqual(code, 130)
        self.assertFalse(receipt['success'])

    def test_busy_host_lock_never_spawns_another_probe(self):
        root, directory = self.fixture()
        path = root / 'host.lock'
        with path.open('a') as holder, patch.object(launcher, 'LOCK_PATH', path), \
             patch.object(launcher.subprocess, 'Popen') as popen:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(launcher.run(directory), 75)
            popen.assert_not_called()
        self.assertFalse(json.loads((directory / 'EXIT.json').read_text())['success'])

    def test_failed_spawn_and_malformed_intent_emit_exit(self):
        root, directory = self.fixture()
        with patch.object(launcher, 'LOCK_PATH', root / 'host.lock'), \
             patch.object(launcher.subprocess, 'Popen', side_effect=OSError('spawn failed')):
            self.assertEqual(launcher.run(directory), 1)
        self.assertEqual(json.loads((directory / 'EXIT.json').read_text())['error_type'], 'OSError')
        root, directory = self.fixture()
        spec = json.loads((directory / 'INTENT.json').read_text()); spec['timeout'] = 0
        (directory / 'INTENT.json').write_text(json.dumps(spec))
        with patch.object(launcher.subprocess, 'Popen') as popen:
            self.assertEqual(launcher.run(directory), 1)
            popen.assert_not_called()
        self.assertEqual(json.loads((directory / 'EXIT.json').read_text())['error_type'], 'ValueError')

    def test_duplicate_runner_cannot_replace_completed_receipt(self):
        root, directory = self.fixture()
        child = self.fake_child([0])
        with patch.object(launcher, 'LOCK_PATH', root / 'host.lock'), \
             patch.object(launcher.subprocess, 'Popen', return_value=child):
            self.assertEqual(launcher.run(directory), 0)
        before = (directory / 'EXIT.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'already started'):
            launcher.run(directory)
        self.assertEqual(before, (directory / 'EXIT.json').read_bytes())

    def test_parent_death_guard_precedes_exec_and_retains_lock(self):
        _, directory = self.fixture()
        libc = MagicMock(); libc.prctl.return_value = 0
        with patch.object(launcher.ctypes, 'CDLL', return_value=libc), \
             patch.object(launcher.os, 'getppid', return_value=100), \
             patch.object(launcher.os, 'set_inheritable') as inherit, \
             patch.object(launcher.os, 'execv') as execute:
            launcher.child_exec(directory, 100, 7)
            libc.prctl.assert_called_once_with(1, signal.SIGTERM, 0, 0, 0)
            inherit.assert_called_once_with(7, True)
            self.assertEqual(execute.call_args.args[0], sys.executable)
        with patch.object(launcher.ctypes, 'CDLL', return_value=libc), \
             patch.object(launcher.os, 'getppid', return_value=1), \
             patch.object(launcher.os, 'execv') as execute:
            with self.assertRaisesRegex(RuntimeError, 'owner exited'):
                launcher.child_exec(directory, 100, 7)
            execute.assert_not_called()


if __name__ == '__main__': unittest.main()
