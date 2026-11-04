import socket
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import sscd_recovery_guard as g


class RecoveryTests(unittest.TestCase):
    def test_recall_failure_is_await_calibration_never_restart(self):
        for state in ('alive', 'dead', 'unknown', 'foreign_host'):
            result = g.classify({'phase': 'failed', 'error': 'Full catalog recall gate not met; no unqualified search released'},
                                now=1000, process_status=state, child_status='dead')
            self.assertEqual(result, ('await_calibration', 'full_catalog_recall_gate_failed', False))

    def test_explicit_failures_stops_children_and_unverifiable_pid_fail_closed(self):
        cases = [({'phase': 'failed'}, 'dead', 'dead'), ({'phase': 'stopped'}, 'dead', 'dead'),
                 ({'phase': 'train'}, 'dead', 'alive'), ({'phase': 'train'}, 'unknown', 'dead'),
                 ({'phase': 'train'}, 'foreign_host', 'dead')]
        for progress, co, child in cases:
            self.assertFalse(g.classify(progress, now=1000, process_status=co, child_status=child)[2])
        self.assertTrue(g.classify({'phase': 'search', 'updated_at': 0}, now=1000,
                                   process_status='dead', child_status='dead')[2])
        self.assertFalse(g.classify({'phase': 'search', 'updated_at': 900}, now=1000,
                                    process_status='dead', child_status='dead')[2])

    def test_workers_are_not_active_just_because_fleet_exists(self):
        rows = [{'phase': 'search', 'status': 'running', 'child_pid': 1, 'updated_at': 1000},
                {'phase': 'search', 'status': 'idle', 'child_pid': None, 'updated_at': 1000},
                {'phase': 'search', 'status': 'running', 'child_pid': 2, 'updated_at': 0},
                {'phase': 'search', 'status': 'failed', 'fatal': True, 'updated_at': 1000}]
        counts = g.worker_counts(rows, 'search', 1001)
        self.assertEqual([counts[k] for k in ('active_stage_workers', 'idle_workers', 'stale_workers', 'failed_workers')], [1, 1, 1, 1])
        self.assertEqual(g.worker_counts(rows, 'failed', 1001)['active_stage_workers'], 0)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.cp = self.root / 'config.json'
        g.save(self.cp, {'root': str(self.root), 'output': str(self.root / 'index'),
                        'runtime': str(self.root), 'runtime_sha256': {}, 'threads': 14,
                        'coordinator_host': socket.gethostname(), 'python': '/usr/bin/python3'})
        g.save(self.root / 'controller' / 'INPUT.json', {'config_sha256': g.sha(self.cp)})
        self.launches = []
        self.guard = g.Guard(self.cp, launcher=self.launch)

    def launch(self):
        self.launches.append(True); return {'submitted_pid': 1234}

    def tearDown(self):
        self.tmp.cleanup()

    def progress(self, **kwargs):
        g.save(self.root / 'controller' / 'PROGRESS.json', {'phase': 'search', 'updated_at': 0, **kwargs})

    def test_bounded_recovery_persists_across_guard_restart(self):
        self.progress()
        with patch.object(g, 'local_process', return_value='dead'):
            self.assertEqual(self.guard.tick(1000)['state'], 'recovery_requested')
            self.assertEqual(self.guard.tick(1100)['reason'], 'recovery_cooldown')
            self.assertEqual(self.guard.tick(1300)['state'], 'recovery_requested')
            resumed = g.Guard(self.cp, launcher=self.launch)
            self.assertEqual(resumed.tick(1600)['state'], 'recovery_requested')
            self.assertEqual(resumed.tick(1900)['reason'], 'bounded_recovery_attempts_exhausted')
        self.assertEqual(len(self.launches), 3)
        self.assertTrue(g.read(self.guard.d / 'ALARM.json')['active'])

    def test_failed_recall_emits_machine_readable_alarm_without_launch(self):
        self.progress(phase='failed', error='Full catalog recall gate not met; no unqualified search released')
        with patch.object(g, 'local_process', return_value='dead'):
            for now in (1000, 2000, 3000):
                self.assertEqual(self.guard.tick(now)['state'], 'await_calibration')
        self.assertEqual(self.launches, [])
        self.assertTrue(g.read(self.guard.d / 'ALARM.json')['active'])
        self.assertEqual(g.read(self.root / 'controller' / 'PROGRESS.json')['phase'], 'failed')

    def test_stop_and_changed_configuration_never_launch(self):
        self.progress(); (self.root / 'controller' / 'STOP').touch()
        with patch.object(g, 'local_process', return_value='dead'):
            self.assertEqual(self.guard.tick(1000)['state'], 'stopped')
        self.assertEqual(self.launches, [])
        self.cp.write_text(self.cp.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            self.guard.tick(2000)


if __name__ == '__main__':
    unittest.main()
