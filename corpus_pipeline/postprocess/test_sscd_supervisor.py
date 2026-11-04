"""Bounded orchestration tests; fake stage receipts are never corpus evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import sscd_supervisor as s

WORKDIR = Path(__file__).resolve().parents[2]


class FakeStages:
    def __init__(self, pass_trial=0, fail_stage=None, dishonest_pass=False, interrupt_after=None):
        self.calls = []
        self.pass_trial = pass_trial
        self.fail_stage = fail_stage
        self.dishonest_pass = dishonest_pass
        self.interrupt_after = interrupt_after

    def run(self, supervisor, command, log):
        stage = supervisor.stage
        self.calls.append(supervisor.stage_key)
        if stage == self.fail_stage:
            return 17
        trial = None
        calibration = None
        if stage == 'calibrate':
            number = int(supervisor.stage_key.rsplit('_', 1)[1])
            trial = supervisor.config['calibration_candidates'][number]
        if stage == 'search':
            calibration = Path(command[command.index('--calibration') + 1])
        _, path, identity = supervisor.spec(stage, trial=trial, calibration=calibration)
        value = {'complete': True, 'identity': identity, 'artifacts': [], 'scope': 'full_run'}
        if stage == 'catalog':
            value = {'complete': True, 'scope': 'full_run', 'identity_sha256': s.digest(identity),
                     'input_sha256': identity['input_sha256'], 'vectors': 10_000, 'blocks': []}
        elif stage == 'build':
            value['shards'] = []
        elif stage == 'calibrate':
            passed = self.pass_trial is not None and number >= self.pass_trial
            value.update(passed=passed or self.dishonest_pass, qualified_for_full_run=passed or self.dishonest_pass,
                         recall_at_k=.97 if passed else .4, threshold_positive_recall=.995 if passed else .1,
                         threshold_positive_count=10)
        elif stage == 'search':
            path.parent.mkdir(parents=True, exist_ok=True)
            leaf = path.parent / '000000000000.parquet'
            leaf.write_bytes(b'fixture-only candidate-edge artifact; not actual parquet')
            receipt = path.parent / '000000000000.json'
            s.atomic_json(receipt, {'complete': True, 'artifacts': [{'path': str(leaf.relative_to(supervisor.output)),
                'sha256': s.file_sha(leaf), 'size_bytes': leaf.stat().st_size}]})
            value.update(query_count=10_000, edge_rows=42, automatic_merge=False, training_records_deleted=0,
                         batches=[{'path': str(receipt.relative_to(supervisor.output)), 'sha256': s.file_sha(receipt)}])
        s.atomic_json(path, value)
        if stage == self.interrupt_after:
            raise InterruptedError('Fixture crash after index completion, before supervisor journal')
        return 0


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        upstream = self.root / 'reconciliation'
        s.atomic_json(upstream / 'COMPLETE.json', {'schema': 'corpus.asset_reconciliation.v1',
                      'complete': True, 'scope': 'full_run', 'buckets': []})
        self.config = {'schema_version': s.CONFIG_SCHEMA, 'reconciliation_root': str(upstream),
            'output': str(self.root / 'index'), 'python': sys.executable, 'workdir': str(WORKDIR),
            'disk_headroom_bytes': {stage: 0 for stage in s.STAGES}, 'heartbeat_seconds': 1}
        self.config_path = self.root / 'config.json'
        s.atomic_json(self.config_path, self.config)

    def run_fake(self, fake):
        supervisor = s.Supervisor(self.config_path)
        with patch.object(s.Supervisor, 'run_child', lambda instance, command, log: fake.run(instance, command, log)):
            result = supervisor.run()
        return supervisor, result

    def test_full_workflow_strengthens_calibration_and_resumes_without_rerunning_stages(self):
        fake = FakeStages(pass_trial=1)
        supervisor, result = self.run_fake(fake)
        self.assertEqual(fake.calls, ['catalog', 'train', 'build', 'merge', 'calibrate_00', 'calibrate_01', 'search'])
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['full_query_completed'])
        self.assertFalse(result['near_duplicates_resolved'])
        self.assertEqual(result['training_records_deleted'], 0)
        again = FakeStages()
        _, resumed = self.run_fake(again)
        self.assertEqual(again.calls, [])
        self.assertEqual(resumed['query_count'], 10_000)
        self.assertTrue((supervisor.directory / 'COMPLETE.json').exists())

    def test_failed_calibration_never_searches_or_claims_completion(self):
        fake = FakeStages(pass_trial=None, dishonest_pass=True)
        supervisor, result = self.run_fake(fake)
        self.assertEqual(result['status'], 'await_calibration')
        self.assertNotIn('search', fake.calls)
        self.assertEqual(len(result['calibration_attempts']), 6)
        self.assertFalse(result['full_query_completed'])
        self.assertFalse((supervisor.directory / 'COMPLETE.json').exists())
        again = FakeStages()
        _, second = self.run_fake(again)
        self.assertEqual(second['status'], 'await_calibration')
        self.assertEqual(again.calls, [])

    def test_permanent_child_failure_does_not_loop_and_resume_reuses_prior_stages(self):
        fake = FakeStages(fail_stage='build')
        supervisor, result = self.run_fake(fake)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(fake.calls, ['catalog', 'train', 'build'])
        self.assertIn('exit=17', result['error'])
        self.assertFalse(result['automatic_retry'])
        self.assertFalse((supervisor.directory / 'COMPLETE.json').exists())
        again = FakeStages()
        _, resumed = self.run_fake(again)
        self.assertEqual(resumed['status'], 'complete')
        self.assertEqual(again.calls, ['build', 'merge', 'calibrate_00', 'search'])

    def test_completed_index_stage_without_supervisor_journal_is_revalidated_by_child(self):
        fake = FakeStages(interrupt_after='build')
        supervisor, result = self.run_fake(fake)
        self.assertEqual(result['status'], 'stopped')
        self.assertTrue((supervisor.output / 'shards/COMPLETE.json').exists())
        self.assertFalse((supervisor.directory / 'stages/build.json').exists())
        again = FakeStages()
        _, resumed = self.run_fake(again)
        self.assertEqual(resumed['status'], 'complete')
        self.assertEqual(again.calls[0], 'build')

    def test_missing_or_corrupted_final_leaf_prevents_relaunch_complete(self):
        supervisor, result = self.run_fake(FakeStages())
        self.assertEqual(result['status'], 'complete')
        receipt = s.read_json(supervisor.directory / 'COMPLETE.json')['search_receipt']['path']
        leaf = Path(receipt).parent / '000000000000.parquet'
        original = leaf.read_bytes()
        for mode in ('missing', 'same_size_corruption'):
            with self.subTest(mode=mode):
                if mode == 'missing':
                    leaf.unlink()
                else:
                    leaf.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                again = FakeStages()
                _, result = self.run_fake(again)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['full_query_completed'])
                self.assertEqual(again.calls, [])
                leaf.write_bytes(original)

    def test_changed_checkpoint_config_and_reconciliation_fail_closed(self):
        supervisor, _ = self.run_fake(FakeStages())
        path = supervisor.output / 'train/COMPLETE.json'
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        _, result = self.run_fake(FakeStages())
        self.assertEqual(result['status'], 'failed')
        self.assertIn('checkpoint changed', result['error'])
        path.write_bytes(original)
        changed = {**self.config, 'threshold': .8}
        s.atomic_json(self.config_path, changed)
        _, result = self.run_fake(FakeStages())
        self.assertEqual(result['status'], 'failed')
        self.assertIn('input/config/runtime changed', result['error'])
        s.atomic_json(self.config_path, self.config)
        reconciliation = Path(self.config['reconciliation_root']) / 'COMPLETE.json'
        value = s.read_json(reconciliation); value['changed'] = True; s.atomic_json(reconciliation, value)
        _, result = self.run_fake(FakeStages())
        self.assertEqual(result['status'], 'failed')

    def test_recall_gates_cannot_be_lowered_and_candidates_must_strengthen(self):
        for update in ({'min_recall': .949}, {'min_positive_recall': .98},
                       {'calibration_candidates': [{'nprobe': 64, 'candidate_k': 512, 'ef_search': 128},
                                                   {'nprobe': 32, 'candidate_k': 256, 'ef_search': 128}]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                s.checked_config({**self.config, **update})
        result = s.checked_config(self.config)
        self.assertEqual(result['search_batch_size'], 1024)
        self.assertEqual(result['threads'], 14)
        self.assertEqual(result['python'], sys.executable)

    def test_stop_file_prevents_next_stage_and_low_disk_fails_without_launch(self):
        supervisor = s.Supervisor(self.config_path)
        (supervisor.directory / 'STOP').touch()
        fake = FakeStages()
        _, result = self.run_fake(fake)
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(fake.calls, [])
        (supervisor.directory / 'STOP').unlink()
        config = deepcopy(self.config); config['disk_headroom_bytes']['catalog'] = 10
        # Fresh run needed because disk limits are part of the immutable config.
        config['output'] = str(self.root / 'disk-limited-index'); s.atomic_json(self.config_path, config)
        with patch.object(s.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 0})()):
            _, result = self.run_fake(fake)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('disk headroom', result['error'])
        self.assertEqual(fake.calls, [])

    def test_pid_reuse_never_counts_as_live_and_launch_does_not_duplicate_daemon(self):
        identity = s.process_identity(os.getpid())
        self.assertTrue(s.process_alive(identity))
        self.assertFalse(s.process_alive({**identity, 'start_ticks': identity['start_ticks'] + 1}))
        self.assertFalse(s.process_alive({**identity, 'argv_sha256': '0' * 64}))
        with patch.object(s, 'status', return_value={'supervisor_alive': True}), patch.object(s.subprocess, 'Popen') as popen:
            result = s.launch(self.config_path)
        self.assertFalse(result['launched'])
        popen.assert_not_called()

    def test_inherited_lock_blocks_replacement_until_child_exits(self):
        path = self.root / 'inherited.lock'
        with s.exclusive(path) as stream:
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.15)'], pass_fds=(stream.fileno(),))
        try:
            with self.assertRaises(BlockingIOError):
                with s.exclusive(path):
                    pass
        finally:
            child.wait(timeout=2)
        with s.exclusive(path):
            pass


if __name__ == '__main__':
    unittest.main()
