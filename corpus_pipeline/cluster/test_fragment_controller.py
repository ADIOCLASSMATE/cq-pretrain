from copy import deepcopy
import contextlib
import io
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .fragment_controller import (
    EXCLUDED, GROUP, PROJECT, WORKSPACE, completed, load_submissions,
    Platform, merge_managed, parse_args, parse_worker_receipt, release_jobs, step, validate_live, worker_command,
)


ITEM = {'notebook_id': 'accepted-id', 'name': 'owned-fragment', 'devices': 1,
        'priority': 3, 'node': 'infra-gpu-npu-001.host.shzhisuan.com'}


class FakePlatform:
    def __init__(self, status='RUNNING', worker=None):
        self.live = {'notebook_id': ITEM['notebook_id'], 'name': ITEM['name'],
            'workspace': WORKSPACE, 'project': PROJECT, 'compute_group': GROUP,
            'priority': '3', 'priority_level': 'LOW', 'status': status,
            'resource': {'gpu_count': 1}, 'node': {'name': ITEM['node']}}
        self.worker_value = worker or {'name': ITEM['name'], 'hostname': 'pod-a',
            'status': 'unknown', 'lock_busy': False}
        self.calls = []
        self.start_error = False
        self.stop_error = False
        self.fit = True

    def observe(self, item): return deepcopy(self.live)
    def budget(self, item):
        self.calls.append('budget'); return {'total_with_restart': 166}
    def start(self, item):
        self.calls.append('start')
        if self.start_error: raise TimeoutError('must not persist this URL/token')
    def stop(self, item):
        self.calls.append('stop')
        if self.stop_error: raise TimeoutError('uncertain stop')
    def pending_fit(self, item): return {'fits': self.fit, 'observed_at': 1000}
    def retirement_cooldown(self, item): return 120
    def worker(self, root, item, action):
        self.calls.append(action)
        if action == 'launch':
            return {'name': item['name'], 'hostname': self.worker_value['hostname'],
                    'status': 'running', 'lock_busy': True, 'launched': True,
                    'attempt': 'new', 'devices': 1}
        return deepcopy(self.worker_value)


def run(state, platform, *, now=1000, done=False):
    step(Path('/run'), ITEM, state, platform, is_complete=done, save=lambda: None, now=now)


class FragmentControllerTests(unittest.TestCase):
    def test_automatic_allocation_requires_explicit_valid_card_limit(self):
        base = ['--root', '/run', '--execution', '/execution']
        for extra in ([], ['--max-cards', '0'], ['--max-cards', '-1'],
                      ['--max-cards', '257'], ['--max-cards', '1.5']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(base + extra)
                self.assertEqual(error.exception.code, 2)
        args = parse_args(base + ['--auto-scale', '--max-cards', '48'])
        self.assertTrue(args.auto_scale)
        self.assertEqual(args.max_cards, 48)
        self.assertTrue(args.consolidate)
        self.assertFalse(parse_args(base + ['--max-cards', '48', '--no-consolidate']).consolidate)
        self.assertIsNone(parse_args(base + ['--no-auto-scale']).max_cards)

    def test_restart_counts_pending_and_unknown_reservations_against_pipeline_cap(self):
        module = 'corpus_pipeline.serving.inspire_serving.downsize'
        live = Mock(return_value={'records': {}, 'observable_requested_cards': 100})
        # 47 accepted cards plus an uncertain two-card Create reserve 49.
        allocator = SimpleNamespace(
            managed_budget=Mock(return_value={'max_cards': 48, 'reserved_total_cards': 49}),
            reserved_unconfirmed_cards=lambda: 2)
        platform = Platform(None, [ITEM | {'devices': 47}], allocator)
        with patch.dict(sys.modules, {module: SimpleNamespace(observable_project_budget=live)}):
            with self.assertRaisesRegex(ValueError, 'pipeline card cap'):
                platform.budget(ITEM)
        live.assert_not_called()

    def test_restart_at_pipeline_cap_still_checks_independent_account_ceiling(self):
        module = 'corpus_pipeline.serving.inspire_serving.downsize'
        allocator = SimpleNamespace(
            managed_budget=lambda *_, **__: {'max_cards': 48, 'reserved_total_cards': 48},
            reserved_budget=lambda *_, **__: {
                'managed_budget': {'max_cards': 48, 'reserved_total_cards': 48},
                'managed_missing_from_live_cards': 47, 'uncertain_cards': 1},
            reserved_unconfirmed_cards=lambda: 1)
        platform = Platform(None, [ITEM | {'devices': 47}], allocator)
        live = Mock(return_value={'records': {}, 'observable_requested_cards': 208})
        with patch.dict(sys.modules, {module: SimpleNamespace(observable_project_budget=live)}):
            result = platform.budget(ITEM)
            self.assertEqual(result['total_with_restart'], 256)
            self.assertEqual(result['managed_budget']['reserved_total_cards'], 48)
            live.return_value['observable_requested_cards'] = 209
            with self.assertRaisesRegex(ValueError, '256-card ceiling'):
                platform.budget(ITEM)

    def test_restart_rechecks_run_cap_against_fresh_live_observation(self):
        module = 'corpus_pipeline.serving.inspire_serving.downsize'
        allocator = SimpleNamespace(
            managed_budget=Mock(return_value={'max_cards': 48, 'reserved_total_cards': 48}),
            reserved_budget=Mock(return_value={
                'managed_budget': {'max_cards': 48, 'reserved_total_cards': 49}}),
            reserved_unconfirmed_cards=lambda: 0)
        platform = Platform(None, [ITEM], allocator)
        live = {'records': {'externally-restarted-retired-id': {}},
                'observable_requested_cards': 100}
        with patch.dict(sys.modules, {module: SimpleNamespace(observable_project_budget=lambda *_: live)}):
            with self.assertRaisesRegex(ValueError, 'pipeline card cap'):
                platform.budget(ITEM)
        self.assertEqual(allocator.reserved_budget.call_args.args, (live, platform.managed, platform.states))

    def test_only_exact_low_identity_can_be_mutated(self):
        for key, bad in [('notebook_id', 'unrelated'), ('project', 'other'),
                         ('priority', '6'), ('priority_level', 'HIGH')]:
            platform = FakePlatform('STOPPED'); platform.live[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): run({}, platform)
            self.assertEqual(platform.calls, [])

    def test_excluded_node_never_launched(self):
        platform = FakePlatform(); platform.live['node']['name'] = next(iter(EXCLUDED))
        with self.assertRaises(ValueError): run({}, platform)
        self.assertEqual(platform.calls, [])

    def test_unknown_platform_state_never_started(self):
        platform = FakePlatform('UNKNOWN'); state = {}
        run(state, platform)
        self.assertEqual(platform.calls, [])
        self.assertEqual(state['phase'], 'wait_for_running')

    def test_stopped_checks_budget_before_start(self):
        platform = FakePlatform('STOPPED'); state = {}
        run(state, platform)
        self.assertEqual(platform.calls, ['budget', 'start'])
        self.assertFalse(state['platform_action']['uncertain'])

    def test_uncertain_start_is_not_replayed(self):
        platform = FakePlatform('STOPPED'); platform.start_error = True; state = {}
        with self.assertRaises(TimeoutError): run(state, platform)
        platform.start_error = False
        run(state, platform, now=1100)
        run(state, platform, now=1400)
        self.assertEqual(platform.calls.count('start'), 1)
        self.assertEqual(state['action_required'], 'platform_start_remains_uncertain')
        platform.live['status'] = 'RUNNING'
        run(state, platform, now=1500)
        self.assertNotIn('action_required', state)
        self.assertEqual(platform.calls[-1], 'launch')

    def test_unknown_busy_worker_never_relaunched(self):
        platform = FakePlatform(worker={'name': ITEM['name'], 'hostname': 'pod-a',
                                       'status': 'unknown', 'lock_busy': True})
        run({}, platform)
        self.assertEqual(platform.calls, ['status'])

    def test_fifth_distinct_failure_stops_restarts(self):
        platform = FakePlatform(); state = {}
        for index in range(5):
            platform.worker_value = {'name': ITEM['name'], 'hostname': 'pod-a',
                'status': 'exited', 'lock_busy': False, 'returncode': 1, 'attempt': str(index)}
            run(state, platform, now=1000+index)
        self.assertEqual(state['consecutive_worker_errors'], 5)
        self.assertEqual(state['action_required'], 'five_consecutive_worker_errors')
        self.assertEqual(platform.calls.count('launch'), 4)
        run(state, platform, now=1100)
        self.assertEqual(platform.calls.count('launch'), 4)

    def test_same_failed_attempt_counted_once(self):
        platform = FakePlatform(worker={'name': ITEM['name'], 'hostname': 'pod-a',
            'status': 'exited', 'lock_busy': False, 'returncode': 1, 'attempt': 'a'})
        state = {}; run(state, platform); run(state, platform, now=1100)
        self.assertEqual(state['consecutive_worker_errors'], 1)

    def test_preemption_hostname_change_is_not_model_failure(self):
        platform = FakePlatform(worker={'name': ITEM['name'], 'hostname': 'pod-b',
            'status': 'exited', 'lock_busy': False, 'returncode': -9, 'attempt': 'b'})
        state = {'worker_hostname': 'pod-a', 'consecutive_worker_errors': 4}
        run(state, platform)
        self.assertEqual(state['consecutive_worker_errors'], 0)
        self.assertEqual(platform.calls[-1], 'launch')

    def test_running_short_period_does_not_reset_error_budget(self):
        platform = FakePlatform(worker={'name': ITEM['name'], 'hostname': 'pod-a',
            'status': 'running', 'lock_busy': True, 'attempt': 'a'})
        state = {'consecutive_worker_errors': 4}
        run(state, platform); run(state, platform, now=1100)
        self.assertEqual(state['consecutive_worker_errors'], 4)
        run(state, platform, now=1601)
        self.assertEqual(state['consecutive_worker_errors'], 0)

    def test_complete_stops_only_owned_instance_then_releases(self):
        platform = FakePlatform(); state = {}
        run(state, platform, done=True)
        self.assertEqual(platform.calls, ['stop'])
        platform.live['status'] = 'STOPPED'
        run(state, platform, now=1100, done=True)
        self.assertTrue(state['released'])
        self.assertEqual(platform.calls, ['stop'])

    def test_receipt_filters_arbitrary_output_and_rejects_ambiguity(self):
        value = {'name': ITEM['name'], 'hostname': 'pod-a', 'status': 'running',
                 'devices': 1, 'secret': 'not persisted'}
        receipt = {'completed': True, 'returncode': 0,
                   'output': 'runtime notice\n' + json.dumps(value)}
        self.assertNotIn('secret', parse_worker_receipt(receipt, ITEM))
        receipt['output'] += '\n' + json.dumps(value)
        with self.assertRaises(ValueError): parse_worker_receipt(receipt, ITEM)

    def test_completed_requires_consistent_full_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(completed(root))
            (root/'FRONTPASS_COMPLETE.json').write_text('{}')
            (root/'plan.json').write_text(json.dumps({'record_count': 3, 'snapshot_fingerprint': {}}))
            with self.assertRaises(ValueError): completed(root)

    def test_submission_receipts_fence_ids_and_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'SUBMITTED.json').write_text(json.dumps({'notebooks': [ITEM], 'total_cards': 1}))
            payload = {'name': ITEM['name'], 'gpu_count': 1, 'task_priority': 3,
                'cpu_count': 8, 'memory_size': 64,
                'project_name': PROJECT, 'node_id': ITEM['node'], 'is_publicpath_readonly': True,
                'workspace_id': 'ws-expected', 'project_id': 'p-expected', 'logic_compute_group_id': 'g-expected'}
            (root/(ITEM['name']+'-INTENT.json')).write_text(json.dumps({'create_kwargs': payload}))
            accepted = root/(ITEM['name']+'-ACCEPTED.json')
            accepted.write_text(json.dumps(ITEM))
            self.assertEqual(load_submissions(root)[0]['notebook_id'], ITEM['notebook_id'])
            accepted.write_text(json.dumps(ITEM | {'notebook_id': 'unrelated'}))
            with self.assertRaises(ValueError): load_submissions(root)

    def test_worker_launch_uses_validated_root_and_devices(self):
        command = worker_command(Path('/run with spaces'), ITEM, 'launch')
        self.assertIn("--root '/run with spaces'", command)
        self.assertIn('--devices 1', command)
        self.assertIn('npu_env.sh', command)

    def test_text_progress_cannot_trigger_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'PROGRESS.json').write_text(json.dumps({'processed_records': 21_000_000}))
            self.assertFalse(completed(root, {'all_tasks_completed': True, 'processed_records': 21_000_000}))

    def test_final_sentinel_requires_full_queue_and_index_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {'record_count': 119_010_208, 'task_count': 24235,
                    'snapshot_fingerprint': {'manifest.json': 'frozen'}}
            (root/'plan.json').write_text(json.dumps(plan))
            index = b'{"all_parts": "bound_by_real_monitor"}'
            (root/'frontpass_index.json').write_bytes(index)
            sentinel = {'complete': True, 'scope': 'offline_frontpass_only',
                'counts': {'records': plan['record_count']},
                'snapshot_fingerprint': plan['snapshot_fingerprint'],
                'index_sha256': hashlib.sha256(index).hexdigest()}
            (root/'FRONTPASS_COMPLETE.json').write_text(json.dumps(sentinel))
            queue = {'all_tasks_completed': True, 'task_count': plan['task_count'],
                     'processed_records': plan['record_count'],
                     'counts': {'completed': plan['task_count'], 'pending': 0, 'running': 0, 'failed': 0}}
            self.assertTrue(completed(root, queue))
            queue['counts']['running'] = 1
            with self.assertRaises(ValueError): completed(root, queue)
            queue['counts']['running'] = 0
            (root/'frontpass_index.json').write_bytes(b'changed')
            with self.assertRaises(ValueError): completed(root, queue)

    def test_registry_merge_rejects_rebound_identity(self):
        self.assertEqual(merge_managed([ITEM], [ITEM]), [ITEM])
        with self.assertRaises(ValueError): merge_managed([ITEM], [ITEM | {'devices': 4}])
        with self.assertRaises(ValueError): merge_managed([ITEM], [ITEM | {'notebook_id': 'new-id'}])

    def test_job_release_only_saved_ids_and_uncertain_stop_is_not_replayed(self):
        class Jobs:
            def __init__(self): self.status = 'QUEUING'; self.stopped = []; self.timeout = True
            def observe_job(self, item): return {'status': self.status, 'native_retry': True}
            def stop_job(self, item):
                self.stopped.append(item['job_id'])
                if self.timeout: raise TimeoutError()
        jobs = Jobs(); runtime = {}; items = [{'job_id': 'owned-job', 'name': 'owned'}]
        self.assertFalse(release_jobs(items, runtime, jobs, lambda: None, now=1000))
        self.assertFalse(release_jobs(items, runtime, jobs, lambda: None, now=2000))
        self.assertEqual(jobs.stopped, ['owned-job'])
        jobs.status = 'CANCELLED'
        self.assertTrue(release_jobs(items, runtime, jobs, lambda: None, now=2100))

    def test_exceeded_budget_prevents_restart(self):
        class Full(FakePlatform):
            def budget(self, item): raise ValueError('ceiling')
        platform = Full('STOPPED')
        with self.assertRaises(ValueError): run({}, platform)
        self.assertNotIn('start', platform.calls)

    def test_pending_lost_fit_has_finite_grace_and_durable_stop_intent(self):
        platform = FakePlatform('PENDING'); platform.fit = False
        state = {}; saved = []
        run(state, platform, now=1000)
        run(state, platform, now=1089)
        self.assertEqual(platform.calls, [])
        def stop(item):
            self.assertEqual(saved[-1]['retirement']['phase'], 'stop_sending')
            self.assertEqual(saved[-1]['retirement']['notebook_id'], ITEM['notebook_id'])
            platform.calls.append('stop')
        platform.stop = stop
        step(Path('/run'), ITEM, state, platform, is_complete=False,
             save=lambda: saved.append(deepcopy(state)), now=1090)
        self.assertEqual(platform.calls, ['stop'])
        self.assertEqual(state['retirement']['reason'], 'fresh_node_no_longer_fits')
        self.assertFalse(state['released'])

    def test_max_pending_wait_applies_even_when_capacity_claims_free(self):
        platform = FakePlatform('PENDING'); state = {}
        run(state, platform, now=1000)
        run(state, platform, now=1239)
        self.assertEqual(platform.calls, [])
        run(state, platform, now=1240)
        self.assertEqual(platform.calls, ['stop'])
        self.assertEqual(state['retirement']['reason'], 'pending_deadline_exceeded')

    def test_max_pending_wait_survives_capacity_api_error(self):
        platform = FakePlatform('PENDING'); state = {'pending_since': 700}
        def error(item): raise TimeoutError()
        platform.pending_fit = error
        run(state, platform)
        self.assertEqual(platform.calls, ['stop'])
        self.assertEqual(state['pending_placement_check']['error_type'], 'TimeoutError')

    def test_initial_accepted_notebooks_get_existing_pending_age(self):
        platform = FakePlatform('PENDING'); state = {}
        step(Path('/run'), ITEM | {'accepted_at': 500}, state, platform,
             is_complete=False, save=lambda: None, now=1000)
        self.assertEqual(platform.calls, ['stop'])

    def test_pending_to_running_race_is_reobserved_before_stop(self):
        platform = FakePlatform('PENDING'); state = {'pending_since': 700}
        original = platform.observe
        calls = []
        def observe(item):
            calls.append('get')
            if len(calls) == 2: platform.live['status'] = 'RUNNING'
            return original(item)
        platform.observe = observe
        run(state, platform)
        self.assertEqual(calls, ['get', 'get'])
        self.assertEqual(platform.calls, [])
        self.assertNotIn('retirement', state)
        run(state, platform, now=1060)
        self.assertEqual(platform.calls, ['status', 'launch'])

    def test_running_instances_never_get_stale_pending_retirement(self):
        platform = FakePlatform('RUNNING'); state = {'pending_since': 1}
        run(state, platform)
        self.assertNotIn('stop', platform.calls)
        self.assertNotIn('retirement', state)
        self.assertIsNone(state['pending_since'])

    def test_preempted_running_instance_gets_fresh_pending_grace(self):
        platform = FakePlatform('PENDING'); platform.fit = False
        state = {'platform_status': 'RUNNING', 'pending_since': None}
        old_item = ITEM | {'accepted_at': 1}
        step(Path('/run'), old_item, state, platform, is_complete=False,
             save=lambda: None, now=1000)
        self.assertEqual(state['pending_since'], 1000)
        self.assertEqual(platform.calls, [])
        step(Path('/run'), old_item, state, platform, is_complete=False,
             save=lambda: None, now=1090)
        self.assertEqual(platform.calls, ['stop'])

    def test_uncertain_retirement_survives_restart_and_never_replays_stop(self):
        platform = FakePlatform('PENDING'); platform.stop_error = True
        state = {'pending_since': 700}
        with self.assertRaises(TimeoutError): run(state, platform)
        # JSON round trip represents a controller process restart.
        state = json.loads(json.dumps(state)); platform.stop_error = False
        run(state, platform, now=1500)
        self.assertEqual(platform.calls, ['stop'])
        self.assertFalse(state['released'])
        self.assertEqual(state['action_required'], 'retirement_stop_remains_unconfirmed')
        platform.live['status'] = 'STOPPED'
        run(state, platform, now=1600)
        self.assertTrue(state['released'])
        self.assertEqual(state['retirement']['cooldown_until'], 1720)
        self.assertNotIn('action_required', state)
        run(state, platform, now=2000)
        run(state, platform, now=2100, done=True)
        self.assertEqual(platform.calls, ['stop'])
        self.assertEqual(state['retirement']['cooldown_until'], 1720)

    def test_crash_after_retirement_intent_rechecks_before_send(self):
        platform = FakePlatform('RUNNING')
        state = {'retirement': {'notebook_id': ITEM['notebook_id'], 'node': ITEM['node'],
                  'phase': 'intent', 'cooldown_seconds': 120}}
        run(state, platform)
        self.assertNotIn('stop', platform.calls)
        self.assertNotIn('retirement', state)

    def test_retired_instance_external_restart_keeps_reservation_no_mutation(self):
        platform = FakePlatform('PENDING'); state = {'pending_since': 700}
        run(state, platform)
        platform.live['status'] = 'STOPPED'; run(state, platform, now=1100)
        platform.live['status'] = 'RUNNING'; run(state, platform, now=1200)
        self.assertFalse(state['released'])
        self.assertEqual(platform.calls, ['stop'])
        self.assertEqual(state['action_required'], 'retirement_previously_stopped_now_active')

    def test_retirement_cooldown_exponentially_backs_off_losing_node(self):
        states = {}
        platform = Platform(None, [], states=states)
        for index, expected in enumerate((120, 240, 480, 600, 600)):
            self.assertEqual(platform.retirement_cooldown(ITEM), expected)
            states[str(index)] = {'retirement': {'node': ITEM['node'], 'stop_attempted_at': 1000}}
        states['other-node'] = {'retirement': {'node': 'other', 'stop_attempted_at': 1000}}
        self.assertEqual(platform.retirement_cooldown(ITEM), 600)


if __name__ == '__main__': unittest.main()
