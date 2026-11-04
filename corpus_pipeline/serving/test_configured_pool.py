"""Fake SDK tests for the configurable LOW pool; never submit platform work."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import configured_pool as module
from .infra_tuning_20261024.test_high_lab import FakeServings, payload


class Servings(FakeServings):
    def __init__(self):
        super().__init__()
        self.starts = []; self.event_rows = []; self.start_hook = None

    def start(self, ref, **kwargs):
        self.starts.append(ref.key)
        if self.start_hook:
            self.start_hook()
        self.get(ref).status = 'PENDING'

    def events(self, ref, **kwargs):
        return SimpleNamespace(items=self.event_rows)


class ConfiguredPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.lab = self.directory / 'lab'; self.old = self.directory / 'old'; self.root = self.directory / 'new'
        for directory in (self.lab / 'runtime', self.lab / 'profiles', self.old, self.root):
            directory.mkdir(parents=True)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(module, '_NATIVE_SCOPE', None))
        self.stack.enter_context(patch.object(module.native, 'RUNTIME', self.root))
        self.servings = Servings()
        class Client:
            servings = self.servings
            def __enter__(self): return self
            def __exit__(self, *_): pass
        self.client = Client()
        self.stack.enter_context(patch.object(module, 'InspireClient', side_effect=lambda **_: Client()))
        self.stack.enter_context(patch.object(module, 'ServingCreateSpec', side_effect=lambda **kw: SimpleNamespace(**kw)))
        self.budget = self.stack.enter_context(patch.object(module, 'observable_project_budget', return_value={'records': {}, 'ceiling': 256}))
        self.base = {'schema_version': 'corpus_pipeline.native_serving.v1', 'role': 'generator',
            'model': 'Qwen3.5-397B-A17B', 'revision': 'revision-fixed', 'processor_revision': 'processor-fixed',
            'precision': 'BF16', 'max_images': 16, 'api_key_file': '/fixture/no-credential', 'api_key_env': 'INF_API_KEY',
            'request_overrides': {}, 'name': 'old-owned', 'service_name': 'old-owned', 'service_id': 'owned-old',
            'payload': payload('old-owned', 3), 'tier': 'low', 'priority': 3, 'cards': 32,
            'expected_nodes': 2, 'expected_replicas': 1}
        module.native.save(self.old / 'qwen-01.json', self.base)
        (self.old / 'DISABLED').touch()
        module.native.save(self.old / 'STOPPED.json', {'complete': True, 'services': [{'service_id': 'owned-old'}]})
        self.servings.add(self.base['payload'], 'owned-old', 'STOPPED')
        for name in module.RUNTIME_FILES:
            (self.lab / 'runtime' / name).write_text('fixture ' + name)
        self.profile = {'model_name': self.base['model'], 'revision': self.base['revision'],
            'processor_revision': self.base['processor_revision'], 'precision': 'BF16', 'nodes': 2,
            'data_parallel_size': 4, 'epoch': 'infra-20261024', 'max_model_len': 65536, 'max_images': 16,
            'max_num_seqs': 64, 'max_num_batched_tokens': 4096, 'cudagraph_capture_sizes': [1,2,4,8,16,32,64]}
        module.native.save(self.lab / 'profiles/s64.json', self.profile)
        files = {str(p.relative_to(self.lab)): module.sha(p) for sub in ('runtime', 'profiles') for p in (self.lab / sub).iterdir()}
        module.native.save(self.lab / 'RUNTIME_MANIFEST.json', {'files': files})
        (self.lab / 'high_lab.py').write_text('# Tested HIGH gate fixture: PRODUCTION_HANDOFF.json\n')
        self.gate()
        self.raw = {'schema': 'corpus.native_serving_pool.v1', 'scope_root': str(self.root),
            'name_prefix': 'corpus-review-test', 'runtime_dir': str(self.lab / 'runtime'),
            'runtime_manifest_path': str(self.lab / 'RUNTIME_MANIFEST.json'),
            'runtime_manifest_sha256': module.sha(self.lab / 'RUNTIME_MANIFEST.json'),
            'profile_path': str(self.lab / 'profiles/s64.json'), 'profile_sha256': module.sha(self.lab / 'profiles/s64.json'),
            'baseline_state_path': str(self.old / 'qwen-01.json'), 'retired_pool_roots': [str(self.old)],
            'retired_lab_root': str(self.lab), 'high_gate_receipt_path': str(self.lab / 'HANDOFF_GATE_READY.json'),
            'replicas': 4, 'max_cards': 128, 'cards_per_replica': 32, 'priority': 3}
        self.config = self.root / 'config.json'; module.native.save(self.config, self.raw)
        self.pool = module.NativeServingPool(module.PoolSpec.load(self.config))

    def gate(self):
        module.native.save(self.lab / 'HANDOFF_GATE_READY.json', {
            'schema': 'corpus.high_lab.handoff_gate.v1', 'enabled': True, 'tests_passed': True,
            'gate_inside_lifecycle_lock': True, 'gate_marker': 'PRODUCTION_HANDOFF.json',
            'entry_path': str(self.lab / 'high_lab.py'), 'entry_sha256': module.sha(self.lab / 'high_lab.py'),
            'tested_entry_sha256': module.sha(self.lab / 'high_lab.py')})

    def evidence(self, number=1, *, boot='boot-one', changed=False):
        state = self.pool.load(number)
        config = {k: v for k, v in state['config_snapshot'].items() if 'key' not in k and k != 'processor_files'}
        if changed: config['max_num_seqs'] = 128
        report = {'runtime_config': config, 'serving_code_revision': state['binding']['serving_code_revision']}
        return {'incarnation': boot, 'expected_boots': [boot], 'runtime_revision': 'fixture-runtime',
                'reports': [deepcopy(report), deepcopy(report)]}

    def probe(self, state, evidence):
        state['qualification_receipt'] = str(self.root / 'fixture-qualified.json')
        state['probe_status'] = 'PASSED'
        return True

    def qualify(self, number=1):
        state = self.pool.load(number)
        self.servings.resources[state['name']].status = 'RUNNING'
        with patch.object(module.native, 'attest', return_value=self.evidence(number)), \
                patch.object(module.native, 'health', return_value=True), \
                patch.object(module.native, 'probe', side_effect=self.probe):
            self.assertTrue(self.pool.observe_once(number)['canary_passed'])

    def expansion(self):
        with patch.object(module.native, 'attest', return_value=self.evidence()), \
                patch.object(module.native, 'health', return_value=True):
            return self.pool.provision()

    def high(self, *, released=True, known=True):
        name = 'high-owned'; body = payload(name, 6)
        state = {**self.base, 'scope': str(self.lab), 'profile': 's64', 'name': name, 'service_name': name,
            'operation_id': name, 'payload': body, 'tier': 'high', 'priority': 6, 'binding': {'profile': 'frozen'},
            'deployment_id': 'high-deploy', 'release_verified': released, 'service_id': 'owned-high'}
        if not known: state.pop('service_id')
        module.native.save(self.lab / 'intents/s64.json', {k: v for k, v in state.items() if k != 'service_id'})
        if released:
            receipt = self.lab / 'observations/s64/STOPPED.json'
            module.native.save(receipt, {'complete': True, 'service_id': 'owned-high', 'binding': state['binding']})
            state['release_receipt_sha256'] = module.sha(receipt)
        module.native.save(self.lab / 'states/s64.json', state)
        if known: self.servings.add(body, 'owned-high', 'STOPPED' if released else 'RUNNING')
        return state

    def preempt(self):
        state = self.pool.load(1)
        self.servings.resources[state['name']].status = 'STOPPED'
        self.servings.event_rows = [{'reason': 'Preempted', 'message': 'was preempted by higher priority',
            'created_at': int(time.time() * 1000) + 1, 'object_id': state['service_id'] + '/pod'}]
        # Event timestamp must be after the last RUNNING sample but not in the future.
        time.sleep(0.005)
        return state

    def test_first_create_returns_pending_and_handoff_is_bound(self):
        result = self.pool.provision()
        self.assertEqual(result, {'reserved_cards': 32, 'replicas': 1})
        state = self.pool.load(1); spec, operation = self.servings.creates[0]
        self.assertEqual((spec.priority, spec.nodes_per_replica, spec.quota), (3, 2, '16,128,1024'))
        self.assertIn('CORPUS_CONFIG=' + str(self.lab / 'profiles/s64.json'), spec.command)
        self.assertEqual(operation, state['name']); self.assertEqual(state['role'], 'generator')
        handoff = module.native.load(self.lab / 'PRODUCTION_HANDOFF.json')
        self.assertEqual(handoff['config_sha256'], module.sha(self.config))
        self.assertEqual(handoff['guard_command'], self.pool.command('guard'))
        self.assertNotIn('endpoint', state)
        self.pool.provision(1); self.assertEqual(len(self.servings.creates), 1)

    def test_missing_or_changed_high_gate_blocks_all_creates(self):
        (self.lab / 'HANDOFF_GATE_READY.json').unlink()
        with self.assertRaises(FileNotFoundError): self.pool.provision()
        self.gate(); (self.lab / 'high_lab.py').write_text('changed gate')
        with self.assertRaisesRegex(ValueError, 'gate'): self.pool.provision()
        self.assertEqual(self.servings.creates, [])

    def test_gate_must_be_tested_same_source_under_lifecycle_lock(self):
        for field in ('tests_passed', 'gate_inside_lifecycle_lock', 'enabled'):
            self.gate(); path = self.lab / 'HANDOFF_GATE_READY.json'; value = module.native.load(path)
            value[field] = False; module.native.save(path, value)
            with self.assertRaisesRegex(ValueError, 'gate'): self.pool.provision()
        self.assertEqual(self.servings.creates, [])

    def test_high_unknown_and_stale_release_never_free_budget(self):
        self.high(released=False, known=False)
        with self.assertRaisesRegex(ValueError, 'unresolved'): self.pool.provision()
        self.high(released=True)
        self.servings.resources['high-owned'].status = 'RUNNING'
        with self.assertRaisesRegex(ValueError, 'HIGH allocation'): self.pool.provision()
        self.servings.resources['high-owned'].status = 'STOPPED'
        self.servings.instance_states['owned-high'] = ['INSTANCE_UNKNOWN']
        with self.assertRaisesRegex(ValueError, 'HIGH allocation'): self.pool.provision()
        self.assertEqual(self.servings.creates, [])

    def test_old_low_disabled_and_fresh_terminal_release_required(self):
        (self.old / 'DISABLED').unlink()
        with self.assertRaisesRegex(ValueError, 'disabled'): self.pool.provision()
        (self.old / 'DISABLED').touch(); self.servings.resources['old-owned'].status = 'RUNNING'
        with self.assertRaisesRegex(ValueError, 'fully stopped'): self.pool.provision()
        self.assertEqual(self.servings.creates, [])

    def test_plan_race_rechecks_predecessors_before_transmission(self):
        self.servings.plan_hook = lambda _: setattr(self.servings.resources['old-owned'], 'status', 'RUNNING')
        with self.assertRaisesRegex(ValueError, 'fully stopped'): self.pool.provision()
        self.assertEqual(self.servings.creates, []); self.assertEqual(self.pool.numbers(), [])

    def test_handoff_cannot_be_reused_by_a_different_config(self):
        self.pool.provision(1)
        handoff = self.lab / 'PRODUCTION_HANDOFF.json'; value = module.native.load(handoff)
        value['config_sha256'] = 'another'; module.native.save(handoff, value)
        with self.assertRaisesRegex(ValueError, 'another pool'): self.pool.provision(1)
        self.assertEqual(len(self.servings.creates), 1)

    def test_four_replicas_only_after_current_boot_qualified(self):
        self.pool.provision()
        with self.assertRaisesRegex(ValueError, 'qualification'): self.pool.provision()
        self.qualify(); self.assertEqual(self.expansion()['reserved_cards'], 128)
        self.assertEqual(len(self.servings.creates), 4)
        with self.assertRaisesRegex(ValueError, 'exceed'): self.pool.provision(5)
        with self.assertRaisesRegex(ValueError, 'cap'): self.pool.budget(self.client, 32)
        registry = module.native.load(self.root / 'registry-01.json')['endpoints'][0]
        self.assertEqual(registry['role'], 'verifier'); self.assertTrue(registry['ready'])

    def test_first_boot_changes_block_expansion_even_with_cached_qualification(self):
        self.pool.provision(1); self.qualify()
        with patch.object(module.native, 'attest', return_value=self.evidence(boot='new-boot')), \
                patch.object(module.native, 'health', return_value=True):
            with self.assertRaisesRegex(ValueError, 'expired'): self.pool.provision()
        self.assertEqual(len(self.servings.creates), 1)

    def test_unknown_create_not_repeated_even_if_mutable_state_is_missing(self):
        self.servings.create_error = TimeoutError('fake transport')
        with self.assertRaises(TimeoutError): self.pool.provision(1)
        self.pool.paths(1)[0].unlink()
        with self.assertRaisesRegex(ValueError, 'unresolved'): self.pool.provision(1)
        self.assertEqual(len(self.servings.creates), 1)
        self.assertEqual(self.pool.budget(self.client)['pool_reserved_cards'], 32)
        self.assertEqual(self.pool.budget(self.client)['unreflected_reserved_cards'], 32)

    def test_lost_create_reply_adopts_exact_owned_id_without_retransmission(self):
        self.servings.create_error = TimeoutError('fake transport'); self.servings.accept_before_error = True
        with self.assertRaises(TimeoutError): self.pool.provision(1)
        self.pool.provision(1)
        self.assertTrue(self.pool.load(1)['service_id']); self.assertEqual(len(self.servings.creates), 1)

    def test_account_cap_adds_unobserved_reserved_cards_without_double_count(self):
        self.pool.provision(1); state = self.pool.load(1)
        self.budget.return_value = {'records': {'other': {'cards': 224}}}
        with self.assertRaisesRegex(ValueError, 'Account budget'): self.pool.budget(self.client, 32)
        self.budget.return_value = {'records': {'other': {'cards': 192}, state['service_id']: {'cards': 32}}}
        self.assertEqual(self.pool.budget(self.client, 32)['conservative_total_cards'], 256)

    def test_loaded_profile_mismatch_and_new_boot_with_pending_probe_are_not_published(self):
        self.pool.provision(1); self.qualify()
        with patch.object(module.native, 'attest', return_value=self.evidence(changed=True)), \
                patch.object(module.native, 'probe') as probe:
            self.assertFalse(self.pool.observe_once(1)['canary_passed']); probe.assert_not_called()
        with patch.object(module.native, 'attest', return_value=self.evidence(boot='new-boot')), \
                patch.object(module.native, 'health', return_value=True), patch.object(module.native, 'probe', return_value=False):
            self.assertFalse(self.pool.observe_once(1)['canary_passed'])
        self.assertIsNone(self.pool.load(1)['qualification_receipt'])
        self.assertFalse(module.native.load(self.root / 'registry-01.json')['endpoints'][0]['ready'])

    def test_profile_modification_blocks_create_and_ready_but_not_owned_stop(self):
        self.pool.provision(1)
        profile = self.lab / 'profiles/s64.json'; value = module.native.load(profile)
        value['precision'] = 'INT8'; module.native.save(profile, value)
        with self.assertRaisesRegex(ValueError, 'profile changed'): self.pool.provision(1)
        self.assertFalse(self.pool.observe_once(1)['canary_passed'])
        self.assertTrue(self.pool.stop(timeout=0)['complete'])

    def test_manual_stop_does_not_restart_and_preempt_event_restarts_only_same_id_once(self):
        self.pool.provision(1); self.qualify(); state = self.pool.load(1)
        self.servings.resources[state['name']].status = 'STOPPED'
        self.pool.observe_once(1); self.assertEqual(self.servings.starts, [])
        # Re-observe a RUNNING boot so the following event has a fresh transition.
        self.qualify(); self.preempt()
        self.pool.observe_once(1); self.assertEqual(self.servings.starts, [state['service_id']])
        self.servings.resources[state['name']].status = 'STOPPED'
        self.pool.observe_once(1); self.assertEqual(self.servings.starts, [state['service_id']])
        self.assertEqual(len(self.servings.creates), 1)

    def test_disabled_during_recovery_budget_prevents_start(self):
        self.pool.provision(1); self.qualify(); self.preempt()
        def disable(*_):
            (self.root / 'DISABLED').touch()
            return {'records': {}}
        self.budget.side_effect = disable
        self.pool.observe_once(1)
        self.assertEqual(self.servings.starts, [])

    def test_inflight_start_finishes_before_stop_and_no_late_restart(self):
        self.pool.provision(1); self.qualify(); state = self.preempt()
        entered = threading.Event(); finish = threading.Event(); stopped = threading.Event(); errors = []
        def starting():
            entered.set()
            if not finish.wait(3): raise TimeoutError('test did not unblock Start')
        self.servings.start_hook = starting
        worker = threading.Thread(target=lambda: self.pool.observe_once(1)); worker.start()
        self.assertTrue(entered.wait(3))
        def stopping():
            try: self.pool.stop(timeout=0)
            except Exception as exc: errors.append(exc)
            finally: stopped.set()
        stopper = threading.Thread(target=stopping); stopper.start()
        deadline = time.monotonic() + 3
        while not self.pool.disabled() and time.monotonic() < deadline: time.sleep(.005)
        self.assertTrue(self.pool.disabled()); self.assertFalse(stopped.is_set()); self.assertEqual(self.servings.stops, [])
        finish.set(); worker.join(3); stopper.join(3)
        self.assertFalse(errors); self.assertTrue(stopped.is_set())
        self.assertEqual(self.servings.starts, [state['service_id']]); self.assertEqual(self.servings.stops, [state['service_id']])
        self.pool.observe_once(1); self.assertEqual(len(self.servings.starts), 1)

    def test_stop_rejects_identity_drift_and_unknown_instance_release(self):
        self.pool.provision(1); state = self.pool.load(1); detail = self.servings.resources[state['name']]
        detail.ref.key = 'not-owned'
        with self.assertRaisesRegex(RuntimeError, 'not verified'): self.pool.stop(timeout=0)
        self.assertEqual(self.servings.stops, [])
        self.assertFalse(module.native.load(self.root / 'STOPPED.json')['complete'])
        detail.ref.key = state['service_id']; detail.status = 'STOPPED'
        self.servings.instance_states[state['service_id']] = ['INSTANCE_UNKNOWN']
        with self.assertRaisesRegex(RuntimeError, 'not verified'): self.pool.stop(timeout=0)
        self.assertFalse(module.native.load(self.root / 'STOPPED.json')['complete'])

    def test_stop_only_owned_resources_and_guard_uses_this_config(self):
        self.pool.provision(1); state = self.pool.load(1)
        self.servings.add(payload('unrelated', 3), 'other-service', 'RUNNING')
        self.assertTrue(self.pool.stop(timeout=0)['complete'])
        self.assertEqual(self.servings.stops, [state['service_id']])
        self.assertEqual(self.servings.resources['unrelated'].status, 'RUNNING')
        self.assertIn('corpus_pipeline.serving.configured_pool', self.pool.command('run'))
        self.assertEqual(self.pool.command('run')[-1], str(self.config))

    def two_slots(self):
        self.pool.provision(1); self.qualify()
        with patch.object(module.native, 'attest', return_value=self.evidence()), \
                patch.object(module.native, 'health', return_value=True):
            self.pool.provision(2)
        return self.pool.load(1), self.pool.load(2)

    def test_unknown_first_slot_does_not_prevent_stopping_known_second(self):
        first, second = self.two_slots()
        self.pool.paths(1)[0].unlink()
        self.servings.resources.pop(first['name'])
        with self.assertRaisesRegex(RuntimeError, 'not verified'): self.pool.stop(timeout=0)
        self.assertEqual(self.servings.stops, [second['service_id']])
        receipt = module.native.load(self.root / 'STOPPED.json')
        self.assertFalse(receipt['complete'])
        self.assertFalse(receipt['services'][0]['terminal']); self.assertTrue(receipt['services'][1]['terminal'])

    def test_unknown_stop_reply_does_not_block_next_slot_or_retransmit(self):
        first, second = self.two_slots()
        original = self.servings.stop
        calls = []
        def stopping(ref, **kwargs):
            calls.append(ref.key)
            if ref.key == first['service_id']: raise TimeoutError('lost stop reply')
            return original(ref, **kwargs)
        with patch.object(self.servings, 'stop', side_effect=stopping):
            with self.assertRaisesRegex(RuntimeError, 'not verified'): self.pool.stop(timeout=0)
            self.assertEqual(calls, [first['service_id'], second['service_id']])
            with self.assertRaisesRegex(RuntimeError, 'not verified'): self.pool.stop(timeout=0)
            self.assertEqual(calls, [first['service_id'], second['service_id']])
        intent = module.native.load(self.root / 'stop-intent-01.json')
        self.assertEqual(intent['phase'], 'UNKNOWN')
        self.assertEqual(self.servings.resources[second['name']].status, 'STOPPED')

    def test_scope_isolation_config_budget_and_probe_runtime_guard(self):
        self.raw['scope_root'] = str(self.lab / 'production')
        module.native.save(self.config, self.raw)
        with self.assertRaisesRegex(ValueError, 'separate'): module.PoolSpec.load(self.config)
        self.raw['scope_root'] = str(self.root); self.raw['max_cards'] = 256
        module.native.save(self.config, self.raw)
        with self.assertRaisesRegex(ValueError, '128'): module.PoolSpec.load(self.config)
        self.pool.runtime_scope()
        with patch.object(module, '_NATIVE_SCOPE', self.old):
            with self.assertRaisesRegex(ValueError, 'one pool'): self.pool.runtime_scope()


if __name__ == '__main__':
    unittest.main()
