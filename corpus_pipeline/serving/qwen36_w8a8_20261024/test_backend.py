import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

from corpus_pipeline.serving.qwen36_w8a8_20261024 import backend as mod
from corpus_pipeline.serving.qwen36_w8a8_20261024 import high_control as high
from corpus_pipeline.serving.qwen36_w8a8_20261024.runtime.contract import engine_argv, validate_config


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = mod.Backend(self.root, self.root / 'config.json')
        self.state = dict(cards=2, expected_nodes=1, expected_replicas=1, tier='high', priority=6,
            precision='W8A8_DYNAMIC', role='verifier', name='owned', service_name='owned', operation_id='owned',
            scope=str(self.root), binding={'frozen': True}, create_spec={'name': 'owned'}, service_id='sv-owned',
            model='model', revision='revision', processor_revision='processor', runtime_revision='runtime',
            incarnation='incarnation', expected_boots=['boot'], was_running=True, last_running_at=100,
            payload={'workspace_id':'workspace-fixture'}, platform_status='RUNNING', api_key_file='/private/key', auto_recovery_enabled=False)
        self.client = MagicMock()
        self.context = patch.object(mod, 'InspireClient')
        self.factory = self.context.start(); self.addCleanup(self.context.stop)
        self.factory.return_value.__enter__.return_value = self.client
        self.detail = NS(ref=NS(key='sv-owned'), status='STOPPED',
            to_dict=lambda: {'endpoint': 'https://private.invalid'})
        self.client.servings.get.return_value = self.detail
        self.client.servings.instances.return_value = [NS(status='INSTANCE_STOPPED')]
        self.client.servings.events.return_value = NS(items=[])
        self.owner = patch.object(mod.native, 'assert_owned').start(); self.addCleanup(patch.stopall)

    def write_intent(self):
        mod.native.save(self.root / 'CREATE_INTENT.json', {})
        mod.native.save(self.root / 'state.json', self.state)

    def test_constructor_initializes_proxy_bypass(self):
        with patch.object(mod.pool, 'configure_management_proxy_bypass') as setup:
            mod.Backend(self.root)
        setup.assert_called_once()

    def test_create_unknown_never_resends(self):
        self.write_intent()
        with patch.object(self.backend, 'binding', return_value=({}, self.state['binding'])), \
             patch.object(mod.pool, 'observable_project_budget'), patch.object(mod, 'ServingCreateSpec', side_effect=lambda **x:x):
            self.client.servings.create.side_effect = TimeoutError()
            with self.assertRaises(TimeoutError): self.backend.create(self.root, self.state)
            self.assertTrue((self.root / 'CREATE_TRANSMISSION.json').exists())
            with self.assertRaises(ValueError): self.backend.create(self.root, self.state)
        self.assertEqual(self.client.servings.create.call_count, 1)

    def test_create_reservation_required(self):
        with patch.object(self.backend, 'binding', return_value=({}, self.state['binding'])):
            with self.assertRaises(ValueError): self.backend.create(self.root, self.state)
        self.client.servings.create.assert_not_called()

    def test_create_budget_failure_happens_before_transmission(self):
        self.write_intent()
        with patch.object(self.backend, 'binding', return_value=({}, self.state['binding'])), \
             patch.object(mod.pool, 'observable_project_budget', side_effect=ValueError()):
            with self.assertRaises(ValueError): self.backend.create(self.root, self.state)
        self.assertFalse((self.root / 'CREATE_TRANSMISSION.json').exists())
        self.client.servings.create.assert_not_called()

    def test_scope_disabled_cannot_create(self):
        self.write_intent(); (self.root / 'DISABLED').touch()
        with patch.object(self.backend, 'binding', return_value=({}, self.state['binding'])), \
             patch.object(mod.pool, 'observable_project_budget'):
            with self.assertRaises(ValueError): self.backend.create(self.root, self.state)
        self.client.servings.create.assert_not_called()

    def test_pool_disabled_during_budget_check_cancels_create(self):
        self.write_intent()
        pool_root = self.root / 'pool'; pool_root.mkdir()
        self.state['pool_scope'] = str(pool_root)
        with patch.object(self.backend, 'binding', return_value=({}, self.state['binding'])), \
             patch.object(mod.pool, 'observable_project_budget', side_effect=lambda *_, **__: (pool_root / 'DISABLED').touch()):
            with self.assertRaises(ValueError): self.backend.create(self.root, self.state)
        self.assertFalse((self.root / 'CREATE_TRANSMISSION.json').exists())
        self.client.servings.create.assert_not_called()

    def test_pool_disabled_during_terminal_check_cancels_restart(self):
        state = self.recovery_state(); event = 'e' * 64
        pool_root = self.root / 'pool'; pool_root.mkdir(); state['pool_scope'] = str(pool_root)
        def instances(*args, **kwargs):
            (pool_root / 'DISABLED').touch()
            return [NS(status='INSTANCE_STOPPED')]
        self.client.servings.instances.side_effect = instances
        with patch.object(self.backend, 'can_restart', return_value=event), patch.object(self.backend, 'publish'):
            with self.assertRaises(ValueError): self.backend.restart(self.root, state, event)
        self.client.servings.start.assert_not_called()

    def test_unknown_create_observe_adopts_exact_owned_only(self):
        state = dict(self.state); state.pop('service_id')
        self.client.servings.iter.return_value = [NS(name='owned', ref='reference')]
        result, _ = self.backend.reconcile(self.client, state)
        self.assertEqual(result['service_id'], 'sv-owned')
        self.owner.assert_called_once()
        self.client.servings.create.assert_not_called()
        self.client.servings.iter.return_value *= 2
        with self.assertRaises(ValueError): self.backend.reconcile(self.client, state)

    def test_identity_drift_prevents_stop(self):
        self.owner.side_effect = AssertionError()
        with patch.object(self.backend, 'publish'):
            with self.assertRaises(AssertionError): self.backend.stop(self.root, self.state)
        self.client.servings.stop.assert_not_called()

    def test_stop_unknown_is_single_send_and_released_only_terminal(self):
        self.detail.status = 'RUNNING'
        self.client.servings.instances.return_value = [NS(status='INSTANCE_RUNNING')]
        self.client.servings.stop.side_effect = TimeoutError()
        with patch.object(self.backend, 'publish'):
            with self.assertRaises(TimeoutError): self.backend.stop(self.root, self.state)
            result = self.backend.stop(self.root, self.state)
            self.assertFalse(result['release_verified'])
            self.detail.status = 'STOPPED'
            result = self.backend.stop(self.root, self.state)
            self.assertFalse(result['release_verified'])
            self.client.servings.instances.return_value = [NS(status='INSTANCE_SUCCEEDED')]
            result = self.backend.stop(self.root, self.state)
        self.assertTrue(result['release_verified']); self.assertEqual(result['phase'], 'RELEASED')
        self.assertEqual(self.client.servings.stop.call_count, 1)

    def test_stopping_intermediate_preserves_preemption_transition(self):
        self.state['platform_status'] = 'STOPPING'
        with patch.object(mod.time, 'time', return_value=1000), patch.object(self.backend, 'publish'):
            result = self.backend.observe(self.root, self.state)
        self.assertEqual(result['stopped_transition_at'], 1000)
        self.client.servings.start.assert_not_called()

    def recovery_state(self):
        return dict(self.state, tier='low', priority=3, auto_recovery_enabled=True, platform_status='STOPPED',
            stopped_transition_at=1000, last_observed_at=1000, events=[{'reason':'Preempted',
            'message':'was preempted by another job', 'object_id':'instance/sv-owned/0', 'last_timestamp':200000}])

    def test_explicit_old_preemption_recovers_after_controller_gap(self):
        state = self.recovery_state()
        with patch.object(mod.time, 'time', return_value=1001):
            event = self.backend.can_restart(state)
            self.assertIsNotNone(event)
            self.assertIsNone(self.backend.can_restart(dict(state, restart_event=event)))
            state['events'][0]['object_id'] = 'another-service'
            self.assertIsNone(self.backend.can_restart(state))

    def test_old_boot_or_manual_stop_does_not_restart(self):
        state = self.recovery_state()
        with patch.object(mod.time, 'time', return_value=1001):
            state['last_running_at'] = 300
            self.assertIsNone(self.backend.can_restart(state))
            state['last_running_at'] = 100
            (self.root / 'STOP_INTENT.json').touch()
            self.assertIsNone(self.backend.can_restart(state))

    def test_restart_event_is_single_send_even_if_start_reply_lost(self):
        state = self.recovery_state(); event = 'e' * 64
        self.client.servings.start.side_effect = TimeoutError()
        with patch.object(self.backend, 'can_restart', return_value=event), patch.object(self.backend, 'publish'):
            with self.assertRaises(TimeoutError): self.backend.restart(self.root, state, event)
            self.backend.restart(self.root, state, event)
        self.assertEqual(self.client.servings.start.call_count, 1)
        self.assertEqual(self.client.servings.start.call_args.args[0].key, 'sv-owned')

    def test_smoke_intent_without_process_is_not_replayed(self):
        directory = self.root / 'smoke/incarnation'; directory.mkdir(parents=True)
        (directory / 'SMOKE_INTENT.json').touch()
        with patch.object(mod.subprocess, 'Popen') as spawn:
            self.assertFalse(self.backend.smoke(self.root, self.state, {'incarnation': 'incarnation'}))
        spawn.assert_not_called()

    def test_smoke_receipt_cannot_cross_boot(self):
        directory = self.root / 'smoke/incarnation'; directory.mkdir(parents=True)
        mod.native.save(directory / 'RESULT.json', {**self.state, 'expected_boots':['different'],
            'passed':True, 'qualification_kind':'basic_smoke', 'request_count':1})
        with self.assertRaises(ValueError): self.backend.smoke(self.root, self.state, {'incarnation':'incarnation'})

    def test_attestation_binds_fresh_boot_runtime_and_allocated_topology(self):
        run = self.root / 'runs/boot'; run.mkdir(parents=True)
        self.state.update(runtime_root=str(run.parent), deployment_id='owned', config_snapshot={})
        self.state['binding']['serving_code_revision'] = 'code'
        beat = dict(time=1000, hostname='pod', deployment_id='owned', boot_id='boot')
        report = {k:self.state[k] for k in ('model','revision','precision','processor_revision')}
        report.update({k:'test-version' for k in mod.native.PACKAGES})
        report.update(deployment_id='owned', boot_id='boot', runtime_config={}, serving_code_revision='code',
            visible_npu_count=2, tensor_parallel=2, expert_parallel=True, data_parallel=1,
            processor_options={'max_pixels':1048576}, communication={})
        mod.native.save(run / 'node0.json', beat); mod.native.save(run / 'versions-node0.json', report)
        instance = NS(status='INSTANCE_RUNNING', raw={'name':'scope/pod','node':'node',
            'created_at':'a','started_at':'b','restarts':0,'term':1})
        with patch.object(self.backend, 'binding', return_value=({}, self.state['binding'])), \
             patch.object(mod.time, 'time', return_value=1001):
            proof = self.backend.attest(self.state, [instance])
            self.assertEqual(proof['expected_boots'], ['boot'])
            report['visible_npu_count'] = 4
            mod.native.save(run / 'versions-node0.json', report)
            with self.assertRaises(ValueError): self.backend.attest(self.state, [instance])
            report['visible_npu_count'] = 2; report['boot_id'] = 'another-boot'
            mod.native.save(run / 'versions-node0.json', report)
            with self.assertRaises(ValueError): self.backend.attest(self.state, [instance])

    def test_publish_revokes_even_if_endpoint_removed_from_state(self):
        mod.native.save(self.root / 'registry.json', {'registry_generation':1,
            'endpoints':[{'base_url':'https://private.invalid/v1'}]})
        self.state.update(max_images=16, deployment_id='owned', request_overrides={})
        self.backend.publish(self.root, self.state, False)
        row = mod.native.load(self.root / 'registry.json')['endpoints'][0]
        self.assertFalse(row['ready']); self.assertEqual(row['ready_expires_at'], 0)
        self.assertEqual(row['max_output_tokens'], 3000)
        self.assertNotIn('qualification_only', row)


class ContractTests(unittest.TestCase):
    def config(self):
        return dict(model_name='Qwen3.6-35B-A3B-W8A8', model_path='/weights', precision='W8A8_DYNAMIC',
            quantization='ascend', nodes=1, tensor_parallel_size=2, data_parallel_size=1, enable_expert_parallel=True,
            dtype='bfloat16', max_num_seqs=128, max_num_batched_tokens=4096, max_model_len=16384,
            gpu_memory_utilization=.90, num_speculative_tokens=3, enable_json_keepalive=True,
            max_images=16, max_pixels=1048576, cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256,512])

    def test_official_tp2_w8a8_recipe_and_root_context_override(self):
        argv = engine_argv(self.config())
        self.assertEqual(argv[argv.index('--tensor-parallel-size')+1], '2')
        self.assertEqual(argv[argv.index('--quantization')+1], 'ascend')
        self.assertEqual(argv[argv.index('--dtype')+1], 'bfloat16')
        self.assertEqual(argv[argv.index('--max-model-len')+1], '16384')
        self.assertIn('--enable-expert-parallel', argv)
        self.assertEqual(json.loads(argv[argv.index('--speculative-config')+1])['num_speculative_tokens'], 3)
        compilation = json.loads(argv[argv.index('--compilation-config')+1])
        self.assertEqual(compilation['max_cudagraph_capture_size'], 128 * (3 + 1))
        self.assertEqual(compilation['cudagraph_capture_sizes'], [1,2,4,8,16,32,64,128,256,512])
        self.assertEqual(compilation['cudagraph_mode'], 'FULL_AND_PIECEWISE')

    def test_precision_topology_or_context_mutation_rejected(self):
        for field, value in [('precision','BF16'),('dtype','float16'),('tensor_parallel_size',4),('max_model_len',135168)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(dict(self.config(), **{field:value}))

    def test_high_create_existing_intent_does_not_retransmit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(high, 'HIGH', Path(tmp)), patch.object(high, 'Backend') as cls:
            high.native.save(Path(tmp) / 'CREATE_INTENT.json', {})
            high.native.save(Path(tmp) / 'state.json', {'phase':'CREATE_UNKNOWN'})
            self.assertEqual(high.create()['phase'], 'CREATE_UNKNOWN')
            cls.return_value.create.assert_not_called()

    def test_repair_unknown_start_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(high, 'HIGH', Path(tmp)), \
             patch.object(high, 'Backend'), patch.object(high, 'InspireClient') as client:
            state = {'service_id':'exact-id','name':high.NAME,'tier':'high','phase':'REPAIR_START_INTENT'}
            high.native.save(Path(tmp) / 'state.json', state)
            high.native.save(Path(tmp) / 'BF16_REPAIR_START_INTENT.json', {'service_id':'exact-id'})
            self.assertEqual(high.repair_start()['phase'], 'REPAIR_START_INTENT')
            client.assert_not_called()

    def test_repair_marker_for_another_resource_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(high, 'HIGH', Path(tmp)), \
             patch.object(high, 'Backend'), patch.object(high, 'InspireClient') as client:
            high.native.save(Path(tmp) / 'state.json', {'service_id':'exact-id','name':high.NAME,'tier':'high'})
            high.native.save(Path(tmp) / 'BF16_REPAIR_START_INTENT.json', {'service_id':'other-id'})
            with self.assertRaises(ValueError): high.repair_start()
            client.assert_not_called()

    def test_capture_repair_has_separate_single_send_marker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(high, 'HIGH', Path(tmp)), \
             patch.object(high, 'Backend'), patch.object(high, 'InspireClient') as client:
            high.native.save(Path(tmp) / 'state.json', {'service_id':'exact-id','name':high.NAME,
                'tier':'high','phase':'REPAIR_START_INTENT'})
            high.native.save(Path(tmp) / 'CAPTURE_REPAIR_START_INTENT.json', {'service_id':'exact-id'})
            self.assertEqual(high.repair_start('capture')['phase'], 'REPAIR_START_INTENT')
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
