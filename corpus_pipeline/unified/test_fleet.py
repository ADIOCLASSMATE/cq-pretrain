"""Offline fleet fencing tests. No platform resources or SDK calls are made."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from corpus_pipeline.production.common import atomic_json, file_sha
from corpus_pipeline.resource_manager.engine import Controller, request_stop
from corpus_pipeline.resource_manager.models import WorkloadSpec
from corpus_pipeline.resource_manager.test_engine import FakeAdapter
from . import SCHEMA
from . import fleet


def fixture(root):
    root = Path(root)
    source = root / 'source'; source.mkdir()
    repo = Path(fleet.__file__).resolve().parents[2]
    config = {'schema': SCHEMA, 'scope': 'validation', 'output': str(root / 'output'),
              'inputs': [{'collection': 'dcvlm', 'adapter': 'dcvlm_native', 'root': str(source)}],
              'resources': {'max_cards': 8, 'frontpass_max_cards': 7, 'review_max_cards': 8,
                            'sscd_max_cards': 5, 'prefer_occupied_nodes': True}}
    template = {'adapter': fleet.ADAPTER, 'state_dir': str(root / 'old_controller'),
                'workload': {'scope': 'old-scope', 'profile': 'old-profile', 'max_cards': 128,
                             'allowed_cards': [1, 2, 4, 8], 'command': ['old-worker'], 'poll_seconds': 10},
                'adapter_config': {'backend': 'notebook', 'account': 'fixture-account', 'workspace': 'fixture-workspace',
                    'project': 'fixture-project', 'group': 'fixture-group', 'image': 'fixture-image',
                    'priority': 3, 'global_max_cards': 256, 'repository': str(repo),
                    'remote_python': str(repo / '.venv/bin/python'), 'worker_state_dir': str(root / 'remote_workers'),
                    'registry_dir': str(root / 'old_registry'), 'excluded_nodes': ['bad-node']}}
    cfg_path, template_path = root / 'config.json', root / 'template.json'
    atomic_json(cfg_path, config); atomic_json(template_path, template)
    return cfg_path, template_path


class FleetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.template = fixture(self.root)
        relative = str(Path(fleet.__file__).resolve().relative_to(Path(fleet.__file__).resolve().parents[2]))
        mock = patch.object(fleet, '_runtime_bindings', return_value=[{'relative_path': relative, 'sha256': file_sha(fleet.__file__)}])
        mock.start(); self.addCleanup(mock.stop)
        self.generated = fleet.build(self.config, self.template)

    def generated_config(self, stage):
        return json.loads(Path(self.generated['manager_configs'][stage]['path']).read_text())

    def controller(self, stage, fake):
        config = self.generated_config(stage)
        with patch.object(fleet, '_factory', return_value=fake):
            adapter = fleet.StageAdapter(config['adapter_config'])
        return Controller(adapter, WorkloadSpec(**config['workload']), config['state_dir'])

    def test_build_is_repeatable_generates_only_and_preserves_platform_contract(self):
        with patch.object(fleet, '_factory', side_effect=AssertionError('No platform initialization allowed')):
            repeated = fleet.build(self.config, self.template)
        self.assertEqual(repeated, self.generated)
        self.assertTrue(repeated['generated_only'])
        self.assertFalse(repeated['resources_started'])
        for stage, budget in (('frontpass', 7), ('sscd_search', 5)):
            config = self.generated_config(stage)
            self.assertEqual(config['workload']['max_cards'], budget)
            self.assertEqual(config['workload']['allowed_cards'], [4, 2, 1])
            self.assertEqual(config['workload']['migration_policy'], 'restartable')
            self.assertEqual(config['workload']['command'][-5:], ['node-worker', '--config', str(self.config), '--stage', stage])
            self.assertEqual(config['adapter_config']['inner_config']['global_max_cards'], 8)
            self.assertEqual(config['adapter_config']['inner_config']['excluded_nodes'], ['bad-node'])

    def test_seven_cards_pack_on_one_node_then_other_stage_waits_for_verified_retirement(self):
        fake = FakeAdapter(); fake.global_limit = 8
        first = self.controller('frontpass', fake)
        second = self.controller('sscd_search', fake)
        result = first.poll()
        self.assertEqual(result['reserved_cards'], 7)
        self.assertEqual([row['quota']['cards'] for row in fake.resources.values()], [4, 2, 1])
        self.assertEqual({row['node'] for row in fake.resources.values()}, {'node-a'})
        with self.assertRaisesRegex(ValueError, 'Previous stage still owns'):
            second.poll()
        self.assertEqual(fake.spawned, 3)
        request_stop(first.directory, scope=first.spec.scope, profile=first.spec.profile)
        first.poll(); first.poll()
        self.assertEqual(first.state['phase'], 'completed')
        self.assertEqual(second.poll()['reserved_cards'], 5)

    def test_unknown_create_and_missing_required_state_never_release_budget(self):
        fake = FakeAdapter(); fake.global_limit = 8
        fake.unknown_create = True; fake.reconcile_enabled = False
        first = self.controller('frontpass', fake)
        second = self.controller('sscd_search', fake)
        self.assertEqual(first.poll()['unknown_creates'], 1)
        with self.assertRaisesRegex(ValueError, 'Previous stage still owns'):
            second.poll()
        (first.directory / 'STATE.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Previous stage still owns'):
            second.poll()
        self.assertEqual(fake.spawned, 1)

    def test_plan_without_any_durable_intent_does_not_permanently_pin_stage(self):
        fake = FakeAdapter()
        first = self.controller('frontpass', fake)
        second = self.controller('sscd_search', fake)
        first.plan(); second.plan()
        self.assertEqual(fake.spawned, 0)

    def test_missing_fence_ledger_cannot_reset_reserved_budget(self):
        fake = FakeAdapter()
        first = self.controller('frontpass', fake)
        second = self.controller('sscd_search', fake)
        first.poll()
        (self.root / 'output/fleet/guard/ACTIVE.json').unlink()
        with self.assertRaisesRegex(ValueError, 'stage ledger is missing'):
            second.poll()
        self.assertEqual(fake.spawned, 3)

    def test_arbitrary_state_directory_override_cannot_allocate_outside_bound_ledger(self):
        fake = FakeAdapter()
        original = self.controller('frontpass', fake)
        altered = Controller(original.adapter, original.spec, self.root / 'unbound-state')
        self.assertEqual(altered.poll()['unknown_creates'], 1)
        self.assertEqual(fake.spawned, 0)

    def test_preemption_replenishes_with_same_manager_and_no_double_create(self):
        fake = FakeAdapter(); fake.global_limit = 8
        now = [1000]
        first = self.controller('frontpass', fake); first.clock = lambda: now[0]
        first.poll()
        item = next(iter(fake.resources.values())); item['state'] = 'STOPPED'
        first.poll()
        now[0] += 11
        self.assertEqual(first.poll()['reserved_cards'], 7)
        self.assertEqual(fake.spawned, 4)

    def test_remote_config_runtime_and_mount_preflight(self):
        self.assertTrue(fleet.preflight(self.config, 'frontpass')['passed'])
        other = self.root / 'copy.json'; other.write_bytes(self.config.read_bytes())
        with self.assertRaisesRegex(ValueError, 'path/content differs'):
            fleet.preflight(other, 'frontpass')
        with patch.dict(os.environ, {'RESOURCE_MANAGER_CARDS': '8'}):
            with self.assertRaisesRegex(ValueError, 'allocation exceeds budget'):
                fleet.preflight(self.config, 'frontpass')
        (self.root / 'source').rmdir()
        with self.assertRaisesRegex(ValueError, 'Dataset mount missing'):
            fleet.preflight(self.config, 'sscd_search')

    def test_changed_runtime_or_platform_contract_fails_closed(self):
        config = self.generated_config('frontpass')
        config['adapter_config']['inner_config']['image'] = 'another-image'
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            fleet.StageAdapter(config['adapter_config'])
        config = json.loads(self.config.read_text())
        config['resources']['max_cards'] = 513
        atomic_json(self.config, config)
        with self.assertRaisesRegex(ValueError, '512 cards'):
                fleet.build(self.config, self.template)

    def test_remote_preflight_accepts_authorized_budget_above_old_128_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,template=fixture(tmp)
            value=json.loads(config.read_text())
            value['resources'].update(max_cards=512,frontpass_max_cards=512,review_max_cards=512)
            atomic_json(config,value)
            fleet.build(config,template)
            with patch.dict(os.environ,{'RESOURCE_MANAGER_CARDS':'1'}):
                self.assertTrue(fleet.preflight(config,'frontpass')['passed'])

    def test_runtime_binding_list_cannot_be_removed_from_preflight(self):
        path = Path(self.generated['preflight_path'])
        value = json.loads(path.read_text()); value['runtime'] = []
        atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, 'runtime binding list changed'):
            fleet.preflight(self.config, 'frontpass')

    def test_completed_search_requires_final_verified_manifest_not_task_counts(self):
        self.assertFalse(fleet.completion(self.config, 'sscd_search'))
        area = self.root / 'output/sscd_search'
        area.mkdir()
        atomic_json(area / 'PLAN.json', {'total_tasks': 3, 'total_pairs': 100})
        atomic_json(area / 'calibration/COMPLETE.json', {'passed': True})
        result = {'complete': True, 'all_pairs_covered': True, 'total_tasks': 3, 'completed_tasks': 3,
                  'total_pairs': 100, 'completed_pairs': 100, 'plan_sha256': file_sha(area / 'PLAN.json'),
                  'calibration_sha256': file_sha(area / 'calibration/COMPLETE.json')}
        atomic_json(area / 'COMPLETE.json', result)
        self.assertTrue(fleet.completion(self.config, 'sscd_search'))
        result['completed_pairs'] = 99
        atomic_json(area / 'COMPLETE.json', result)
        self.assertFalse(fleet.completion(self.config, 'sscd_search'))
        with patch('corpus_pipeline.cluster.queue_rpc.status', return_value={'all_tasks_completed': False, 'counts': {'completed': 99}}):
            self.assertFalse(fleet.completion(self.config, 'frontpass'))

    def test_root_stop_requests_both_managers_and_verified_drain_without_false_completion(self):
        fake = FakeAdapter()
        first = self.controller('frontpass', fake)
        self.assertEqual(first.poll()['reserved_cards'], 7)
        result = fleet.request_stop(self.config)
        self.assertTrue(result['stop_requested'])
        self.assertFalse(result['resources_released_verified'])
        self.assertEqual(len(result['manager_stop_requests']), 2)
        self.assertTrue(all(Path(path).exists() for path in result['business_stop_markers']))
        with self.assertRaisesRegex(ValueError, 'STOP forbids worker startup'):
            fleet.preflight(self.config, 'frontpass')
        with patch('corpus_pipeline.cluster.queue_rpc.status', return_value={'all_tasks_completed': True, 'counts': {}}):
            self.assertFalse(fleet.completion(self.config, 'frontpass'))
        self.assertEqual(first.poll()['phase'], 'stopping')
        self.assertEqual(first.poll()['phase'], 'completed')
        self.assertEqual(first.state['finish_reason'], 'operator_stop')
        self.assertEqual(fake.spawned, 3)

    def test_stage_disk_guard_stop_propagates_before_new_allocation_or_launch(self):
        fake = FakeAdapter()
        first = self.controller('frontpass', fake)
        first.poll()
        path = self.root / 'output/frontpass/STOP'; path.parent.mkdir(parents=True); path.touch()
        self.assertEqual(first.adapter.snapshot()['global_remaining_cards'], 0)
        self.assertTrue((first.directory / 'STOP_REQUEST.json').exists())
        with self.assertRaisesRegex(ValueError, 'STOP forbids new worker launches'):
            first.adapter.launch('resource-1', list(first.spec.command), 'new-launch')
        intent = next(iter(first.state['resources'].values()))['intent']
        with self.assertRaisesRegex(ValueError, 'STOP forbids new allocations'):
            first.adapter.create(intent)
        first.poll(); first.poll()
        self.assertEqual(first.state['phase'], 'completed')
        self.assertEqual(fake.spawned, 3)

    def test_completion_probe_transmits_exact_allocation_environment(self):
        fake = FakeAdapter(); first = self.controller('frontpass', fake); first.poll()
        for resource in fake.resources.values(): resource['state'] = 'RUNNING'
        first.poll()
        operation, row = next(iter(first.state['resources'].items()))
        with patch.object(fake, 'completion', return_value=True) as callback:
            self.assertFalse(first.adapter.completion(row['resource_id'], ['python', 'completion-probe']))
            command = callback.call_args.args[1]
            self.assertEqual(command[:3], ['env', 'RESOURCE_MANAGER_ALLOCATION_ID=' + operation,
                                          'RESOURCE_MANAGER_CARDS=' + str(row['intent']['placement']['quota']['cards'])])
            self.assertEqual(command[3:], ['python', 'completion-probe'])

    def test_fast_pool_cannot_stop_other_allocations_that_are_still_sealing(self):
        fake = FakeAdapter(); first = self.controller('frontpass', fake); first.poll()
        for resource in fake.resources.values(): resource['state'] = 'RUNNING'
        first.poll()
        identifiers = list(fake.resources)
        ready = {identifiers[0]}
        with patch.object(fake, 'completion', side_effect=lambda identifier, command: identifier in ready):
            self.assertNotEqual(first.poll()['phase'], 'stopping')
            self.assertFalse(any(call[0] == 'stop' for call in fake.calls))
            self.assertEqual(first.adapter.snapshot()['global_remaining_cards'], 0)
            self.assertFalse(first.adapter.capabilities['node_placement'])
            ready.update(identifiers)
            self.assertEqual(first.poll()['phase'], 'stopping')
            self.assertEqual(first.poll()['phase'], 'completed')

    def test_global_done_waits_for_current_allocation_pool_cache_seal(self):
        root = self.root / 'output'; execution = root / 'EXECUTION.json'
        atomic_json(execution, {'fixture': True})
        identifier = 'test-allocation'
        path = root / 'fleet/pools/frontpass' / (identifier + '.json')
        binding = {'schema': 'corpus.unified.worker_pool.v1', 'allocation': identifier, 'stage': 'frontpass', 'cards': 4,
                   'config_path': str(self.config), 'config_sha256': file_sha(self.config), 'execution_sha256': file_sha(execution)}
        value = {'binding': binding, 'allocation': identifier, 'stage': 'frontpass', 'cards': 4,
                 'host': socket.gethostname(), 'runtime_id': socket.gethostname() + ':' + Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 'status': 'draining_completed_children', 'active_children': {'0': {'pid': 123}}, 'fatal': False}
        with patch.dict(os.environ, {'RESOURCE_MANAGER_ALLOCATION_ID': identifier, 'RESOURCE_MANAGER_CARDS': '4'}), \
                patch('corpus_pipeline.cluster.queue_rpc.status', return_value={'all_tasks_completed': True, 'counts': {}}):
            self.assertFalse(fleet.completion(self.config, 'frontpass'))
            atomic_json(path, value)
            self.assertFalse(fleet.completion(self.config, 'frontpass'))
            value.update(status='complete', active_children={}); atomic_json(path, value)
            self.assertTrue(fleet.completion(self.config, 'frontpass'))
            value['runtime_id'] = 'previous-boot'; atomic_json(path, value)
            self.assertFalse(fleet.completion(self.config, 'frontpass'))


if __name__ == '__main__':
    unittest.main()
