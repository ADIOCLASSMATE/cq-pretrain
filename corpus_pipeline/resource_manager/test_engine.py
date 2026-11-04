from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .engine import Controller, request_stop, atomic_json
from .models import WorkloadSpec


class FakeAdapter:
    capabilities = {'node_placement': True, 'drain': True}
    def __init__(self, config=None):
        self.nodes = {'node-a': (7, 56, 448)}
        self.quotas = [dict(cards=n, cpu=n*8, memory=n*64) for n in (4, 2, 1)]
        self.resources = {}
        self.intents = {}
        self.calls = []
        self.unknown_create = False
        self.unknown_stop = False
        self.reconcile_enabled = True
        self.global_limit = 256
        self.foreign_cards = 0
        self.complete = False
        self.spawned = 0

    def snapshot(self):
        free = {node: list(values) for node, values in self.nodes.items()}
        visible = {}
        for identifier, resource in self.resources.items():
            if resource['state'] == 'STOPPED':
                continue
            allocated = resource['state'] == 'RUNNING'
            visible[identifier] = {'state': resource['state'], 'node': resource['node'],
                                   'capacity_accounted': allocated}
            if allocated:
                quota = resource['quota']
                free[resource['node']] = [value - quota[key] for value, key in
                                         zip(free[resource['node']], ('cards', 'cpu', 'memory'))]
        used = sum(row['quota']['cards'] for row in self.resources.values() if row['state'] != 'STOPPED')
        return {'nodes': [dict(name=name, gpu=values[0], cpu=values[1], memory=values[2])
                          for name, values in free.items()],
                'quotas': self.quotas, 'resources': visible,
                'global_remaining_cards': max(0, self.global_limit - self.foreign_cards - used)}

    def create(self, intent):
        self.calls.append(('create', intent['operation_id']))
        self.spawned += 1
        identifier = 'resource-' + str(self.spawned)
        placement = intent['placement']
        self.resources[identifier] = {'state': 'PENDING', 'node': placement['node'], 'quota': placement['quota'],
                                      'health': {'running': False, 'ready': False, 'completed': False,
                                                 'runtime_id': identifier + ':boot1'}}
        self.intents[intent['operation_id']] = identifier
        if self.unknown_create:
            raise TimeoutError('ambiguous fixture Create')
        return identifier

    def reconcile(self, intent):
        return self.intents.get(intent['operation_id']) if self.reconcile_enabled else None

    def observe(self, identifier):
        row = self.resources[identifier]
        return {'state': row['state'], 'node': row['node']}

    def stop(self, identifier, operation_id):
        self.calls.append(('stop', identifier, operation_id))
        if self.unknown_stop:
            raise TimeoutError('ambiguous fixture stop')
        self.resources[identifier]['state'] = 'STOPPED'

    def launch(self, identifier, command, operation_id):
        self.calls.append(('launch', identifier, tuple(command), operation_id))
        self.resources[identifier]['health'].update(running=True, ready=True, completed=False,
                                                   exit_code=None, launch_operation_id=operation_id)

    def health(self, identifier, readiness_command):
        return dict(self.resources[identifier]['health'])

    def drain(self, identifier, command, operation_id):
        self.calls.append(('drain', identifier, operation_id))
        self.resources[identifier]['health'].update(drained=True, drain_operation_id=operation_id)

    def completion(self, identifier, command):
        return self.complete


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = FakeAdapter()
        self.now = 1000
        self.spec = WorkloadSpec('test-scope', 'worker', 7, ('python', 'worker.py'), (4, 2, 1))
        self.control = self.make()

    def make(self, spec=None):
        return Controller(self.adapter, spec or self.spec, self.root, clock=lambda: self.now)

    def running(self):
        for row in self.adapter.resources.values():
            if row['state'] != 'STOPPED':
                row['state'] = 'RUNNING'
        self.control.poll(); self.control.poll()

    def calls(self, kind):
        return [call for call in self.adapter.calls if call[0] == kind]

    def test_unknown_stop_retries_same_operation_only_after_fresh_observation(self):
        self.adapter.capabilities={**self.adapter.capabilities,'retry_owned_stop':True}
        self.control.poll();self.adapter.unknown_stop=True
        request_stop(self.root,scope=self.spec.scope,profile=self.spec.profile)
        first=self.control.poll();self.assertGreater(first['reserved_cards'],0)
        before={call[1]:call[2] for call in self.calls('stop')}
        self.now+=10;self.control.poll()
        self.assertEqual(len(self.calls('stop')),len(before))
        self.adapter.unknown_stop=False;self.now+=30;self.control.poll()
        for _,identifier,operation in self.calls('stop'):self.assertEqual(operation,before[identifier])
        result=self.control.poll();self.assertEqual(result['reserved_cards'],0)
        self.assertEqual(result['phase'],'completed')

    def test_fresh_owned_stop_clears_stale_launch_warning(self):
        self.control.poll()
        with self.control.locked():
            self.control.state=self.control._load()
            row=next(iter(self.control.state['resources'].values()))
            row['action_required']='launch_result_unconfirmed'
            self.adapter.resources[row['resource_id']]['state']='STOPPED'
            self.control._observe(row)
            self.assertTrue(row['retired_verified'])
            self.assertNotIn('action_required',row)

    def test_seven_cards_pack_four_two_one_on_same_node(self):
        result = self.control.poll()
        self.assertEqual(result['reserved_cards'], 7)
        self.assertEqual([row['quota']['cards'] for row in self.adapter.resources.values()], [4, 2, 1])
        self.assertEqual({row['node'] for row in self.adapter.resources.values()}, {'node-a'})
        self.assertEqual(self.make().poll()['reserved_cards'], 7)
        self.assertEqual(len(self.calls('create')), 3)

    def test_unknown_create_is_reserved_and_never_replayed_after_restart(self):
        self.adapter.unknown_create = True
        self.adapter.reconcile_enabled = False
        first = self.control.poll()
        self.assertEqual(first['unknown_creates'], 1)
        self.assertEqual(first['reserved_cards'], 4)
        # Existing unknown reservations may allow independent residual cards,
        # but the first operation itself is never sent a second time.
        original = self.calls('create')[0][1]
        self.make().poll()
        self.assertEqual(sum(call[1] == original for call in self.calls('create')), 1)
        self.adapter.unknown_create = False
        self.adapter.reconcile_enabled = True
        self.assertLessEqual(self.make().poll()['reserved_cards'], 7)
        self.assertEqual(sum(call[1] == original for call in self.calls('create')), 1)

    def test_account_headroom_and_run_budget_are_independent(self):
        self.adapter.foreign_cards = 251
        result = self.control.poll()
        self.assertEqual(result['reserved_cards'], 5)
        self.assertEqual([row['quota']['cards'] for row in self.adapter.resources.values()], [4, 1])

    def test_fixed_replica_never_splits_for_leftover_cards(self):
        self.control = self.make(replace(self.spec, replica_cards=4, workload_kind='service'))
        result = self.control.poll()
        self.assertEqual(result['reserved_cards'], 4)
        self.assertEqual(len(self.calls('create')), 1)

    def test_successful_batch_completes_and_is_not_recreated(self):
        self.control.poll(); self.running()
        for row in self.adapter.resources.values():
            row['health'].update(running=False, ready=False, completed=True, exit_code=0)
        self.assertEqual(self.control.poll()['phase'], 'stopping')
        self.assertEqual(self.make().poll()['phase'], 'completed')
        self.assertEqual(self.make().poll()['phase'], 'completed')
        self.assertEqual(len(self.calls('create')), 3)
        self.assertEqual(len(self.calls('stop')), 3)

    def test_stop_request_never_launches_new_commands_and_unknown_stop_is_not_replayed(self):
        self.control.poll()
        for row in self.adapter.resources.values(): row['state'] = 'RUNNING'
        self.adapter.unknown_stop = True
        request_stop(self.root, scope=self.spec.scope, profile=self.spec.profile)
        self.control.poll(); self.make().poll()
        self.assertEqual(len(self.calls('stop')), 3)
        self.assertEqual(len(self.calls('launch')), 0)
        self.assertEqual(self.make().poll()['reserved_cards'], 7)

    def test_nonzero_process_exit_retries_after_backoff_with_new_operation(self):
        self.control.poll(); self.running()
        resource = next(iter(self.adapter.resources))
        previous = self.adapter.resources[resource]['health']['launch_operation_id']
        self.adapter.resources[resource]['health'].update(running=False, completed=True, exit_code=3)
        self.control.poll()
        before = len(self.calls('launch'))
        self.control.poll()
        self.assertEqual(len(self.calls('launch')), before)
        self.now += 11
        self.control.poll()
        self.assertEqual(len(self.calls('launch')), before + 1)
        self.assertNotEqual(self.adapter.resources[resource]['health']['launch_operation_id'], previous)

    def test_proven_process_gone_retries_but_unconfirmed_absence_does_not(self):
        self.control.poll(); self.running()
        resource = next(iter(self.adapter.resources))
        row = self.adapter.resources[resource]
        row['health'].update(running=False, completed=False)
        count = len(self.calls('launch'))
        self.control.poll()
        self.assertEqual(len(self.calls('launch')), count)
        row['health']['process_gone'] = True
        self.control.poll()
        self.now += 11
        self.control.poll()
        self.assertEqual(len(self.calls('launch')), count + 1)

    def test_confirmed_container_replacement_relaunches_without_new_resource(self):
        self.control.poll(); self.running()
        resource = next(iter(self.adapter.resources))
        previous = self.adapter.resources[resource]['health']['launch_operation_id']
        self.adapter.resources[resource]['health'] = {'running': False, 'ready': False,
                                                      'completed': False, 'runtime_id': resource + ':boot2'}
        self.control.poll()
        self.assertNotEqual(self.adapter.resources[resource]['health']['launch_operation_id'], previous)
        self.assertEqual(len(self.calls('create')), 3)

    def test_unknown_launch_does_not_replay_without_new_runtime(self):
        self.control.poll()
        for row in self.adapter.resources.values(): row['state'] = 'RUNNING'
        def uncertain(identifier, command, operation_id):
            self.adapter.calls.append(('launch', identifier, tuple(command), operation_id))
            raise TimeoutError('fixture launch unknown')
        self.adapter.launch = uncertain
        self.control.poll(); self.make().poll()
        self.assertEqual(len(self.calls('launch')), 3)
        self.now += 301
        self.assertEqual(len(self.make().poll()['action_required']), 3)

    def test_platform_preemption_releases_only_stopped_and_refills_after_backoff(self):
        self.control.poll(); self.running()
        resource = next(iter(self.adapter.resources))
        self.adapter.resources[resource]['state'] = 'PENDING'
        self.assertEqual(self.control.poll()['reserved_cards'], 7)
        self.assertEqual(len(self.calls('create')), 3)
        self.adapter.resources[resource]['state'] = 'STOPPED'
        self.assertEqual(self.control.poll()['reserved_cards'], 3)
        self.assertEqual(len(self.calls('create')), 3)
        self.now += 11
        self.assertEqual(self.make().poll()['reserved_cards'], 7)
        self.assertEqual(len(self.calls('create')), 4)

    def test_service_requires_readiness_probe_and_does_not_complete_on_exit_zero(self):
        self.control = self.make(replace(self.spec, workload_kind='service', replica_cards=1))
        self.control.poll(); self.running()
        self.assertEqual(self.control.poll()['ready'], 0)
        for row in self.adapter.resources.values():
            row['health'].update(running=False, completed=True, exit_code=0)
        self.assertEqual(self.control.poll()['phase'], 'running')

    def test_service_migration_requires_drain_and_ready_floor(self):
        for policy, drain, ready in (('restartable', None, 0), ('drain', ('drain',), 0)):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'Service migration'):
                self.make(replace(self.spec, workload_kind='service', replica_cards=1,
                                  migration_policy=policy, drain_command=drain, min_ready_replicas=ready))

    def test_completion_probe_releases_workers(self):
        self.control = self.make(replace(self.spec, completion_command=('test', '-f', 'DONE')))
        self.control.poll(); self.running()
        self.adapter.complete = True
        self.assertEqual(self.control.poll()['phase'], 'stopping')
        self.assertEqual(len(self.calls('stop')), 3)

    def test_flexible_migration_preserves_heterogeneous_existing_quotas(self):
        self.adapter.nodes = {'a': (1, 8, 64), 'b': (2, 16, 128), 'c': (4, 32, 256)}
        self.control = self.make(replace(self.spec, migration_policy='restartable'))
        self.control.poll(); self.running()
        self.adapter.nodes['c'] = (7, 56, 448)
        for _ in range(10):
            result = self.control.poll()
            for row in self.adapter.resources.values():
                if row['state'] == 'PENDING': row['state'] = 'RUNNING'
        self.assertEqual(result['reserved_cards'], 7)
        remaining = [row for row in self.adapter.resources.values() if row['state'] != 'STOPPED']
        self.assertEqual({row['node'] for row in remaining}, {'c'})
        self.assertEqual(sorted(row['quota']['cards'] for row in remaining), [1, 2, 4])
        self.assertEqual(result['migration']['phase'], 'completed')

    def test_service_migration_drains_before_stop_and_preserves_ready_floor(self):
        self.adapter.nodes = {'a': (2, 16, 128), 'b': (1, 8, 64)}
        spec = replace(self.spec, max_cards=3, replica_cards=1, workload_kind='service',
                       migration_policy='drain', min_ready_replicas=2,
                       readiness_command=('ready',), drain_command=('drain',))
        self.control = self.make(spec)
        self.control.poll(); self.running()
        self.adapter.nodes['a'] = (3, 24, 192)
        result = self.control.poll()
        self.assertEqual(len(self.calls('drain')), 1)
        self.assertEqual(len(self.calls('stop')), 1)
        self.assertLess(self.adapter.calls.index(self.calls('drain')[0]),
                        self.adapter.calls.index(self.calls('stop')[0]))
        self.assertEqual(len(self.calls('create')), 3)
        self.assertGreaterEqual(sum(row['health'].get('ready', False) for row in self.adapter.resources.values()
                                    if row['state'] == 'RUNNING'), 2)
        self.control.poll()
        self.assertEqual(len(self.calls('create')), 4)

    def test_migration_create_crash_preserves_replacement_binding(self):
        self.adapter.nodes = {'a': (1, 8, 64), 'b': (2, 16, 128)}
        self.control = self.make(replace(self.spec, max_cards=3, migration_policy='restartable'))
        self.control.poll(); self.running()
        self.adapter.nodes['b'] = (3, 24, 192)
        self.control.poll()  # source stop
        original_create = self.adapter.create
        def crash(intent):
            original_create(intent)
            raise SystemExit('fixture process crash after Create')
        self.adapter.create = crash
        with self.assertRaises(SystemExit):
            self.control.poll()
        saved = json.loads((self.root / 'STATE.json').read_text())
        self.assertEqual(len(saved['migration']['replacement_ids']), 1)
        replacement_id = saved['migration']['replacement_ids'][0]
        self.assertIn(replacement_id, saved['resources'])
        count = len(self.calls('create'))
        self.adapter.create = original_create
        self.control = self.make(replace(self.spec, max_cards=3, migration_policy='restartable'))
        self.control.poll()
        self.assertEqual(len(self.calls('create')), count)
        self.assertEqual(self.control.state['migration']['phase'], 'completed')

    def test_same_host_controller_lock_and_identity_are_enforced(self):
        with self.control.locked():
            with self.assertRaises(BlockingIOError):
                self.make().poll()
        self.control.poll()
        with self.assertRaisesRegex(ValueError, 'identity changed|another workload'):
            self.make(replace(self.spec, command=('different',))).poll()

    def test_external_restart_of_retired_resource_is_reserved_but_not_migrated(self):
        self.adapter.nodes = {'a': (1, 8, 64), 'b': (2, 16, 128)}
        self.control = self.make(replace(self.spec, max_cards=3, migration_policy='restartable'))
        self.control.poll(); self.running()
        source = next(key for key, row in self.adapter.resources.items() if row['node'] == 'a')
        self.adapter.resources[source]['state'] = 'STOPPED'
        self.control.poll()
        self.adapter.resources[source]['state'] = 'RUNNING'
        self.adapter.nodes['b'] = (3, 24, 192)
        self.now += 11
        result = self.control.poll()
        self.assertEqual(result['reserved_cards'], 3)
        self.assertTrue(result['action_required'])
        self.assertEqual(self.calls('stop'), [])
        self.assertEqual(len(self.calls('create')), 2)

    def test_failed_identity_observation_blocks_new_allocation(self):
        self.make(replace(self.spec, max_cards=4)).poll()
        def unknown(identifier):
            raise ValueError('fixture identity mismatch')
        self.adapter.observe = unknown
        result = self.control.poll()
        self.assertEqual(result['reserved_cards'], 4)
        self.assertEqual(len(self.calls('create')), 1)

    def test_required_state_marker_prevents_budget_reset_and_precedes_first_send(self):
        original_create = self.adapter.create
        def checked_create(intent):
            self.assertTrue((self.root / 'STATE_REQUIRED.json').exists())
            self.assertTrue((self.root / 'STATE.json').exists())
            self.assertEqual((self.root / 'STATE.json').stat().st_mode & 0o777, 0o600)
            return original_create(intent)
        self.adapter.create = checked_create
        self.control.poll()
        before = len(self.calls('create'))
        (self.root / 'STATE.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'refusing an empty budget'):
            self.make().poll()
        self.assertEqual(len(self.calls('create')), before)
        # A crash after the required marker but before the first STATE write
        # leaves no platform request and also fails closed when resumed.
        with tempfile.TemporaryDirectory() as directory:
            controller = Controller(self.adapter, self.spec, directory, clock=lambda: self.now)
            def fail_state(path, value):
                if Path(path).name == 'STATE.json':
                    raise KeyboardInterrupt('fixture initialization crash')
                return atomic_json(path, value)
            with patch('corpus_pipeline.resource_manager.engine.atomic_json', side_effect=fail_state):
                with self.assertRaises(KeyboardInterrupt): controller.poll()
            self.assertTrue((Path(directory) / 'STATE_REQUIRED.json').exists())
            self.assertFalse((Path(directory) / 'STATE.json').exists())
            with self.assertRaisesRegex(RuntimeError, 'refusing an empty budget'):
                controller.poll()
            self.assertEqual(len(self.calls('create')), before)

    def test_continuous_run_never_overwrites_a_lost_or_corrupt_ledger(self):
        for damage in ('missing', 'corrupt'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                adapter = FakeAdapter()
                controller = Controller(adapter, replace(self.spec, poll_seconds=.001), directory,
                                        clock=lambda: self.now)
                state_path = Path(directory) / 'STATE.json'
                original_poll = controller.poll
                polls = [0]
                damaged_content = '{"schema":"corrupt-ledger"}\n'
                def damage_between_polls(**kwargs):
                    polls[0] += 1
                    result = original_poll(**kwargs)
                    if polls[0] == 1:
                        if damage == 'missing':
                            state_path.unlink()
                        else:
                            state_path.write_text(damaged_content)
                    return result
                controller.poll = damage_between_polls
                with self.assertRaises((ValueError, RuntimeError)):
                    controller.run(should_stop=lambda: polls[0] >= 2)
                self.assertEqual(polls[0], 2)
                self.assertIsNone(controller.state)
                self.assertEqual(len([call for call in adapter.calls if call[0] == 'create']), 3)
                if damage == 'missing':
                    self.assertFalse(state_path.exists())
                else:
                    self.assertEqual(state_path.read_text(), damaged_content)

    def test_changed_budget_never_releases_existing_reservations(self):
        self.control.poll()
        result = self.make(replace(self.spec, max_cards=4)).poll()
        self.assertEqual(result['reserved_cards'], 7)
        self.assertEqual(len(self.calls('create')), 3)
        self.assertEqual(len(self.calls('stop')), 0)


if __name__ == '__main__':
    unittest.main()
