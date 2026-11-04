"""No platform writes: injected snapshots cover quota, budget and send fencing."""
from contextlib import nullcontext
from copy import deepcopy
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import fragment_allocator as allocator
from corpus_pipeline.production.common import file_sha


def node(name, gpu=4, cpu=64, memory=512):
    return {'name': name, 'status': 'Ready', 'cordon_type': '', 'gpu_type': 'ASCEND_910B_64G',
            'gpu': {'available': gpu}, 'cpu': {'available': cpu}, 'memory': {'available': memory}}


def quotas():
    return allocator.quota_options([{'compute_group': allocator.GROUP,
        'quota': f'{cards},{cards * 8},{cards * 64}', 'allowed_priority_levels': ['LOW']}
        for cards in (1, 2, 4, 8, 16)])


class FakeLive:
    def __init__(self, nodes, *, used=0, records=None):
        self.inventory = nodes
        self.context = {'options': quotas(), 'owner_id': 'owner', 'workspace_id': 'ws',
            'project_id': 'project', 'group_id': 'group',
            'image': SimpleNamespace(ref=SimpleNamespace(key='image-fixture'), url=allocator.IMAGE_URL)}
        self.used, self.records = used, records or {}
        self.creates, self.reconciliations = [], []
        self.failure = None
        self.resolved = None
        self.discoveries = 0
        self.budgets = []

    def discover(self):
        self.discoveries += 1
        return self.context, deepcopy(self.inventory)

    def nodes(self, context):
        return deepcopy(self.inventory)

    def budget(self):
        used = self.budgets.pop(0) if self.budgets else self.used
        return {'observable_requested_cards': used, 'records': self.records}

    def plan(self, name, option, node_name, image):
        return {'name': name, 'project_id': 'project', 'project_name': allocator.PROJECT,
            'workspace_id': 'ws', 'logic_compute_group_id': 'group', 'node_id': node_name,
            'gpu_count': option['gpu'], 'cpu_count': option['cpu'], 'memory_size': option['memory'],
            'shared_memory_size': option['cpu'], 'image_id': image.ref.key, 'image_url': image.url,
            'task_priority': 3, 'is_publicpath_readonly': True, 'auto_stop': False,
            'quota_id': 'quota-' + option['quota']}

    def create(self, payload, operation_id):
        self.creates.append((payload, operation_id))
        if self.failure:
            raise self.failure
        return 'notebook-' + operation_id

    def reconcile(self, intent):
        self.reconciliations.append(intent)
        return self.resolved


class AllocatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'run'
        self.root.mkdir()
        (self.root / 'QUALIFIED.json').write_text(json.dumps({'success': True, 'dtype': 'float32',
            'inference_code_sha256': file_sha(Path(allocator.__file__).with_name('inference.py'))}))
        self.engine = allocator.Allocator(None, self.root, Path(self.temp.name) / 'execution', max_cards=256)

    def tearDown(self):
        self.temp.cleanup()

    def test_legal_live_quota_uses_smallest_cpu_for_largest_fitting_gpu(self):
        rows = [{'compute_group': allocator.GROUP, 'quota': value, 'allowed_priority_levels': ['LOW']}
                for value in ['1,4,64', '1,8,64', '4,16,256', '4,32,256', '3,4,64']]
        options = allocator.quota_options(rows)
        self.assertEqual(allocator.fit(node('fit', gpu=4, cpu=17), options, 256)['quota'], '4,16,256')
        self.assertEqual(allocator.fit(node('small', gpu=4, cpu=6), options, 256)['quota'], '1,4,64')
        self.assertIsNone(allocator.fit(node('memory', memory=63), options, 256))
        for invalid in [{**node('bad'), 'status': 'NotReady'}, {**node('bad'), 'cordon_type': 'CORDON'},
                        node(next(iter(allocator.EXCLUDED))), {**node('bad'), 'gpu_type': 'H100'}]:
            self.assertIsNone(allocator.fit(invalid, options, 256))
        unrestricted = allocator.quota_options([{'compute_group': allocator.GROUP,
            'quota': '16,64,1024', 'allowed_priority_levels': []}])
        self.assertEqual(unrestricted[0]['cpu'], 64)

    def test_pending_and_unknown_nodes_reserve_capacity_and_new_receipts_resume(self):
        live = self.engine.live = FakeLive([node('initial-a'), node('initial-b'), node('free')])
        managed = [{'notebook_id': 'initial-a-id', 'node': 'initial-a', 'devices': 4},
                   {'notebook_id': 'initial-b-id', 'node': 'initial-b', 'devices': 1}]
        added = self.engine._discover_and_allocate(managed,
            {'initial-a-id': {'platform_status': 'PENDING', 'observed_at': time.time()}})
        self.assertEqual([(row['node'], row['devices']) for row in added],
                         [('initial-b', 2), ('initial-b', 1), ('free', 4)])
        self.assertNotIn('initial-a', [row['node'] for row in added])
        resumed = allocator.Allocator(None, self.root, self.engine.execution, max_cards=256)
        self.assertEqual(resumed.load_managed(), managed + added)
        self.assertEqual(self.engine._discover_and_allocate(managed, {}), [])
        self.assertEqual(len(live.creates), 3)
        # A fresh RUNNING observation says its quota is now included in live
        # free capacity; exactly the newly available four cards can be filled.
        live.inventory = [node('initial-a'), node('free')]
        extra = self.engine._discover_and_allocate(managed,
            {'initial-a-id': {'platform_status': 'RUNNING', 'observed_at': time.time()}})
        self.assertEqual([(row['node'], row['devices']) for row in extra], [('initial-a', 4)])

    def test_budget_includes_missing_managed_and_newly_accepted_requests(self):
        live = self.engine.live = FakeLive([node('a', gpu=16), node('b', gpu=16)], used=250)
        managed = [{'notebook_id': 'queued-not-yet-listed', 'node': 'other', 'devices': 4}]
        added = self.engine._discover_and_allocate(managed, {'queued-not-yet-listed': {
            'platform_status': 'RUNNING', 'observed_at': time.time()}})
        self.assertEqual(sum(row['devices'] for row in added), 2)
        self.assertEqual(len(live.creates), 1)
        self.assertEqual(allocator.reserved_budget(live.budget(), managed + added, [])['reserved_total_cards'], 256)

    def test_budget_is_rechecked_after_planning(self):
        live = self.engine.live = FakeLive([node('a', gpu=16)], used=255)
        live.budgets = [100, 255]
        self.assertEqual(self.engine._discover_and_allocate([], {}), [])
        self.assertEqual(live.creates, [])

    def test_unknown_create_is_single_send_reserved_and_read_only_reconciled(self):
        live = self.engine.live = FakeLive([node('a'), node('b')])
        live.failure = TimeoutError('fixture uncertain create')
        with self.assertRaises(RuntimeError):
            self.engine._discover_and_allocate([], {})
        self.assertEqual(len(live.creates), 1)  # Error aborts the remaining candidate cycle.
        self.assertEqual(self.engine.reserved_unconfirmed_cards(), 4)
        live.inventory = [node('a')]
        self.engine._discover_and_allocate([], {})
        self.assertEqual(len(live.creates), 1)
        self.assertEqual(self.engine.reserved_unconfirmed_cards(), 4)
        live.resolved = 'late-accepted-id'
        self.root.joinpath('QUALIFIED.json').unlink()  # Cleanup reconciliation does not need model code.
        discovery_count = live.discoveries
        added = self.engine._discover_and_allocate([], {}, allow_create=False)
        self.assertEqual(added[0]['notebook_id'], 'late-accepted-id')
        self.assertTrue(added[0]['reconciled_read_only'])
        self.assertEqual(self.engine.reserved_unconfirmed_cards(), 0)
        self.assertEqual(live.discoveries, discovery_count)
        self.assertEqual(len(live.creates), 1)

    def test_exact_intent_is_durable_before_send(self):
        live = self.engine.live = FakeLive([node('a')])
        original = live.create
        def create(payload, operation_id):
            path = self.engine.directory / operation_id / 'INTENT.json'
            self.assertTrue(path.is_file())
            saved = json.loads(path.read_text())
            self.assertEqual(saved['create_kwargs'], payload)
            self.assertFalse(saved['retry_allowed'])
            self.assertEqual(payload['image_id'], 'image-fixture')
            self.assertEqual(payload['image_url'], allocator.IMAGE_URL)
            return original(payload, operation_id)
        live.create = create
        added = self.engine._discover_and_allocate([], {})
        self.assertEqual(len(added), 1)

    def test_explicit_rejection_releases_reservation_and_aborts_cycle(self):
        live = self.engine.live = FakeLive([node('a'), node('b')])
        live.failure = ValueError('fixture typed rejection supplied by boundary mock')
        with patch.object(allocator, 'definite_rejection', return_value='InvalidParameter'):
            with self.assertRaises(RuntimeError):
                self.engine._discover_and_allocate([], {})
        self.assertEqual(len(live.creates), 1)
        self.assertEqual(self.engine.reserved_unconfirmed_cards(), 0)
        self.assertEqual(len(list(self.engine.directory.glob('*/REJECTED.json'))), 1)
        self.engine._discover_and_allocate([], {}, allow_create=False)
        self.assertEqual(live.reconciliations, [])

    def test_busy_nodes_do_not_trigger_budget_or_node_requeries(self):
        live = self.engine.live = FakeLive([node(f'busy-{i}', gpu=0) for i in range(100)])
        with patch.object(live, 'budget') as budget, patch.object(live, 'nodes') as nodes:
            self.assertEqual(self.engine._discover_and_allocate([], {}), [])
            budget.assert_not_called()
            nodes.assert_not_called()

    def test_only_fresh_running_observation_releases_node_reservation(self):
        live = self.engine.live = FakeLive([node('a')], records={'existing': {'cards': 4}}, used=4)
        managed = [{'notebook_id': 'existing', 'node': 'a', 'devices': 4}]
        self.engine._discover_and_allocate(managed, {'existing': {'platform_status': 'RUNNING', 'observed_at': 0}})
        self.assertEqual(live.creates, [])
        added = self.engine._discover_and_allocate(managed,
            {'existing': {'platform_status': 'RUNNING', 'observed_at': time.time()}})
        self.assertEqual(len(added), 1)

    def test_qualification_failure_prevents_any_platform_probe(self):
        live = self.engine.live = FakeLive([node('a')])
        (self.root / 'QUALIFIED.json').write_text(json.dumps({'success': False}))
        with self.assertRaises(ValueError):
            self.engine._discover_and_allocate([], {})
        self.assertEqual(live.discoveries, 0)
        self.assertEqual(live.creates, [])

    def test_mutated_accepted_registry_is_rejected(self):
        self.engine.live = FakeLive([node('a')])
        self.engine._discover_and_allocate([], {})
        path = next(self.engine.directory.glob('*/ACCEPTED.json'))
        data = json.loads(path.read_text()); data['node'] = 'other'
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.engine.load_managed()

    def test_verified_retirement_releases_missing_budget_but_not_live_active_usage(self):
        item = {'notebook_id': 'retired-id', 'node': 'a', 'devices': 4}
        states = {item['notebook_id']: {'platform_status': 'STOPPED', 'retirement': {
            'notebook_id': item['notebook_id'], 'node': 'a', 'phase': 'verified_stopped',
            'verified_stopped_at': 1000, 'cooldown_until': 1120}}}
        snapshot = {'observable_requested_cards': 100, 'records': {}}
        self.assertEqual(allocator.reserved_budget(snapshot, [item], [], states)['reserved_total_cards'], 100)
        # Active platform inventory always remains authoritative even if our
        # earlier exact observation said STOPPED.
        snapshot = {'observable_requested_cards': 104, 'records': {'retired-id': {}}}
        self.assertEqual(allocator.reserved_budget(snapshot, [item], [], states)['reserved_total_cards'], 104)
        for change in ({'phase': 'stop_requested'}, {'notebook_id': 'wrong'},
                       {'node': 'wrong'}, {'verified_stopped_at': None}):
            bad = deepcopy(states)
            bad['retired-id']['retirement'].update(change)
            self.assertEqual(allocator.reserved_budget(
                {'observable_requested_cards': 100, 'records': {}}, [item], [], bad)['reserved_total_cards'], 104)

    def test_retired_cooldown_survives_restart_then_allows_fresh_replacement(self):
        item = {'notebook_id': 'retired-id', 'node': 'a', 'devices': 4}
        states = {item['notebook_id']: {'platform_status': 'STOPPED', 'retirement': {
            'notebook_id': item['notebook_id'], 'node': 'a', 'phase': 'verified_stopped',
            'verified_stopped_at': 1000, 'cooldown_until': 1120}}}
        states = json.loads(json.dumps(states))
        live = self.engine.live = FakeLive([node('a'), node('b')], used=252)
        with patch.object(allocator.time, 'time', return_value=1100):
            added = self.engine._discover_and_allocate([item], states)
        self.assertEqual([row['node'] for row in added], ['b'])
        self.assertEqual(allocator.node_cooldowns(states, 1100), {'a': 1120})
        self.assertEqual(allocator.node_cooldowns(states, 1120), {})
        # A new controller sees the persisted retirement and can use the node
        # after cooldown, if a fresh quota and account budget still fit.
        resumed = allocator.Allocator(None, self.root, self.engine.execution, max_cards=256)
        resumed.live = FakeLive([node('a')], used=248)
        states[added[0]['notebook_id']] = {'platform_status': 'RUNNING', 'observed_at': 1121}
        with patch.object(allocator.time, 'time', return_value=1121):
            replacement = resumed._discover_and_allocate([item], states)
        self.assertEqual([row['node'] for row in replacement], ['a'])

    def test_unconfirmed_retirement_keeps_node_and_budget_reserved(self):
        item = {'notebook_id': 'retiring-id', 'node': 'a', 'devices': 4}
        states = {'retiring-id': {'platform_status': 'PENDING', 'retirement': {
            'notebook_id': 'retiring-id', 'node': 'a', 'phase': 'stop_sending',
            'stop_attempted_at': 1000, 'cooldown_until': 1120}}}
        live = self.engine.live = FakeLive([node('a'), node('b')], used=252)
        with patch.object(allocator.time, 'time', return_value=2000):
            self.assertEqual(self.engine._discover_and_allocate([item], states), [])
        self.assertEqual(live.creates, [])
        self.assertEqual(allocator.reserved_budget(live.budget(), [item], [], states)['reserved_total_cards'], 256)

    def test_pending_fit_uses_exact_accepted_cpu_memory_and_fresh_nodes(self):
        item = {'notebook_id': 'id', 'node': 'a', 'devices': 4,
                'cpu': 32, 'memory': 256, 'workspace_id': 'ws', 'group_id': 'group'}
        live = self.engine.live = FakeLive([node('a', gpu=4, cpu=31)])
        self.assertFalse(self.engine.pending_fit(item)['fits'])
        live.inventory = [node('a', gpu=4, cpu=32)]
        self.assertTrue(self.engine.pending_fit(item)['fits'])
        live.inventory = [node('a', gpu=0, cpu=64)]
        self.assertFalse(self.engine.pending_fit(item)['fits'])

    def test_missing_cap_prevents_create_but_allows_read_only_reconciliation(self):
        engine = allocator.Allocator(None, self.root, self.engine.execution)
        live = engine.live = FakeLive([node('free')])
        with self.assertRaises(ValueError):
            engine._discover_and_allocate([], {})
        self.assertEqual(live.discoveries, 0)
        self.assertEqual(live.creates, [])
        self.root.joinpath('QUALIFIED.json').unlink()
        self.assertEqual(engine._discover_and_allocate([], {}, allow_create=False), [])
        for invalid in (0, -1, 257, True, False, 48.0, '48'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                allocator.Allocator(None, self.root, self.engine.execution, max_cards=invalid)

    def test_run_cap_48_limits_all_creates_in_one_cycle_and_next_restart(self):
        self.engine.max_cards = 48
        live = self.engine.live = FakeLive([node(name, gpu=16, cpu=128, memory=1024)
                                           for name in ('a', 'b', 'c', 'd')])
        managed = [{'notebook_id': 'existing-' + name, 'node': name, 'devices': cards}
                   for name, cards in (('a', 16), ('b', 16), ('c', 4), ('d', 1))]
        states = {item['notebook_id']: {'platform_status': 'RUNNING', 'observed_at': time.time()}
                  for item in managed}
        added = self.engine._discover_and_allocate(managed, states)
        self.assertEqual([item['devices'] for item in added], [8, 2, 1])
        self.assertEqual(self.engine.managed_budget([], {})['reserved_total_cards'], 48)
        self.assertEqual(self.engine.managed_budget([], {})['remaining_cards'], 0)
        resumed = allocator.Allocator(None, self.root, self.engine.execution, max_cards=48)
        resumed.live = live
        discovery_count = live.discoveries
        self.assertEqual(resumed._discover_and_allocate([], {}), [])
        self.assertEqual(live.discoveries, discovery_count)
        self.assertEqual(len(live.creates), 3)

    def test_account_ceiling_can_be_stricter_than_run_cap(self):
        self.engine.max_cards = 48
        live = self.engine.live = FakeLive([node('a', gpu=16, cpu=128, memory=1024)], used=254)
        added = self.engine._discover_and_allocate([], {})
        self.assertEqual([row['devices'] for row in added], [2])
        budget = self.engine.reserved_budget(live.budget(), [], {})
        self.assertEqual(budget['reserved_total_cards'], 256)
        self.assertEqual(budget['managed_budget']['reserved_total_cards'], 2)
        self.assertEqual(budget['remaining_cards'], 0)

    def test_run_cap_is_rechecked_after_planning(self):
        self.engine.max_cards = 48
        live = self.engine.live = FakeLive([node('a', gpu=16, cpu=128, memory=1024)])
        plan = live.plan
        def changed_budget(*args):
            payload = plan(*args)
            self.engine.managed_budget([{'notebook_id': 'accepted-during-plan',
                                         'node': 'elsewhere', 'devices': 47}], {})
            return payload
        live.plan = changed_budget
        self.assertEqual(self.engine._discover_and_allocate([], {}), [])
        self.assertEqual(live.creates, [])
        self.assertEqual(list(self.engine.directory.glob('*/INTENT.json')), [])

    def test_unknown_stopped_failed_and_pending_all_reserve_deduplicated_ids(self):
        managed = [{'notebook_id': str(i), 'node': 'old-' + str(i), 'devices': 8}
                   for i in range(4)]
        states = {str(i): {'platform_status': state}
                  for i, state in enumerate(('PENDING', 'STOPPED', 'FAILED', 'UNKNOWN'))}
        budget = allocator.managed_reserved_budget(managed + managed, [], states, 48)
        self.assertEqual(budget['accepted_reserved_cards'], 32)
        self.assertEqual(budget['remaining_cards'], 16)
        conflict = managed[0] | {'devices': 1}
        with self.assertRaises(ValueError):
            allocator.managed_reserved_budget(managed + [conflict], max_cards=48)

    def test_only_verified_retirement_releases_run_budget_and_live_reactivation_reclaims_it(self):
        item = {'notebook_id': 'old', 'node': 'a', 'devices': 8}
        state = {'platform_status': 'STOPPED', 'retirement': {'phase': 'verified_stopped',
            'notebook_id': 'old', 'node': 'a', 'verified_stopped_at': 123}}
        self.engine.max_cards = 8
        states = {'old': state}
        self.assertEqual(self.engine.managed_budget([item], states)['remaining_cards'], 8)
        snapshot = {'records': {'old': {'cards': 8}}, 'observable_requested_cards': 8}
        budget = self.engine.reserved_budget(snapshot, [], states)
        self.assertEqual(budget['managed_budget']['reserved_total_cards'], 8)
        self.assertEqual(budget['remaining_cards'], 0)
        self.assertEqual(self.engine.managed_budget([], states, live=snapshot)['remaining_cards'], 0)
        self.assertEqual(states['old'], state)  # Budget never mutates retirement evidence.
        self.engine.live = FakeLive([node('new')], used=8, records=snapshot['records'])
        self.assertEqual(self.engine._discover_and_allocate([], states), [])
        self.assertEqual(self.engine.live.creates, [])
        self.assertEqual(self.engine.managed_budget([], {})['remaining_cards'], 0)

    def test_accepted_directory_disappearance_does_not_release_run_cap(self):
        self.engine.max_cards = 4
        live = self.engine.live = FakeLive([node('a')])
        accepted = self.engine._discover_and_allocate([], {})
        shutil.rmtree(self.engine.directory)
        resumed = allocator.Allocator(None, self.root, Path(self.temp.name) / 'new-execution', max_cards=4)
        resumed.live = FakeLive([node('free')])
        self.assertEqual(resumed.load_managed(), accepted)
        self.assertEqual(resumed.managed_budget([], {})['reserved_total_cards'], 4)
        self.assertEqual(resumed._discover_and_allocate([], {}), [])
        self.assertEqual(resumed.live.creates, [])

    def test_unconfirmed_directory_disappearance_keeps_cap_and_new_execution_reconciles_existing(self):
        self.engine.max_cards = 4
        live = self.engine.live = FakeLive([node('a')])
        live.failure = TimeoutError('uncertain')
        with self.assertRaises(RuntimeError):
            self.engine._discover_and_allocate([], {})
        resumed = allocator.Allocator(None, self.root, Path(self.temp.name) / 'new-execution', max_cards=4)
        resumed.live = FakeLive([node('free')])
        self.assertEqual(resumed.reserved_unconfirmed_cards(), 4)
        self.assertEqual(resumed._discover_and_allocate([], {}), [])
        self.assertEqual(len(resumed.live.reconciliations), 1)
        self.assertEqual(resumed.live.creates, [])
        shutil.rmtree(self.engine.directory)
        self.assertEqual(resumed._discover_and_allocate([], {}), [])
        self.assertEqual(resumed.managed_budget([], {})['reserved_total_cards'], 4)

    def test_at_cap_still_reconciles_pending_into_same_accepted_reservation(self):
        self.engine.max_cards = 4
        live = self.engine.live = FakeLive([node('a')])
        live.failure = TimeoutError('uncertain')
        with self.assertRaises(RuntimeError):
            self.engine._discover_and_allocate([], {})
        live.failure, live.resolved = None, 'late-id'
        added = self.engine._discover_and_allocate([], {})
        self.assertEqual([row['notebook_id'] for row in added], ['late-id'])
        self.assertEqual(self.engine.managed_budget([], {})['accepted_reserved_cards'], 4)
        self.assertEqual(self.engine.reserved_unconfirmed_cards(), 0)
        self.assertEqual(len(live.creates), 1)

    def test_confirmed_rejection_stays_released_if_receipt_disappears(self):
        live = self.engine.live = FakeLive([node('a')])
        live.failure = ValueError('rejected')
        with patch.object(allocator, 'definite_rejection', return_value='InvalidParameter'):
            with self.assertRaises(RuntimeError):
                self.engine._discover_and_allocate([], {})
        next(self.engine.directory.glob('*/REJECTED.json')).unlink()
        self.assertEqual(self.engine.reserved_unconfirmed_cards(), 0)
        self.engine._discover_and_allocate([], {}, allow_create=False)
        self.assertEqual(live.reconciliations, [])

    def test_ledger_loss_or_corruption_fails_closed(self):
        item = {'notebook_id': 'old', 'node': 'a', 'devices': 8}
        self.engine.managed_budget([item], {})
        saved = self.engine.reservations_path.read_text()
        self.engine.reservations_path.unlink()
        resumed = allocator.Allocator(None, self.root, self.engine.execution, max_cards=48)
        resumed.live = FakeLive([node('free')])
        with self.assertRaises(ValueError):
            resumed._discover_and_allocate([], {})
        resumed.reservations_path.write_text(saved)
        ledger = json.loads(saved)
        ledger['notebooks'] = []
        resumed.reservations_path.write_text(json.dumps(ledger))
        with self.assertRaises(ValueError):
            resumed.managed_budget([], {})
        self.assertEqual(resumed.live.creates, [])

    def test_stop_requested_during_planning_prevents_intent_and_create(self):
        live = self.engine.live = FakeLive([node('a')])
        stopped = False
        plan = live.plan
        def stop_during_plan(*args):
            nonlocal stopped
            payload = plan(*args)
            stopped = True
            return payload
        live.plan = stop_during_plan
        self.assertEqual(self.engine._discover_and_allocate([], {}, should_stop=lambda: stopped), [])
        self.assertEqual(live.creates, [])
        self.assertEqual(list(self.engine.directory.glob('*/INTENT.json')), [])

    def test_stop_after_first_accept_does_not_create_remaining_candidates(self):
        live = self.engine.live = FakeLive([node('a'), node('b')])
        added = self.engine._discover_and_allocate([], {}, should_stop=lambda: bool(live.creates))
        self.assertEqual(len(added), 1)
        self.assertEqual(len(live.creates), 1)

    def test_allocation_packs_existing_node_before_larger_new_node(self):
        self.engine.max_cards = 8
        item = {'notebook_id': 'existing', 'node': 'packed', 'devices': 4}
        states = {'existing': {'platform_status': 'RUNNING', 'observed_at': time.time()}}
        live = self.engine.live = FakeLive([node('new', gpu=16, cpu=128, memory=1024), node('packed')])
        added = self.engine._discover_and_allocate([item], states)
        self.assertEqual([(row['node'], row['devices']) for row in added], [('packed', 4)])
        self.assertEqual(len(live.creates), 1)

    def test_packing_prefers_node_with_most_existing_owned_cards(self):
        self.engine.max_cards = 11
        items = [{'notebook_id': 'a', 'node': 'small', 'devices': 2},
                 {'notebook_id': 'b', 'node': 'large', 'devices': 8}]
        states = {item['notebook_id']: {'platform_status': 'RUNNING', 'observed_at': time.time()}
                  for item in items}
        self.engine.live = FakeLive([node('small'), node('large', gpu=1), node('unused', gpu=16)])
        added = self.engine._discover_and_allocate(items, states)
        self.assertEqual([(row['node'], row['devices']) for row in added], [('large', 1)])

    def test_new_node_order_prefers_legal_quota_fit_over_declared_free_cards(self):
        self.engine.max_cards = 4
        self.engine.live = FakeLive([node('sixteen-gpu-one-quota', gpu=16, cpu=8),
                                    node('four-gpu-four-quota', gpu=4)])
        added = self.engine._discover_and_allocate([], {})
        self.assertEqual([(row['node'], row['devices']) for row in added], [('four-gpu-four-quota', 4)])

    def test_targeted_allocation_creates_one_exact_quota_only_at_destination(self):
        self.engine.max_cards = 48
        live = self.engine.live = FakeLive([node('elsewhere', gpu=16, cpu=128, memory=1024),
                                           node('destination', gpu=8)])
        added = self.engine._discover_and_allocate([], {}, target_node='destination', target_cards=4)
        self.assertEqual([(row['node'], row['devices']) for row in added], [('destination', 4)])
        self.assertEqual(len(live.creates), 1)

    def test_targeted_allocation_never_falls_back_to_other_node_or_smaller_quota(self):
        self.engine.max_cards = 48
        live = self.engine.live = FakeLive([node('elsewhere', gpu=16, cpu=128, memory=1024),
                                           node('destination', gpu=2)])
        self.assertEqual(self.engine._discover_and_allocate(
            [], {}, target_node='destination', target_cards=4), [])
        self.assertEqual(self.engine._discover_and_allocate(
            [], {}, target_node='absent', target_cards=4), [])
        live.inventory = [node('destination', gpu=8)]
        self.engine.max_cards = 2
        self.assertEqual(self.engine._discover_and_allocate(
            [], {}, target_node='destination', target_cards=4), [])
        self.assertEqual(live.creates, [])

    def test_targeted_unknown_create_is_not_resent(self):
        self.engine.max_cards = 48
        live = self.engine.live = FakeLive([node('destination', gpu=8), node('elsewhere')])
        live.failure = TimeoutError('uncertain')
        with self.assertRaises(RuntimeError):
            self.engine._discover_and_allocate([], {}, target_node='destination', target_cards=4)
        live.failure = None
        self.assertEqual(self.engine._discover_and_allocate(
            [], {}, target_node='destination', target_cards=4), [])
        self.assertEqual(len(live.creates), 1)
        self.assertEqual(self.engine.reserved_unconfirmed_cards(), 4)

    def test_targeted_interface_rejects_partial_or_invalid_placement(self):
        live = self.engine.live = FakeLive([node('a')])
        for target, cards in (('a', None), (None, 4), ('', 4), ('a', 3), ('a', True),
                              (next(iter(allocator.EXCLUDED)), 4)):
            with self.subTest(target=target, cards=cards), self.assertRaises(ValueError):
                self.engine._discover_and_allocate([], {}, target_node=target, target_cards=cards)
        self.assertEqual(live.discoveries, 0)
        self.assertEqual(live.creates, [])

    def test_seven_cards_pack_one_node_without_waiting_for_each_smaller_request(self):
        self.engine.max_cards = 7
        live = self.engine.live = FakeLive([node(name, gpu=16, cpu=128, memory=1024)
                                           for name in ('a', 'b', 'c')])
        accepted = self.engine._discover_and_allocate([], {})
        self.assertEqual([(item['node'], item['devices']) for item in accepted],
                         [('a', 4), ('a', 2), ('a', 1)])
        self.assertEqual(self.engine._discover_and_allocate([], {}), [])
        self.assertEqual(len(live.creates), 3)
        self.assertEqual(self.engine.managed_budget([], {})['reserved_total_cards'], 7)

    def test_one_running_card_plus_seven_free_packs_four_two_one_on_same_node(self):
        self.engine.max_cards = 8
        item = {'notebook_id': 'one', 'node': 'existing', 'devices': 1}
        state = {'one': {'platform_status': 'RUNNING', 'observed_at': time.time()}}
        self.engine.live = FakeLive([node('new', gpu=16, cpu=128, memory=1024),
                                    node('existing', gpu=7, cpu=56, memory=448)])
        rows = self.engine._discover_and_allocate([item], state)
        self.assertEqual([(row['node'], row['devices']) for row in rows],
                         [('existing', 4), ('existing', 2), ('existing', 1)])

    def test_exact_target_quota_preserves_cpu_and_memory_of_migration_plan(self):
        self.engine.max_cards = 4
        live = self.engine.live = FakeLive([node('target', gpu=4)])
        requested = {'quota': '4,64,256', 'gpu': 4, 'cpu': 64, 'memory': 256}
        live.context['options'].append(requested)
        rows = self.engine._discover_and_allocate([], {}, target_node='target', target_cards=4,
            target_quota=requested, allocation_tag='exact-quota')
        self.assertEqual((rows[0]['cpu'], rows[0]['memory']), (64, 256))
        changed = {'quota': '4,32,256', 'gpu': 4, 'cpu': 32, 'memory': 256}
        with self.assertRaisesRegex(ValueError, 'CPU or memory'):
            self.engine._discover_and_allocate([], {}, target_node='target', target_cards=4,
                target_quota=changed, allocation_tag='exact-quota')

    def test_pending_reservations_do_not_prevent_using_other_available_capacity(self):
        self.engine.max_cards = 16
        managed = [{'notebook_id': 'pending', 'node': 'pending-node', 'devices': 4},
                   {'notebook_id': 'running', 'node': 'running-node', 'devices': 4}]
        states = {'pending': {'platform_status': 'PENDING', 'observed_at': time.time()},
                  'running': {'platform_status': 'RUNNING', 'observed_at': time.time()}}
        self.engine.live = FakeLive([node('new-node', gpu=16, cpu=128, memory=1024),
                                    node('running-node', gpu=2), node('pending-node', gpu=4)])
        added = self.engine._discover_and_allocate(managed, states)
        self.assertEqual([(item['node'], item['devices']) for item in added],
                         [('running-node', 2), ('new-node', 4), ('new-node', 2)])

    def test_allocation_tag_is_durable_and_never_sends_a_second_create(self):
        self.engine.max_cards = 48
        live = self.engine.live = FakeLive([node('target', gpu=8)])
        self.assertEqual(self.engine.allocation_tag_status('plan-1')['status'], 'not_sent')
        accepted = self.engine._discover_and_allocate([], {}, target_node='target', target_cards=4,
                                                       allocation_tag='plan-1')
        self.assertEqual(accepted[0]['allocation_tag'], 'plan-1')
        status = self.engine.allocation_tag_status('plan-1')
        self.assertEqual(status['status'], 'accepted')
        self.assertEqual(status['notebook_id'], accepted[0]['notebook_id'])
        states = {accepted[0]['notebook_id']: {'platform_status': 'RUNNING', 'observed_at': time.time()}}
        self.assertEqual(self.engine._discover_and_allocate([], states, target_node='target',
                         target_cards=4, allocation_tag='plan-1'), accepted)
        shutil.rmtree(self.engine.directory)
        resumed = allocator.Allocator(None, self.root, Path(self.temp.name) / 'new-execution', max_cards=48)
        resumed.live = live
        self.assertEqual(resumed._discover_and_allocate([], states, target_node='target',
                         target_cards=4, allocation_tag='plan-1'), accepted)
        self.assertEqual(len(live.creates), 1)
        with self.assertRaises(ValueError):
            resumed._discover_and_allocate([], states, target_node='other', target_cards=4,
                                            allocation_tag='plan-1')

    def test_allocation_tag_pending_reconciles_to_accepted_without_resend(self):
        live = self.engine.live = FakeLive([node('target')])
        live.failure = TimeoutError('uncertain')
        with self.assertRaises(RuntimeError):
            self.engine._discover_and_allocate([], {}, target_node='target', target_cards=4,
                                                allocation_tag='plan-1')
        self.assertEqual(self.engine.allocation_tag_status('plan-1')['status'], 'pending')
        live.failure, live.resolved = None, 'resolved-id'
        accepted = self.engine._discover_and_allocate([], {}, target_node='target', target_cards=4,
                                                       allocation_tag='plan-1')
        self.assertEqual(accepted[0]['notebook_id'], 'resolved-id')
        self.assertEqual(accepted[0]['allocation_tag'], 'plan-1')
        self.assertEqual(self.engine.allocation_tag_status('plan-1')['status'], 'accepted')
        self.assertEqual(len(live.creates), 1)

    def test_allocation_tag_rejection_releases_cards_but_not_permission_to_resend(self):
        live = self.engine.live = FakeLive([node('target')])
        live.failure = ValueError('rejected')
        with patch.object(allocator, 'definite_rejection', return_value='InvalidParameter'):
            with self.assertRaises(RuntimeError):
                self.engine._discover_and_allocate([], {}, target_node='target', target_cards=4,
                                                    allocation_tag='plan-1')
        self.assertEqual(self.engine.allocation_tag_status('plan-1')['status'], 'rejected')
        next(self.engine.directory.glob('*/REJECTED.json')).unlink()
        self.assertEqual(self.engine._discover_and_allocate([], {}, target_node='target',
                         target_cards=4, allocation_tag='plan-1'), [])
        self.assertEqual(len(live.creates), 1)
        self.assertEqual(self.engine.reserved_unconfirmed_cards(), 0)

    def test_allocation_tag_requires_exact_placement_and_bounded_identifier(self):
        self.engine.live = FakeLive([node('a')])
        with self.assertRaises(ValueError):
            self.engine._discover_and_allocate([], {}, allocation_tag='plan')
        for invalid in ('', 'bad/path', 'a' * 129, 1):
            with self.subTest(tag=invalid), self.assertRaises(ValueError):
                self.engine._discover_and_allocate([], {}, target_node='a', target_cards=4,
                                                    allocation_tag=invalid)
        self.assertEqual(self.engine.live.creates, [])

    def test_same_run_different_executions_share_nonblocking_control_host_lock(self):
        other = allocator.Allocator(None, self.root, Path(self.temp.name) / 'other-execution', max_cards=48)
        other.live = FakeLive([node('free')])
        self.assertEqual(self.engine.lock_path, other.lock_path)
        self.assertEqual(self.engine.lock_path.parent, Path(tempfile.gettempdir()))
        different_run = allocator.Allocator(None, Path(self.temp.name) / 'other-run',
                                           Path(self.temp.name) / 'third-execution', max_cards=48)
        self.assertNotEqual(other.lock_path, different_run.lock_path)
        self.addCleanup(lambda: self.engine.lock_path.unlink(missing_ok=True))
        with self.engine.lock_path.open('a') as locked:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch('corpus_pipeline.cluster.text_worker.verify_control_host') as host_check:
                with self.assertRaises(BlockingIOError):
                    other.discover_and_allocate([], {}, allow_create=False)
                host_check.assert_called_once_with(self.root)
        self.assertEqual(other.live.discoveries, 0)
        self.assertEqual(other.live.creates, [])
        with patch('corpus_pipeline.cluster.text_worker.verify_control_host'):
            self.assertEqual(other.discover_and_allocate([], {}, allow_create=False), [])


@unittest.skipUnless(importlib.util.find_spec('inspire'), 'SDK interpreter required for SDK boundary test')
class SDKBoundaryTests(unittest.TestCase):
    def test_only_typed_definite_business_rejection_releases_budget(self):
        from inspire.platform.errors import ValidationError, SubmissionUncertainError
        self.assertEqual(allocator.definite_rejection(ValidationError('API error: InvalidParameter: fixture')), 'InvalidParameter')
        for error in (ValueError('API error: InvalidParameter: fixture'), TimeoutError('timeout'),
                      ValidationError('HTTP 500: ambiguous'), SubmissionUncertainError('id')):
            self.assertIsNone(allocator.definite_rejection(error))

    def test_plan_binds_actual_image_id_separately_from_url(self):
        image = SimpleNamespace(ref=SimpleNamespace(key='image-actual'), url=allocator.IMAGE_URL)
        option = {'quota': '1,4,64', 'gpu': 1, 'cpu': 4, 'memory': 64}
        def plan(spec):
            self.assertIs(spec.image, image.ref)
            self.assertEqual(spec.quota, '1,4,64')
            self.assertEqual(spec.priority, 3)
            return SimpleNamespace(workspace=SimpleNamespace(name=allocator.WORKSPACE),
                project=SimpleNamespace(name=allocator.PROJECT), group=SimpleNamespace(name=allocator.GROUP),
                priority=3, create_kwargs={'gpu_count': 1, 'cpu_count': 4, 'memory_size': 64,
                    'task_priority': 3, 'is_publicpath_readonly': True, 'node_id': 'fixture-node',
                    'image_id': allocator.IMAGE_URL, 'image_url': allocator.IMAGE_URL})
        live = allocator.Live(SimpleNamespace(notebooks=SimpleNamespace(plan=plan)))
        payload = live.plan('fixture', option, 'fixture-node', image)
        self.assertEqual(payload['image_id'], 'image-actual')
        self.assertEqual(payload['image_url'], allocator.IMAGE_URL)

    def test_reconcile_requires_exact_owner_project_and_payload(self):
        payload = FakeLive([]).plan('fixture', quotas()[-1], 'node',
            SimpleNamespace(ref=SimpleNamespace(key='image-fixture'), url=allocator.IMAGE_URL))
        intent = {'create_kwargs': payload, 'owner_id': 'owner'}
        raw = {**payload, 'user_id': 'owner', 'mirror_id': payload['image_id'], 'mirror_url': payload['image_url']}
        client = SimpleNamespace(notebooks=SimpleNamespace(iter=lambda **_: [SimpleNamespace(name='fixture',
            ref=SimpleNamespace(key='exact-id'), priority=3, priority_level='LOW')]),
            _transport=SimpleNamespace(scope=lambda **_: nullcontext(), session=object()))
        live = allocator.Live(client)
        target = 'inspire.platform.web.browser_api.notebooks.get_notebook_detail'
        with patch(target, return_value=raw):
            self.assertEqual(live.reconcile(intent), 'exact-id')
        for change in ({'user_id': 'another-owner'}, {'project_id': 'wrong-project'},
                       {'cpu_count': 999}, {'mirror_id': allocator.IMAGE_URL}):
            with patch(target, return_value=raw | change):
                self.assertIsNone(live.reconcile(intent))
        nested = {'name': payload['name'], 'creator': {'id': 'owner'},
            'project': {'id': payload['project_id']}, 'workspace': {'id': payload['workspace_id']},
            'is_publicpath_readonly': True,
            'start_config': {**payload, 'mirror_id': payload['image_id'],
                             'mirror_url': payload['image_url'], 'project_priority': {'priority': 13}}}
        with patch(target, return_value=nested):
            self.assertEqual(live.reconcile(intent), 'exact-id')
        nested['start_config']['node_id'] = 'wrong-node'
        with patch(target, return_value=nested):
            self.assertIsNone(live.reconcile(intent))


if __name__ == '__main__':
    unittest.main()
