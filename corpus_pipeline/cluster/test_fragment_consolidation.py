from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from corpus_pipeline.cluster.fragment_allocator import managed_reserved_budget
from corpus_pipeline.cluster.fragment_controller import WORKSPACE, PROJECT, GROUP
from corpus_pipeline.cluster.fragment_consolidation import Consolidator, plan_consolidation
from corpus_pipeline.production.common import digest


def item(identifier, node, cards):
    return {'notebook_id': identifier, 'name': 'test-' + identifier,
            'node': node, 'devices': cards, 'priority': 3,
            'workspace_id': 'workspace', 'project_id': 'project', 'group_id': 'group',
            'cpu': cards * 8, 'memory': cards * 64}


def node(name, free, *, cpu=None, memory=None):
    return {'name': name, 'status': 'Ready', 'cordon_type': '',
            'gpu_type': 'ASCEND_910B_64G', 'gpu': {'available': free},
            'cpu': {'available': free * 8 if cpu is None else cpu},
            'memory': {'available': free * 64 if memory is None else memory}}


OPTIONS = [{'gpu': n, 'cpu': 8*n, 'memory': 64*n, 'quota': f'{n},{8*n},{64*n}'}
           for n in (16, 8, 4, 2, 1)]


def fresh(rows, now=1000):
    return {row['notebook_id']: {'platform_status': 'RUNNING', 'observed_at': now,
                                'node': {'name': row['node']}}
            for row in rows}


class FakeLive:
    def __init__(self, nodes):
        self.node_rows = nodes
        self.calls = 0

    def discover(self):
        self.calls += 1
        return {'options': OPTIONS}, copy.deepcopy(self.node_rows)


class FakeAllocator:
    def __init__(self, managed, nodes, max_cards=48):
        self.managed = managed
        self.live = FakeLive(nodes)
        self.max_cards = max_cards
        self.tags = {}
        self.create_count = 0
        self.mode = 'accepted'
        self.before_create = None
        self.requests = []

    def managed_budget(self, managed, states):
        return managed_reserved_budget(managed, (), states, self.max_cards)

    def allocation_tag_status(self, tag):
        return self.tags.get(tag, {'status': 'not_sent'})

    def discover_and_allocate(self, managed, states, *, target_node, target_cards,
                              target_quota, allocation_tag, should_stop):
        if should_stop():
            return []
        current = self.tags.get(allocation_tag)
        if current:
            return []
        self.create_count += 1
        self.requests.append({'node': target_node, 'cards': target_cards,
                              'quota': target_quota, 'tag': allocation_tag})
        if self.before_create:
            self.before_create(managed, states, target_cards)
        if self.mode == 'pending':
            self.tags[allocation_tag] = {'status': 'pending'}
            raise TimeoutError('fixture unknown Create')
        if self.mode == 'rejected':
            self.tags[allocation_tag] = {'status': 'rejected'}
            return []
        if self.mode == 'not_sent':
            return []
        identifier = 'replacement' if self.create_count == 1 else f'replacement-{self.create_count}'
        replacement = item(identifier, target_node, target_cards)
        replacement.update(cpu=target_quota['cpu'], memory=target_quota['memory'])
        replacement['allocation_tag'] = allocation_tag
        self.managed.append(replacement)
        self.tags[allocation_tag] = {'status': 'accepted', 'notebook_id': identifier}
        return [replacement]

    def load_managed(self):
        return self.managed


class FakePlatform:
    def __init__(self, managed):
        self.status = {row['notebook_id']: 'RUNNING' for row in managed}
        self.stop_calls = []
        self.error = False

    def observe(self, row):
        return {'notebook_id': row['notebook_id'], 'name': row['name'],
                'workspace': WORKSPACE, 'project': PROJECT, 'compute_group': GROUP,
                'priority': 3, 'priority_level': 'LOW',
                'resource': {'gpu_count': row['devices']}, 'node': {'name': row['node']},
                'status': self.status.get(row['notebook_id'], 'PENDING')}

    def stop(self, row):
        self.stop_calls.append(row['notebook_id'])
        if self.error:
            raise TimeoutError('fixture uncertain stop')
        self.status[row['notebook_id']] = 'STOPPED'


class PlannerTests(unittest.TestCase):
    def test_clear_whole_sparse_nodes_into_one_legal_instance(self):
        rows = [item('target', 'target', 8), item('a', 'a', 2),
                item('b', 'b', 1), item('c', 'c', 1)]
        result = plan_consolidation(rows, fresh(rows), [node('target', 4), node('a', 0),
                                    node('b', 0), node('c', 0)], OPTIONS, 48, now=1000)
        self.assertEqual(result['target_node'], 'target')
        self.assertEqual(result['moved_cards'], 4)
        self.assertEqual(set(result['source_nodes']), {'a', 'b', 'c'})
        self.assertEqual(result['owned_nodes_before'] - result['owned_nodes_after'], 3)
        self.assertEqual(sum(row['devices'] for row in result['sources']), 4)

    def test_full_node_group_is_never_partially_moved(self):
        rows = [item('target', 'target', 8), item('a1', 'a', 2), item('a2', 'a', 1)]
        result = plan_consolidation(rows, fresh(rows), [node('target', 4), node('a', 0)],
                                    OPTIONS, 48, now=1000)
        self.assertEqual({row['notebook_id'] for row in result['sources']}, {'a1', 'a2'})
        self.assertEqual([quota['gpu'] for quota in result['target_quotas']], [2, 1])
        rows.append(item('b', 'b', 1))
        result = plan_consolidation(rows, fresh(rows), [node('target', 4), node('a', 0), node('b', 0)],
                                    OPTIONS, 48, now=1000)
        self.assertEqual({row['notebook_id'] for row in result['sources']}, {'a1', 'a2', 'b'})

    def test_uncertain_pending_stale_and_partial_node_observations_are_excluded(self):
        rows = [item('target', 'target', 8), item('a1', 'a', 1), item('a2', 'a', 1)]
        for change in ({'platform_status': 'PENDING'}, {'observed_at': 800},
                       {'platform_action': {'action': 'start', 'uncertain': True}},
                       {'action_required': 'unknown'}, {'retirement': {'phase': 'intent'}}):
            with self.subTest(change=change):
                states = fresh(rows)
                states['a2'].update(change)
                self.assertIsNone(plan_consolidation(rows, states, [node('target', 2), node('a', 0)],
                                                    OPTIONS, 48, now=1000))

    def test_target_must_be_owned_and_fit_all_three_resources(self):
        rows = [item('target', 'target', 8), item('source', 'source', 2)]
        for target in (node('target', 1), node('target', 2, cpu=15),
                       node('target', 2, memory=127), node('unowned-free-node', 16)):
            with self.subTest(target=target):
                self.assertIsNone(plan_consolidation(rows, fresh(rows), [target, node('source', 0)],
                                                    OPTIONS, 48, now=1000))

    def test_live_scope_ignores_unmanaged_resources_and_over_cap_fails_closed(self):
        rows = [item('target', 'target', 8), item('source', 'source', 2)]
        nodes = [node('target', 2), node('source', 0), node('unmanaged', 16)]
        result = plan_consolidation(rows, fresh(rows), nodes, OPTIONS, 10, now=1000)
        self.assertEqual([row['notebook_id'] for row in result['sources']], ['source'])
        self.assertIsNone(plan_consolidation(rows, fresh(rows), nodes, OPTIONS, 9, now=1000))
        self.assertEqual(plan_consolidation(rows + rows, fresh(rows), nodes, OPTIONS, 10, now=1000), result)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [item('target', 'target', 8), item('a', 'a', 1), item('b', 'b', 1)]
        self.states = fresh(self.rows)
        self.allocator = FakeAllocator(self.rows, [node('target', 2), node('a', 0), node('b', 0)])
        self.platform = FakePlatform(self.rows)
        self.saves = []
        self.control = Consolidator(self.root)

    def advance(self, now=1000, **kwargs):
        return self.control.advance(self.rows, self.states, self.platform, self.allocator,
                                    lambda: self.saves.append(copy.deepcopy(self.states)), now=now, **kwargs)

    def seven_card_source(self):
        self.rows = [item('target', 'target', 8), item('a', 'source', 4),
                     item('b', 'source', 2), item('c', 'source', 1)]
        self.states = fresh(self.rows)
        self.allocator = FakeAllocator(self.rows, [node('target', 7), node('source', 0)], max_cards=15)
        self.platform = FakePlatform(self.rows)

    def test_stop_then_confirm_all_then_create_same_cards_without_oversubscription(self):
        self.allocator.before_create = lambda rows, states, requested: self.assertEqual(
            managed_reserved_budget(rows, (), states, 48)['reserved_total_cards'] + requested, 10)
        first = self.advance()
        self.assertFalse(first['allow_normal_allocation'])
        self.assertEqual(len(self.platform.stop_calls), 1)
        self.assertEqual(self.allocator.create_count, 0)
        self.advance(1001)
        self.assertEqual(len(self.platform.stop_calls), 2)
        self.assertEqual(self.allocator.create_count, 0)
        final = self.advance(1002)
        self.assertEqual(final['phase'], 'completed')
        self.assertTrue(final['allow_normal_allocation'])
        self.assertEqual(self.allocator.create_count, 1)
        self.assertEqual(self.allocator.managed_budget(self.rows, self.states)['reserved_total_cards'], 10)
        self.assertNotIn('target', self.platform.stop_calls)
        before = len(self.platform.stop_calls)
        self.assertEqual(self.advance(1003)['phase'], 'cooldown')
        self.assertEqual(len(self.platform.stop_calls), before)

    def test_unknown_stop_survives_controller_state_loss_without_replay(self):
        self.platform.error = True
        self.advance()
        source = self.platform.stop_calls[0]
        self.states[source].pop('retirement')
        self.control = Consolidator(self.root)
        result = self.advance(1001)
        self.assertEqual(self.platform.stop_calls, [source])
        self.assertEqual(self.states[source]['retirement']['consolidation_id'], result['operation_id'])
        self.assertEqual(self.allocator.create_count, 0)
        self.assertFalse(result['allow_normal_allocation'])
        result = self.advance(1901)
        self.assertTrue(result['allow_normal_allocation'])
        self.assertEqual(self.allocator.managed_budget(self.rows, self.states)['reserved_total_cards'], 10)
        self.assertIn('retirement', self.states[source])

    def test_pending_create_is_reconciled_without_second_send(self):
        self.allocator.mode = 'pending'
        self.advance(); self.advance(1001)
        self.assertFalse(self.advance(1002)['allow_normal_allocation'])
        self.control = Consolidator(self.root)
        self.assertFalse(self.advance(1003)['allow_normal_allocation'])
        self.assertEqual(self.allocator.create_count, 1)
        self.assertEqual(len(self.platform.stop_calls), 2)

    def test_target_lost_after_first_stop_aborts_and_allows_normal_refill(self):
        self.advance()
        self.allocator.live.node_rows[0]['cpu']['available'] = 0
        result = self.advance(1001)
        self.assertEqual(result['phase'], 'aborted')
        self.assertTrue(result['allow_normal_allocation'])
        self.assertEqual(len(self.platform.stop_calls), 1)
        self.assertEqual(self.allocator.create_count, 0)
        self.assertEqual(self.allocator.managed_budget(self.rows, self.states)['reserved_total_cards'], 9)

    def test_fresh_target_preemption_overrides_stale_running_state(self):
        self.advance()
        self.platform.status['target'] = 'PENDING'
        self.assertEqual(self.states['target']['platform_status'], 'RUNNING')
        result = self.advance(1001)
        self.assertEqual(result['phase'], 'aborted')
        self.assertTrue(result['allow_normal_allocation'])
        self.assertEqual(self.states['target']['platform_status'], 'PENDING')
        self.assertEqual(self.states['target']['pending_since'], 1001)
        self.assertEqual(len(self.platform.stop_calls), 1)
        self.assertEqual(self.allocator.create_count, 0)

    def test_no_mutation_when_disabled_complete_stopping_or_no_cap(self):
        for kwargs in ({'enabled': False}, {'is_complete': True}, {'should_stop': lambda: True}):
            with self.subTest(kwargs=kwargs):
                self.advance(**kwargs)
                self.assertEqual(self.platform.stop_calls, [])
                self.assertEqual(self.allocator.create_count, 0)
        self.allocator.max_cards = None
        self.advance()
        self.assertEqual(self.platform.stop_calls, [])
        self.assertFalse(self.control.path.exists())

    def test_disable_during_plan_preserves_retirement_without_more_stops(self):
        self.advance()
        self.assertEqual(self.advance(1001, enabled=False)['phase'], 'aborted')
        self.assertEqual(len(self.platform.stop_calls), 1)
        self.assertEqual(self.allocator.create_count, 0)
        self.assertTrue(any(row.get('retirement') for row in self.states.values()))

    def test_identity_change_rejects_before_stop_or_create(self):
        self.advance()
        self.rows[2]['devices'] = 2
        with self.assertRaisesRegex(ValueError, 'source identity'):
            self.advance(1001)
        self.assertEqual(len(self.platform.stop_calls), 1)
        self.assertEqual(self.allocator.create_count, 0)

    def test_corrupt_plan_rejects_before_stop_or_create(self):
        self.advance()
        value = json.loads(self.control.path.read_text())
        value['contract']['moved_cards'] = 4
        self.control.path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'contract identity'):
            self.advance(1001)
        self.assertEqual(len(self.platform.stop_calls), 1)

    def test_source_preemption_before_stop_aborts_without_stopping_it(self):
        self.platform.status['a'] = 'PENDING'
        result = self.advance()
        self.assertEqual(result['phase'], 'aborted')
        self.assertEqual(self.platform.stop_calls, [])
        self.assertEqual(self.allocator.create_count, 0)

    def test_rejected_or_unavailable_replacement_allows_normal_refill(self):
        for mode in ('rejected', 'not_sent'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                rows = copy.deepcopy(self.rows)
                states = fresh(rows)
                allocator = FakeAllocator(rows, copy.deepcopy(self.allocator.live.node_rows))
                allocator.mode = mode
                platform = FakePlatform(rows)
                control = Consolidator(directory)
                control.advance(rows, states, platform, allocator, lambda: None, now=1000)
                control.advance(rows, states, platform, allocator, lambda: None, now=1001)
                result = control.advance(rows, states, platform, allocator, lambda: None, now=1002)
                self.assertEqual(result['phase'], 'aborted')
                self.assertTrue(result['allow_normal_allocation'])

    def test_seven_card_source_is_fully_stopped_then_created_four_two_one_on_target(self):
        self.seven_card_source()
        self.allocator.before_create = lambda rows, states, requested: self.assertLessEqual(
            managed_reserved_budget(rows, (), states, 15)['reserved_total_cards'] + requested, 15)
        for tick in range(3):
            result = self.advance(1000 + tick)
            self.assertFalse(result['allow_normal_allocation'])
            self.assertEqual(self.allocator.create_count, 0)
        self.assertEqual(set(self.platform.stop_calls), {'a', 'b', 'c'})
        for tick, expected in ((1003, [4]), (1004, [4, 2]), (1005, [4, 2, 1])):
            result = self.advance(tick)
            self.assertEqual([request['cards'] for request in self.allocator.requests], expected)
            self.assertEqual({request['node'] for request in self.allocator.requests}, {'target'})
        self.assertEqual(result['phase'], 'completed')
        self.assertEqual(self.allocator.managed_budget(self.rows, self.states)['reserved_total_cards'], 15)
        self.assertEqual(len({request['tag'] for request in self.allocator.requests}), 3)
        saved = json.loads(self.control.path.read_text())
        self.assertEqual(len(saved['replacement_notebook_ids']), 3)
        self.assertEqual(len(saved['replacements']), 3)

    def test_unknown_middle_quota_never_repeats_or_skips_to_third_quota(self):
        self.seven_card_source()
        for tick in range(4):
            self.advance(1000 + tick)
        self.allocator.mode = 'pending'
        result = self.advance(1004)
        self.assertFalse(result['allow_normal_allocation'])
        self.assertEqual([request['cards'] for request in self.allocator.requests], [4, 2])
        self.allocator.mode = 'accepted'
        self.control = Consolidator(self.root)
        self.advance(1005); self.advance(1006)
        self.assertEqual([request['cards'] for request in self.allocator.requests], [4, 2])
        tag = self.allocator.requests[1]['tag']
        replacement = item('late-two', 'target', 2)
        replacement['allocation_tag'] = tag
        self.rows.append(replacement)
        self.allocator.tags[tag] = {'status': 'accepted', 'notebook_id': 'late-two'}
        result = self.advance(1007)
        self.assertEqual(result['phase'], 'completed')
        self.assertEqual([request['cards'] for request in self.allocator.requests], [4, 2, 1])

    def test_target_capacity_loss_after_first_quota_aborts_without_duplicate_or_extra_stop(self):
        self.seven_card_source()
        for tick in range(4):
            self.advance(1000 + tick)
        # Four requested cards remain PENDING and must be deducted from this
        # raw free snapshot. Only two slots then remain for a three-card tail.
        self.allocator.live.node_rows[0] = node('target', 6)
        result = self.advance(1004)
        self.assertEqual(result['phase'], 'aborted')
        self.assertTrue(result['allow_normal_allocation'])
        self.assertEqual([request['cards'] for request in self.allocator.requests], [4])
        self.assertEqual(len(self.platform.stop_calls), 3)

    def test_running_first_quota_is_not_deducted_twice_from_live_free_resources(self):
        self.seven_card_source()
        for tick in range(4):
            self.advance(1000 + tick)
        self.platform.status['replacement'] = 'RUNNING'
        self.allocator.live.node_rows[0] = node('target', 3)
        result = self.advance(1004)
        self.assertEqual(result['phase'], 'replacement_pending')
        self.assertEqual([request['cards'] for request in self.allocator.requests], [4, 2])

    def test_accepted_quota_is_recovered_from_tag_after_plan_receipt_loss(self):
        self.seven_card_source()
        for tick in range(4):
            self.advance(1000 + tick)
        saved = json.loads(self.control.path.read_text())
        saved.pop('replacements')
        saved.pop('replacement_notebook_ids')
        self.control.path.write_text(json.dumps(saved))
        self.control = Consolidator(self.root)
        self.advance(1004)
        self.assertEqual([request['cards'] for request in self.allocator.requests], [4, 2])
        self.assertEqual(len(json.loads(self.control.path.read_text())['replacements']), 2)

    def test_old_v1_single_quota_plan_uses_its_original_allocation_tag(self):
        self.advance()
        saved = json.loads(self.control.path.read_text())
        saved['contract']['target_quota'] = saved['contract'].pop('target_quotas')[0]
        saved['contract_sha256'] = digest(saved['contract'])
        self.control.path.write_text(json.dumps(saved))
        self.control = Consolidator(self.root)
        self.advance(1001)
        result = self.advance(1002)
        self.assertEqual(result['phase'], 'completed')
        self.assertEqual(self.allocator.requests[0]['tag'],
                         'consolidation:' + saved['contract']['operation_id'])
        self.assertEqual(self.allocator.create_count, 1)


if __name__ == '__main__':
    unittest.main()
