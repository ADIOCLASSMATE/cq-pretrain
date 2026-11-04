from dataclasses import replace
import random
import unittest

from .models import NodeCapacity, Quota, Reservation, WorkloadSpec
from .planning import available_nodes, deduplicate_reservations, plan_allocations, plan_consolidation, reserved_cards


def spec(**changes):
    return replace(WorkloadSpec('run-a', 'profile-a', 48, ('python', 'worker.py'), (8, 4, 2, 1)), **changes)


def row(identifier, node, cards, **changes):
    return Reservation(identifier, node, cards, cards * 8, cards * 64,
                       changes.pop('scope', 'run-a'), changes.pop('profile', 'profile-a'), **changes)


def node(name, cards, **changes):
    return NodeCapacity(name, cards, changes.get('cpu', cards * 8), changes.get('memory', cards * 64),
                        changes.get('eligible', True))


QUOTAS = tuple(Quota(cards, cards * 8, cards * 64) for cards in (16, 8, 4, 2, 1))


class ContractTests(unittest.TestCase):
    def test_explicit_budget_scope_profile_and_argv_are_required(self):
        for changes in ({'max_cards': 0}, {'max_cards': True}, {'scope': ''}, {'profile': ''},
                        {'command': 'python worker.py'}, {'command': ()}, {'allowed_cards': ()},
                        {'allowed_cards': (True,)}, {'poll_seconds': float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                spec(**changes)
        self.assertEqual(spec(command=['python', 'worker.py']).command, ('python', 'worker.py'))

    def test_service_is_fixed_and_migration_boundaries_are_explicit(self):
        with self.assertRaises(ValueError):
            spec(workload_kind='service')
        with self.assertRaises(ValueError):
            spec(migration_policy='drain')
        with self.assertRaises(ValueError):
            spec(replica_cards=3)
        with self.assertRaises(ValueError):
            spec(replica_cards=8, min_ready_replicas=7)
        value = spec(workload_kind='service', replica_cards=8, migration_policy='drain',
                     drain_command=('python', 'drain.py'), min_ready_replicas=1)
        self.assertEqual(value.effective_cards, (8,))

    def test_invalid_resource_numbers_reject(self):
        for values in ((-1, 8, 64), (True, 8, 64), (1, float('inf'), 64), (1, 8, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                NodeCapacity('node', *values)


class ReservationTests(unittest.TestCase):
    def test_stopped_failed_unknown_pending_and_running_all_reserve(self):
        rows = [row(str(i), str(i), 1, state=state)
                for i, state in enumerate(('STOPPED', 'FAILED', 'UNKNOWN', 'PENDING', 'RUNNING'))]
        self.assertEqual(reserved_cards(spec(), rows + rows), 5)
        self.assertEqual(reserved_cards(spec(), rows + [row('other', 'x', 8, scope='another-run')]), 5)

    def test_retirement_requires_confirmation_and_stopped_observation(self):
        self.assertEqual(reserved_cards(spec(), [row('x', 'a', 8, state='STOPPED', retired_verified=True)]), 0)
        self.assertEqual(reserved_cards(spec(), [row('x', 'a', 8, state='RUNNING', retired_verified=True)]), 8)
        self.assertEqual(reserved_cards(spec(), [row('x', 'a', 8, state='UNKNOWN', retired_verified=True)]), 8)

    def test_aliases_deduplicate_and_conflicting_identity_fails_closed(self):
        first = row('op', 'a', 1, operation_id='op')
        accepted = row('platform-id', 'a', 1, operation_id='op', resource_id='platform-id', state='RUNNING')
        values = deduplicate_reservations([first, accepted, accepted])
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].state, 'UNKNOWN')
        self.assertEqual(reserved_cards(spec(), [first, accepted]), 1)
        with self.assertRaises(ValueError):
            deduplicate_reservations([first, replace(first, cards=2)])
        with self.assertRaises(ValueError):
            deduplicate_reservations([first, replace(first, scope='other')])

    def test_conflicting_retirement_observations_keep_capacity_reserved(self):
        retired = row('x', 'a', 1, state='STOPPED', retired_verified=True, capacity_accounted=True)
        pending = replace(retired, state='PENDING', retired_verified=False, capacity_accounted=False)
        self.assertEqual(reserved_cards(spec(), [retired, pending]), 1)
        self.assertEqual(available_nodes([node('a', 8)], [retired, pending])[0].gpu, 7)


class AllocationTests(unittest.TestCase):
    def test_existing_one_card_plus_free_seven_produces_four_two_one_on_same_node(self):
        existing = row('existing', 'a', 1, state='RUNNING', capacity_accounted=True)
        result = plan_allocations(spec(max_cards=8), [node('a', 7), node('new', 16)], QUOTAS, [existing])
        self.assertEqual([(p.node, p.quota.cards) for p in result.placements], [('a', 4), ('a', 2), ('a', 1)])
        self.assertEqual((result.reserved_cards, result.added_cards, result.remaining_cards), (1, 7, 0))

    def test_pending_or_unknown_one_card_deducts_resources_but_does_not_block_node(self):
        for state in ('PENDING', 'UNKNOWN', 'STOPPED'):
            with self.subTest(state=state):
                pending = row('pending', 'a', 1, state=state)
                result = plan_allocations(spec(max_cards=8), [node('a', 8), node('new', 16)], QUOTAS, [pending])
                self.assertEqual([(p.node, p.quota.cards) for p in result.placements], [('a', 4), ('a', 2), ('a', 1)])

    def test_pending_reservations_apply_cpu_and_memory_as_well_as_cards(self):
        pending = row('pending', 'a', 1, state='PENDING')
        result = plan_allocations(spec(max_cards=8), [node('a', 8, cpu=24, memory=192)], QUOTAS, [pending])
        self.assertEqual(result.added_cards, 2)
        self.assertEqual(result.remaining_cards, 5)

    def test_external_scope_occupies_capacity_but_never_our_budget(self):
        external = row('external', 'a', 4, scope='other-run', state='UNKNOWN')
        result = plan_allocations(spec(max_cards=8), [node('a', 8)], QUOTAS, [external])
        self.assertEqual(result.reserved_cards, 0)
        self.assertEqual(result.added_cards, 4)

    def test_workload_and_global_budgets_both_bound_additions(self):
        result = plan_allocations(spec(max_cards=48), [node('a', 64)], QUOTAS, [], global_remaining_cards=7)
        self.assertEqual(result.added_cards, 7)
        self.assertEqual(result.remaining_cards, 0)
        self.assertEqual(plan_allocations(spec(max_cards=1), [node('a', 64)], QUOTAS,
                                         [row('x', 'b', 2)]).reason, 'over_budget')
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                plan_allocations(spec(), [], QUOTAS, [], global_remaining_cards=invalid)

    def test_new_nodes_pack_before_spreading_and_all_three_resources_limit_fit(self):
        result = plan_allocations(spec(max_cards=7), [node('a', 16, cpu=16), node('b', 8)], QUOTAS, [])
        self.assertEqual({p.node for p in result.placements}, {'b'})
        self.assertEqual([p.quota.cards for p in result.placements], [4, 2, 1])

    def test_knapsack_avoids_cpu_heavy_largest_quota_stranding_cards(self):
        result = plan_allocations(spec(max_cards=6, allowed_cards=(5, 3)),
                                 [node('a', 6, cpu=8, memory=20)],
                                 [Quota(5, 8, 5), Quota(3, 4, 3)], [])
        self.assertEqual([p.quota.cards for p in result.placements], [3, 3])

    def test_fixed_eight_card_service_never_splits_into_small_replicas(self):
        service = spec(max_cards=16, workload_kind='service', replica_cards=8)
        result = plan_allocations(service, [node('a', 7), node('b', 8), node('c', 4)], QUOTAS, [])
        self.assertEqual([(p.node, p.quota.cards) for p in result.placements], [('b', 8)])
        self.assertEqual(result.remaining_cards, 8)
        mismatch = plan_allocations(service, [node('a', 16)], QUOTAS, [row('legacy', 'b', 4)])
        self.assertEqual(mismatch.reason, 'configuration_mismatch')
        self.assertEqual(mismatch.placements, ())

    def test_ineligible_and_ambiguous_nodes_are_not_used(self):
        self.assertEqual(plan_allocations(spec(), [node('a', 8, eligible=False)], QUOTAS, []).placements, ())
        with self.assertRaises(ValueError):
            plan_allocations(spec(), [node('a', 8), node('a', 4)], QUOTAS, [])

    def test_generated_small_cases_never_exceed_any_budget_or_node_resource(self):
        rng = random.Random(20261023)
        for _ in range(80):
            nodes = [node(str(i), rng.randrange(1, 17), cpu=rng.randrange(1, 129), memory=rng.randrange(1, 1025))
                     for i in range(3)]
            limit, global_limit = rng.randrange(1, 49), rng.randrange(0, 49)
            result = plan_allocations(spec(max_cards=limit), nodes, QUOTAS, [], global_remaining_cards=global_limit)
            self.assertLessEqual(result.added_cards, min(limit, global_limit))
            for candidate in nodes:
                chosen = [p.quota for p in result.placements if p.node == candidate.name]
                self.assertLessEqual(sum(q.cards for q in chosen), candidate.gpu)
                self.assertLessEqual(sum(q.cpu for q in chosen), candidate.cpu)
                self.assertLessEqual(sum(q.memory for q in chosen), candidate.memory)


class MigrationTests(unittest.TestCase):
    def test_three_card_complete_source_group_moves_as_two_plus_one(self):
        rows = [row('target', 'target', 8, state='RUNNING', capacity_accounted=True),
                row('a', 'source', 2, state='RUNNING', capacity_accounted=True),
                row('b', 'source', 1, state='RUNNING', capacity_accounted=True)]
        result = plan_consolidation(spec(migration_policy='restartable'),
                                    [node('target', 3), node('source', 0)], QUOTAS, rows)
        self.assertEqual(set(result.source_ids), {'a', 'b'})
        self.assertEqual(result.moved_cards, 3)
        self.assertEqual([p.quota.cards for p in result.placements], [2, 1])
        self.assertEqual((result.nodes_before, result.nodes_after), (2, 1))

    def test_seven_cards_can_move_to_multiple_existing_destinations(self):
        rows = [row('ta', 'ta', 8, state='RUNNING', capacity_accounted=True),
                row('tb', 'tb', 8, state='RUNNING', capacity_accounted=True),
                row('s4', 'source', 4, state='RUNNING', capacity_accounted=True),
                row('s2', 'source', 2, state='RUNNING', capacity_accounted=True),
                row('s1', 'source', 1, state='RUNNING', capacity_accounted=True)]
        result = plan_consolidation(spec(migration_policy='restartable'),
                                   [node('ta', 4), node('tb', 3), node('source', 0)], QUOTAS, rows,
                                   global_remaining_cards=0)
        self.assertEqual(result.moved_cards, 7)
        self.assertEqual({p.node for p in result.placements}, {'ta', 'tb'})
        self.assertEqual([p.quota.cards for p in result.placements], [4, 2, 1])
        self.assertEqual(result.nodes_before - result.nodes_after, 1)

    def test_pending_member_prevents_partial_source_node_migration(self):
        rows = [row('target', 'target', 8, state='RUNNING', capacity_accounted=True),
                row('a', 'source', 2, state='RUNNING', capacity_accounted=True),
                row('b', 'source', 1, state='UNKNOWN')]
        self.assertIsNone(plan_consolidation(spec(migration_policy='restartable'),
                                            [node('target', 3), node('source', 0)], QUOTAS, rows))

    def test_other_scope_resources_never_appear_in_sources(self):
        rows = [row('target', 'target', 8, state='RUNNING', capacity_accounted=True),
                row('our', 'source', 2, state='RUNNING', capacity_accounted=True),
                row('unrelated', 'other', 4, scope='other-run', state='RUNNING', capacity_accounted=True)]
        result = plan_consolidation(spec(migration_policy='restartable'),
                                   [node('target', 2), node('source', 0), node('other', 0)], QUOTAS, rows)
        self.assertEqual(result.source_ids, ('our',))

    def test_disabled_policy_and_no_node_reduction_produce_no_move(self):
        rows = [row('a', 'a', 1, state='RUNNING', capacity_accounted=True),
                row('b', 'b', 1, state='RUNNING', capacity_accounted=True)]
        self.assertIsNone(plan_consolidation(spec(), [node('a', 1), node('b', 1)], QUOTAS, rows))
        self.assertIsNone(plan_consolidation(spec(migration_policy='restartable'),
                                            [node('a', 0), node('b', 0)], QUOTAS, rows))

    def test_fixed_service_migration_keeps_replica_shape_and_minimum_ready(self):
        rows = [row('a', 'a', 8, state='RUNNING', capacity_accounted=True, ready=True),
                row('b', 'b', 8, state='RUNNING', capacity_accounted=True, ready=True)]
        service = spec(max_cards=16, replica_cards=8, workload_kind='service', migration_policy='drain',
                       drain_command=('drain',), min_ready_replicas=1)
        result = plan_consolidation(service, [node('a', 8), node('b', 0)], QUOTAS, rows)
        self.assertEqual([p.quota.cards for p in result.placements], [8])
        self.assertEqual(result.source_ids, ('b',))
        self.assertIsNone(plan_consolidation(replace(service, min_ready_replicas=2),
                                            [node('a', 8), node('b', 0)], QUOTAS, rows))
        self.assertIsNone(plan_consolidation(service, [node('a', 7), node('b', 0), node('c', 1)], QUOTAS, rows))

    def test_move_whole_sparse_nodes_to_one_new_node_only_when_count_decreases(self):
        rows = [row('a', 'a', 1, state='RUNNING', capacity_accounted=True),
                row('b', 'b', 1, state='RUNNING', capacity_accounted=True)]
        result = plan_consolidation(spec(migration_policy='restartable'),
                                   [node('a', 0), node('b', 0), node('new', 2)], QUOTAS, rows)
        self.assertEqual(set(result.source_ids), {'a', 'b'})
        self.assertEqual([(p.node, p.quota.cards) for p in result.placements], [('new', 2)])
        self.assertEqual((result.nodes_before, result.nodes_after), (2, 1))

    def test_bounded_search_preserves_the_only_large_destination(self):
        rows = [row(str(i), f'n{i:02}', 1, state='RUNNING', capacity_accounted=True) for i in range(12)]
        nodes = [node(f'n{i:02}', 11 if i == 0 else 0) for i in range(12)]
        result = plan_consolidation(spec(migration_policy='restartable'), nodes, QUOTAS, rows)
        self.assertEqual((result.moved_cards, result.nodes_after), (11, 1))
        self.assertEqual({p.node for p in result.placements}, {'n00'})


if __name__ == '__main__':
    unittest.main()
