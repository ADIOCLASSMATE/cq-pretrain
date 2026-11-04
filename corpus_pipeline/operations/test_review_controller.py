"""Operational scheduling/SDK optimizations, with no network or model calls."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from corpus_pipeline.operations import review_controller as control
from corpus_pipeline.serving.budget_pool import BudgetPool
from corpus_pipeline.serving.test_budget_pool import FakeBackend


@dataclass(frozen=True)
class Ref:
    key: str


class RawServings:
    def __init__(self):
        self.ref = Ref('owned-id')
        self.status = 'RUNNING'
        self.calls = []
        self.error = None

    def get(self, ref, **kwargs):
        self.calls.append(('get', ref, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(ref=self.ref, status=self.status)

    def instances(self, ref, **kwargs):
        self.calls.append(('instances', ref, kwargs))
        return ['fresh-instance']

    def events(self, ref, **kwargs):
        self.calls.append(('events', ref, kwargs))
        return SimpleNamespace(items=[len(self.calls)])


class ReadsTests(unittest.TestCase):
    def setUp(self):
        self.raw = RawServings()
        self.reads = control.ServingReads(self.raw, events_ttl=60)

    def test_details_and_instances_always_fresh_but_ref_avoids_repeated_discovery(self):
        self.reads.get('owned-name', workspace='owned-workspace')
        self.reads.instances(self.raw.ref, workspace='owned-workspace')
        self.reads.get('owned-name', workspace='owned-workspace')
        self.reads.instances(self.raw.ref, workspace='owned-workspace')
        self.assertEqual(self.raw.calls, [
            ('get', 'owned-name', {'workspace': 'owned-workspace'}),
            ('instances', self.raw.ref, {}), ('get', self.raw.ref, {}),
            ('instances', self.raw.ref, {}),
        ])

    def test_foreign_workspace_is_not_stripped_or_name_cache_reused(self):
        self.reads.get('owned-name', workspace='owned-workspace')
        self.reads.instances(self.raw.ref, workspace='different-workspace')
        self.reads.get('owned-name', workspace='different-workspace')
        self.assertEqual(self.raw.calls[-2][2], {'workspace': 'different-workspace'})
        self.assertEqual(self.raw.calls[-1], ('get', 'owned-name', {'workspace': 'different-workspace'}))

    def test_failed_detail_never_falls_back_to_stale_success(self):
        self.reads.get('owned-name', workspace='owned-workspace')
        self.raw.error = RuntimeError('not found')
        with self.assertRaises(RuntimeError):
            self.reads.get('owned-name', workspace='owned-workspace')
        self.assertEqual(self.reads.counts['fresh_details'], 1)

    def test_only_running_event_queries_are_cached_with_bounded_ttl(self):
        self.reads.get('owned-name', workspace='owned-workspace')
        with patch.object(control.time, 'monotonic', return_value=100):
            first = self.reads.events(self.raw.ref, workspace='owned-workspace', limit=30)
        with patch.object(control.time, 'monotonic', return_value=159):
            self.assertIs(self.reads.events(self.raw.ref, workspace='owned-workspace', limit=30), first)
        with patch.object(control.time, 'monotonic', return_value=160):
            self.assertIsNot(self.reads.events(self.raw.ref, workspace='owned-workspace', limit=30), first)
        self.assertEqual(self.reads.counts['fresh_events'], 2)

    def test_nonrunning_transition_invalidates_cache_before_recovery(self):
        self.reads.get('owned-name', workspace='owned-workspace')
        first = self.reads.events(self.raw.ref, workspace='owned-workspace', limit=30)
        self.raw.status = 'STOPPED'
        self.reads.get('owned-name', workspace='owned-workspace')
        second = self.reads.events(self.raw.ref, workspace='owned-workspace', limit=30)
        third = self.reads.events(self.raw.ref, workspace='owned-workspace', limit=30)
        self.assertIsNot(first, second)
        self.assertIsNot(second, third)
        self.assertEqual(self.reads.counts['cached_events'], 0)

    def test_event_filter_changes_require_new_query(self):
        self.reads.get('owned-name', workspace='owned-workspace')
        self.reads.events(self.raw.ref, workspace='owned-workspace', limit=30)
        self.reads.events(self.raw.ref, workspace='owned-workspace', limit=10)
        self.assertEqual(self.reads.counts['fresh_events'], 2)

    def test_clients_are_reused_in_thread_but_not_shared_between_threads(self):
        clients = []
        class RawClient:
            def __init__(self, **kwargs):
                self.servings = RawServings()
                self.entered = 0
                clients.append(self)
            def __enter__(self):
                self.entered += 1
                return self
        factory = control.ThreadClients(RawClient)
        one = factory(account='test')
        self.assertIs(one, factory(account='test'))
        result = []
        thread = threading.Thread(target=lambda: result.append(factory(account='test')))
        thread.start(); thread.join()
        self.assertIsNot(one, result[0])
        self.assertEqual([x.entered for x in clients], [1, 1])


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / 'config.json'
        self.policy = self.root / 'policy.json'
        self.cfg = {'schema_version': 'corpus.native_budget_pool.v1', 'root': str(self.root / 'pool'),
                    'max_cards': 128, 'cards_per_replica': 2, 'replicas': 64, 'priority': 1,
                    'per_replica_max_inflight': 128, 'name_prefix': 'test', 'poll_seconds': 8}
        control.original.atomic_json(self.config, self.cfg)
        self.config_before = self.config.read_bytes()
        self.sha = hashlib.sha256(self.config_before).hexdigest()
        self.value = {'schema': control.SCHEMA, 'config_sha256': self.sha, 'max_cards': 128,
                      'target_replicas': 32, 'poll_seconds': 15, 'provision_seconds': 20,
                      'observer_start_interval': .25, 'events_ttl_seconds': 60}
        self.save()
        self.backend = FakeBackend()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        control.original.atomic_json(self.policy, self.value)

    def pool(self):
        return control.OperationalPool(self.config, self.policy, self.backend)

    def test_existing_ledger_and_model_states_keep_original_identity(self):
        before = BudgetPool(self.config, self.backend)
        before.provision_one()
        ledger = (before.root / 'RESERVATIONS.json').read_bytes()
        state = (before.directory('1') / 'state.json').read_bytes()
        pool = self.pool()
        self.assertEqual(pool.config_sha, self.sha)
        self.assertEqual(pool.active_rows()[0]['service_id'], 'id-test-01')
        self.assertEqual(pool.target(), 32)
        self.assertEqual((pool.root / 'RESERVATIONS.json').read_bytes(), ledger)
        self.assertEqual((pool.directory('1') / 'state.json').read_bytes(), state)
        self.assertEqual(self.config.read_bytes(), self.config_before)

    def test_policy_target_hot_reload_and_target_file_honor_lower_bound(self):
        pool = self.pool()
        self.value['target_replicas'] = 64; self.save(); pool.refresh_policy()
        self.assertEqual(pool.target(), 64)
        control.original.atomic_json(pool.root / 'TARGET.json', {'replicas': 0})
        self.assertEqual(pool.target(), 0)
        control.original.atomic_json(pool.root / 'TARGET.json', {'replicas': 48})
        self.assertEqual(pool.target(), 48)

    def test_invalid_or_missing_policy_pauses_allocation_preserving_observation(self):
        pool = self.pool()
        self.value['max_cards'] = 256; self.save(); pool.refresh_policy()
        self.assertEqual(pool.target(), 0)
        self.assertEqual(pool.config['poll_seconds'], 15)
        self.assertFalse(pool.provision_one())
        self.assertEqual(self.backend.created, [])
        self.policy.unlink(); pool.refresh_policy()
        self.assertEqual(pool.target(), 0)

    def test_scope_and_unsafe_poll_interval_rejected(self):
        for key, value in [('config_sha256', 'wrong'), ('poll_seconds', 31),
                           ('target_replicas', 65), ('events_ttl_seconds', 0),
                           ('observer_start_interval', True), ('max_cards', 32)]:
            with self.subTest(key=key):
                saved = self.value[key]
                self.value[key] = value; self.save()
                with self.assertRaises(ValueError):
                    self.pool()
                self.value[key] = saved

    def test_uses_original_lifecycle_run_and_records_explicit_operational_identity(self):
        pool = self.pool()
        self.assertIs(type(pool).run, BudgetPool.run)
        status = pool.status()
        self.assertEqual(status['config_sha256'], self.sha)
        self.assertEqual(status['operational_controller']['source_sha256'], control.SOURCE_SHA)
        self.assertEqual(status['operational_controller']['ready_lease_seconds_unchanged'], 30)


def serving(key, *, owner='owner', gpu=2, replicas=1, nodes=1, status='RUNNING'):
    return SimpleNamespace(ref=Ref(key), name=key, status=status,
        raw={'user_id': owner, 'resource_spec_price': {'gpu_count': gpu},
             'replicas': replicas, 'node_num_per_replica': nodes})


class PagedService:
    def __init__(self, pages=()):
        self.pages = list(pages) or [[]]
        self.details = {}
        self.calls = []
        self.instances_calls = 0
        self.forced_cursor = None

    def list(self, workspace, *, limit, cursor):
        self.calls.append(cursor)
        index = 0 if cursor is None else int(cursor)
        return SimpleNamespace(items=self.pages[index], next_cursor=(
            self.forced_cursor if self.forced_cursor is not None else
            str(index + 1) if index + 1 < len(self.pages) else None))

    def get(self, ref, **kwargs):
        value = self.details[ref.key]
        if isinstance(value, Exception):
            raise value
        return value

    def instances(self, ref, **kwargs):
        self.instances_calls += 1
        return [SimpleNamespace(node='node')]


class BudgetScanTests(unittest.TestCase):
    def client(self, pages=()):
        return SimpleNamespace(servings=PagedService(pages), notebooks=PagedService(), jobs=PagedService())

    def scan(self, client, requested=2):
        return control.requested_card_budget(client, requested, 'owner')

    def test_full_fresh_pagination_repeated_per_scan_with_no_serving_detail_or_instances(self):
        client = self.client([[serving('one', replicas=2)], [serving('two', nodes=3)]])
        result = self.scan(client)
        self.assertEqual(result['observable_requested_cards'], 10)
        self.assertEqual(result['observable_total_with_request'], 12)
        self.assertEqual(client.servings.calls, [None, '1'])
        self.assertEqual(result['scan_counts']['serving_detail_fallbacks'], 0)
        self.assertEqual(client.servings.instances_calls, 0)
        self.assertIsNone(result['observable_allocated_cards'])
        self.assertEqual(result['observable_allocated_cards_upper_bound'], 10)
        self.scan(client)
        self.assertEqual(client.servings.calls, [None, '1', None, '1'])

    def test_foreign_owner_skipped_without_extra_detail(self):
        client = self.client([[serving('foreign', owner='else', gpu=256), serving('owned')]])
        self.assertEqual(self.scan(client)['observable_requested_cards'], 2)

    def test_missing_owner_or_topology_falls_back_to_exact_identity(self):
        client = self.client([[serving('owner-missing', owner=None), serving('topology-missing', nodes=None)]])
        client.servings.details = {'owner-missing': serving('owner-missing'),
                                   'topology-missing': serving('topology-missing', nodes=2)}
        result = self.scan(client)
        self.assertEqual(result['observable_requested_cards'], 6)
        self.assertEqual(result['scan_counts']['serving_detail_fallbacks'], 2)

    def test_missing_owner_in_detail_refuses_allocation(self):
        client = self.client([[serving('missing', owner=None)]])
        client.servings.details['missing'] = serving('missing', owner=None)
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.scan(client)

    def test_fallback_unknown_or_wrong_identity_never_becomes_zero_cards(self):
        client = self.client([[serving('missing', nodes=None)]])
        for value in [TimeoutError(), serving('different'), serving('missing', gpu=-1)]:
            client.servings.details['missing'] = value
            with self.assertRaises((ValueError, TimeoutError)):
                self.scan(client)

    def test_unknown_queued_and_error_states_reserve_full_topology(self):
        client = self.client([[serving('unknown', status='UNKNOWN', gpu=8),
                               serving('queued', status='PENDING', gpu=8),
                               serving('error', status='ERROR', gpu=8),
                               serving('stopped', status='STOPPED', gpu=200)]])
        self.assertEqual(self.scan(client)['observable_requested_cards'], 24)

    def test_bad_topology_remains_rejected_after_detail_fallback(self):
        for field, value in [('gpu_count', True), ('gpu_count', -1), ('replicas', 0),
                             ('replicas', 1.5), ('node_num_per_replica', '1')]:
            with self.subTest(field=field, value=value):
                item = serving('bad')
                (item.raw['resource_spec_price'] if field == 'gpu_count' else item.raw)[field] = value
                client = self.client([[item]])
                client.servings.details['bad'] = item
                with self.assertRaises(ValueError):
                    self.scan(client)

    def test_budget_limit_includes_other_workloads_and_additional_request(self):
        client = self.client([[serving('s', gpu=240)]])
        notebook = SimpleNamespace(ref=Ref('n'), name='n', status='RUNNING',
                                   resource={'gpu_count': 8}, node={'name': 'node'})
        client.notebooks.pages = [[notebook]]
        job = SimpleNamespace(ref=Ref('j'), name='j', status='RUNNING',
                              raw={'user_id': 'owner', 'framework_config': [{'gpu_count': 4, 'instance_count': 1}]})
        client.jobs.pages = [[job]]; client.jobs.details['j'] = job
        result = self.scan(client, requested=4)
        self.assertEqual(result['observable_total_with_request'], 256)
        self.assertEqual(client.jobs.instances_calls, 1)
        self.assertEqual(result['records']['j']['allocated_cards'], 4)
        self.assertEqual(result['records']['n']['allocated_cards'], 8)
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            self.scan(client, requested=5)

    def test_duplicate_resource_or_cursor_and_empty_middle_page_fail_closed(self):
        for pages in [[[serving('same')], [serving('same')]], [[], [serving('missing')]]]:
            with self.assertRaises(ValueError):
                self.scan(self.client(pages))
        client = self.client([[serving('one')], [serving('two')]])
        client.servings.forced_cursor = '1'
        with self.assertRaises(ValueError):
            self.scan(client)

    def test_later_page_error_prevents_partial_budget_result(self):
        client = self.client([[serving('one')], [serving('two')]])
        original_list = client.servings.list
        def fail_later(workspace, **kwargs):
            if kwargs['cursor'] is not None:
                raise TimeoutError('next page')
            return original_list(workspace, **kwargs)
        client.servings.list = fail_later
        with self.assertRaises(TimeoutError):
            self.scan(client)


if __name__ == '__main__':
    unittest.main()
