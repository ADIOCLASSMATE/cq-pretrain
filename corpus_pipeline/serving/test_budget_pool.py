"""Effectful pool invariants using a fake native backend; no platform calls."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from corpus_pipeline.resource_manager.engine import atomic_json
from corpus_pipeline.serving.budget_pool import (
    BudgetPool, reservation_total, audited_unsent_backend, safe_error_chain)

try:
    from inspire.platform.errors import TransportError, AuthenticationError
except ImportError:
    TransportError = AuthenticationError = None


class FakeBackend:
    def __init__(self):
        self.created = []
        self.restarts = []
        self.raise_create = False
        self.external_released = False
        self.disable_prepare = None
        self.should_restart = None
        self.published = []
        self.fail_stop_name = None

    def prepare(self, directory, number, name, priority):
        if self.disable_prepare:
            self.disable_prepare.write_text('stop')
        return {'name': name, 'cards': 2, 'priority': priority, 'phase': 'CREATE_INTENT',
                'operation_id': name, 'binding': {'review': 'one'}}

    def validate(self, state):
        pass

    def binding(self):
        return {}, {'review': 'one'}

    def create(self, directory, state):
        self.created.append(state['name'])
        if self.raise_create:
            raise TimeoutError('accepted but response lost')
        return {**state, 'service_id': 'id-' + state['name'], 'platform_status': 'RUNNING'}

    def observe(self, directory, state):
        return {**state, 'service_id': 'id-' + state['name'], 'platform_status': 'RUNNING'}

    def can_restart(self, state):
        return self.should_restart

    def restart(self, directory, state, event):
        self.restarts.append((state['service_id'], event))
        return state

    def released(self, state):
        return self.external_released

    def publish(self, directory, state, ready):
        self.published.append(ready)

    def stop(self, directory, state):
        if state['name'] == self.fail_stop_name:
            raise ValueError('identity mismatch')
        self.publish(directory, state, False)
        return {**state, 'phase': 'RELEASED', 'release_verified': True}


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = {'schema_version': 'corpus.native_budget_pool.v1', 'root': str(self.root / 'pool'),
            'max_cards': 128, 'cards_per_replica': 2, 'replicas': 64, 'priority': 3,
            'per_replica_max_inflight': 32, 'name_prefix': 'test', 'poll_seconds': 8}
        self.backend = FakeBackend()

    def tearDown(self):
        self.tmp.cleanup()

    def pool(self):
        path = self.root / 'config.json'
        atomic_json(path, self.cfg)
        return BudgetPool(path, self.backend)

    def test_current_low_priority_provisions_without_rewriting_identity(self):
        self.cfg.update(priority=1, replicas=1, max_cards=2)
        pool = self.pool()
        self.assertTrue(pool.provision_one())
        self.assertEqual(pool.state('1')['priority'], 1)
        self.assertEqual(pool.stop_once()['complete'], True)

    def test_high_or_ambiguous_priority_cannot_enter_low_pool(self):
        for priority in (True, 0, 2, 4, 6, '1'):
            with self.subTest(priority=priority):
                self.cfg['priority'] = priority
                with self.assertRaises(ValueError):
                    self.pool()
        self.assertEqual(self.backend.created, [])

    def test_high_two_cards_counts_until_live_verified_release(self):
        high = self.root / 'high'
        high.mkdir()
        self.cfg['external_reservations'] = [{'state': str(high / 'state.json'),
            'registry': str(high / 'registry.json'), 'cards': 2, 'service_id': 'high-id'}]
        pool = self.pool()
        for _ in range(63):
            self.assertTrue(pool.provision_one())
        self.assertFalse(pool.provision_one())  # Missing HIGH evidence still holds 2.
        atomic_json(high / 'state.json', {'cards': 2, 'service_id': 'high-id',
            'release_verified': True, 'phase': 'RELEASED'})
        (high / 'DISABLED').write_text('explicit stop')
        self.assertFalse(pool.provision_one())  # Local receipt alone is insufficient.
        self.backend.external_released = True
        self.assertTrue(pool.provision_one())
        self.assertFalse(pool.provision_one())
        self.assertEqual(len(self.backend.created), 64)

    def test_unknown_create_holds_budget_and_only_reconciles(self):
        self.cfg.update(max_cards=2, replicas=1)
        pool = self.pool()
        self.backend.raise_create = True
        with self.assertRaises(TimeoutError):
            pool.provision_one()
        self.assertFalse(pool.provision_one())
        self.assertEqual(pool.observe_one('1')['service_id'], 'id-test-01')
        self.assertEqual(self.backend.created, ['test-01'])

    def test_drain_withdraws_external_traffic_but_holds_cards(self):
        high = self.root / 'high'
        high.mkdir()
        marker = high / 'DRAIN'
        self.cfg['external_reservations'] = [{'state': str(high / 'state.json'),
            'registry': str(high / 'registry.json'), 'cards': 2, 'drain_marker': str(marker)}]
        pool = self.pool()
        pool.publish_index()
        self.assertEqual(len(json.loads((pool.root / 'pool.json').read_text())['registries']), 1)
        marker.write_text('drain')
        pool.publish_index()
        self.assertEqual(json.loads((pool.root / 'pool.json').read_text())['registries'], [])
        self.assertEqual(pool.external_cards(), 2)

    def test_preempted_stopped_slot_remains_reserved(self):
        self.cfg.update(max_cards=2, replicas=1)
        pool = self.pool()
        pool.provision_one()
        state = pool.state('1')
        state['platform_status'] = 'STOPPED'
        pool.save_state('1', state)
        self.assertFalse(pool.provision_one())
        self.backend.should_restart = 'victim-event'
        pool.observe_one('1')
        self.assertEqual(self.backend.restarts, [('id-test-01', 'victim-event')])
        self.assertEqual(len(self.backend.created), 1)

    def test_stop_before_create_has_no_mutation(self):
        pool = self.pool()
        self.backend.disable_prepare = pool.root / 'DISABLED'
        with self.assertRaises(ValueError):
            pool.provision_one()
        self.assertEqual(self.backend.created, [])

    def test_disabled_cannot_restart(self):
        pool = self.pool()
        pool.provision_one()
        (pool.root / 'DISABLED').write_text('stop')
        self.backend.should_restart = 'new-victim-event'
        pool.observe_one('1')
        self.assertEqual(self.backend.restarts, [])
        self.assertFalse(self.backend.published[-1])

    def test_missing_ledger_fails_closed(self):
        pool = self.pool()
        pool.provision_one()
        (pool.root / 'RESERVATIONS.json').unlink()
        with self.assertRaises(ValueError):
            pool.provision_one()

    def test_intent_retained_when_state_missing(self):
        self.cfg.update(max_cards=2, replicas=1)
        pool = self.pool()
        pool.provision_one()
        (pool.directory('1') / 'state.json').unlink()
        self.assertFalse(pool.provision_one())
        self.assertEqual(pool.observe_one('1')['service_id'], 'id-test-01')
        self.assertEqual(len(self.backend.created), 1)

    def test_altered_scope_or_card_count_refused(self):
        pool = self.pool()
        pool.provision_one()
        state = pool.state('1')
        state['cards'] = 1
        pool.save_state('1', state)
        with self.assertRaises(ValueError):
            pool.provision_one()

    def test_stop_continues_when_one_identity_invalid(self):
        pool = self.pool()
        pool.provision_one(); pool.provision_one()
        self.backend.fail_stop_name = 'test-01'
        result = pool.stop_once()
        self.assertFalse(result['complete'])
        self.assertTrue(result['slots'][1]['released'])
        self.assertFalse(result['external_reservations_owned'])

    def test_identity_conflicts_and_overbudget_refused(self):
        rows = [{'name': 'a', 'cards': 2, 'service_id': 'same'},
                {'name': 'b', 'cards': 2, 'service_id': 'same'}]
        with self.assertRaises(ValueError):
            reservation_total(rows)
        with self.assertRaises(ValueError):
            reservation_total([{'name': 'a', 'cards': 128}], 2)

    def unsent_pool(self):
        self.cfg.update(max_cards=2, replicas=1)
        pool = self.pool()
        self.backend.raise_create = True
        with self.assertRaises(TimeoutError):
            pool.provision_one()
        self.backend.raise_create = False
        client = Mock()
        client.servings.iter.return_value = []
        factory = Mock()
        factory.return_value.__enter__ = Mock(return_value=client)
        factory.return_value.__exit__ = Mock(return_value=False)
        adapter = SimpleNamespace(InspireClient=factory, native=SimpleNamespace(WORKSPACE='test-workspace'))
        return pool, client, adapter

    def test_audited_unsent_recovery_reuses_reservation_under_both_locks(self):
        import fcntl
        pool, client, adapter = self.unsent_pool()
        ledger_before = (pool.root / 'RESERVATIONS.json').read_bytes()
        original = self.backend.create
        def create(directory, state):
            for lock in (directory / 'lifecycle.lock', pool.root / 'provision.lock'):
                with lock.open('a') as stream:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(directory / 'CREATE_TRANSMISSION.json', {'operation_id': state['operation_id']})
            return original(directory, state)
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter), \
             patch.object(self.backend, 'create', side_effect=create):
            state = pool.observe_one('1')
            pool.observe_one('1')
        self.assertEqual(state['service_id'], 'id-test-01')
        self.assertEqual(self.backend.created, ['test-01', 'test-01'])  # Initial pre-send failure + one send.
        self.assertEqual((pool.root / 'RESERVATIONS.json').read_bytes(), ledger_before)
        self.assertEqual(state['pool_last_error']['stage'], 'CREATE_UNKNOWN')
        client.servings.iter.assert_called_once_with(workspace='test-workspace')

    def test_unsent_recovery_pause_marker_name_collision_and_disable_prevent_send(self):
        pool, client, adapter = self.unsent_pool()
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter), \
             patch.object(self.backend, 'observe', side_effect=ValueError('unresolved')):
            atomic_json(pool.root / 'TARGET.json', {'replicas': 0})
            pool.observe_one('1')
            client.servings.iter.assert_not_called()
            (pool.root / 'TARGET.json').unlink()
            atomic_json(pool.directory('1') / 'CREATE_TRANSMISSION.json', {'sent': True})
            pool.observe_one('1')
            client.servings.iter.assert_not_called()
            (pool.directory('1') / 'CREATE_TRANSMISSION.json').unlink()
            client.servings.iter.return_value = [SimpleNamespace(name='test-01')]
            pool.observe_one('1')
            client.servings.iter.return_value = []
            (pool.root / 'DISABLED').write_text('stop')
            pool.observe_one('1')
        self.assertEqual(self.backend.created, ['test-01'])

    def test_unsent_recovery_checks_fresh_scope_budget_and_binding(self):
        pool, client, adapter = self.unsent_pool()
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter), \
             patch.object(pool, 'external_cards', return_value=2):
            result = pool.observe_one('1')
        self.assertEqual(result['pool_error_stage'], 'UNSENT_RECOVERY')
        self.assertEqual(result['pool_error_type'], 'ValueError')
        client.servings.iter.assert_not_called()
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter), \
             patch.object(self.backend, 'binding', return_value=({}, {'review': 'changed'})):
            pool.observe_one('1')
        self.assertEqual(self.backend.created, ['test-01'])
        client.servings.iter.assert_not_called()

    def test_unsent_recovery_lost_reply_is_never_retransmitted(self):
        pool, client, adapter = self.unsent_pool()
        def create(directory, state):
            self.backend.created.append(state['name'])
            atomic_json(directory / 'CREATE_TRANSMISSION.json', {'operation_id': state['operation_id']})
            raise TimeoutError('response lost')
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter), \
             patch.object(self.backend, 'create', side_effect=create), \
             patch.object(self.backend, 'observe', side_effect=ValueError('not yet visible')):
            pool.observe_one('1')
            pool.observe_one('1')
        self.assertEqual(self.backend.created, ['test-01', 'test-01'])
        self.assertEqual(len(pool.ledger()['slots']), 1)

    def test_unknown_backend_does_not_treat_missing_marker_as_proof(self):
        pool, _, _ = self.unsent_pool()
        self.assertIsNone(audited_unsent_backend(self.backend, None))
        pool.observe_one('1')
        self.assertEqual(self.backend.created, ['test-01'])

    @unittest.skipIf(TransportError is None, 'Inspire SDK required for audited adapter identity')
    def test_only_exact_audited_adapter_implementation_opts_in(self):
        from corpus_pipeline.serving.qwen36_w8a8_20261024.official_v023 import backend as official
        backend = object.__new__(official.Backend)  # No initialization or platform/network calls.
        self.assertIs(audited_unsent_backend(backend, official.__file__), official.lifecycle)
        self.assertIsNone(audited_unsent_backend(backend, Path(official.__file__).parent / 'other.py'))
        self.assertIsNone(audited_unsent_backend(self.backend, official.__file__))
        with patch.object(backend, 'create', Mock()):
            self.assertIsNone(audited_unsent_backend(backend, official.__file__))

    def ready_pool(self):
        pool = self.pool()
        pool.provision_one()
        state = pool.state('1')
        state.update(deployment_id='deployment', incarnation='boot-1')
        pool.save_state('1', state)
        registry = pool.directory('1') / 'registry.json'
        atomic_json(registry, {'endpoints': [{'ready': True, 'ready_expires_at': time.time() + 30,
            'service_id': state['service_id'], 'deployment_id': 'deployment', 'incarnation': 'boot-1'}]})
        return pool, registry

    @unittest.skipIf(TransportError is None, 'Inspire SDK required for actual exception type')
    def test_transport_error_preserves_existing_lease_bytes_without_renewal(self):
        pool, registry = self.ready_pool()
        before = registry.read_bytes()
        before_stat = registry.stat().st_mtime_ns
        with patch.object(self.backend, 'observe', side_effect=TransportError('temporary')):
            state = pool.observe_one('1')
        self.assertTrue(state['pool_registry_lease_preserved'])
        self.assertEqual(registry.read_bytes(), before)
        self.assertEqual(registry.stat().st_mtime_ns, before_stat)
        self.assertEqual(self.backend.published, [])
        self.assertEqual(state['pool_last_error']['type'], 'TransportError')
        self.assertGreater(state['pool_last_error']['at'], 0)
        pool.observe_one('1')
        self.assertEqual(pool.state('1')['pool_last_error'], state['pool_last_error'])

    @unittest.skipIf(TransportError is None, 'Inspire SDK required for actual exception type')
    def test_transport_error_never_preserves_expired_wrong_identity_or_disabled_lease(self):
        pool, registry = self.ready_pool()
        original = json.loads(registry.read_text())
        cases = [('expired', 'ready_expires_at', 0), ('wrong-service', 'service_id', 'other'),
                 ('wrong-boot', 'incarnation', 'other'), ('not-ready', 'ready', False)]
        for label, key, value in cases:
            with self.subTest(label=label):
                changed = json.loads(json.dumps(original))
                changed['endpoints'][0][key] = value
                atomic_json(registry, changed)
                with patch.object(self.backend, 'observe', side_effect=TransportError('temporary')):
                    self.assertFalse(pool.observe_one('1')['pool_registry_lease_preserved'])
                self.assertFalse(self.backend.published[-1])
        atomic_json(registry, original)
        (pool.root / 'DISABLED').write_text('stop')
        with patch.object(self.backend, 'observe', side_effect=TransportError('temporary')):
            self.assertFalse(pool.observe_one('1')['pool_registry_lease_preserved'])

    @unittest.skipIf(TransportError is None, 'Inspire SDK required for actual exception types')
    def test_identity_authentication_errors_and_explicit_stop_immediately_withdraw(self):
        pool, registry = self.ready_pool()
        for error in (ValueError('binding changed'), AuthenticationError('authentication'), TimeoutError('other')):
            with patch.object(self.backend, 'observe', side_effect=error):
                self.assertFalse(pool.observe_one('1')['pool_registry_lease_preserved'])
            self.assertFalse(self.backend.published[-1])
        def stopped(directory, state):
            self.backend.publish(directory, state, False)
            return {**state, 'platform_status': 'STOPPED'}
        with patch.object(self.backend, 'observe', side_effect=stopped):
            self.assertEqual(pool.observe_one('1')['platform_status'], 'STOPPED')
        self.assertFalse(self.backend.published[-1])

    def test_request_limit_changes_traffic_only_without_rewriting_reservations(self):
        pool = self.pool()
        pool.provision_one()
        ledger = (pool.root / 'RESERVATIONS.json').read_bytes()
        state = (pool.directory('1') / 'state.json').read_bytes()
        for limit in (1, 16, 32):
            atomic_json(pool.root / 'REQUEST_LIMIT.json', {'max_inflight': limit})
            pool.publish_index()
            self.assertEqual(json.loads((pool.root / 'pool.json').read_text())['registries'][0]['max_inflight'], limit)
        (pool.root / 'REQUEST_LIMIT.json').unlink()
        pool.publish_index()
        self.assertEqual(json.loads((pool.root / 'pool.json').read_text())['registries'][0]['max_inflight'], 32)
        self.assertEqual((pool.root / 'RESERVATIONS.json').read_bytes(), ledger)
        self.assertEqual((pool.directory('1') / 'state.json').read_bytes(), state)

    def test_request_limit_invalid_values_cannot_replace_published_index(self):
        pool = self.pool()
        pool.provision_one()
        before = (pool.root / 'pool.json').read_bytes()
        for value in (0, -1, 33, True, 1.5, '16'):
            with self.subTest(value=value):
                atomic_json(pool.root / 'REQUEST_LIMIT.json', {'max_inflight': value})
                with self.assertRaises(ValueError):
                    pool.publish_index()
                self.assertEqual((pool.root / 'pool.json').read_bytes(), before)

    def test_live_observer_starts_are_spaced_and_at_most_eight_are_active(self):
        from concurrent.futures import ThreadPoolExecutor
        pool = self.pool()
        self.assertEqual(pool._observer_start_interval, 0.35)
        pool._observer_start_interval = 0.01  # Exercise the same algorithm without a long test.
        lock = threading.Lock()
        starts = []
        active = maximum = 0
        def observe(number):
            nonlocal active, maximum
            with lock:
                starts.append(time.monotonic())
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.12)
            with lock:
                active -= 1
            return number
        with patch.object(pool, 'state', return_value={'platform_status':'RUNNING'}), \
                patch.object(pool, '_observe_one', side_effect=observe):
            with ThreadPoolExecutor(max_workers=12) as executor:
                self.assertEqual(list(executor.map(pool.observe_one, range(12))), list(range(12)))
        self.assertEqual(maximum, 8)
        self.assertTrue(all(b - a >= 0.009 for a, b in zip(starts, starts[1:])))
        self.assertEqual(active, 0)

    def test_shutdown_interrupts_rate_wait_without_holding_resource_locks(self):
        import fcntl
        pool = self.pool()
        pool.provision_one()
        pool._observer_last_start = time.monotonic() + 60
        with patch.object(pool, '_observe_one') as observe:
            thread = threading.Thread(target=pool.observe_one, args=('1',))
            thread.start()
            time.sleep(0.03)
            for path in (pool.root / 'provision.lock', pool.directory('1') / 'lifecycle.lock'):
                with path.open('a') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pool.exit.set()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            observe.assert_not_called()
        self.assertTrue(pool._observer_slots.acquire(blocking=False))
        pool._observer_slots.release()

    def test_pending_observation_cannot_occupy_live_readiness_permits(self):
        pool=self.pool();pool._observer_start_interval=0
        pool._waiting_observers.acquire()
        try:
            with patch.object(pool,'state',return_value={'platform_status':'RUNNING'}), \
                    patch.object(pool,'_observe_one',return_value='renewed'):
                self.assertEqual(pool.observe_one('1'),'renewed')
        finally:
            pool._waiting_observers.release()
        from .budget_pool import observation_interval
        self.assertEqual(observation_interval({'platform_status':'RUNNING'}),8)
        for state in ('PENDING','STOPPED','FAILED',None):
            self.assertGreaterEqual(observation_interval({'platform_status':state}),60)

    def test_run_starts_existing_observers_before_provision(self):
        pool = self.pool()
        pool.provision_one()
        events = []
        class Thread:
            def __init__(self, target, args, daemon):
                self.target, self.args = target, args
            def start(self):
                self.target(*self.args)
            def is_alive(self):
                return True
        def provision():
            events.append('provision')
            pool.exit.set()
        with patch('corpus_pipeline.serving.budget_pool.threading.Thread', Thread), \
             patch.object(pool, 'worker', side_effect=lambda n: events.append('observe-' + n)), \
             patch.object(pool, 'provision_one', side_effect=provision):
            pool.run()
        self.assertEqual(events, ['observe-1', 'provision'])

    def test_error_chain_is_bounded_and_excludes_sensitive_messages_and_non_numeric_codes(self):
        error = ValueError('secret-url-token')
        error.status = 503
        error.status_code = 'secret-header'
        cause = OSError(111, 'secret-path')
        error.__cause__ = cause
        for _ in range(5):
            cause.__cause__ = RuntimeError('secret-key')
            cause = cause.__cause__
        chain = safe_error_chain(error)
        self.assertEqual(len(chain), 4)
        self.assertEqual(chain[0], {'type': 'ValueError', 'status': 503})
        self.assertEqual(chain[1]['errno'], 111)
        self.assertNotIn('secret', json.dumps(chain))
        pool, _ = self.ready_pool()
        with patch.object(self.backend, 'observe', side_effect=error):
            state = pool.observe_one('1')
        self.assertEqual(state['pool_last_error']['cause_chain'], chain)

    def test_busy_provision_defers_unsent_while_ready_observer_renews(self):
        import fcntl
        pool, registry = self.ready_pool()
        self.backend.raise_create = True
        with self.assertRaises(TimeoutError):
            pool.provision_one()
        self.backend.raise_create = False
        pending_bytes = (pool.directory('2') / 'state.json').read_bytes()
        prior_expiry = json.loads(registry.read_text())['endpoints'][0]['ready_expires_at']
        renewed = threading.Event()
        results = {}
        def observe(directory, state):
            self.assertEqual(state['service_id'], 'id-test-01')
            value = json.loads(registry.read_text())
            value['endpoints'][0]['ready_expires_at'] = prior_expiry + 30
            atomic_json(registry, value)
            self.backend.publish(directory, state, True)
            renewed.set()
            return {**state, 'heartbeat_at': time.time()}
        def call(number):
            results[number] = pool.observe_one(number)
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=object()), \
             patch.object(self.backend, 'observe', side_effect=observe):
            threads = [threading.Thread(target=call, args=(n,), daemon=True) for n in ('2', '1')]
            with (pool.root / 'provision.lock').open('a') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                for thread in threads:
                    thread.start()
                try:
                    self.assertTrue(renewed.wait(2), 'Ready renewal blocked behind allocation lock')
                    for thread in threads:
                        thread.join(timeout=1)
                        self.assertFalse(thread.is_alive())
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
        self.assertIsNone(results['2'])
        self.assertEqual((pool.directory('2') / 'state.json').read_bytes(), pending_bytes)
        self.assertEqual(self.backend.published, [True])
        self.assertEqual(json.loads(registry.read_text())['endpoints'][0]['ready_expires_at'], prior_expiry + 30)
        self.assertEqual(len(self.backend.created), 2)

    def test_busy_lifecycle_and_restart_locks_defer_without_false_error(self):
        import fcntl
        pool, registry = self.ready_pool()
        before_state = (pool.directory('1') / 'state.json').read_bytes()
        before_registry = registry.read_bytes()
        with (pool.directory('1') / 'lifecycle.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            self.assertIsNone(pool.observe_one('1'))
        self.assertEqual((pool.directory('1') / 'state.json').read_bytes(), before_state)
        self.assertEqual(registry.read_bytes(), before_registry)
        self.backend.should_restart = 'victim-event'
        with (pool.root / 'provision.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            state = pool.observe_one('1')
        self.assertNotIn('pool_error_type', state)
        self.assertEqual(self.backend.restarts, [])
        self.assertEqual(self.backend.published, [])
        pool.observe_one('1')
        self.assertEqual(self.backend.restarts, [('id-test-01', 'victim-event')])

    def rate_error(self, status=429):
        error = TransportError('hidden endpoint') if TransportError else RuntimeError('hidden endpoint')
        cause = RuntimeError('hidden response')
        cause.status = status
        error.__cause__ = cause
        return error

    def test_rate_backoff_persists_grows_once_per_window_and_is_capped(self):
        pool = self.pool()
        with patch('corpus_pipeline.serving.budget_pool.time.time', return_value=1000), \
             patch('corpus_pipeline.serving.budget_pool.random.uniform', side_effect=lambda a, b: b):
            self.assertTrue(pool.record_rate_limit(self.rate_error()))
            first = pool._backoff_state()
            self.assertEqual(first['not_before_epoch'], 1072)
            self.assertTrue(pool.record_rate_limit(self.rate_error()))
            self.assertEqual(pool._backoff_state()['consecutive_429'], 1)
            self.assertEqual(pool._backoff_state()['not_before_epoch'], 1072)
            restarted = BudgetPool(pool.config_path, self.backend)
            self.assertTrue(restarted.provisioning_delayed())
            restarted.provisioning_succeeded()  # Cannot erase a newer/current active cooldown.
            self.assertTrue(restarted.provisioning_delayed())
        with patch('corpus_pipeline.serving.budget_pool.random.uniform', side_effect=lambda a, b: b):
            for expected in (144, 288, 576, 900, 900):
                now = pool._backoff_state()['not_before_epoch'] + 1
                with patch('corpus_pipeline.serving.budget_pool.time.time', return_value=now):
                    pool.record_rate_limit(self.rate_error())
                    self.assertEqual(pool._backoff_state()['cooldown_seconds'], expected)
            now = pool._backoff_state()['not_before_epoch'] + 1
            with patch('corpus_pipeline.serving.budget_pool.time.time', return_value=now):
                pool.provisioning_succeeded()
                self.assertEqual(pool._backoff_state()['consecutive_429'], 0)
                self.assertFalse(pool.provisioning_delayed())
        self.assertNotIn('hidden', (pool.root / 'PROVISION_BACKOFF.json').read_text())

    def test_prepare_429_holds_global_cooldown_without_creating_reservation(self):
        pool = self.pool()
        with patch.object(self.backend, 'prepare', side_effect=self.rate_error()) as prepare:
            with self.assertRaises(Exception):
                pool.provision_one()
            self.assertFalse(pool.provision_one())
            prepare.assert_called_once()
        self.assertEqual(pool.ledger()['slots'], {})
        self.assertEqual(self.backend.created, [])
        self.assertEqual(pool._backoff_state()['consecutive_429'], 1)

    def test_paused_or_full_pool_does_not_query_external_budget(self):
        self.cfg.update(max_cards=2, replicas=1)
        pool = self.pool()
        atomic_json(pool.root / 'TARGET.json', {'replicas': 0})
        with patch.object(pool, 'external_cards', side_effect=AssertionError('unnecessary platform query')):
            self.assertFalse(pool.provision_one())
        (pool.root / 'TARGET.json').unlink()
        pool.provision_one()
        with patch.object(pool, 'external_cards', side_effect=AssertionError('unnecessary platform query')):
            self.assertFalse(pool.provision_one())

    def test_concurrent_429_during_prepare_prevents_new_reservation_and_create(self):
        pool = self.pool()
        prepare = self.backend.prepare
        def ready_but_throttled(*args):
            state = prepare(*args)
            pool.record_rate_limit(self.rate_error())  # A concurrent ready observer's failure.
            return state
        with patch.object(self.backend, 'prepare', side_effect=ready_but_throttled):
            self.assertFalse(pool.provision_one())
        self.assertEqual(pool.ledger()['slots'], {})
        self.assertEqual(self.backend.created, [])

    def test_concurrent_429_during_name_lookup_defers_recovery_before_create(self):
        pool, client, adapter = self.unsent_pool()
        before = (pool.directory('1') / 'state.json').read_bytes()
        def absent_but_throttled(**kwargs):
            pool.record_rate_limit(self.rate_error())
            return []
        client.servings.iter.side_effect = absent_but_throttled
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter):
            self.assertIsNone(pool.observe_one('1'))
        self.assertEqual(self.backend.created, ['test-01'])
        self.assertEqual((pool.directory('1') / 'state.json').read_bytes(), before)
        self.assertFalse((pool.directory('1') / 'CREATE_TRANSMISSION.json').exists())

    def test_recovery_429_pauses_other_unsent_and_allocation_but_not_ready_observation(self):
        pool, registry = self.ready_pool()
        self.backend.raise_create = True
        for _ in range(2):
            with self.assertRaises(TimeoutError):
                pool.provision_one()
        self.backend.raise_create = False
        original_ledger = (pool.root / 'RESERVATIONS.json').read_bytes()
        pending3 = (pool.directory('3') / 'state.json').read_bytes()
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=object()), \
             patch.object(self.backend, 'binding', side_effect=self.rate_error()):
            result = pool.observe_one('2')
        self.assertEqual(result['pool_error_stage'], 'UNSENT_RECOVERY')
        observe = Mock(wraps=self.backend.observe)
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=object()), \
             patch.object(self.backend, 'observe', observe), \
             patch.object(self.backend, 'prepare') as prepare:
            self.assertIsNone(pool.observe_one('3'))
            self.assertFalse(pool.provision_one())
            self.assertEqual(pool.observe_one('1')['platform_status'], 'RUNNING')
            observe.assert_called_once()
            prepare.assert_not_called()
        self.assertEqual((pool.root / 'RESERVATIONS.json').read_bytes(), original_ledger)
        self.assertEqual((pool.directory('3') / 'state.json').read_bytes(), pending3)
        self.assertEqual(pool._backoff_state()['consecutive_429'], 1)  # Healthy reads do not reset it.

    def test_cooldown_reconciles_sent_unknown_but_defers_restart_and_non429_does_not_trigger(self):
        pool, _, adapter = self.unsent_pool()
        atomic_json(pool.directory('1') / 'CREATE_TRANSMISSION.json', {'operation_id': 'test-01'})
        self.assertFalse(pool.record_rate_limit(self.rate_error(503)))
        self.assertFalse((pool.root / 'PROVISION_BACKOFF.json').exists())
        pool.record_rate_limit(self.rate_error())
        self.backend.should_restart = 'preempted'
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter):
            self.assertEqual(pool.observe_one('1')['service_id'], 'id-test-01')
        self.assertEqual(self.backend.created, ['test-01'])
        self.assertEqual(self.backend.restarts, [])
        with patch('corpus_pipeline.serving.budget_pool.time.time',
                   return_value=pool._backoff_state()['not_before_epoch'] + 1):
            pool.observe_one('1')
        self.assertEqual(self.backend.created, ['test-01'])  # Expiry is never permission to replay Create.
        self.assertEqual(self.backend.restarts, [('id-test-01', 'preempted')])

    @unittest.skipIf(TransportError is None, 'Actual SDK TransportError needed')
    def test_observation_429_keeps_valid_lease_and_pauses_allocation(self):
        pool, registry = self.ready_pool()
        before = registry.read_bytes()
        with patch.object(self.backend, 'observe', side_effect=self.rate_error()):
            self.assertTrue(pool.observe_one('1')['pool_registry_lease_preserved'])
        self.assertEqual(registry.read_bytes(), before)
        self.assertEqual(self.backend.published, [])
        self.assertTrue(pool.provisioning_delayed())

    def test_stop_cancels_proven_never_sent_without_claiming_platform_stop_or_reusing_slot(self):
        import fcntl
        pool, client, adapter = self.unsent_pool()
        original_ledger = (pool.root / 'RESERVATIONS.json').read_bytes()
        intent_path = pool.directory('1') / 'CREATE_INTENT.json'
        intent = json.loads(intent_path.read_text())
        intent['pool_config_sha256'] = 'previous-control-only-config'
        atomic_json(intent_path, intent)
        def absent(**kwargs):
            for path in (pool.root / 'cleanup.lock', pool.directory('1') / 'lifecycle.lock'):
                with path.open('a') as stream:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return []
        client.servings.iter.side_effect = absent
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter), \
             patch.object(self.backend, 'stop') as stop:
            result = pool.stop_once()
            stop.assert_not_called()
        self.assertTrue(result['complete'])
        self.assertTrue(result['slots'][0]['cancelled_never_sent'])
        state = pool.state('1')
        self.assertEqual(state['phase'], 'CANCELLED_UNSENT')
        self.assertNotEqual(state.get('platform_status'), 'STOPPED')
        proof = json.loads((pool.directory('1') / 'CANCELLED_NEVER_SENT.json').read_text())
        self.assertFalse(proof['platform_stop_claimed'])
        self.assertEqual(proof['original_intent_pool_config_sha256'], 'previous-control-only-config')
        self.assertEqual((pool.root / 'RESERVATIONS.json').read_bytes(), original_ledger)
        self.assertEqual(reservation_total([state]), 2)
        with self.assertRaises(ValueError):
            pool.provision_one()

    def test_never_sent_cancel_refuses_drift_or_failed_lookup_and_keeps_unknown_transmission(self):
        pool, client, adapter = self.unsent_pool()
        intent_path = pool.directory('1') / 'CREATE_INTENT.json'
        original = intent_path.read_bytes()
        changed = json.loads(original)
        changed['operation_id'] = 'different-operation'
        atomic_json(intent_path, changed)
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter), \
             patch.object(self.backend, 'stop', side_effect=ValueError('unresolved')) as stop:
            self.assertFalse(pool.stop_once()['complete'])
            stop.assert_not_called()
            intent_path.write_bytes(original)
            client.servings.iter.side_effect = self.rate_error()
            self.assertFalse(pool.stop_once()['complete'])
            stop.assert_not_called()
            client.servings.iter.side_effect = None
            atomic_json(pool.directory('1') / 'CREATE_TRANSMISSION.json', {'sent': True})
            self.assertFalse(pool.stop_once()['complete'])
            stop.assert_called_once()
        self.assertFalse((pool.directory('1') / 'CANCELLED_NEVER_SENT.json').exists())
        self.assertFalse(pool.state('1').get('release_verified', False))

    def test_never_sent_name_match_uses_normal_owned_stop_not_cancellation(self):
        pool, client, adapter = self.unsent_pool()
        client.servings.iter.return_value = [SimpleNamespace(name='test-01')]
        with patch('corpus_pipeline.serving.budget_pool.audited_unsent_backend', return_value=adapter), \
             patch.object(self.backend, 'stop', wraps=self.backend.stop) as stop:
            self.assertTrue(pool.stop_once()['complete'])
            stop.assert_called_once()
        self.assertFalse((pool.directory('1') / 'CANCELLED_NEVER_SENT.json').exists())


if __name__ == '__main__':
    unittest.main()
