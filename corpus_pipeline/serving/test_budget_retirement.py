"""Retirement integration with a fake native SDK and temporary filesystem.

No real SDK client is constructed and no platform/network request is issued.
No production configuration or process is used.
"""
from contextlib import nullcontext
import itertools
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from corpus_pipeline.resource_manager.engine import atomic_json
from corpus_pipeline.serving import budget_pool as subject
from corpus_pipeline.postprocess.full_run_watchdog import Watchdog


class FakeNative:
    def __init__(self):
        self.created, self.stopped, self.observed = [], [], []
        self.reads = []
        self.platform = {}
        self.before_send_failure = False
        self.after_send_failure = False
        self.fail_fresh_read = False
        self.binding_value = {'immutable': 'fake-native-contract'}
        def forbidden(*args, **kwargs):
            raise AssertionError('Retire must not perform any native mutation')
        self.client = SimpleNamespace(servings=SimpleNamespace(
            instances=self.instances, iter=self.iter_services,
            stop=forbidden, start=forbidden, create=forbidden, delete=forbidden))
        self.adapter = SimpleNamespace(InspireClient=lambda **_: nullcontext(self.client),
                                       native=SimpleNamespace(WORKSPACE='fake-owned-workspace'))

    def binding(self):
        return {}, self.binding_value

    def validate(self, state):
        if state['binding'] != self.binding_value:
            raise ValueError('Fake binding changed')

    def prepare(self, directory, number, name, priority):
        return {'name': name, 'service_name': name, 'operation_id': name,
                'cards': 2, 'priority': priority, 'tier': 'low', 'binding': self.binding_value,
                'scope': str(directory), 'phase': 'PREPARED', 'release_verified': False,
                'canary_passed': False}

    def create(self, directory, state):
        if self.before_send_failure:
            raise RuntimeError('fake-preflight-unsent')
        atomic_json(Path(directory) / 'CREATE_TRANSMISSION.json', {'name': state['name']})
        self.created.append(state['name'])
        if self.after_send_failure:
            raise RuntimeError('fake-dispatched-unknown')
        sid = 'id-' + state['name']
        self.platform[sid] = {'status': 'RUNNING', 'instances': ['INSTANCE_RUNNING'], 'name': state['name']}
        return {**state, 'service_id': sid, 'platform_status': 'RUNNING', 'phase': 'QUALIFIED',
                'canary_passed': True, 'instance_statuses': ['INSTANCE_RUNNING']}

    def reconcile(self, client, state):
        self.reads.append(('detail', state['service_id']))
        if self.fail_fresh_read:
            raise TimeoutError('fake-native-read-unavailable')
        actual = self.platform[state['service_id']]
        if actual['name'] != state['name']:
            raise ValueError('Fake native ownership mismatch')
        return state, SimpleNamespace(ref=state['service_id'], status=actual['status'])

    def instances(self, ref, **kwargs):
        self.reads.append(('instances', ref))
        return [SimpleNamespace(status=s) for s in self.platform[ref]['instances']]

    def iter_services(self, **kwargs):
        self.reads.append(('list', None))
        return [SimpleNamespace(name=v['name']) for v in self.platform.values()]

    def publish(self, directory, state, ready):
        atomic_json(Path(directory) / 'registry.json', {'ready': ready, 'service_id': state.get('service_id')})

    def observe(self, directory, state):
        self.observed.append(state['name'])
        return state

    def can_restart(self, state):
        return None

    def released(self, state):
        return False

    def stop(self, directory, state):
        self.stopped.append(state['name'])
        directory = Path(directory)
        (directory / 'DISABLED').write_text('fake explicit stop')
        sid = state['service_id']
        self.platform[sid].update(status='STOPPED', instances=['INSTANCE_SUCCEEDED'])
        stopped = {'complete': True, 'service_id': sid, 'status': 'STOPPED',
                   'instance_statuses': ['INSTANCE_SUCCEEDED'], 'time': time.time()}
        atomic_json(directory / 'STOP_INTENT.json', {'service_id': sid})
        atomic_json(directory / 'STOPPED.json', stopped)
        self.publish(directory, state, False)
        return {**state, 'phase': 'RELEASED', 'platform_status': 'STOPPED',
                'release_verified': True, 'canary_passed': False,
                'instance_statuses': stopped['instance_statuses']}


class RetirementIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='candidate-retirement-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {'schema_version': 'corpus.native_budget_pool.v1',
            'root': str(self.root / 'pool'), 'max_cards': 128, 'cards_per_replica': 2,
            'replicas': 64, 'priority': 3, 'per_replica_max_inflight': 64,
            'name_prefix': 'fake-owned', 'poll_seconds': 8}
        self.config_path = self.root / 'config.json'
        atomic_json(self.config_path, self.config)
        self.backend = FakeNative()
        self.pool = subject.BudgetPool(self.config_path, self.backend)
        audited = patch.object(subject, 'audited_unsent_backend', return_value=self.backend.adapter)
        audited.start()
        self.addCleanup(audited.stop)

    def fill(self, number):
        for _ in range(number):
            self.assertTrue(self.pool.provision_one())

    def release(self, number):
        state = self.backend.stop(self.pool.directory(number), self.pool.state(number))
        self.pool.save_state(number, state)
        return state

    def test_retire_performs_fresh_reads_and_no_mutation_or_ledger_deletion(self):
        self.fill(1)
        self.release(1)
        writes_before = list(self.backend.created), list(self.backend.stopped)
        intent_before = (self.pool.directory(1) / 'CREATE_INTENT.json').read_bytes()
        proof = self.pool.retire(1)
        self.assertEqual(writes_before, (self.backend.created, self.backend.stopped))
        self.assertEqual([x[0] for x in self.backend.reads], ['detail', 'instances'])
        self.assertTrue(proof['complete'])
        self.assertEqual(set(self.pool.ledger()['slots']), {'1'})
        self.assertEqual((self.pool.directory(1) / 'CREATE_INTENT.json').read_bytes(), intent_before)
        self.assertEqual(self.pool.state(1)['phase'], 'RELEASED')
        self.assertTrue(self.pool.retired(1))
        self.assertEqual(self.pool.status()['slot_reserved_cards'], 0)
        self.assertEqual(subject.read(self.pool.root / 'pool.json')['registries'], [])
        self.pool._observe_one(1)
        self.assertEqual(self.backend.observed, [])
        reads = list(self.backend.reads)
        self.assertEqual(self.pool.retire(1), proof)
        self.assertEqual(reads, self.backend.reads)  # committed idempotence does not re-query

    def test_sixty_four_to_sixty_five_restores_one_hundred_twenty_eight_active_cards(self):
        self.fill(64)
        self.release(20)
        self.assertFalse(self.pool.provision_one())  # stop alone never frees a reservation
        self.pool.retire(20)
        self.assertTrue(self.pool.provision_one())
        self.assertEqual(self.pool.state(65)['name'], 'fake-owned-65')
        counts = self.pool.status()['reservation_accounting']
        self.assertEqual((counts['historical_slot_count'], counts['historical_reserved_cards'],
                          counts['retired_slot_count'], counts['active_reserved_slots'],
                          counts['active_reserved_cards']), (65, 130, 1, 64, 128))
        self.assertFalse(self.pool.provision_one())
        self.assertEqual(len(self.pool.ledger()['slots']), 65)

    def test_unknown_dispatched_create_still_holds_budget_and_is_not_retransmitted(self):
        self.fill(63)
        self.backend.after_send_failure = True
        with self.assertRaisesRegex(RuntimeError, 'unknown'):
            self.pool.provision_one()
        self.assertEqual(self.pool.status()['slot_reserved_cards'], 128)
        self.assertFalse(self.pool.provision_one())
        self.backend.after_send_failure = False
        self.pool.recover_unsent(64, self.pool.state(64))
        self.assertEqual(self.backend.created.count('fake-owned-64'), 1)

    def test_fresh_native_running_or_read_failure_cannot_retire_old_local_proof(self):
        self.fill(1)
        state = self.release(1)
        self.backend.platform[state['service_id']]['status'] = 'RUNNING'
        with self.assertRaisesRegex(ValueError, 'freshly terminal'):
            self.pool.retire(1)
        self.assertNotIn('retirement', self.pool.ledger()['slots']['1'])
        self.assertEqual(self.pool.status()['slot_reserved_cards'], 2)
        self.backend.fail_fresh_read = True
        with self.assertRaises(TimeoutError):
            self.pool.retire(1)
        self.assertEqual(self.pool.status()['slot_reserved_cards'], 2)

    def test_archive_corruption_and_reenable_fail_closed_before_allocation(self):
        self.fill(1)
        self.release(1)
        self.pool.retire(1)
        archive = self.pool.directory(1) / 'retirement/STOPPED_SNAPSHOT.json'
        before = archive.read_bytes()
        archive.write_bytes(before + b' ')
        with self.assertRaisesRegex(ValueError, 'evidence_changed'):
            self.pool.provision_one()
        archive.write_bytes(before)
        (self.pool.directory(1) / 'DISABLED').unlink()
        with self.assertRaisesRegex(ValueError, 'reenabled'):
            self.pool.provision_one()
        self.assertEqual(self.backend.created, ['fake-owned-01'])

    def test_original_ledger_binding_drift_cannot_keep_retirement_exemption(self):
        self.fill(1)
        self.release(1)
        self.pool.retire(1)
        ledger = self.pool.ledger()
        ledger['slots']['1']['intent']['binding'] = {'immutable': 'different-contract'}
        self.pool.save_ledger(ledger)
        with self.assertRaises(ValueError):
            self.pool.provision_one()
        self.assertEqual(self.backend.created, ['fake-owned-01'])

    def test_uncommitted_proof_cannot_free_budget_but_can_resume_after_ledger_crash(self):
        self.fill(1)
        self.release(1)
        with patch.object(self.pool, 'save_ledger', side_effect=OSError('fake-commit-interrupted')):
            with self.assertRaises(OSError):
                self.pool.retire(1)
        proof_path = self.pool.directory(1) / 'retirement/COMMITTED.json'
        proof_bytes = proof_path.read_bytes()
        self.assertEqual(self.pool.status()['slot_reserved_cards'], 2)
        self.assertFalse(self.pool.retired(1))
        self.pool.retire(1)
        self.assertEqual(proof_path.read_bytes(), proof_bytes)
        self.assertEqual(self.pool.status()['slot_reserved_cards'], 0)

    def test_slot_sixty_five_proven_unsent_recovers_without_new_reservation(self):
        self.fill(64)
        self.release(20)
        self.pool.retire(20)
        self.backend.before_send_failure = True
        with self.assertRaisesRegex(RuntimeError, 'unsent'):
            self.pool.provision_one()
        self.backend.before_send_failure = False
        self.assertFalse((self.pool.directory(65) / 'CREATE_TRANSMISSION.json').exists())
        self.assertTrue(self.pool.unsent_allowed(65))
        atomic_json(self.pool.root / 'TARGET.json', {'replicas': 0})
        self.assertFalse(self.pool.unsent_allowed(65))
        self.pool.recover_unsent(65, self.pool.state(65))
        self.assertNotIn('fake-owned-65', self.backend.created)
        atomic_json(self.pool.root / 'TARGET.json', {'replicas': 64})
        state = self.pool.recover_unsent(65, self.pool.state(65))
        self.pool.save_state(65, state)
        self.assertEqual(self.backend.created.count('fake-owned-65'), 1)
        self.assertEqual(len(self.pool.ledger()['slots']), 65)
        self.assertEqual(self.pool.status()['slot_reserved_cards'], 128)

    def test_final_cleanup_and_existing_watchdog_cover_retired_slot(self):
        self.fill(64)
        self.release(20)
        self.pool.retire(20)
        self.pool.provision_one()
        # Avoid a 0.35-second dispatch wait per fake cleanup request. The thread
        # synchronization and actual file operations still execute normally.
        with patch.object(subject.time, 'monotonic', side_effect=itertools.count().__next__):
            stopped = self.pool.stop_once()
        self.assertTrue(stopped['complete'])
        self.assertEqual({r['slot'] for r in stopped['slots']}, {str(n) for n in range(1, 66)})
        self.assertIn('fake-owned-20', self.backend.stopped)
        self.assertTrue(self.pool.retired(20))  # refreshed live STOPPED does not corrupt archive
        guard = object.__new__(Watchdog)
        guard.pool_root, guard.pool_sha, guard.pool = self.pool.root, self.pool.config_sha, self.config
        self.assertIsNotNone(guard.released())
        stopped['slots'] = [r for r in stopped['slots'] if str(r['slot']) != '20']
        atomic_json(self.pool.root / 'STOPPED.json', stopped)
        self.assertIsNone(guard.released())


if __name__ == '__main__':
    unittest.main()
