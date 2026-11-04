import copy
from dataclasses import asdict
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import socket
import tempfile
import unittest

from corpus_pipeline.resource_manager.engine import Controller, _digest
from corpus_pipeline.resource_manager.models import WorkloadSpec
from corpus_pipeline.unified.baseline_fleet_ops import status


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hold_lock(path, ready, finish):
    with open(path, 'r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        ready.set()
        finish.wait(20)


class FleetReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manager = self.root / 'fleet/frontpass.json'
        self.directory = self.root / 'fleet/frontpass/controller'
        self.pipeline = self.root / 'config.json'
        self.template = self.root / 'fleet-template.json'
        _write(self.pipeline, {'output': str(self.root)})
        _write(self.template, {'template': True})
        self.identity = self.root / 'fleet/IDENTITY.json'
        _write(self.identity, {'schema': 'corpus.unified.fleet.v1',
            'manager_directory': str(self.manager.parent), 'config_path': str(self.pipeline),
            'config_sha256': _sha(self.pipeline), 'template_path': str(self.template),
            'template_sha256': _sha(self.template), 'stage_budgets': {'frontpass': 48, 'sscd_search': 48},
            'total_max_cards': 128, 'account_ceiling': 128})
        self.spec = WorkloadSpec(scope='unified-' + _sha(self.identity)[:16], profile='frontpass',
            max_cards=48, command=('python', 'worker.py'), allowed_cards=(8, 4, 2, 1))
        contract = {'schema': 'corpus.unified.fleet.v1', 'stage': 'frontpass',
            'state_directory': str(self.directory), 'scope': self.spec.scope, 'profile': self.spec.profile,
            'stage_max_cards': 48, 'identity_path': str(self.identity),
            'identity_sha256': _sha(self.identity), 'config_path': str(self.pipeline),
            'config_sha256': _sha(self.pipeline), 'total_max_cards': 128, 'account_ceiling': 128}
        _write(self.manager, {'adapter': 'corpus_pipeline.unified.fleet:StageAdapter',
            'state_dir': str(self.directory), 'workload': asdict(self.spec),
            'adapter_config': {'contract': contract}})
        self.controller = Controller(None, self.spec, self.directory)
        with self.controller.locked():
            self.controller.state = self.controller._load()
            intent = {'operation_id': 'allocation-a', 'scope': self.spec.scope,
                'profile': self.spec.profile, 'placement': {'node': 'node-a',
                'quota': {'cards': 8, 'cpu': 64, 'memory': 512}}, 'command': list(self.spec.command)}
            self.controller.state['resources']['allocation-a'] = {'intent': intent,
                'intent_sha256': _digest(intent), 'resource_id': 'notebook-a', 'state': 'STOPPED',
                'retired_verified': True, 'observed_at': 100}
            self.controller.state['phase'] = 'completed'
            self.controller._save()

    def change(self, callback):
        path = self.directory / 'STATE.json'
        value = json.loads(path.read_text())
        callback(value)
        _write(path, value)

    def test_verified_ledger_releases_without_business_receipt(self):
        result = status(self.manager)
        self.assertTrue(result['released'], result)
        self.assertEqual(result['reserved'], 0)
        self.assertEqual(result['verified_retired_resources'], 1)

    def test_stage_complete_never_releases_active_resource(self):
        _write(self.root / 'frontpass/COMPLETE.json', {'complete': True})
        self.change(lambda state: state['resources']['allocation-a'].update(state='RUNNING'))
        result = status(self.manager)
        self.assertFalse(result['released'])
        self.assertEqual(result['reserved'], 8)

    def test_unknown_create_remains_reserved(self):
        self.change(lambda state: state['resources']['allocation-a'].update(
            resource_id=None, state='UNKNOWN', retired_verified=False))
        self.assertEqual(status(self.manager)['reserved'], 8)
        self.assertFalse(status(self.manager)['released'])

    def test_unverified_stop_remains_reserved(self):
        self.change(lambda state: state['resources']['allocation-a'].update(retired_verified=False))
        self.assertEqual(status(self.manager)['reserved'], 8)

    def test_observation_error_blocks_stale_retirement(self):
        self.change(lambda state: state['resources']['allocation-a'].update(observation_error_type='TimeoutError'))
        self.assertEqual(status(self.manager)['reserved'], 8)

    def test_missing_required_state_fails_closed(self):
        (self.directory / 'STATE.json').unlink()
        result = status(self.manager)
        self.assertFalse(result['released'])
        self.assertIsNone(result['reserved'])

    def test_no_state_not_misreported_as_released(self):
        (self.directory / 'STATE.json').unlink()
        (self.directory / 'STATE_REQUIRED.json').unlink()
        result = status(self.manager)
        self.assertFalse(result['released'])
        self.assertEqual(result['phase'], 'not_started')

    def test_foreign_host_cannot_prove_controller_exit(self):
        _write(self.directory / '.owner/HOST.json', {'hostname': socket.gethostname() + '-other'})
        result = status(self.manager)
        self.assertFalse(result['released'])
        self.assertEqual(result['reason'], 'query_requires_controller_host')

    def test_live_controller_holding_lock_blocks_release(self):
        context = multiprocessing.get_context('fork')
        ready, finish = context.Event(), context.Event()
        process = context.Process(target=_hold_lock,
            args=(str(self.directory / '.controller.lock'), ready, finish))
        process.start()
        try:
            self.assertTrue(ready.wait(5))
            result = status(self.manager)
            self.assertFalse(result['released'])
            self.assertTrue(result['controller_running'])
        finally:
            finish.set(); process.join(5)
            if process.is_alive():
                process.kill(); process.join()
        self.assertTrue(status(self.manager)['released'])

    def test_intent_tamper_fails_closed(self):
        self.change(lambda state: state['resources']['allocation-a']['intent']['placement']['quota'].update(cards=1))
        self.assertIsNone(status(self.manager)['reserved'])
        self.assertFalse(status(self.manager)['released'])

    def test_config_or_marker_identity_change_fails_closed(self):
        original = json.loads(self.pipeline.read_text())
        _write(self.pipeline, {'different': True})
        self.assertFalse(status(self.manager)['released'])
        _write(self.pipeline, original)
        required = self.directory / 'STATE_REQUIRED.json'
        value = json.loads(required.read_text()); value['identity_sha256'] = 'wrong'
        _write(required, value)
        self.assertFalse(status(self.manager)['released'])

    def test_completed_ledger_without_final_observation_fails_closed(self):
        self.change(lambda state: state['resources']['allocation-a'].pop('observed_at'))
        self.assertEqual(status(self.manager)['reserved'], 8)


if __name__ == '__main__':
    unittest.main()
