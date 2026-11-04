import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from .budget_pool import BudgetPool
from .test_budget_pool import FakeBackend
from . import review_control as mod
from corpus_pipeline.resource_manager.engine import atomic_json


class ReviewControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = {'schema_version': 'corpus.native_budget_pool.v1', 'root': str(self.root / 'pool'),
            'max_cards': 2, 'cards_per_replica': 2, 'replicas': 1, 'priority': 1,
            'per_replica_max_inflight': 128, 'name_prefix': 'baseline-test', 'poll_seconds': 8}
        atomic_json(self.root / 'config.json', config)
        self.backend = FakeBackend()
        self.pool = BudgetPool(self.root / 'config.json', self.backend)
        self.pool.provision_one()

    def qualify(self):
        state = self.pool.state('1')
        keys = ('model', 'revision', 'precision', 'processor_revision', 'runtime_revision',
                'incarnation', 'platform_instance_id', 'launch_contract_sha256')
        state.update({key: key + '-bound' for key in keys})
        receipt = self.pool.directory('1') / 'smoke/current/RESULT.json'
        state.update(canary_passed=True, phase='QUALIFIED', qualification_receipt=str(receipt))
        self.pool.save_state('1', state)
        proof = {key: state[key] for key in keys}
        proof.update(passed=True, request_count=1, qualification_kind='basic_smoke',
                     standard_sse_done=True, image_input={'sha256': 'real-image'},
                     usage={'prompt_tokens': 42}, finish_reason='stop', response_text='Image objects.')
        atomic_json(receipt, proof)
        endpoint = {key: state[key] for key in (*keys, 'service_id')}
        endpoint.update(ready=True, ready_expires_at=time.time()+30, supports_images=True,
                        transport_mode='native_openai', native_stream=True, backend='inspire_serving',
                        qualification_receipt=str(receipt))
        atomic_json(self.pool.directory('1') / 'registry.json', {'endpoints': [endpoint]})
        return state, endpoint, receipt

    def test_running_alone_is_not_ready_then_current_image_proof_is(self):
        with patch.object(mod, 'controller_alive', return_value=True):
            self.assertFalse(mod.ready(self.pool)['ready'])
            self.qualify()
            self.assertTrue(mod.ready(self.pool)['ready'])

    def test_stale_or_missing_image_proof_rejected(self):
        _, endpoint, receipt = self.qualify()
        with patch.object(mod, 'controller_alive', return_value=True):
            proof = mod.read(receipt); proof['incarnation'] = 'old'
            atomic_json(receipt, proof)
            self.assertFalse(mod.ready(self.pool)['ready'])
            self.qualify()
            proof = mod.read(receipt); proof.pop('image_input')
            atomic_json(receipt, proof)
            self.assertFalse(mod.ready(self.pool)['ready'])
            self.qualify()
            endpoint['ready_expires_at'] = 0
            atomic_json(self.pool.directory('1') / 'registry.json', {'endpoints': [endpoint]})
            self.assertFalse(mod.ready(self.pool)['ready'])

    def test_release_requires_disabled_live_zero_and_controller_exit(self):
        self.assertFalse(mod.released(self.pool)['released'])
        self.pool.stop_once()
        with patch.object(mod, 'controller_alive', return_value=False):
            pending = mod.released(self.pool)
            self.assertFalse(pending['released'])
            self.assertEqual(pending['reserved'], 2)
            self.backend.external_released = True
            self.assertEqual(mod.released(self.pool)['reserved'], 0)
            self.assertTrue(mod.released(self.pool)['released'])
        with patch.object(mod, 'controller_alive', return_value=True):
            self.assertFalse(mod.released(self.pool)['released'])


if __name__ == '__main__':
    unittest.main()
