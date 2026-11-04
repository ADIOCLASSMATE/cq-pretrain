from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from corpus_pipeline.serving.qwen36_w8a8_20261024 import backend

RUN = Path(__file__).resolve().parents[1]


class BoundServiceTests(unittest.TestCase):
    def test_known_id_skips_name_inventory_but_checks_live_ownership(self):
        state = {'service_id': 'observed-id', 'name': 'genund-six-fixture',
                 'scope': str(RUN / 'sources/fixture/review_pool/slots/1'),
                 'payload': {'workspace_id': 'observed-workspace'}}
        bound = object()
        detail = object()
        services = NS(_make_ref=Mock(return_value=bound), get=Mock(return_value=detail),
                      iter=Mock(side_effect=AssertionError('Name inventory must not be queried')))
        instance = backend.Backend.__new__(backend.Backend)
        with patch.object(instance, 'validate') as validate, patch.object(backend.native, 'assert_owned') as owned:
            self.assertEqual(instance.reconcile(NS(servings=services), state), (state, detail))
            validate.assert_called_once_with(state)
            services.get.assert_called_once_with(bound, workspace=backend.native.WORKSPACE)
            owned.assert_called_once_with(state, detail)
        with patch.object(instance, 'validate'), patch.object(backend.native, 'assert_owned', side_effect=AssertionError('wrong owner')):
            with self.assertRaisesRegex(AssertionError, 'wrong owner'):
                instance.reconcile(NS(servings=services), state)

    def test_unknown_creation_still_requires_exact_name_reconciliation(self):
        state = {'service_id': None, 'name': 'genund-six-fixture',
                 'scope': str(RUN / 'sources/fixture/review_pool/slots/1')}
        services = NS(iter=Mock(return_value=[]), get=Mock())
        instance = backend.Backend.__new__(backend.Backend)
        with patch.object(instance, 'validate'), self.assertRaisesRegex(ValueError, 'Create intent unresolved'):
            instance.reconcile(NS(servings=services), state)
        services.iter.assert_called_once()
        services.get.assert_not_called()


if __name__ == '__main__':
    unittest.main()
