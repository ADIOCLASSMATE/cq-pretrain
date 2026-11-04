from contextlib import nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from corpus_pipeline.serving.account_budget import observable_project_budget


class Resources:
    def __init__(self, rows):
        self.rows = rows
    def iter(self, **kwargs):
        return iter(self.rows)
    def get(self, ref, **kwargs):
        return next(row for row in self.rows if row.ref == ref)
    def instances(self, *args, **kwargs):
        raise AssertionError('Admission must not query physical placement')


def serving(name, cards, status='PENDING', owner='self'):
    return NS(name=name, ref=NS(key=name), status=status,
              raw={'user_id': owner, 'resource_spec_price': {'gpu_count': cards},
                   'replicas': 1, 'node_num_per_replica': 1})


class RequestedBudgetTests(unittest.TestCase):
    def setUp(self):
        self.client = NS(_transport=NS(scope=lambda **kwargs: nullcontext(), session=None),
            servings=Resources([serving('running', 64, 'RUNNING'), serving('pending', 384),
                                serving('unknown', 32, 'UNKNOWN'), serving('old', 500, 'STOPPED'),
                                serving('foreign', 500, owner='someone-else')]),
            notebooks=Resources([NS(ref=NS(key='notebook'), name='notebook', status='PENDING',
                                     resource={'gpu_count': 16}, node={})]),
            jobs=Resources([NS(ref=NS(key='job'), name='job', status='PENDING',
                               raw={'user_id': 'self', 'framework_config': [{'gpu_count': 8, 'instance_count': 2}]})]))
        self.current = patch('inspire.platform.web.browser_api.jobs.get_current_user', return_value={'id': 'self'})
        self.current.start()
        self.addCleanup(self.current.stop)

    def test_all_requested_cards_including_pending_and_unknown_count(self):
        result = observable_project_budget(self.client, 0, ceiling=512)
        self.assertEqual(result['observable_total_with_request'], 512)
        self.assertEqual(result['ceiling'], 512)
        self.assertIsNone(result['observable_allocated_cards'])
        self.assertEqual(set(result['records']), {'running', 'pending', 'unknown', 'notebook', 'job'})

    def test_new_request_over_512_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            observable_project_budget(self.client, 2, ceiling=512)

    def test_missing_ownership_still_refused(self):
        self.client.servings.rows[0].raw.pop('user_id')
        with self.assertRaisesRegex(AssertionError, 'ownership'):
            observable_project_budget(self.client, 0, ceiling=512)

    def test_malformed_job_resource_contract_still_refused(self):
        self.client.jobs.rows[0].raw['framework_config'] = []
        with self.assertRaisesRegex(ValueError, 'resource contract'):
            observable_project_budget(self.client, 0, ceiling=512)


if __name__ == '__main__':
    unittest.main()
