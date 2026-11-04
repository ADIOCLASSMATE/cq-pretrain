"""SDK compatibility preserves single-send Create and ambiguous outcomes."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

from .platform_client import NativeServings, SubmissionUncertainError, workspace_entries, ValidationError


class PlatformClientTests(unittest.TestCase):
    def service(self):
        transport=NS(session=object(),scope=Mock(side_effect=lambda **_:nullcontext()),
                     single_send=Mock(side_effect=lambda *_,**__:nullcontext()))
        service=NativeServings(NS(_transport=transport,operation_timeout=60,_catalog_context=None))
        service.plan=Mock(return_value=NS(name='owned',workspace=NS(ref=NS(key='workspace')),
            create_kwargs={'name':'owned','is_publicpath_readonly':True,'gpu_count':2,'priority':1}))
        service._make_ref=Mock(return_value=object())
        return service,transport

    def test_only_unsupported_field_removed_and_create_sent_once(self):
        service,transport=self.service()
        with patch('corpus_pipeline.serving.platform_client.api.create_serving',return_value={'id':'service'}) as send:
            service.create(object(),operation_id='owned-operation')
        send.assert_called_once_with(name='owned',gpu_count=2,priority=1,session=transport.session)
        transport.single_send.assert_called_once_with('owned-operation',create=True,inspect='servings')
        self.assertIn('is_publicpath_readonly',service.plan.return_value.create_kwargs)

    def test_missing_identifier_never_retries_create(self):
        service,transport=self.service()
        with patch('corpus_pipeline.serving.platform_client.api.create_serving',return_value={}) as send:
            with self.assertRaises(SubmissionUncertainError):service.create(object(),operation_id='owned-operation')
        self.assertEqual(send.call_count,1)
        service._make_ref.assert_not_called()

    def test_workspace_catalog_uses_live_pages_with_scheduling_policy(self):
        session=NS(base_url='https://fixture.invalid')
        rows=[{'id':'ws-00000001-0000-0000-0000-000000000001','name':'first','is_fair_workspace':False},
              {'id':'ws-00000002-0000-0000-0000-000000000002','name':'second','is_fair_workspace':True}]
        with patch('corpus_pipeline.serving.platform_client._request_json',side_effect=[{'items':[r],'total':'2'} for r in rows]) as request, \
             patch('corpus_pipeline.serving.platform_client._v2_result',side_effect=lambda x:x):
            self.assertEqual(workspace_entries(session),rows)
        self.assertEqual([call.kwargs['body']['PageNumber'] for call in request.call_args_list],[1,2])
        self.assertEqual(session.all_workspace_fair_scheduling[rows[1]['id']],True)

    def test_incomplete_workspace_inventory_cannot_authorize_selection(self):
        with patch('corpus_pipeline.serving.platform_client._request_json',return_value={'items':[],'total':2}), \
             patch('corpus_pipeline.serving.platform_client._v2_result',side_effect=lambda x:x):
            with self.assertRaisesRegex(ValidationError,'pagination'):
                workspace_entries(NS(base_url='https://fixture.invalid'))


if __name__=='__main__':unittest.main()
