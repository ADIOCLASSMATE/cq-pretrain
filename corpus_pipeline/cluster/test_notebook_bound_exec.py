"""Gateway timeout stays local to the authorized Notebook connection."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest
import requests
from . import notebook_bound_exec as bound


class NotebookTransportTests(unittest.TestCase):
    def test_gateway_adapter_keeps_read_timeout_and_does_not_retry(self):
        request=requests.Request('GET','https://notebook-inspire-sj.sii.edu.cn/fixture').prepare()
        with patch.object(requests.adapters.HTTPAdapter,'send',return_value=object()) as send:
            bound.GatewayAdapter().send(request,timeout=(5,90),verify=True)
        send.assert_called_once_with(request,timeout=(30,90),verify=True)

    def test_only_gateway_mount_uses_extended_connect_timeout(self):
        http=requests.Session()
        @contextmanager
        def connection(*_):yield NS(http=http)
        transport=bound.NotebookTransport.__new__(bound.NotebookTransport)
        with patch.object(bound.Transport,'application_connection',connection):
            with transport.application_connection('https://notebook-inspire-sj.sii.edu.cn/fixture') as session:
                self.assertIsInstance(session.http.get_adapter('https://notebook-inspire-sj.sii.edu.cn/'),bound.GatewayAdapter)
                self.assertNotIsInstance(session.http.get_adapter('https://elsewhere.invalid/'),bound.GatewayAdapter)
        http.close()

    def test_wrong_workspace_never_opens_terminal(self):
        with patch.object(bound,'account_scope',return_value=nullcontext()), \
             patch.object(bound,'get_web_session',return_value=NS()), \
             patch.object(bound,'workspace_entries',return_value=[{'name':'expected','id':'other'}]), \
             patch.object(bound,'run_command_capture_in_notebook') as command:
            result=bound.execute('true',notebook='owned',account='fixture',workspace='expected',
                                 notebook_id='accepted',workspace_id='accepted-ws')
        self.assertFalse(result['completed']);command.assert_not_called()


if __name__=='__main__':unittest.main()
