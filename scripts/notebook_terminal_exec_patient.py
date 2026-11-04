"""Use the existing authorized terminal helper with a longer TLS handshake budget.

The CLI application transport caps its connection/TLS timeout at five seconds.
This local compatibility wrapper changes only that timeout for the official
Notebook gateway; account, live resource and command authorization stay in the
original helper.
"""
from pathlib import Path
import runpy
from urllib.parse import urlsplit

import requests


_send = requests.Session.send


def patient_send(self, request, **kwargs):
    if urlsplit(request.url).hostname == 'notebook-inspire-sj.sii.edu.cn':
        timeout = kwargs.get('timeout')
        if isinstance(timeout, tuple) and len(timeout) == 2:
            kwargs['timeout'] = (max(30, timeout[0] or 0), timeout[1])
    return _send(self, request, **kwargs)


requests.Session.send = patient_send
runpy.run_path('/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/'
               'genund_mount_preflight_20261103_v1/notebook_terminal_exec.py',
               run_name='__main__')
