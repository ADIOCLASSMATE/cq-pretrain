"""Run-owned CLI-context terminal compatibility; exact live ownership is mandatory."""
from __future__ import annotations
import hashlib,logging,os,shlex,time
from contextlib import contextmanager
from requests.adapters import HTTPAdapter
from inspire.accounts import account_scope
from inspire.platform.web.session import get_web_session
from inspire.platform.web.transport import Transport
from inspire.platform.web.browser_api import get_notebook_detail,run_command_capture_in_notebook
from inspire.cli.commands.notebook.notebook_lookup import _get_current_user_detail,_validate_notebook_account_access
from corpus_pipeline.serving.platform_client import workspace_entries


class GatewayAdapter(HTTPAdapter):
    def send(self,request,**kwargs):
        timeout=kwargs.get('timeout')
        if isinstance(timeout,tuple) and len(timeout)==2:
            kwargs['timeout']=(max(30,timeout[0] or 0),timeout[1])
        return super().send(request,**kwargs)


class NotebookTransport(Transport):
    @contextmanager
    def application_connection(self,url):
        with super().application_connection(url) as connection:
            connection.http.mount('https://notebook-inspire-sj.sii.edu.cn/',GatewayAdapter())
            yield connection

def execute(command,*,notebook,account,workspace,notebook_id=None,workspace_id=None,timeout=40,max_output_bytes=8192):
    if not notebook_id or not workspace_id:raise ValueError('Exact accepted resource and workspace IDs required')
    if not command.strip() or not 1<=timeout<=1800:raise ValueError('Invalid command or timeout')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
    logging.getLogger('inspire').setLevel(logging.CRITICAL)
    start=time.time();receipt={'transport':'authenticated_cli_context_jupyter_terminal','notebook':notebook,
        'account':account,'workspace':workspace,'notebook_id':notebook_id,'started_at':start,'command_sha256':hashlib.sha256(command.encode()).hexdigest()}
    try:
        with account_scope(account):
            session=get_web_session(account=account,require_workspace=False)
            matches=[row for row in workspace_entries(session) if row['name']==workspace]
            if len(matches)!=1 or matches[0]['id']!=workspace_id:raise ValueError('Workspace binding differs')
            detail=get_notebook_detail(notebook_id=notebook_id,session=session)
            if (detail.get('name')!=notebook or (detail.get('notebook_id')or detail.get('id'))!=notebook_id
                    or detail.get('status')!='RUNNING'):
                raise ValueError('Exact owned RUNNING Notebook differs')
            allowed,_=_validate_notebook_account_access(current_user=_get_current_user_detail(session,base_url=session.base_url),notebook_detail=detail)
            if not allowed:raise ValueError('Notebook account access denied')
            wrapped='timeout --signal=TERM --kill-after=10s '+str(timeout)+'s bash -c '+shlex.quote(command)
            transport=NotebookTransport(account,session.base_url,username='',allow_browser=False,cli_compat=True)
            transport.adopt_session(session)
            try:
                with transport.scope(timeout=timeout+60):
                    result=run_command_capture_in_notebook(notebook_id=notebook_id,command=wrapped,session=session,
                        timeout=timeout+25,max_output_bytes=max_output_bytes)
            finally:transport.close()
            receipt.update(completed=result.completed,returncode=result.returncode,output=result.output,
                truncated=result.truncated,total_output_bytes=result.total_output_bytes)
    except Exception as error:
        receipt.update(completed=False,returncode=125,error_type=type(error).__name__)
    receipt.update(ended_at=time.time(),elapsed_seconds=time.time()-start)
    return receipt
