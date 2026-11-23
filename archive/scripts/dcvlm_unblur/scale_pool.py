"""Expand this repair's CPU pool to 256 explicitly named instances."""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from inspire.accounts import account_scope
from inspire.config.workspaces import resolve_workspace_operation_scope
from inspire.platform.web.session import get_web_session
from inspire.cli.commands.notebook.notebook_lookup import _list_notebooks_for_workspace, _try_get_current_user_ids

RUN = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/_repairs/dcvlm_datacomp_unblur_20261104')
for key in list(os.environ):
    if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
        del os.environ[key]

with account_scope('wjx-ascend'):
    session = get_web_session(account='wjx-ascend')
    workspace = resolve_workspace_operation_scope(workspace='昇腾卡公共空间', session=session)
    def listing():
        return _list_notebooks_for_workspace(session=session, workspace_id=workspace,
            user_ids=_try_get_current_user_ids(session, base_url=session.base_url), keyword='dcvlm-unblur-cpu-1104-')
    existing = {r['name'] for r in listing()}
    def create(i):
        name = f'dcvlm-unblur-cpu-1104-{i:02d}'
        if name in existing:
            return
        result = subprocess.run(['inspire', '--account', 'wjx-ascend', '--json', 'notebook', 'create',
            '--name', name, '--workspace', '昇腾卡公共空间', '--project', '多模态大模型新架构评测探索与scaling-law',
            '--group', '910B资源', '--quota', '0,8,128', '--image', 'dev-wjx-ascend:v-1.4',
            '--shm-size', '32', '--priority', '10', '--no-auto-stop', '--no-enable-notification', '--no-wait'],
            capture_output=True, text=True, timeout=90)
        (RUN / (name + '.create.json')).write_text(result.stdout or result.stderr)
        try:
            success = json.loads(result.stdout).get('success')
        except json.JSONDecodeError:
            success = False
        print(name, result.returncode, success, flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(create, range(32, 256)))
    rows = [dict(name=r['name'], notebook_id=r.get('notebook_id') or r['id'],
        workspace_id=workspace, status=r['status']) for r in listing()
        if r['name'].startswith('dcvlm-unblur-cpu-1104-') and int(r['name'].rsplit('-', 1)[-1]) < 256
        and not r['name'].endswith('-112')]
    assert len({r['name'] for r in rows}) == len(rows)
    temp = RUN / 'workers.json.tmp'
    temp.write_text(json.dumps(rows, indent=2))
    os.replace(temp, RUN / 'workers.json')
    print('BOUND', len(rows), flush=True)
