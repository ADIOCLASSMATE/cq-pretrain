"""Use authenticated platform terminals when the installed SSH kit is unavailable."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

RUN = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/_repairs/dcvlm_datacomp_unblur_20261104')
REPO = Path(__file__).resolve().parents[2]
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    os.environ.pop(key, None)


def one(index):
    from inspire.accounts import account_scope
    from inspire.config.workspaces import resolve_workspace_operation_scope
    from inspire.platform.web.session import get_web_session
    from inspire.platform.web.browser_api import get_notebook_detail, run_command_capture_in_notebook
    from inspire.cli.commands.notebook.notebook_lookup import _get_current_user_detail, _validate_notebook_account_access
    worker = next(r for r in json.loads((RUN / 'workers.json').read_text()) if r['name'].endswith(f'-{index:02d}'))
    with account_scope('wjx-ascend'):
        session = get_web_session(account='wjx-ascend')
        workspace = resolve_workspace_operation_scope(workspace='昇腾卡公共空间', session=session)
        assert workspace == worker['workspace_id']
        detail = get_notebook_detail(notebook_id=worker['notebook_id'], session=session)
        assert detail['name'] == worker['name']
        assert _validate_notebook_account_access(current_user=_get_current_user_detail(session, base_url=session.base_url), notebook_detail=detail)[0]
        if detail['status'] == 'STOPPED':
            result = subprocess.run(['inspire', '--account', 'wjx-ascend', '--json', 'notebook', 'start',
                worker['name'], '--workspace', '昇腾卡公共空间', '--no-wait'], capture_output=True, timeout=60)
            print(json.dumps(dict(worker=worker['name'], restart_requested=result.returncode == 0)), flush=True)
            return False
        assert detail['status'] == 'RUNNING', detail['status']
        if detail.get('node', {}).get('name') == 'infra-gpu-npu-248.host.shzhisuan.com':
            result = subprocess.run(['inspire', '--account', 'wjx-ascend', '--json', 'notebook', 'stop',
                worker['name'], '--workspace', '昇腾卡公共空间'], capture_output=True, text=True, timeout=60)
            assert json.loads(result.stdout).get('success'), 'Failed to stop excluded node'
            (RUN / f'worker_{index:02d}.disabled.json').write_text(json.dumps(dict(worker=worker,
                reason='excluded faulty node 248', stopped=True)))
            return True
        command = f'{REPO}/.venv/bin/python {REPO}/scripts/dcvlm_unblur/launch.py {index}'
        result = run_command_capture_in_notebook(notebook_id=worker['notebook_id'], command=command,
                    session=session, timeout=45, max_output_bytes=4000)
        record = dict(worker=worker['name'], completed=result.completed, returncode=result.returncode, output=result.output,
                      node=detail.get('node', {}).get('name'))
        (RUN / f'worker_{index:02d}.terminal.json').write_text(json.dumps(record))
        print(json.dumps(record), flush=True)
        return result.completed and result.returncode == 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            ok = one(int(sys.argv[1]))
        except Exception as error:
            print(json.dumps(dict(worker=sys.argv[1], error_type=type(error).__name__, reason=str(error)[:150])), flush=True)
            ok = False
        raise SystemExit(0 if ok else 1)
    def start(index):
        if (RUN / f'worker_{index:02d}.disabled.json').exists():
            return
        heartbeat = RUN / f'worker_{index:02d}.heartbeat.json'
        if heartbeat.exists():
            state = json.loads(heartbeat.read_text())
            if time.time() - state['updated'] < 90 and state['code_sha256'] == hashlib.sha256(Path(__file__).with_name('worker.py').read_bytes()).hexdigest():
                return
        for attempt in range(3):
            try:
                result = subprocess.run([sys.executable, __file__, str(index)], timeout=100)
            except subprocess.TimeoutExpired:
                continue
            if result.returncode == 0:
                return
            time.sleep(5)
    with ThreadPoolExecutor(max_workers=16) as pool:
        indices = [int(w['name'].rsplit('-', 1)[-1]) for w in json.loads((RUN / 'workers.json').read_text())]
        list(pool.map(start, sorted(indices)))
