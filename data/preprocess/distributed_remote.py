"""Bounded authenticated Inspire terminal commands for this eight-CPU run.

Execute using the installed Inspire Python. Shared scripts carry the payload;
no stdin, authentication URLs, cookies or request headers are printed or saved.
"""
import argparse
import json
import re
from inspire.config.workspaces import resolve_workspace_operation_scope
from inspire.platform.web.session import get_web_session
from inspire.platform.web.browser_api.notebooks import list_notebooks
from inspire.services.execution.remote_exec import exec_in_notebook_jupyter


def execute(name, command, timeout=30):
    if not re.fullmatch(r"openvidhd-cpu-20261030-0[1-8]", name):
        raise ValueError("Target is not one of this task's CPU instances")
    session = get_web_session(account="lipengfei")
    workspace = resolve_workspace_operation_scope(workspace="昇腾卡公共空间", session=session)
    user = session.user_detail.get("id") or session.user_detail.get("user_id")
    rows, _ = list_notebooks(workspace, user_ids=[user], keyword=name, session=session)
    rows = [r for r in rows if r["name"] == name]
    if len(rows) != 1 or rows[0]["status"] != "RUNNING":
        raise ValueError("Expected one running CPU instance with this exact name")
    result = exec_in_notebook_jupyter(session=session, notebook_id=rows[0]["notebook_id"],
        command=command, timeout=timeout, max_output_bytes=16000)
    return dict(completed=result.completed, returncode=result.returncode, output=result.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("command")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    result = execute(args.name, args.command, args.timeout)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["completed"] and result["returncode"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"Remote command failed: {type(error).__name__}") from None
