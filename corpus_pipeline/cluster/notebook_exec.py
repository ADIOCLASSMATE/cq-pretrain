"""Bounded authenticated JupyterTerminal execution; no SSH/bootstrap changes.

Run with the Inspire SDK Python interpreter. The platform access URL, cookies
and authentication state remain inside the SDK and are never included in the
receipt. The requested command's stdout is included, so do not print secrets.
The SDK creates one temporary Jupyter terminal and deletes it in its finally
block. A remote GNU timeout also bounds the command if the socket disconnects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import time


PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
              "https_proxy", "all_proxy")


def execute(command: str, *, notebook="dev-wjx-ascend", account="wjx-ascend",
            workspace="昇腾卡公共空间", timeout=60, max_output_bytes=65536,
            notebook_id=None, workspace_id=None):
    if not command.strip() or not 1 <= timeout <= 1800:
        raise ValueError("Nonempty command and timeout in [1, 1800] required")
    if not 1 <= max_output_bytes <= 8 * 1024 * 1024:
        raise ValueError("Output limit must be within 1 byte and 8 MiB")
    if bool(notebook_id) != bool(workspace_id):
        raise ValueError("Exact notebook and workspace IDs must be provided together")
    if notebook_id:
        # The current SDK generic context can lose the Notebook account scope.
        # Use the CLI's authenticated context with the same exact ownership
        # checks, never SSH or an unauthenticated terminal fallback.
        from .notebook_bound_exec import execute as execute_bound
        return execute_bound(command, notebook=notebook, account=account, workspace=workspace,
            notebook_id=notebook_id, workspace_id=workspace_id, timeout=timeout,
            max_output_bytes=max_output_bytes)
    for key in PROXY_KEYS:
        os.environ.pop(key, None)
    # Avoid SDK diagnostics containing access URLs or underlying HTTP details.
    logging.getLogger("inspire").setLevel(logging.CRITICAL)
    from corpus_pipeline.serving.platform_client import InspireClient
    from inspire.platform.web.browser_api.jupyter_terminal import run_command_capture_in_notebook

    start = time.time()
    receipt = {
        "schema_version": "authenticated_jupyter_exec.v1",
        "transport": "inspire_sdk_jupyter_terminal",
        "notebook": notebook, "account": account, "workspace": workspace,
        "started_at": start, "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "remote_timeout_seconds": timeout, "proxies_removed": list(PROXY_KEYS),
        "creates_compute_resource": False,
    }
    try:
        with InspireClient(account=account) as client:
            if notebook_id:
                from inspire.sdk.models_notebooks import NotebookRef
                ref = NotebookRef(name=notebook, account=account,
                    base_url=client.base_url, key=notebook_id, workspace_id=workspace_id)
                target = client.notebooks.get(ref, workspace=workspace)
                if target.ref.key != notebook_id or target.name != notebook:
                    raise RuntimeError("Notebook identity differs from the accepted receipt")
            else:
                target = client.notebooks.get(notebook, workspace=workspace)
            receipt.update(notebook_id=target.ref.key, notebook_status=target.status,
                           node=target.node.get("name"))
            if target.status != "RUNNING":
                raise RuntimeError("Notebook must already be RUNNING")
            # Target script cannot consume the SDK's terminal status marker.
            wrapped = "timeout --signal=TERM --kill-after=10s " + str(timeout) + "s bash -c " + shlex.quote(command)
            with client._transport.scope(timeout=timeout + 45):
                result = run_command_capture_in_notebook(
                    notebook_id=target.ref.key, command=wrapped,
                    session=client._transport.session, timeout=timeout + 25,
                    max_output_bytes=max_output_bytes,
                )
            receipt.update(completed=result.completed, returncode=result.returncode,
                           output=result.output, truncated=result.truncated,
                           total_output_bytes=result.total_output_bytes)
    except Exception as exc:
        # Exception messages may contain authenticated URLs. Never persist them.
        receipt.update(completed=False, returncode=125, error_type=type(exc).__name__)
    receipt.update(ended_at=time.time(), elapsed_seconds=time.time() - start)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--command")
    commands.add_argument("--command-file", type=Path)
    parser.add_argument("--notebook", default="dev-wjx-ascend")
    parser.add_argument("--account", default="wjx-ascend")
    parser.add_argument("--workspace", default="昇腾卡公共空间")
    parser.add_argument("--notebook-id")
    parser.add_argument("--workspace-id")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-output-bytes", type=int, default=65536)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Refusing to overwrite an existing receipt")
    command = args.command if args.command is not None else args.command_file.read_text()
    receipt = execute(command, notebook=args.notebook, account=args.account,
                      workspace=args.workspace, timeout=args.timeout,
                      max_output_bytes=args.max_output_bytes,
                      notebook_id=args.notebook_id, workspace_id=args.workspace_id)
    rendered = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return receipt["returncode"] if receipt["completed"] else 125


if __name__ == "__main__":
    raise SystemExit(main())
