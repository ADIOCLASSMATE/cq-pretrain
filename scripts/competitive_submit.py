#!/usr/bin/env python3
"""Quick node discovery and competitive job submission.

Reads live Ascend node status, identifies idle nodes, and submits a job with
exclude_nodes targeting free capacity. Latency from read to job creation is
kept under 10 seconds to claim capacity before other submissions arrive.

Usage:
  python3 competitive_submit.py \\
    --job-name cq-train-1234 \\
    --command "bash scripts/my_training.sh" \\
    --nodes 2 \\
    --priority 4 \\
    [--dry-run]

The script uses the same node discovery as watch_ascend_nodes.py but adds
competitive submission: it excludes occupied nodes to steer the job toward
free ones, without waiting for the platform scheduler to pick them randomly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import sys
import time
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc)


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def classify_node(row, gpus_per_node=16):
    """Fail closed on missing allocation fields; never use GPU utilization."""
    gpu = row.get("gpu") or {}
    associated = row.get("tasks_associated")
    tasks = associated.get("tasks") if isinstance(associated, dict) else None
    count = number(associated.get("count")) if isinstance(associated, dict) else None
    result = {
        "name": row.get("name"),
        "state": "unknown",
        "gpu_total": number(gpu.get("total")),
        "gpu_used": number(gpu.get("used")),
        "task_count": count,
    }
    if not result["name"]:
        return result
    if (
        str(row.get("status", "")).upper() != "READY"
        or bool(row.get("cordon_type"))
        or row.get("is_maint") not in (None, False)
        or str(row.get("resource_pool", "")).lower() == "fault"
    ):
        result["state"] = "unschedulable"
        return result
    if result["gpu_total"] != gpus_per_node:
        result["state"] = "unschedulable"
        return result
    used = [number((row.get(k) or {}).get("used")) for k in ("gpu", "cpu", "memory")]
    if (
        count is None or not count.is_integer() or not isinstance(tasks, list)
        or any(v is None for v in used)
        or "cordon_type" not in row or not row.get("resource_pool")
    ):
        result["state"] = "incomplete_data"
        return result
    if count > 0 or tasks or any(v > 0 for v in used):
        result["state"] = "occupied"
    elif number(gpu.get("available")) != gpus_per_node:
        result["state"] = "inconsistent"
    else:
        result["state"] = "idle"
    return result


class NodeReader:
    """Uses the installed client's authentication, pooling and retry handling."""

    def __init__(self, account, workspace, group):
        from inspire import InspireClient
        from inspire.platform.web.browser_api.availability.api import list_compute_groups

        self.client = InspireClient(account=account, timeout=10, operation_timeout=20)
        try:
            self.workspace_id = self.client.workspaces.get(workspace).ref.key
            with self.client._transport.scope(timeout=20):
                groups = list_compute_groups(
                    workspace_id=self.workspace_id, session=self.client._transport.session
                )
            matches = [g for g in groups if g["name"] == group]
            if len(matches) != 1:
                raise ValueError(f"Compute group {group} must match exactly one live name")
            self.group_id = matches[0]["logic_compute_group_id"]
        except BaseException:
            self.client.close()
            raise

    def read(self):
        from inspire.platform.web.browser_api.core import _request_json, _v2_result

        rows = {}
        with self.client._transport.scope(timeout=20):
            for page in range(1, 41):
                result = _v2_result(_request_json(
                    self.client._transport.session,
                    "POST",
                    "/api/v2/workspace?Action=ListNodeDimension",
                    referer=self.client.base_url + "/jobs/distributedTraining",
                    body={
                        "filter": {
                            "workspace_id": self.workspace_id,
                            "logic_compute_group_id": self.group_id,
                        },
                        "PageNumber": page,
                        "page_size": 5000,
                    },
                    timeout=10,
                ))
                items = result.get("node_dimensions")
                total = number(result.get("total"))
                if not isinstance(items, list) or total is None or not total.is_integer():
                    raise ValueError("Incomplete node response")
                before = len(rows)
                for item in items:
                    if not isinstance(item, dict) or not item.get("name"):
                        raise ValueError("Invalid node record")
                    rows[item["name"]] = item
                if len(rows) == total:
                    return list(rows.values())
                if len(rows) > total or len(rows) == before:
                    raise ValueError("Inconsistent pagination")
        raise ValueError("Pagination exceeded limit")

    def close(self):
        self.client.close()


def find_exclusions(reader, needed_nodes, gpus_per_node=16):
    """Read nodes and return exclude_nodes list to target free capacity.

    Returns:
        (idle_nodes, exclude_nodes, state_counts)
    """
    rows = reader.read()
    nodes = {r["name"]: classify_node(r, gpus_per_node) for r in rows}

    idle = sorted(n for n, r in nodes.items() if r["state"] == "idle")
    occupied = sorted(n for n, r in nodes.items() if r["state"] == "occupied")
    unschedulable = sorted(n for n, r in nodes.items() if r["state"] == "unschedulable")

    counts = {
        "idle": len(idle),
        "occupied": len(occupied),
        "unschedulable": len(unschedulable),
        "incomplete_data": sum(1 for r in nodes.values() if r["state"] == "incomplete_data"),
        "inconsistent": sum(1 for r in nodes.values() if r["state"] == "inconsistent"),
        "unknown": sum(1 for r in nodes.values() if r["state"] == "unknown"),
    }

    # Warn if not enough idle nodes, but still submit — platform will queue
    if len(idle) < needed_nodes:
        print(f"⚠ Warning: only {len(idle)} idle nodes, but need {needed_nodes}; "
              "job will queue until they become available", file=sys.stderr)

    # Exclude occupied nodes to steer toward idle ones
    exclude = occupied + unschedulable

    return idle, exclude, counts


def submit_job(args, exclude_nodes):
    """Call inspire job create with the competitive exclude list."""
    from subprocess import run, PIPE

    cmd = [
        "inspire", "job", "create",
        "-n", args.job_name,
        "--workspace", args.workspace,
        "-p", args.project,
        "--group", args.group,
        "-q", args.quota,
        "--nodes", str(args.nodes),
        "-i", args.image,
        "--priority", str(args.priority),
        "--shm-size", str(args.shm_size),
    ]

    if args.datasets:
        for dataset in args.datasets:
            cmd.extend(["--dataset", dataset])

    for node in exclude_nodes:
        cmd.extend(["--exclude-node", node])

    if args.auto_fault_tolerance:
        cmd.append("--auto-fault-tolerance")
        if args.fault_tolerance_max_retry:
            cmd.extend(["--fault-tolerance-max-retry", str(args.fault_tolerance_max_retry)])
        if args.fault_tolerance_retry_interval:
            cmd.extend(["--fault-tolerance-retry-interval", str(args.fault_tolerance_retry_interval)])

    if args.dry_run:
        cmd.append("--dry-run")

    if args.max_time:
        cmd.extend(["--max-time", str(args.max_time)])

    cmd.extend(["-c", args.command])

    # Print what we're about to do
    print(f"Submitting with {len(exclude_nodes)} excluded nodes: {', '.join(exclude_nodes[:3])}",
          end="")
    if len(exclude_nodes) > 3:
        print(f", ... ({len(exclude_nodes)} total)")
    else:
        print()

    if args.dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return 0

    result = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    if result.returncode != 0:
        print(f"❌ Job creation failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    # Parse output to find job name
    for line in result.stdout.split('\n'):
        if 'Job created:' in line or 'name' in line.lower():
            print(f"✓ {line}")

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Node discovery
    parser.add_argument("--account", default="wjx-ascend", help="Inspire account")
    parser.add_argument("--workspace", default="昇腾卡公共空间", help="Workspace name")
    parser.add_argument("--group", default="910B资源", help="Compute group")
    parser.add_argument("--gpus-per-node", type=int, default=16, help="Expected GPU count per node")

    # Job spec
    parser.add_argument("--job-name", required=True, help="Job name")
    parser.add_argument("--command", required=True, help="Job command")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--project", default="公共科研项目", help="Project")
    parser.add_argument("--quota", default="16,180,1800", help="Resource quota")
    parser.add_argument("--priority", type=int, default=4, help="Priority (3=LOW, 4=HIGH)")
    parser.add_argument("--image", default="docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.4")
    parser.add_argument("--shm-size", type=int, default=128, help="Shared memory in GiB")
    parser.add_argument("--max-time", type=int, help="Max time in hours")
    parser.add_argument("--datasets", action="append", help="Datasets to mount (repeatable)")
    parser.add_argument("--auto-fault-tolerance", action="store_true", default=True)
    parser.add_argument("--fault-tolerance-max-retry", type=int)
    parser.add_argument("--fault-tolerance-retry-interval", type=int)

    # Control
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    try:
        reader = None
        start = time.monotonic()

        print(f"[{utc_now():%H:%M:%S}] Reading node status...", file=sys.stderr)
        reader = NodeReader(args.account, args.workspace, args.group)
        idle, exclude, counts = find_exclusions(reader, args.nodes, args.gpus_per_node)
        elapsed_ms = 1000 * (time.monotonic() - start)

        print(f"[{utc_now():%H:%M:%S}] Discovery took {elapsed_ms:.0f} ms: "
              f"{counts['idle']} idle, {counts['occupied']} occupied, "
              f"{counts['unschedulable']} unschedulable, "
              f"{counts['incomplete_data'] + counts['inconsistent'] + counts['unknown']} other",
              file=sys.stderr)

        if args.verbose:
            print(f"Idle nodes: {idle}", file=sys.stderr)
            print(f"Will exclude: {exclude}", file=sys.stderr)

        # Submit quickly
        return submit_job(args, exclude)

    except Exception as error:
        print(f"❌ {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if reader:
            try:
                reader.close()
            except:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
