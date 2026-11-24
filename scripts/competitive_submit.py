#!/usr/bin/env python3
"""Quick node discovery and competitive job submission.

Discovers currently occupied Ascend nodes via inspire CLI, then submits a job
with exclude_nodes targeting free capacity. Latency from discovery to submission
is kept under 10 seconds to claim capacity before other submissions arrive.

Usage:
  python3 competitive_submit.py \\
    --job-name cq-train-1234 \\
    --command "bash scripts/my_training.sh" \\
    --nodes 2 \\
    --priority 4 \\
    [--dry-run]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time


def utc_now():
    return datetime.now(timezone.utc)


def find_excluded_nodes_via_cli(workspace, group):
    """Query inspire CLI for occupied/unschedulable nodes; build exclude list.

    Returns:
        (exclude_nodes_list, state_summary)
    """
    # Call: inspire job list to get running jobs and their node placement
    cmd = ["inspire", "--json", "job", "list", "-n", "100", "--workspace", workspace]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

    if result.returncode != 0:
        raise RuntimeError(f"inspire job list failed: {result.stderr}")

    try:
        jobs_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ValueError("Could not parse job list JSON")

    # Collect nodes that have running jobs
    occupied = set()
    for job in jobs_data.get("data", {}).get("items", []):
        if job.get("status") == "RUNNING":
            nodes = job.get("node_placement", [])
            if isinstance(nodes, list):
                occupied.update(n for n in nodes if isinstance(n, str))
            elif isinstance(nodes, str):
                occupied.add(nodes)

    # Also add known fixed exclusions (node health, maintenance, etc)
    fixed_exclusions = {"infra-gpu-npu-248.host.shzhisuan.com"}

    exclude = sorted(occupied | fixed_exclusions)

    summary = {
        "occupied_nodes": len(occupied),
        "fixed_exclusions": len(fixed_exclusions),
        "total_exclude": len(exclude),
    }

    return exclude, summary


def submit_job(args, exclude_nodes):
    """Call inspire job create with the competitive exclude list."""

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

    if args.max_time:
        cmd.extend(["--max-time", str(args.max_time)])

    cmd.extend(["-c", args.command])

    # Print what we're about to do
    exclude_str = ", ".join(exclude_nodes[:3])
    if len(exclude_nodes) > 3:
        exclude_str += f", ... ({len(exclude_nodes)} total)"
    print(f"Excluding {len(exclude_nodes)} occupied/unschedulable nodes: {exclude_str}",
          file=sys.stderr)

    if args.dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}", file=sys.stderr)
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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

    # Job spec
    parser.add_argument("--job-name", required=True, help="Job name")
    parser.add_argument("--command", required=True, help="Job command")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--workspace", default="昇腾卡公共空间", help="Workspace name")
    parser.add_argument("--project", default="公共科研项目", help="Project")
    parser.add_argument("--group", default="910B资源", help="Compute group")
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
        start = time.monotonic()

        print(f"[{utc_now():%H:%M:%S}] Discovering node occupancy...", file=sys.stderr)
        exclude, summary = find_excluded_nodes_via_cli(args.workspace, args.group)
        elapsed_ms = 1000 * (time.monotonic() - start)

        print(f"[{utc_now():%H:%M:%S}] Discovery took {elapsed_ms:.0f} ms: "
              f"excluding {summary['occupied_nodes']} occupied + "
              f"{summary['fixed_exclusions']} fixed = {summary['total_exclude']} total",
              file=sys.stderr)

        if args.verbose:
            print(f"Exclude list: {exclude}", file=sys.stderr)

        # Submit quickly
        return submit_job(args, exclude)

    except Exception as error:
        print(f"❌ {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
