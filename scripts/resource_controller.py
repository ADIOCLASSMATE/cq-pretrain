#!/usr/bin/env python3
"""Intelligent resource allocation controller for agents.

Agents provide explicit job_type and required parameters; controller handles:
  - Priority auto-decision (based on job_type + cluster state)
  - Competitive node submission (discover + exclude occupied)
  - Fault tolerance and infrastructure lessons

Agent responsibility: provide clear job_type (training/eval/batch)
Controller responsibility: everything else (discover, exclude, switch HIGH/LOW, fault recovery)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc)


def get_cluster_config(cluster_name="ascend"):
    """Get cluster-specific configuration.

    NOTE: Customize workspace, group, image, project to your environment.
    These defaults are examples; always verify against 'inspire resources availability'.
    """
    configs = {
        "ascend": {
            "workspace": "YOUR_WORKSPACE",  # e.g. 昇腾卡公共空间
            "group": "YOUR_COMPUTE_GROUP",  # e.g. 910B资源
            "quota": "QUOTA_SPEC",  # e.g. 16,180,1800
            "image": "YOUR_IMAGE_URI",  # e.g. docker-t.sii.shaipower.online/studio/dev-wjx:v-1.4
            "gpus_per_node": 16,
            "nodes_per_card_hour": 1,
            "points_per_node_hour": 5.6,
            "high_priority": 4,
            "low_priority": 3,
            "fixed_exclusions": [],  # Add problematic node hostnames here
        },
        "l40s": {
            "workspace": "YOUR_WORKSPACE",
            "group": "YOUR_COMPUTE_GROUP",
            "quota": "QUOTA_SPEC",
            "image": "YOUR_IMAGE_URI",
            "gpus_per_node": 8,
            "nodes_per_card_hour": 2,
            "points_per_node_hour": 2.64,
            "high_priority": 4,
            "low_priority": 3,
            "fixed_exclusions": [],
        },
    }
    return configs.get(cluster_name, configs["ascend"])


def get_occupied_nodes(workspace, cluster_name="ascend"):
    """Query occupied nodes via inspire CLI."""
    config = get_cluster_config(cluster_name)

    try:
        cmd = ["inspire", "--json", "job", "list", "-n", "100", "--workspace", workspace]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode != 0:
            # Fallback: just use fixed exclusions if query fails
            return config["fixed_exclusions"], {"status": "query_failed"}

        jobs_data = json.loads(result.stdout)
        occupied = set()

        for job in jobs_data.get("data", {}).get("items", []):
            if job.get("status") == "RUNNING":
                nodes = job.get("node_placement", [])
                if isinstance(nodes, list):
                    occupied.update(n for n in nodes if isinstance(n, str))
                elif isinstance(nodes, str):
                    occupied.add(nodes)

        exclude = sorted(occupied | set(config["fixed_exclusions"]))
        return exclude, {"occupied": len(occupied), "fixed": len(config["fixed_exclusions"]), "status": "ok"}

    except Exception as e:
        # Degrade gracefully: use at least fixed exclusions
        return config["fixed_exclusions"], {"status": "error", "error": str(e)[:100]}


def decide_priority(job_type, cluster_state, user_priority=None):
    """Auto-decide priority based on job type and cluster state.

    Args:
        job_type: "training" | "eval" | "batch"
        cluster_state: {"free_nodes": int, "occupied_nodes": int, ...}
        user_priority: User override (3=LOW, 4=HIGH) or None for auto

    Returns:
        priority_level: 3 (LOW) or 4 (HIGH)
        reason: str explaining the decision
    """

    if user_priority is not None:
        return user_priority, f"user override: {user_priority}"

    if job_type == "training":
        # Training: try LOW if nodes are free, otherwise HIGH
        # This is the primary use case for switching
        if cluster_state.get("free_nodes", 0) > 0:
            return 3, "training + free nodes available → LOW"
        else:
            return 4, "training + cluster full → HIGH (burn points, no preemption)"

    elif job_type == "eval":
        # Eval: prefer LOW (free but preemptible)
        # Eval can resume from checkpoints, so preemption is acceptable
        return 3, "eval → LOW (preemptible, can resume)"

    elif job_type == "batch":
        # Batch jobs: LOW by default, switch to HIGH if urgent
        return 3, "batch → LOW (preemptible by default)"

    else:
        # Unknown: default to LOW
        return 3, "unknown job_type → LOW (safe default)"


def build_exclude_list(cluster_name="ascend"):
    """Get the exclude list for competitive submission."""
    config = get_cluster_config(cluster_name)
    exclude, state = get_occupied_nodes(config["workspace"], cluster_name)
    return exclude, state, config


def submit_job(
    job_name: str,
    command: str,
    job_type: str = "batch",
    required_cards: int = 16,
    cluster_name: str = "ascend",
    priority: int | None = None,
    project: str | None = None,
    workspace: str | None = None,
    datasets: list[str] | None = None,
    max_time: int | None = None,
    auto_fault_tolerance: bool = True,
    fault_tolerance_max_retry: int | None = None,
    fault_tolerance_retry_interval: int | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Submit a job with intelligent resource allocation.

    Args:
        job_name: Job name (required)
        command: Job command (required)
        job_type: "training" | "eval" | "batch" (default: batch)
        required_cards: Total GPU count needed (default: 16)
        cluster_name: "ascend" | "l40s" (default: ascend)
        priority: 3 (LOW) | 4 (HIGH) | None (auto-decide)
        project: Project name | None (use default)
        workspace: Workspace override | None (use cluster default)
        datasets: List of "name:version" strings
        max_time: Max job time in hours
        auto_fault_tolerance: Enable fault tolerance (default: True)
        fault_tolerance_max_retry: Max retries (default: 100 for training, 20 for eval)
        fault_tolerance_retry_interval: Retry interval in seconds (default: 60)
        dry_run: Show what would be submitted
        verbose: Print debug info

    Returns:
        0 on success, 1 on failure
    """

    config = get_cluster_config(cluster_name)
    workspace = workspace or config["workspace"]
    project = project or "DEFAULT_PROJECT"  # customize to your project

    # Calculate nodes needed
    gpus_per_node = config["gpus_per_node"]
    nodes_needed = (required_cards + gpus_per_node - 1) // gpus_per_node

    # Get cluster state
    start = time.monotonic()
    exclude, state, _ = build_exclude_list(cluster_name)
    discovery_ms = 1000 * (time.monotonic() - start)

    if verbose:
        print(f"[Discovery {discovery_ms:.0f}ms] {state}", file=sys.stderr)

    # Decide priority
    cluster_state = {"free_nodes": 0}  # TODO: infer from exclude list
    priority, priority_reason = decide_priority(job_type, cluster_state, priority)

    if verbose:
        print(f"[Priority] {priority_reason}", file=sys.stderr)

    # Set fault tolerance defaults based on job type
    if auto_fault_tolerance:
        if fault_tolerance_max_retry is None:
            fault_tolerance_max_retry = 100 if job_type == "training" else 20
        if fault_tolerance_retry_interval is None:
            fault_tolerance_retry_interval = 60

    # Build inspire job create command
    cmd = [
        "inspire", "job", "create",
        "-n", job_name,
        "--workspace", workspace,
        "-p", project,
        "--group", config["group"],
        "-q", config["quota"],
        "--nodes", str(nodes_needed),
        "-i", config["image"],
        "--priority", str(priority),
        "--shm-size", "128",
    ]

    if datasets:
        for ds in datasets:
            cmd.extend(["--dataset", ds])

    # Add exclusions
    for node in exclude:
        cmd.extend(["--exclude-node", node])

    # Fault tolerance
    if auto_fault_tolerance:
        cmd.append("--auto-fault-tolerance")
        cmd.extend(["--fault-tolerance-max-retry", str(fault_tolerance_max_retry)])
        cmd.extend(["--fault-tolerance-retry-interval", str(fault_tolerance_retry_interval)])

    if max_time:
        cmd.extend(["--max-time", str(max_time)])

    cmd.extend(["-c", command])

    # Submit or dry-run
    exclude_str = ", ".join(exclude[:2]) + ("..." if len(exclude) > 2 else "")
    priority_name = "LOW" if priority == 3 else "HIGH"

    print(f"[{utc_now():%H:%M:%S}] Submitting {job_name} ({job_type}, {nodes_needed}×{gpus_per_node}GPU, {priority_name})",
          file=sys.stderr)
    print(f"  Excluding {len(exclude)} nodes: {exclude_str}", file=sys.stderr)

    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}", file=sys.stderr)
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"❌ Job creation failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    # Success
    for line in result.stdout.split('\n'):
        if 'Job created:' in line or 'name' in line.lower():
            print(f"✓ {line}", file=sys.stderr)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required
    parser.add_argument("--job-name", required=True, help="Job name")
    parser.add_argument("--command", required=True, help="Job command")

    # Job classification (auto-detects resource needs)
    parser.add_argument(
        "--job-type",
        choices=["training", "eval", "batch"],
        default="batch",
        help="Job type (auto-decides priority)"
    )
    parser.add_argument(
        "--required-cards",
        type=int,
        default=16,
        help="Total GPU cards needed"
    )

    # Cluster / resource
    parser.add_argument(
        "--cluster",
        choices=["ascend", "l40s"],
        default="ascend",
        help="Target cluster"
    )
    parser.add_argument("--workspace", help="Workspace override")
    parser.add_argument("--project", help="Project override")
    parser.add_argument("--quota", help="Quota override (format: gpu,cpu,mem)")

    # Priority and control
    parser.add_argument(
        "--priority",
        type=int,
        choices=[3, 4],
        help="3=LOW (preemptible), 4=HIGH (paid). Auto-decided if omitted."
    )
    parser.add_argument(
        "--max-time",
        type=int,
        help="Max job time in hours"
    )

    # Data and fault tolerance
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to mount (name:version). Repeatable."
    )
    parser.add_argument(
        "--no-fault-tolerance",
        action="store_true",
        help="Disable automatic fault tolerance"
    )
    parser.add_argument(
        "--fault-tolerance-max-retry",
        type=int,
        help="Max retries (default: 100 for training, 20 for eval)"
    )
    parser.add_argument(
        "--fault-tolerance-retry-interval",
        type=int,
        help="Retry interval in seconds (default: 60)"
    )

    # Control
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    return submit_job(
        job_name=args.job_name,
        command=args.command,
        job_type=args.job_type,
        required_cards=args.required_cards,
        cluster_name=args.cluster,
        priority=args.priority,
        project=args.project,
        workspace=args.workspace,
        datasets=args.datasets,
        max_time=args.max_time,
        auto_fault_tolerance=not args.no_fault_tolerance,
        fault_tolerance_max_retry=args.fault_tolerance_max_retry,
        fault_tolerance_retry_interval=args.fault_tolerance_retry_interval,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
