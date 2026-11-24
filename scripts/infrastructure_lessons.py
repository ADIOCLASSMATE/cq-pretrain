#!/usr/bin/env python3
"""Infrastructure lessons learned and anti-patterns to avoid.

这个模块记录在昇腾/L40S 集群上踩过的坑和对应的修复策略。
Agent 决策逻辑会引用这些规则。
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Callable


@dataclass
class InfrastructureLesson:
    """A lesson learned from past infrastructure issues."""

    id: str  # unique identifier
    problem: str  # what went wrong
    impact: str  # what it cost (time, points, lost work)
    detection: str  # how to detect it's happening
    fix: str  # how to prevent/recover
    agent_decision: Callable[[dict], bool]  # when this rule applies


LESSONS = [
    InfrastructureLesson(
        id="validation-canary-killed-training",
        problem="Validation suite had frozen snapshots with gitignored test images; validation aborted training every 500 steps",
        impact="Every epoch: 500-step cycle, 450 steps wasted because validation crashed at 450",
        detection="training log shows validation.jsonl every 500 steps but no validation completion, then JOB_FAILED",
        fix="Validation suite now reads images from live dataset, not frozen snapshots. No more abort.",
        agent_decision=lambda ctx: ctx.get("job_type") == "training" and ctx.get("step_interval", 500) % 500 == 0,
    ),

    InfrastructureLesson(
        id="corpus-relocation-hidden-siblings",
        problem="release/ was claimed to be self-contained, but asset paths were buried in resume contract; moving corpus broke training resume",
        impact="Lost 72 hours of training work trying to figure out why checkpoints wouldn't load",
        detection="training fails with 'asset path not found' or 'schema mismatch' on resume from checkpoint",
        fix="All asset paths now explicit in manifest.json; corpus move requires migrate script",
        agent_decision=lambda ctx: ctx.get("checkpoint_resume", False),
    ),

    InfrastructureLesson(
        id="ascend-whole-node-allocation",
        problem="Ascend quota has no 1-card row; every job allocates a whole 16-card node. '1卡低优' was not expressible.",
        impact="Small jobs became HARDER to schedule (need whole free node), not cheaper",
        detection="job queues forever; cluster shows 0 free_nodes even with hundreds of GPUs idle",
        fix="Request whole nodes (16 or 32); use exclude_nodes to steer toward free ones",
        agent_decision=lambda ctx: ctx.get("cluster") == "ascend" and ctx.get("required_cards", 16) < 16,
    ),

    InfrastructureLesson(
        id="preemption-without-checkpoint-is-total-loss",
        problem="LOW priority job runs 10 hours, then gets preempted without saving. Lost work is total.",
        impact="Wasted 10 hours on L40S, burned ~200 points on HIGH to finish the work",
        detection="job status transitions from RUNNING to QUEUING (preempted); no recent checkpoint in run/checkpoints/",
        fix="Always enable --auto-fault-tolerance; checkpoint frequently (every 50-100 steps); resume always reads from checkpoints",
        agent_decision=lambda ctx: ctx.get("job_type") == "training" and not ctx.get("auto_fault_tolerance", True),
    ),

    InfrastructureLesson(
        id="priority-switching-needs-checkpoint-safety",
        problem="Switching from HIGH to LOW mid-epoch: job saves a checkpoint, but platform kills it before that checkpoint fully syncs to disk",
        impact="Checkpoint corruption; can't resume; job redoes work from earlier checkpoint",
        detection="resume from checkpoint fails with 'corrupted' or checksum mismatch",
        fix="Before priority switch: bank one full checkpoint; wait for filesystem sync; THEN switch to LOW",
        agent_decision=lambda ctx: ctx.get("job_type") == "training" and ctx.get("priority_switch", False),
    ),

    InfrastructureLesson(
        id="node-exclusion-must-be-fresh",
        problem="Exclude list was stale; job tried to avoid nodes that just freed up; lost opportunity to schedule immediately",
        impact="Job queued 2 hours waiting for a node that was available after 10 minutes",
        detection="job status stays QUEUING; cluster has free_nodes; job's exclude list doesn't match current occupancy",
        fix="Discover excluded nodes every submission; don't cache exclude list; <1s latency required",
        agent_decision=lambda ctx: True,  # always relevant
    ),

    InfrastructureLesson(
        id="low-priority-needs-quick-resubmit",
        problem="LOW job gets preempted, but controller doesn't resubmit for 5 minutes (waiting for poll interval)",
        impact="Cluster has free nodes for 3 minutes, but job is queued instead of RUNNING",
        detection="job in QUEUING after preemption; cluster has free_nodes",
        fix="Resubmit LOW jobs immediately on preemption detection; don't wait for next poll cycle",
        agent_decision=lambda ctx: ctx.get("job_type") == "training" and ctx.get("priority", 4) == 3,
    ),

    InfrastructureLesson(
        id="training-on-fully-loaded-cluster-pay-high",
        problem="Training submitted to LOW when cluster is 100% full; job queues indefinitely; wastes time",
        impact="Epoch stalled for 24 hours waiting for capacity that never came",
        detection="LOW job in QUEUING status; every node occupied; no preemption happening",
        fix="If cluster is 100% full (0 free nodes), submit training to HIGH instead of LOW. Burn points, don't stall.",
        agent_decision=lambda ctx: ctx.get("job_type") == "training" and ctx.get("free_nodes_pct", 0) == 0,
    ),
]


def get_applicable_lessons(context: dict) -> list[InfrastructureLesson]:
    """Get lessons that apply to the current agent decision context."""
    applicable = []
    for lesson in LESSONS:
        try:
            if lesson.agent_decision(context):
                applicable.append(lesson)
        except Exception:
            pass  # Skip lessons with broken predicates
    return applicable


def format_decision_context(context: dict) -> str:
    """Format the context for a human-readable decision log."""
    return "\n".join(
        f"  {k}: {v}"
        for k, v in sorted(context.items())
        if k not in ("agent_decision",)
    )


def print_lessons(context: dict):
    """Print applicable lessons for debugging/transparency."""
    applicable = get_applicable_lessons(context)
    if not applicable:
        return

    print("# Infrastructure context:", file=sys.stderr)
    for lesson in applicable:
        print(f"  [{lesson.id}] {lesson.problem}", file=sys.stderr)
        print(f"    → Fix: {lesson.fix}", file=sys.stderr)
