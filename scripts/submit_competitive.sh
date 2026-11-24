#!/usr/bin/env bash
# Competitive job submission wrapper for training and evaluation.
#
# Instead of letting the platform's scheduler randomly pick nodes, this script
# discovers free nodes and directs the job toward them by excluding occupied ones.
# It bridges the latency gap: from node discovery to job submission in <10 seconds,
# so the job claims capacity before other submissions arrive.
#
# Usage from a priority controller or eval launcher:
#   bash scripts/submit_competitive.sh \
#     --name my-job-name \
#     --nodes 2 \
#     --command "bash my_job.sh" \
#     [--priority 3|4] [--project ...] [--dry-run] [--verbose]

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# Parse args: forward most to competitive_submit.py, intercept a few
declare -a py_args=()
job_name=""
nodes=1
command=""
priority=4
project="公共科研项目"
workspace="昇腾卡公共空间"
group="910B资源"
dry_run=0
verbose=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            job_name="$2"
            shift 2
            ;;
        --nodes)
            nodes="$2"
            shift 2
            ;;
        --command)
            command="$2"
            shift 2
            ;;
        --priority)
            priority="$2"
            shift 2
            ;;
        --project)
            project="$2"
            shift 2
            ;;
        --workspace)
            workspace="$2"
            shift 2
            ;;
        --group)
            group="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --verbose)
            verbose=1
            shift
            ;;
        --dataset)
            py_args+=(--datasets "$2")
            shift 2
            ;;
        *)
            # Forward unknown args (like --image, --quota, --max-time, etc.)
            py_args+=("$1")
            shift
            ;;
    esac
done

[[ -n "$job_name" ]] || { echo "❌ --name required" >&2; exit 1; }
[[ -n "$command" ]] || { echo "❌ --command required" >&2; exit 1; }

declare -a final_args=(
    --job-name "$job_name"
    --nodes "$nodes"
    --command "$command"
    --priority "$priority"
    --project "$project"
    --workspace "$workspace"
    --group "$group"
    "${py_args[@]}"
)

[[ $dry_run -eq 1 ]] && final_args+=(--dry-run)
[[ $verbose -eq 1 ]] && final_args+=(--verbose)

exec "$PYTHON" "$REPO/scripts/competitive_submit.py" "${final_args[@]}"
