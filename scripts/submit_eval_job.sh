#!/usr/bin/env bash
# Competitive job submission for evaluation tasks.
#
# Discovers free Ascend nodes and submits a job directed toward them.
# Usage:
#   bash scripts/submit_eval_job.sh \
#     --job-name cq-eval-1000 \
#     --task geneval \
#     --weights /path/to/weights \
#     --output /path/to/output \
#     [--priority 3|4] [--dry-run]

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

job_name=""
task=""
weights=""
output=""
priority=3
dry_run=0
verbose=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-name)
            job_name="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --weights)
            weights="$2"
            shift 2
            ;;
        --output)
            output="$2"
            shift 2
            ;;
        --priority)
            priority="$2"
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
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

[[ -n "$job_name" && -n "$task" && -n "$weights" && -n "$output" ]] || {
    echo "Usage: $0 --job-name NAME --task TASK --weights W --output O [--priority 3|4] [--dry-run]" >&2
    exit 1
}

# Build the eval command
eval_cmd="bash $REPO/scripts/eval_geneval_ascend.sh $output $weights"

declare -a py_args=(
    --job-name "$job_name"
    --nodes 1
    --priority "$priority"
    --command "$eval_cmd"
    --workspace "昇腾卡公共空间"
    --project "公共科研项目"
    --group "910B资源"
    --quota "16,180,1800"
    --image "docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.4"
    --shm-size 128
    --datasets "dcvlm-baseline-200b:v2"
    --datasets "gen-und-mm-pretrain:01"
    --datasets "imgedit:v1"
    --datasets "infovis:v1"
    --datasets "textground4m:v1"
    --datasets "textatlas5m:v1"
    --datasets "poster100k:v1"
    --datasets "tide:v1"
    --auto-fault-tolerance
    --fault-tolerance-max-retry 20
    --fault-tolerance-retry-interval 120
)

[[ $dry_run -eq 1 ]] && py_args+=(--dry-run)
[[ $verbose -eq 1 ]] && py_args+=(--verbose)

exec "$PYTHON" "$REPO/scripts/competitive_submit.py" "${py_args[@]}"
