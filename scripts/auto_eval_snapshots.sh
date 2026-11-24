#!/usr/bin/env bash
# Monitor training eval_snapshots and auto-submit evaluation jobs using resource controller.
#
# Usage:
#   bash scripts/auto_eval_snapshots.sh \
#     --run /path/to/training/run \
#     --tasks geneval,qa,vqav2 \
#     --max-concurrent 2
#
# Watches train/eval_snapshots.jsonl; for each new snapshot, submits an eval job
# to Ascend with the resource controller (auto priority, competitive submission).

set -u

RUN=""
TASKS="geneval"
MAX_CONCURRENT=2
ASCEND_CODE=/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain
RESOURCE_CONTROLLER="$ASCEND_CODE/scripts/resource_allocator.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -n "$RUN" ]] || { echo "Usage: $0 --run /path/to/run"; exit 1; }

EVAL_LOG="$RUN/eval_submissions.log"
SUBMITTED_FILE="$RUN/.submitted_steps"
mkdir -p "$RUN"
touch "$SUBMITTED_FILE"

running_evals() {
    ssh ascend "~/.local/bin/inspire job list -n 100 --workspace 昇腾卡公共空间 2>/dev/null" | \
        grep -c "cq-eval-" | head -1
}

log() {
    echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$EVAL_LOG"
}

log "=== Monitoring eval snapshots at $RUN ==="

# Watch eval_snapshots.jsonl and submit jobs for new ones
while true; do
    if [[ ! -f "$RUN/train/eval_snapshots.jsonl" ]]; then
        sleep 10
        continue
    fi

    # Get already-submitted steps
    submitted=$(cat "$SUBMITTED_FILE" 2>/dev/null | tr '\n' ' ')

    # Check for new snapshots
    while IFS= read -r line; do
        step=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('step'))")
        skipped=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('skipped'))")

        [[ "$skipped" == "true" ]] && continue
        [[ " $submitted " == *" $step "* ]] && continue

        # Check concurrent limit
        running=$(running_evals)
        if [[ $running -ge $MAX_CONCURRENT ]]; then
            log "Max concurrent ($MAX_CONCURRENT) reached; waiting..."
            sleep 60
            continue
        fi

        # Submit eval job for this step
        weights="$RUN/eval_snapshots/step_$(printf '%07d' $step)"
        output="$RUN/evaluations/geneval-step$step"

        log "Submitting eval for step $step"
        python3 "$RESOURCE_CONTROLLER" \
            --job-name "cq-eval-step$step" \
            --command "bash $ASCEND_CODE/scripts/eval_geneval_ascend.sh $output $weights" \
            --job-type eval \
            --required-cards 16 \
            --workspace 昇腾卡公共空间 \
            --project 公共科研项目 \
            --verbose 2>&1 | tail -3 >> "$EVAL_LOG"

        echo "$step" >> "$SUBMITTED_FILE"
    done < "$RUN/train/eval_snapshots.jsonl"

    sleep 30
done
