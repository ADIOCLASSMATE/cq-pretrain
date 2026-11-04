#!/usr/bin/env bash
# Start this script inside tmux; logs and exit status outlive terminal detachment.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:?pilot or full}
dataset_root=${2:?output dataset directory}
workers=${3:-4}
pilot_root=${4:-}
if [[ "$mode" != pilot && "$mode" != full ]]; then
    printf '%s\n' 'Mode must be pilot or full' >&2
    exit 2
fi
mkdir -p "$dataset_root/logs"
rm -f "$dataset_root/logs/exit_code"
run_label=$(date -u +%Y%m%dT%H%M%SZ)
exec > >(tee -a "$dataset_root/logs/$run_label.log") 2>&1
trap 'batch_status=$?; printf "%s\n" "$batch_status" > "$dataset_root/logs/exit_code"; exit "$batch_status"' EXIT
if [[ ! -f "$dataset_root/processing_config.json" ]]; then
    if [[ "$mode" == pilot ]]; then
        bash data/preprocess/run.sh select --output "$dataset_root" --pilot
    else
        bash data/preprocess/run.sh select --output "$dataset_root"
    fi
fi
if [[ "$mode" == pilot ]]; then
    bash data/preprocess/run.sh process --output "$dataset_root" --workers "$workers"
    bash data/preprocess/run.sh verify --output "$dataset_root" --source-correspondence
else
    bash data/preprocess/run.sh process --output "$dataset_root" --workers "$workers" --require-pilot "${pilot_root:?passed pilot required}"
fi
