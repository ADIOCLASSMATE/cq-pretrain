#!/usr/bin/env bash
# The shared log and exit status survive closing the Inspire terminal.
set -euo pipefail
cd "$(dirname "$0")/../.."
dataset_root=${1:?dataset root}
instance=${2:?instance name}
stage=${3:?validation or full}
workers=${4:-16}
batch=${5:-}
run_label="${instance}-${stage}-$(date -u +%Y%m%dT%H%M%S)"
log_root="$dataset_root/_distributed/logs"
mkdir -p "$log_root"
exec > >(tee -a "$log_root/$run_label.log") 2>&1
trap 'batch_status=$?; printf "%s\n" "$batch_status" > "$log_root/$run_label.exit_code"; exit "$batch_status"' EXIT
args=(worker --output "$dataset_root" --instance "$instance" --stage "$stage" --workers "$workers")
if [[ -n "$batch" ]]; then args+=(--batch "$batch"); fi
bash data/preprocess/distributed.sh "${args[@]}"
