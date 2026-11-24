#!/usr/bin/env bash
# Usage: scripts/evaluate_ascend.sh TASK WEIGHTS OUTPUT_DIR
#
# Sample one benchmark task on Ascend cards. Unlike scripts/evaluate.sh this
# only produces outputs; it never scores them. GenEval's detector is a pinned
# torch 1.13 / Python 3.9 CUDA runtime that does not exist on this cluster, so
# the images are packed here and scored on H100 instead. The two clusters share
# no filesystem, so the tar is the handoff.
#
# EVAL_LIMIT=N runs the first N items and writes SMOKE.json instead of the
# full GENERATED.json, which is how this path is verified before a real run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task="${1:?Usage: evaluate_ascend.sh TASK WEIGHTS OUTPUT_DIR}"
weights="${2:?missing WEIGHTS}"
out="${3:?missing OUTPUT_DIR}"
config="${EVAL_CONFIG:-$ROOT/configs/evaluation_ascend.yaml}"
cd "$ROOT"
mkdir -p "$out"

export PRETRAIN_MODULE=evaluations.suites.run
export PRETRAIN_BACKEND=npu PRETRAIN_STACK=core17
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
{
  echo "host=$(hostname) task=$task weights=$weights cards=$NPROC_PER_NODE limit=${EVAL_LIMIT:-0} $(date -u +%FT%TZ)"
  echo "commit=$(cat SOURCE_COMMIT 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >>"$out/job.log"

args=("$task" "$weights" "$out" --config "$config")
[[ -n "${EVAL_LIMIT:-}" ]] && args+=(--limit "$EVAL_LIMIT")
bash scripts/launch.sh "${args[@]}" >>"$out/job.log" 2>&1
code=$?

receipt="$out/$([[ -n "${EVAL_LIMIT:-}" ]] && echo SMOKE.json || echo GENERATED.json)"
if [[ ${code} -ne 0 || ! -f "${receipt}" ]]; then
  echo "sampling failed (exit ${code}); no ${receipt##*/}" >>"$out/job.log"
  tail -30 "$out/job.log"
  exit 3
fi

# Pack for the handoff. Relative paths, so the detector can unpack anywhere:
# it wants <root>/images/<prompt>/samples/<n>.png beside the receipt.
if [[ "$task" == geneval ]]; then
  tar czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"
  echo "packed $(du -h "$out.tar.gz" | cut -f1) -> $out.tar.gz" >>"$out/job.log"
  du -h "$out.tar.gz"
fi
echo "OK $(basename "${receipt}")"
