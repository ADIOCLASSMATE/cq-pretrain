#!/usr/bin/env bash
# The Ascend half of the 32-card comparison: 2 nodes x 16 cards.
#
# Every previous Ascend number was 16 cards on a single node, so it never paid
# for a cross-node allreduce while the L40S numbers it was compared against
# always did. This runs the same global batch over the same card count, which
# is the only form in which "which one is faster" has an answer.
#
# It also carries the loader finding: the host, not the NPU, has set the pace
# since varlen cut compute by 26%, and no thread setting moves it (torch
# intra-op threads changed nothing, doubling image workers bought 3.5%). So the
# loader settings here are simply the widest that measured best, and the report
# prints data_wait so a host-bound result cannot be mistaken for a slow card.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu PRETRAIN_STACK=core17
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export WORLD="${WORLD:-32}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:?Set BENCH_ROOT in the job command; every node must get the same one}"
NODE="${PET_NODE_RANK:-${NODE_RANK:-0}}"
mkdir -p "${BENCH_ROOT}"
[[ "${NODE}" == 0 ]] && echo "suite root ${BENCH_ROOT} node=${NODE} cpus=$(nproc)"

failed=()
if [[ "${NODE}" == 0 ]]; then
  echo "=== gate: the Ascend varlen kernel against the per-segment loop ==="
  NPROC_PER_NODE=1 PRETRAIN_MODULE=scripts.validation.validate_varlen \
    bash "${ROOT}/scripts/launch.sh" --output "${BENCH_ROOT}/varlen_gate.json" \
    >"${BENCH_ROOT}/gate.log" 2>&1
  if [[ $? -ne 0 ]]; then
    echo "GATE FAILED -- numerics first; no timing collected."
    tail -40 "${BENCH_ROOT}/gate.log"
    exit 1
  fi
  python3 -c "
import json
report = json.load(open('${BENCH_ROOT}/varlen_gate.json'))
print('  passed =', report['passed'], ' device =', report.get('device'))
" || failed+=(gate-report)
fi

run() {
  local name="$1" steps="$2" warmup="$3"
  shift 3
  [[ "${NODE}" == 0 ]] && echo "=== ${name}: steps=${steps} $* ==="
  BENCH_NAME="${name}" BENCH_STEPS="${steps}" BENCH_WARMUP="${warmup}" \
    bash "${ROOT}/scripts/bench_head_to_head.sh" "$@" 2>&1 || failed+=("${name}")
  [[ "${NODE}" == 0 ]] && echo "=== ${name} done ==="
  return 0
}

PRODUCTION=(--checkpoint --keep-attention-activations 6
  --data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24)

export PRETRAIN_CUDA_VARLEN=1
run varlen-on 20 6 "${PRODUCTION[@]}"
# The same A/B the L40S side runs, so the comparison is like for like. On
# Ascend this knob is already known to matter; measuring it again here costs
# one phase and keeps both halves of the comparison symmetric.
export PRETRAIN_NPU_VARLEN=0
run varlen-off 20 6 "${PRODUCTION[@]}"
export PRETRAIN_NPU_VARLEN=1

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
[[ "${NODE}" == 0 ]] && echo "suite complete: ${BENCH_ROOT}"
exit 0
