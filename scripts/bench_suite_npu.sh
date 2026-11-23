#!/usr/bin/env bash
# Ascend 910B throughput suite: can a 64 GiB card beat the tuned 44 GiB L40S
# configuration once the extra memory is spent on less recompute or more work
# per card? Every configuration runs the real training entry point on the same
# release; summary.json records the operator implementations that ran, and
# PRETRAIN_REQUIRE_FUSED=1 (set by scripts/launch.sh for npu) turns a silent
# fallback to a portable expression into a failure rather than a slow number.
#
# One job runs the whole suite so the model load and the scheduling wait are
# paid once. A configuration that fails (usually out of memory) is recorded
# and the suite continues.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-910b-$(date -u +%Y%m%dT%H%M%SZ)}"
export BENCH_STEPS="${BENCH_STEPS:-12}"
export BENCH_WARMUP="${BENCH_WARMUP:-4}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"

# Step 0: the dispatch itself, on synthetic data, before any long run. Three
# updates are enough to prove every operator reached its Ascend kernel.
echo "=== smoke (synthetic, 3 updates) ==="
NPROC_PER_NODE="${NPROC_PER_NODE}" bash "${ROOT}/scripts/launch.sh" \
  --config "${ROOT}/configs/qwen35_2b_wan.yaml" \
  --text-assets "${BENCH_ASSETS}/Qwen3.5-2B-Base" --vision-assets "${BENCH_ASSETS}/Qwen3.5-2B-Base" \
  --wan-assets "${BENCH_ASSETS}/Wan2.1-T2V-1.3B" --wan-vae-assets "${BENCH_ASSETS}/Wan2.1-VAE" \
  --synthetic --tp 1 --global-batch "${NPROC_PER_NODE}" --sequence-length 8192 --images 1 \
  --steps 3 --warmup 1 --rf 1 --timestep-shift 3.0 --dit-microbatch-tokens 8192 \
  --checkpoint --keep-attention-activations 6 --phase-timing \
  --output "${BENCH_ROOT}/smoke" >"${BENCH_ROOT}/smoke.log" 2>&1
echo "smoke exit=$? (log ${BENCH_ROOT}/smoke.log)"
python3 - "${BENCH_ROOT}/smoke/summary.json" <<'PY' || true
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.exists():
    s = json.loads(path.read_text())
    print("smoke operator_paths =", json.dumps(s.get("operator_paths"), sort_keys=True))
    print("smoke seconds =", s.get("median_seconds"), "peak_gib =", s.get("peak_memory_gib"))
PY

# The L40S production run holds 4 packs of 8192 slots per card with the whole
# backbone and DiT recomputed and 6 attention activations kept: 20.2 s per
# update over 32 cards. Configuration 1 is that, unchanged, on 910B; the rest
# spend the extra 20 GiB.
run() {
  local name="$1" packs="$2"
  shift 2
  echo "=== ${name}: packs_per_card=${packs} $* ==="
  BENCH_NAME="${name}" PACKS_PER_CARD="${packs}" bash "${ROOT}/scripts/bench_accel.sh" "$@" 2>&1
  echo "=== ${name} done ==="
}

run parity 4 --checkpoint --keep-attention-activations 6
run keep-more 4 --checkpoint --keep-attention-activations 16
run no-recompute 4 --no-checkpoint
run packs8-parity 8 --checkpoint --keep-attention-activations 6
run packs8-no-recompute 8 --no-checkpoint
echo "suite complete: ${BENCH_ROOT}"
