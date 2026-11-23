#!/usr/bin/env bash
# One throughput configuration, measured the way production measures itself.
#
# The probe runs the real training entry point on the real release, so the
# numbers include data loading, the VAE, the recurrence and the optimizer, and
# summary.json records which operator implementation each dispatch point took.
#
#   BENCH_NAME=parity bash scripts/bench_accel.sh [extra train args...]
#
# Per-card work is held fixed across accelerators: --global-batch is
# PACKS_PER_CARD x the local rank count, every pack is one sequence_length
# window, so seconds/update is directly comparable between 16 Ascend and 32
# L40S cards once divided by packs per card.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${BENCH_NAME:?Set BENCH_NAME}"
RANKS="${NPROC_PER_NODE:?Set NPROC_PER_NODE}"
PACKS_PER_CARD="${PACKS_PER_CARD:-4}"
STEPS="${BENCH_STEPS:-12}"
WARMUP="${BENCH_WARMUP:-4}"
OUT="${BENCH_ROOT:-${ROOT}/.runs/bench}/${NAME}"
DATA="${BENCH_DATA:?Set BENCH_DATA to the release directory}"
ASSETS="${BENCH_ASSETS:?Set BENCH_ASSETS to the pinned model directory}"
rm -rf "${OUT}"
mkdir -p "${OUT}"
GLOBAL_BATCH=$((RANKS * PACKS_PER_CARD))
{
  echo "name=${NAME} ranks=${RANKS} global_batch=${GLOBAL_BATCH} steps=${STEPS} warmup=${WARMUP}"
  echo "backend=${PRETRAIN_BACKEND:-unset} extra=$*"
} | tee "${OUT}/invocation.txt"
bash "${ROOT}/scripts/launch.sh" --config "${ROOT}/configs/qwen35_2b_wan.yaml" \
  --text-assets "${ASSETS}/Qwen3.5-2B-Base" --vision-assets "${ASSETS}/Qwen3.5-2B-Base" \
  --wan-assets "${ASSETS}/Wan2.1-T2V-1.3B" --wan-vae-assets "${ASSETS}/Wan2.1-VAE" \
  --data-root "${DATA}" --long-document-policy error \
  --tp 1 --global-batch "${GLOBAL_BATCH}" --sequence-length 8192 \
  --steps "${STEPS}" --warmup "${WARMUP}" \
  --rf 1 --timestep-shift 3.0 --dit-microbatch-tokens 8192 \
  --lr 5e-5 --lr-decay-style wsd --lr-warmup-steps 1000 \
  --bf16-grad-reduce --phase-timing \
  --output "${OUT}" "$@" >>"${OUT}/log.txt" 2>&1
code=$?
echo "exit=${code}"
python3 - "${OUT}/summary.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("no summary.json; read log.txt")
    raise SystemExit(1)
s = json.loads(path.read_text())
keys = ("world", "global_batch", "steps", "mean_seconds", "median_seconds", "p95_seconds",
        "peak_memory_gib", "capacity_slots_per_second", "capacity_slots_per_second_per_card",
        "forward_ms", "backward_ms", "optimizer_ms")
for key in keys:
    if key in s:
        print(f"{key} = {s[key]}")
print("operator_paths =", json.dumps(s.get("operator_paths"), indent=2, sort_keys=True))
PY
exit $code
