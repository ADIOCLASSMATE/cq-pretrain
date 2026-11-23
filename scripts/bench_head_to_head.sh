#!/usr/bin/env bash
# One configuration, measured identically on either accelerator, at 32 cards.
#
# Every number this project has comparing the two so far was 16 Ascend cards on
# one node against 32 L40S cards on four, so "Ascend is faster per card" still
# contained "and it did not pay for cross-node allreduce". This runs the same
# global batch over the same card count on both sides, so that term cancels.
#
# Multi-node launch comes from the platform's PET_* variables through
# scripts/launch.sh, exactly as the production job.sh does. Node rank 0 writes
# the report; every node writes its own log.
#
#   BENCH_NAME=l40s-baseline WORLD=32 NPROC_PER_NODE=8 PRETRAIN_BACKEND=cuda \
#     bash scripts/bench_head_to_head.sh --checkpoint --keep-attention-activations 6
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${BENCH_NAME:?Set BENCH_NAME}"
WORLD="${WORLD:?Set WORLD to the total card count}"
PACKS_PER_CARD="${PACKS_PER_CARD:-4}"
STEPS="${BENCH_STEPS:-20}"
WARMUP="${BENCH_WARMUP:-6}"
OUT="${BENCH_ROOT:?Set BENCH_ROOT}/${NAME}"
DATA="${BENCH_DATA:?Set BENCH_DATA}"
ASSETS="${BENCH_ASSETS:?Set BENCH_ASSETS}"
NODE="${PET_NODE_RANK:-${NODE_RANK:-0}}"
GLOBAL_BATCH=$((WORLD * PACKS_PER_CARD))
mkdir -p "${OUT}"
LOG="${OUT}/node${NODE}.log"

if [[ "${NODE}" == 0 ]]; then
  {
    echo "name=${NAME} world=${WORLD} global_batch=${GLOBAL_BATCH} packs/card=${PACKS_PER_CARD}"
    echo "backend=${PRETRAIN_BACKEND:-unset} nodes=${PET_NNODES:-?} per_node=${NPROC_PER_NODE:-?} extra=$*"
    echo "cpus=$(nproc)"
  } | tee "${OUT}/invocation.txt"
fi

bash "${ROOT}/scripts/launch.sh" --config "${ROOT}/configs/qwen35_2b_wan.yaml" \
  --text-assets "${ASSETS}/Qwen3.5-2B-Base" --vision-assets "${ASSETS}/Qwen3.5-2B-Base" \
  --wan-assets "${ASSETS}/Wan2.1-T2V-1.3B" --wan-vae-assets "${ASSETS}/Wan2.1-VAE" \
  --data-root "${DATA}" --long-document-policy error \
  --tp 1 --global-batch "${GLOBAL_BATCH}" --sequence-length 8192 \
  --steps "${STEPS}" --warmup "${WARMUP}" \
  --rf 1 --timestep-shift 3.0 --dit-microbatch-tokens 8192 \
  --lr 5e-5 --lr-decay-style wsd --lr-warmup-steps 1000 \
  --bf16-grad-reduce --phase-timing \
  --output "${OUT}" "$@" >>"${LOG}" 2>&1
code=$?
echo "node=${NODE} exit=${code}"
[[ "${NODE}" == 0 ]] || exit ${code}

python3 - "${OUT}" "${WORLD}" <<'PY'
import json, statistics as st, sys
from pathlib import Path
out, world = Path(sys.argv[1]), int(sys.argv[2])
summary = out / "summary.json"
if summary.exists():
    s = json.loads(summary.read_text())
    for key in ("world", "global_batch", "steps", "mean_seconds", "median_seconds", "p95_seconds",
                "peak_memory_gib", "capacity_slots_per_second_per_card", "forward_ms", "backward_ms",
                "optimizer_ms"):
        if key in s:
            print(f"  {key} = {s[key]}")
    if "operator_paths" in s:
        print("  operator_paths =", json.dumps(s["operator_paths"], sort_keys=True))
else:
    print("  no summary.json; read the node logs")
rows = [json.loads(line) for line in (out / "steps.jsonl").open()] if (out / "steps.jsonl").exists() else []
measured = [r for r in rows if not r.get("warmup")]
if measured:
    def med(key):
        values = [r[key] for r in measured if key in r]
        return round(st.median(values), 3) if values else None
    per_card = st.median(r["occupied_tokens"] for r in measured) / world
    mean_seconds = st.mean(r["seconds"] for r in measured)
    print(f"  measured_steps      = {len(measured)}")
    print(f"  seconds mean/median = {round(mean_seconds, 2)} / {med('seconds')}")
    print(f"  data_wait median    = {med('data_wait_seconds')}  max {round(max(r['data_wait_seconds'] for r in measured), 2)}")
    print(f"  peak_gib median     = {med('peak_memory_gib')}")
    print(f"  occupied/s/card     = mean {round(per_card / mean_seconds, 1)}  median {round(per_card / med('seconds'), 1)}")
PY
exit ${code}
