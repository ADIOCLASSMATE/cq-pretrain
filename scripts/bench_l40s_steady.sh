#!/usr/bin/env bash
# L40S steady-state throughput, long enough to be past its warm-up.
#
# A 14-step probe cannot measure this cluster. Its production run needed about
# 100 steps before the step time settled: inside the warm-up the same
# configuration bounces between 20.5 s and 42 s with no trend, while the
# 724-step steady state is mean 20.49 s, median 20.08, p90 21.97. Ascend shows
# no such warm-up (its steps sit in 14.9-17.2 s from the first one), so a short
# probe is fair there and dishonest here.
#
# So: 150 updates with the first 100 discarded, in the exact configuration the
# training job would use. One number, measured where it is meaningful.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=cuda PRETRAIN_STACK=cuda
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WORLD="${WORLD:-32}"
export BENCH_DATA="${BENCH_DATA:-/inspire/hdd/global_user/wanjiaxin-253108030048/datasets/mm_pretrain/stage1-512-v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-${ROOT}/.assets}"
export BENCH_ROOT="${BENCH_ROOT:?Set BENCH_ROOT in the job command; every node must get the same one}"
NODE="${PET_NODE_RANK:-${NODE_RANK:-0}}"
mkdir -p "${BENCH_ROOT}"

BENCH_NAME=steady BENCH_STEPS="${BENCH_STEPS:-150}" BENCH_WARMUP="${BENCH_WARMUP:-100}" \
  bash "${ROOT}/scripts/bench_head_to_head.sh" \
  --checkpoint --keep-attention-activations 6 \
  --data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24
code=$?

if [[ "${NODE}" == 0 ]]; then
  python3 - "${BENCH_ROOT}/steady/steps.jsonl" "${WORLD}" <<'PY' || true
import json, statistics as st, sys
from pathlib import Path
path, world = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(line) for line in path.open() if line.strip()]
measured = [r for r in rows if not r.get("warmup")]
if not measured:
    print("no measured steps"); raise SystemExit
seconds = sorted(r["seconds"] for r in measured)
occupied = st.median(r["occupied_tokens"] for r in measured) / world
mean = st.mean(seconds)
print(f"  n = {len(measured)}  (first measured update: {measured[0]['step']})")
print(f"  seconds mean/median/p90 = {mean:.2f} / {st.median(seconds):.2f} / {seconds[int(0.9 * len(seconds)) - 1]:.2f}")
print(f"  data_wait median = {st.median(r.get('data_wait_seconds', 0) for r in measured):.3f}")
print(f"  peak_gib median  = {st.median(r['peak_memory_gib'] for r in measured):.2f}")
print(f"  occupied tokens/s/card = mean {occupied / mean:.0f}  median {occupied / st.median(seconds):.0f}")
PY
fi
exit ${code}
