#!/usr/bin/env bash
# What is the varlen change worth in wall time, once the loader keeps up again?
#
# The gate run (cq-npu-varlen-1124) proved the numerics and showed the device
# idle collapse from 18.5 s to 2.6 s per step, but its wall time was unusable:
# forward+backward fell 26%, and the loader settings that had kept the
# pre-change run fed (data_wait median 0.02 s) no longer could -- data_wait
# came back at a 5.4 s median with 25 s spikes. Faster compute consumes packs
# faster; the host pipeline became the binding constraint again.
#
# Two things change here and nothing else:
#   - deeper queues, to ride out the bursts the spikes point at
#   - 48 measured updates instead of 10, so the queue is full for most of them
#     and the comparison is against packs8-wide (n=38), not a 10-step sample
#
# The baseline to beat, same node shape, same configuration, before the change:
#   packs8-wide  n=38  mean 36.72 s  median 37.07 s  1679 occupied tokens/s/card
#   base-full    n=10  mean 36.50 s  median 35.90 s  1688 occupied tokens/s/card
# L40S steady state, for the per-card comparison: 1506 occupied tokens/s/card.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-varlen-fed-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"
echo "cpus=$(nproc) ranks=${NPROC_PER_NODE}"
nproc >"${BENCH_ROOT}/nproc.txt"

# Total CPU utilization once a second: a loader short of cores looks different
# from one short of threads. It read 59% median during the pre-change suite.
(
  previous=""
  while true; do
    read -r _ a b c idle rest </proc/stat
    busy=$((a + b + c))
    total=$((busy + idle))
    if [[ -n "${previous}" ]]; then
      pb=${previous%% *}; pt=${previous##* }
      d=$((total - pt))
      [[ ${d} -gt 0 ]] && echo "$(date -u +%H:%M:%S) cpu_busy_pct=$(( (busy - pb) * 100 / d ))"
    fi
    previous="${busy} ${total}"
    sleep 1
  done
) >"${BENCH_ROOT}/cpu.log" 2>&1 &
SAMPLER=$!
trap 'kill ${SAMPLER} 2>/dev/null' EXIT

failed=()
run() {
  local name="$1" steps="$2" warmup="$3"
  shift 3
  echo "=== ${name}: steps=${steps} $* ==="
  BENCH_NAME="${name}" PACKS_PER_CARD=8 BENCH_STEPS="${steps}" BENCH_WARMUP="${warmup}" \
    bash "${ROOT}/scripts/bench_accel.sh" "$@" 2>&1 || failed+=("${name}")
  python3 - "${BENCH_ROOT}/${name}/steps.jsonl" <<'PY' || true
import json, statistics as st, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("  no steps.jsonl"); raise SystemExit
rows = [json.loads(line) for line in path.open()]
m = [r for r in rows if not r.get("warmup")]
if not m:
    print("  no measured steps"); raise SystemExit
sec = [r["seconds"] for r in m]
wait = [r.get("data_wait_seconds", 0.0) for r in m]
occ = [r["occupied_tokens"] / 16 for r in m]
order = sorted(sec)
print("  n=%d mean=%.2f median=%.2f p90=%.2f max=%.2f" % (
    len(m), st.mean(sec), st.median(sec), order[int(0.9 * len(order)) - 1], max(sec)))
print("  data_wait median=%.2f p90=%.2f max=%.2f" % (
    st.median(wait), sorted(wait)[int(0.9 * len(wait)) - 1], max(wait)))
print("  fwd/bwd median ms = %.0f / %.0f" % (
    st.median([r.get("forward_ms", 0) for r in m]), st.median([r.get("backward_ms", 0) for r in m])))
print("  peak_gib median=%.2f" % st.median([r["peak_memory_gib"] for r in m]))
print("  occupied_tokens/s/card  mean=%.0f  median=%.0f" % (
    st.mean(occ) / st.mean(sec), st.median(occ) / st.median(sec)))
fed = [r for r in m if r.get("data_wait_seconds", 0.0) < 0.5]
if fed:
    fsec = [r["seconds"] for r in fed]
    print("  fed(wait<0.5s) n=%d mean=%.2f -> occ/s/card=%.0f" % (
        len(fed), st.mean(fsec), st.mean([r["occupied_tokens"] / 16 for r in fed]) / st.mean(fsec)))
PY
  echo "=== ${name} done ==="
}

PRODUCTION=(--checkpoint --keep-attention-activations 6)
# Same thread counts that fed the pre-change run; only the queues are deeper,
# because the spikes were bursts rather than a steady shortfall.
DEEP=(--data-workers 8 --image-workers 12 --image-prefetch 96 --data-prefetch 48)
# If depth alone is not enough, more decode threads. 192 cores, 16 ranks.
WIDE=(--data-workers 12 --image-workers 20 --image-prefetch 96 --data-prefetch 48)

run deep-48 48 10 "${PRODUCTION[@]}" "${DEEP[@]}"
run wide-48 48 10 "${PRODUCTION[@]}" "${WIDE[@]}"

echo "=== CPU utilization while the suite ran ==="
python3 - "${BENCH_ROOT}/cpu.log" <<'PY' || true
import statistics as st, sys
from pathlib import Path
values = [int(l.rsplit("=", 1)[1]) for l in Path(sys.argv[1]).read_text().splitlines() if "cpu_busy_pct=" in l]
if values:
    order = sorted(values)
    print("samples=%d median=%s%% p90=%s%% max=%s%%" % (
        len(values), st.median(values), order[int(0.9 * len(order))], max(values)))
PY

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED configurations: ${failed[*]}"
  exit 1
fi
echo "suite complete: ${BENCH_ROOT}"
