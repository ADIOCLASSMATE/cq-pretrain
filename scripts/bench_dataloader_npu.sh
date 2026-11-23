#!/usr/bin/env bash
# Is the Ascend run compute-bound or starved by the host data pipeline?
#
# The first 910B suite (.runs/bench-910b-20261123T131419Z) measured 27.8 s per
# update at 4 packs per card against 19.9 s on L40S, but its steps.jsonl shows
# data_wait_seconds around 14 s on every step except one, and that one step took
# 15.2 s -- faster than L40S. L40S waits 0.08 s. The difference the platform
# gives us is CPU per rank: 210 CPU / 8 cards there, 128 / 16 here.
#
# So this suite changes only the host pipeline, on the wider 180-CPU quota row,
# and samples CPU utilization while it runs. If data_wait falls to ~0 the 910B
# is the faster card for this model; if it does not, the loader is the ceiling
# and the extra device speed cannot be reached on this node shape.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-loader-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"
nproc > "${BENCH_ROOT}/nproc.txt"
echo "cpus=$(nproc) ranks=${NPROC_PER_NODE} cpus_per_rank=$(( $(nproc) / NPROC_PER_NODE ))"

# Sample total CPU utilization once a second. A loader that is starved because
# the node has too few cores looks different from one starved by too few
# threads: the first pins every core, the second leaves cores idle.
(
  previous=""
  while true; do
    read -r _ a b c idle rest < /proc/stat
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
) > "${BENCH_ROOT}/cpu.log" 2>&1 &
SAMPLER=$!
trap 'kill ${SAMPLER} 2>/dev/null' EXIT

failed=()
run() {
  local name="$1" packs="$2" steps="$3" warmup="$4"
  shift 4
  echo "=== ${name}: packs/card=${packs} steps=${steps} $* ==="
  BENCH_NAME="${name}" PACKS_PER_CARD="${packs}" BENCH_STEPS="${steps}" BENCH_WARMUP="${warmup}" \
    bash "${ROOT}/scripts/bench_accel.sh" "$@" 2>&1 || failed+=("${name}")
  python3 - "${BENCH_ROOT}/${name}" <<'PY' || true
import json, statistics as st, sys
from pathlib import Path
out = Path(sys.argv[1])
path = out / "steps.jsonl"
if path.exists():
    rows = [json.loads(line) for line in path.open()]
    measured = [r for r in rows if not r.get("warmup")]
    if measured:
        def median(key):
            values = [r[key] for r in measured if key in r]
            return round(st.median(values), 2) if values else None
        print("  measured_steps =", len(measured))
        for key in ("seconds", "data_wait_seconds", "data_prepare_seconds", "peak_memory_gib"):
            print(f"  {key} median = {median(key)}")
        per_card = st.median(r["occupied_tokens"] for r in measured) / 16
        print("  occupied_tokens_per_second_per_card =", round(per_card / median("seconds"), 1))
PY
  echo "=== ${name} done ==="
}

# The same per-card work as the L40S production run, with the host pipeline
# widened: more decode threads per rank and a deeper queue to ride out the
# bursts. Long enough (40 updates) to show whether the wait settles on its own.
TUNED=(--data-workers 4 --image-workers 6 --image-prefetch 32 --data-prefetch 16)
run baseline-long 4 24 6 --checkpoint --keep-attention-activations 6
run tuned-loader 4 40 8 --checkpoint --keep-attention-activations 6 "${TUNED[@]}"
run tuned-packs8 8 24 6 --checkpoint --keep-attention-activations 6 "${TUNED[@]}"

echo "=== CPU utilization while the suite ran ==="
python3 - "${BENCH_ROOT}/cpu.log" <<'PY' || true
import statistics as st, sys
from pathlib import Path
values = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if "cpu_busy_pct=" in line:
        values.append(int(line.rsplit("=", 1)[1]))
if values:
    ordered = sorted(values)
    print(f"samples={len(values)} median={st.median(values)}% p90={ordered[int(0.9 * len(ordered))]}% max={max(values)}%")
PY

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED configurations: ${failed[*]}"
  exit 1
fi
echo "suite complete: ${BENCH_ROOT}"
