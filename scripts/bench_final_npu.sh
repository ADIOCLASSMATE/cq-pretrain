#!/usr/bin/env bash
# The decisive 910B measurement: per-card throughput with the loader fed.
#
# What the previous two suites established:
#   - 4 packs/card with the stock loader: 27.8 s/update, data_wait ~14 s.
#   - Widening the loader (4 data / 6 image workers) on the 180-CPU row drops
#     data_wait to a 0.01 s median and the update to 18.9 s, with the node's
#     CPUs only 45% busy -- so the ceiling was threads, not cores.
#   - 8 packs/card is the better shape on 64 GiB (1530 vs 1452 tokens/s/card),
#     but it still waited on data for a median 1.2 s and a p90 of 13.9 s.
#   - Spending the extra memory on less recompute is not available: --no-checkpoint
#     runs out of memory at 58.8 of 60.96 GiB, at either pack count.
#
# So this suite widens the loader again at 8 packs/card and runs long enough for
# a stable number, then A/Bs the one remaining dispatch point that still takes
# its reference path on Ascend: the Wan DiT pointwise chain (modulated_norm and
# gated_add), which TorchAIR can fuse under PRETRAIN_COMPILE_MODULATION=1.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-final-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"
echo "cpus=$(nproc) ranks=${NPROC_PER_NODE} cpus_per_rank=$(( $(nproc) / NPROC_PER_NODE ))"

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
  python3 - "${BENCH_ROOT}/${name}" "${NPROC_PER_NODE}" <<'PY' || true
import json, statistics as st, sys
from pathlib import Path

out, ranks = Path(sys.argv[1]), int(sys.argv[2])
path = out / "steps.jsonl"
if not path.exists():
    print("  no steps.jsonl")
    raise SystemExit(0)
rows = [json.loads(line) for line in path.open()]
measured = [r for r in rows if not r.get("warmup")]
if not measured:
    raise SystemExit(0)


def summarize(label, selected):
    if not selected:
        print(f"  {label}: none")
        return
    seconds = [r["seconds"] for r in selected]
    ordered = sorted(seconds)
    # Throughput over elapsed time, not a median of ratios: a long run pays for
    # its slow updates, so the sum is the number that predicts wall clock.
    rate = sum(r["occupied_tokens"] for r in selected) / sum(seconds) / ranks
    print(
        f"  {label}: n={len(selected)} mean={st.mean(seconds):.2f} median={st.median(seconds):.2f} "
        f"p90={ordered[int(0.9 * len(ordered)) - 1]:.2f} max={max(seconds):.2f} "
        f"-> occupied_tokens/s/card={rate:.1f}"
    )


waits = [r.get("data_wait_seconds", 0.0) for r in measured]
print(f"  data_wait median={st.median(waits):.2f} mean={st.mean(waits):.2f} max={max(waits):.2f}")
print(f"  peak_memory_gib median={st.median(r['peak_memory_gib'] for r in measured):.2f}")
summarize("all measured", measured)
# Steps that never waited isolate the device; if the two agree, the loader is
# no longer the ceiling and the 'all measured' number is the real one.
summarize("fed (wait<0.5s)", [r for r in measured if r.get("data_wait_seconds", 0.0) < 0.5])
PY
  echo "=== ${name} done ==="
}

# Twice the decode threads of the previous suite: 8+12 per rank over 16 ranks is
# 320 threads on 192 cores, which is the point of the test. These threads block
# on image reads, so oversubscription is only wrong if the cores saturate --
# cpu.log says whether they did.
WIDE=(--data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24)

export PRETRAIN_COMPILE_MODULATION=0
run packs8-wide 8 48 10 --checkpoint --keep-attention-activations 6 "${WIDE[@]}"

# The DiT pointwise chain is the last dispatch point still evaluating its
# reference expression on Ascend. summary.json's operator_paths says whether
# TorchAIR actually took it over; the timing says whether that is worth doing.
# (An env assignment prefixed to a shell function is not reliably scoped, so
# export it and put it back afterwards.)
export PRETRAIN_COMPILE_MODULATION=1
run packs8-wide-torchair 8 24 6 --checkpoint --keep-attention-activations 6 "${WIDE[@]}"
export PRETRAIN_COMPILE_MODULATION=0

echo "=== CPU utilization while the suite ran ==="
python3 - "${BENCH_ROOT}/cpu.log" <<'PY' || true
import statistics as st, sys
from pathlib import Path
values = [int(line.rsplit("=", 1)[1]) for line in Path(sys.argv[1]).read_text().splitlines() if "cpu_busy_pct=" in line]
if values:
    ordered = sorted(values)
    print(f"samples={len(values)} median={st.median(values)}% p90={ordered[int(0.9 * len(ordered))]}% max={max(values)}%")
PY

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED configurations: ${failed[*]}"
  exit 1
fi
echo "suite complete: ${BENCH_ROOT}"
