#!/usr/bin/env bash
# Where does the Ascend backward actually go, and can the spare memory buy any
# of it back?
#
# Measured on 2026-11-23 (evaluations/benchmarks/accel_910b_vs_l40s_20261123.json):
# backward is 25.1 s of a 37.1 s step, and removing 12 of 30 DiT recompute
# layers moved it by only 5%. So recompute is not the bulk -- but nothing has
# ever profiled this model on 910B, so the remaining 95% is unattributed.
#
# Two things here, in one job:
#   1. An operator profile of the production configuration (op_statistic.csv),
#      which names the kernels that own the device time.
#   2. The one untested memory-for-backward-time trade in the GDN backend.
#      In DELTA_ASCENDC_MODE=fused the backward recomputes recompute_w_u_fwd
#      and chunk_gated_delta_rule_fwd_h on every GDN layer; in `cached` the
#      forward saves w/h/vn instead. Activation checkpointing rebuilds the
#      layer graph during backward, so only one layer's tensors are ever live:
#      the memory cost should be small and the saving is two AscendC calls per
#      GDN layer per microbatch.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-gdn-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"

# The settled production configuration: 8 packs per card, everything
# recomputed, a loader wide enough that no step waits for data.
LOADER=(--data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24)
BASE=(--checkpoint --keep-attention-activations 6 "${LOADER[@]}")

failed=()
summarize() {
  python3 - "${BENCH_ROOT}/$1" <<'PY' || true
import json, statistics as st, sys
from pathlib import Path
rows = Path(sys.argv[1], "steps.jsonl")
if rows.exists():
    measured = [json.loads(line) for line in rows.open()]
    measured = [r for r in measured if not r.get("warmup")]
    if measured:
        seconds = [r["seconds"] for r in measured]
        waits = [r.get("data_wait_seconds", 0.0) for r in measured]
        per_card = st.median(r["occupied_tokens"] for r in measured) / 16
        print(f"  n={len(measured)} mean={st.mean(seconds):.2f} median={st.median(seconds):.2f}")
        print(f"  data_wait median={st.median(waits):.2f} max={max(waits):.2f}")
        print(f"  peak_gib median={st.median(r['peak_memory_gib'] for r in measured):.2f}")
        for key in ("forward_ms", "backward_ms", "optimizer_ms"):
            if key in measured[0]:
                print(f"  {key} median={st.median(r[key] for r in measured):.1f}")
        print(f"  occupied_tokens/s/card mean={per_card / st.mean(seconds):.1f}"
              f" median={per_card / st.median(seconds):.1f}")
PY
}

run() {
  local name="$1" steps="$2" warmup="$3"
  shift 3
  echo "=== ${name}: mode=${DELTA_ASCENDC_MODE:-fused} steps=${steps} $* ==="
  BENCH_NAME="${name}" PACKS_PER_CARD=8 BENCH_STEPS="${steps}" BENCH_WARMUP="${warmup}" \
    bash "${ROOT}/scripts/bench_accel.sh" "$@" 2>&1 || failed+=("${name}")
  summarize "${name}"
  echo "=== ${name} done ==="
}

# A/B in one job so both see the same node, the same release and the same
# scheduling. Control first, so a `cached` failure still leaves a baseline.
DELTA_ASCENDC_MODE=fused run gdn-fused 14 4 "${BASE[@]}"
DELTA_ASCENDC_MODE=cached run gdn-cached 14 4 "${BASE[@]}"

# The profile runs last: its trace parse takes minutes and must not delay the
# timed configurations. Rank 0 only; the schedule needs warmup+3 updates.
echo "=== profile: operator breakdown of the production configuration ==="
BENCH_NAME="profile" PACKS_PER_CARD=8 BENCH_STEPS=9 BENCH_WARMUP=4 \
  bash "${ROOT}/scripts/bench_accel.sh" "${BASE[@]}" --profile --profile-ranks 0 2>&1 || failed+=("profile")
python3 - "${BENCH_ROOT}/profile/profile/rank0" <<'PY' || true
import csv, sys
from pathlib import Path
rows = []
for path in Path(sys.argv[1]).glob("*/ASCEND_PROFILER_OUTPUT/op_statistic.csv"):
    with path.open() as handle:
        rows.extend(dict(row) for row in csv.DictReader(handle))
if not rows:
    print("  no op_statistic.csv; read the profile directory")
    raise SystemExit
def total(row):
    for key in ("Total Time(us)", "Total Time(ns)", "Total Time"):
        if key in row:
            return float(row[key])
    return 0.0
rows.sort(key=total, reverse=True)
grand = sum(total(r) for r in rows) or 1.0
name = next(k for k in rows[0] if "Type" in k or "Name" in k)
print(f"  {'operator':<42}{'share':>8}{'calls':>10}")
for row in rows[:25]:
    calls = row.get("Count") or row.get("Core Type") or ""
    print(f"  {row[name][:42]:<42}{100 * total(row) / grand:>7.1f}%{calls:>10}")
PY
echo "=== profile done ==="

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED configurations: ${failed[*]}"
  exit 1
fi
echo "suite complete: ${BENCH_ROOT}"
