#!/usr/bin/env bash
# Did attending to every packed segment in one call change the numbers, and did
# it buy anything?
#
# Order matters. The numerical gate runs first on one card: if the kernel
# disagrees with the per-segment loop, no timing is worth collecting and the
# suite stops there instead of spending the rest of the allocation. Then the
# dispatch census (did the launches actually go away), the timing against the
# recorded baseline, and a profile (did the device stop idling).
#
# Baselines, same node shape and configuration, measured 2026-11-23 before this
# change (evaluations/benchmarks/npu_910b_{operator_profile,dispatch_attribution}):
#   base-full    n=10  mean 36.50 s  median 35.90 s  1688 occupied tokens/s/card
#   packs8-wide  n=38  mean 36.72 s  median 37.07 s  1679 occupied tokens/s/card
#   census       375,314 dispatches/step
#   profile      Computing 25.8 s, Free ~12.4 s of a 38 s step (33% idle)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-varlen-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"
failed=()

echo "=== gate: varlen kernel against the per-segment loop (1 card) ==="
NPROC_PER_NODE=1 PRETRAIN_MODULE=scripts.validation.validate_varlen \
  bash "${ROOT}/scripts/launch.sh" \
  --output "${BENCH_ROOT}/varlen_gate.json" \
  --vision-assets "${BENCH_ASSETS}/Qwen3.5-2B-Base" >"${BENCH_ROOT}/gate.log" 2>&1
if [[ $? -ne 0 ]]; then
  echo "GATE FAILED -- numerics first; no timing collected."
  tail -40 "${BENCH_ROOT}/gate.log"
  exit 1
fi
python3 - "${BENCH_ROOT}/varlen_gate.json" <<'PY' || failed+=(gate-report)
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
print("  dit =", report["dit"], " vision =", report["vision"])
print(f"  {'case':22s} {'field':6s} {'loop':>9s} {'varlen':>9s} {'agree':>9s}")
for name, case in report["cases"].items():
    if "error" in case:
        print(f"  {name:22s} ERROR {case['error'][:90]}")
        continue
    for field in ("output", "dq", "dk", "dv"):
        print(
            f"  {name:22s} {field:6s} {case['loop'][field]:9.5f} {case['varlen'][field]:9.5f}"
            f" {case['loop_vs_varlen'][field]:9.2e}"
        )
print("  passed =", report["passed"], report["failures"])
PY
echo "=== gate done ==="

TUNED=(--data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24)
PRODUCTION=(--checkpoint --keep-attention-activations 6 "${TUNED[@]}")

run() {
  local name="$1" steps="$2" warmup="$3"
  shift 3
  echo "=== ${name}: steps=${steps} $* ==="
  BENCH_NAME="${name}" PACKS_PER_CARD=8 BENCH_STEPS="${steps}" BENCH_WARMUP="${warmup}" \
    bash "${ROOT}/scripts/bench_accel.sh" "$@" 2>&1 || failed+=("${name}")
  python3 - "${BENCH_ROOT}/${name}" <<'PY' || true
import json, statistics as st, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "steps.jsonl").open()] if (out / "steps.jsonl").exists() else []
measured = [r for r in rows if not r.get("warmup")]
if measured:
    seconds = [r["seconds"] for r in measured]
    per_card = st.median(r["occupied_tokens"] for r in measured) / 16
    print(f"  n={len(measured)} mean={st.mean(seconds):.2f} median={st.median(seconds):.2f} max={max(seconds):.2f}")
    print(f"  data_wait median={st.median(r.get('data_wait_seconds', 0) for r in measured):.2f}")
    print(f"  peak_gib median={st.median(r['peak_memory_gib'] for r in measured):.2f}")
    print(f"  occupied_tokens/s/card mean={per_card / st.mean(seconds):.1f} median={per_card / st.median(seconds):.1f}")
    print("  BASELINE before this change: mean 36.50-36.72 s, 1679-1688 tokens/s/card")
census = out / "census_rank_00000.json"
if census.exists():
    report = json.loads(census.read_text())
    print(f"  dispatches/step = {report['dispatches_per_step']}  (was 375,314)")
    print(f"  device_to_host_syncs/step = {report['device_to_host_syncs_per_step']}  (was 228)")
    for row in report["by_site"][:8]:
        print(f"    {row['site']:44s} {row['per_step']:8d}")
PY
  echo "=== ${name} done ==="
}

run census 6 2 "${PRODUCTION[@]}" --operator-census 2
run timing 14 4 "${PRODUCTION[@]}"
run profile 9 4 "${PRODUCTION[@]}" --profile --profile-ranks 0

echo "=== device occupancy from the profiled step ==="
python3 - "${BENCH_ROOT}/profile/profile/rank0" <<'PY' || failed+=(profile-report)
import csv, sys
from pathlib import Path
rows = []
for path in Path(sys.argv[1]).rglob("ASCEND_PROFILER_OUTPUT/step_trace_time.csv"):
    rows += list(csv.DictReader(path.open()))
if not rows:
    print("  no step_trace_time.csv; the offline parser did not export")
    raise SystemExit(1)
for row in rows:
    compute = float(row["Computing"]) / 1e6
    free = float(row["Free"]) / 1e6
    comm = float(row["Communication(Not Overlapped)"]) / 1e6
    total = compute + free + comm
    print(
        f"  step {row['Step']}: computing {compute:6.2f}s  free {free:6.2f}s  comm {comm:5.2f}s"
        f"  -> {100 * free / total:.1f}% idle   (was about 33%)"
    )
PY

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "suite complete: ${BENCH_ROOT}"
