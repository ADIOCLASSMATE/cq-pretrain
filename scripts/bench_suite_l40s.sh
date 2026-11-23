#!/usr/bin/env bash
# What does an L40S step actually cost, and does it have the Ascend problem?
#
# The Ascend step turned out to be host-bound: 351k kernel launches per update,
# a third of the device time idle, and one per-visual Python attention loop
# responsible for most of it. Whether L40S shares that is unmeasured. Its
# launches are cheaper and its execution runs further ahead, which is exactly
# why it could hide the same op count -- so the count has to be read, not
# inferred from the fact that L40S looked fine.
#
# Three phases on node rank 0's report:
#   probe   which optional kernels this image actually has (flash-attn is NOT
#           in .venv-cuda; ops/cuda_fused silently falls back when it is absent,
#           so "PRETRAIN_CUDA_FUSED_NORM=1" does not mean a fused norm ran)
#   census  dispatched operators per update, attributed to call sites
#   timing  the comparable number, same global batch and card count as Ascend
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=cuda PRETRAIN_STACK=cuda
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WORLD="${WORLD:-32}"
export BENCH_DATA="${BENCH_DATA:-/inspire/hdd/global_user/wanjiaxin-253108030048/datasets/mm_pretrain/stage1-512-v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-${ROOT}/.assets}"
# Not a timestamp: each node would compute its own and the run would land in
# as many directories as there are nodes. The job name is identical everywhere.
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-l40s-${PET_JOB_NAME:-${JOB_NAME:-current}}}"
NODE="${PET_NODE_RANK:-${NODE_RANK:-0}}"
mkdir -p "${BENCH_ROOT}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BENCH_ROOT}/../cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${BENCH_ROOT}/../cache/inductor}"
[[ "${NODE}" == 0 ]] && echo "suite root ${BENCH_ROOT}  node=${NODE} cpus=$(nproc)"

if [[ "${NODE}" == 0 ]]; then
  echo "=== probe: which optional kernels does this image have? ==="
  "${ROOT}/.venv-cuda/bin/python" - <<'PY' 2>&1 | tee "${BENCH_ROOT}/kernels.txt"
import importlib, json
found = {}
for name in ("flash_attn", "flash_attn.ops.triton.layer_norm", "flash_attn.ops.triton.cross_entropy",
             "flash_attn.flash_attn_interface", "fla", "fla.modules.fused_norm_gate",
             "fla.ops.gated_delta_rule", "liger_kernel", "apex", "fused_weight_gradient_mlp_cuda",
             "transformer_engine"):
    try:
        importlib.import_module(name)
        found[name] = True
    except Exception as error:
        found[name] = f"{type(error).__name__}"
import torch
found["torch"] = torch.__version__
found["flex_attention"] = hasattr(getattr(torch.nn.attention, "flex_attention", None), "flex_attention")
print(json.dumps(found, indent=2))
PY
fi

failed=()
# Numerics before timing: if the CUDA varlen kernel disagrees with the loop it
# replaces, no throughput number from it is worth collecting. One rank, no
# collectives; it runs on node 0 and the others wait at the next phase.
if [[ "${NODE}" == 0 ]]; then
  echo "=== gate: flash_attn varlen against the per-segment loop ==="
  NPROC_PER_NODE=1 PRETRAIN_MODULE=scripts.validation.validate_varlen \
    bash "${ROOT}/scripts/launch.sh" --output "${BENCH_ROOT}/varlen_gate.json" \
    >"${BENCH_ROOT}/gate.log" 2>&1
  if [[ $? -ne 0 ]]; then
    echo "GATE FAILED -- numerics first; no timing collected."
    tail -40 "${BENCH_ROOT}/gate.log"
    exit 1
  fi
  python3 -c "
import json, sys
report = json.load(open('${BENCH_ROOT}/varlen_gate.json'))
print('  passed =', report['passed'], ' device =', report.get('device'))
for name, case in report['cases'].items():
    agree = case['loop_vs_varlen']
    print(f\"  {name:22s} loop_out={case['loop']['output']:.5f} \"
          f\"agree_out={agree['output']:.1e} agree_dk={agree['dk']:.1e}\")
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

# Production recompute and the loader settings the Ascend side needed. The
# census costs real time on the updates it covers, so it runs few steps and is
# never the configuration quoted for throughput.
PRODUCTION=(--checkpoint --keep-attention-activations 6
  --data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24)

run census 8 4 "${PRODUCTION[@]}" --operator-census 2 --profile-ranks 0
if [[ "${NODE}" == 0 ]]; then
  python3 - "${BENCH_ROOT}/census/census_rank_00000.json" <<'PY' || failed+=(census-report)
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("  no census written"); raise SystemExit(1)
c = json.loads(path.read_text())
print("  dispatches_per_step =", c["dispatches_per_step"])
print("  syncs_per_step      =", c["device_to_host_syncs_per_step"])
files = {}
for row in c["by_site"]:
    files[row["site"].split(":")[0]] = files.get(row["site"].split(":")[0], 0) + row["per_step"]
print("  --- by file, per step ---")
for name, count in sorted(files.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {name:44s}{count:9d}{100 * count / c['dispatches_per_step']:6.1f}%")
print("  --- top sites ---")
for row in c["by_site"][:10]:
    print(f"  {row['site']:44s}{row['per_step']:9d}  written={row['written_on_this_line']}")
PY
fi

run timing 20 6 "${PRODUCTION[@]}"

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
[[ "${NODE}" == 0 ]] && echo "suite complete: ${BENCH_ROOT}"
exit 0
