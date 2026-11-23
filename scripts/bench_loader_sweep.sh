#!/usr/bin/env bash
# Why can the host not keep 16 ranks fed, and which knob fixes it?
#
# After the varlen attention change the Ascend step computes in ~29 s instead
# of ~37 s, and data_wait went from 0.02 s to a 5.4 s median with 25 s spikes
# on exactly the loader settings that had been keeping up. The loader's own
# counters say the packing thread is busy ~100% of wall time and spends 87% of
# it blocked in image decode.
#
# The suspected cause is that every image worker thread calls the torchvision
# image processor, and torch.set_num_threads is never called in this
# repository: each small resize fans out over the process-wide intra-op pool
# (OMP_NUM_THREADS=4 from launch.sh), so 16 ranks x 12 threads contend. That
# shows up as moderate CPU utilisation, not saturated cores, which is what we
# measured (59% median).
#
# No model, no device, no collectives: this measures only the host pipeline.
# Each configuration runs RANKS concurrent single-rank processes, because one
# rank alone would have the whole node's cores and would not show contention.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND="${PRETRAIN_BACKEND:-npu}"
export BENCH_DATA="${BENCH_DATA:?Set BENCH_DATA to the release directory}"
export BENCH_ASSETS="${BENCH_ASSETS:?Set BENCH_ASSETS to the pinned model directory}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-loader-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
RANKS="${RANKS:-16}"
WORLD="${WORLD:-32}"
PACKS="${PACKS:-24}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"
echo "cpus=$(nproc) ranks=${RANKS} world=${WORLD} packs=${PACKS}"

if [[ "${PRETRAIN_BACKEND}" == npu ]]; then
  PYTHON="${ROOT}/.venv-core17/bin/python"
  export PYTHONPATH="${ROOT}/.deps/wan/site-packages:${ROOT}/.deps/moss/site-packages:${ROOT}"
  export HCCL_CONNECT_TIMEOUT=300
else
  PYTHON="${ROOT}/.venv-cuda/bin/python"
  export PYTHONPATH="${ROOT}"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# The loader never touches the accelerator here; keep its runtime out of the way.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

COMMON=(--config "${ROOT}/configs/qwen35_2b_wan.yaml"
  --text-assets "${BENCH_ASSETS}/Qwen3.5-2B-Base" --vision-assets "${BENCH_ASSETS}/Qwen3.5-2B-Base"
  --wan-assets "${BENCH_ASSETS}/Wan2.1-T2V-1.3B" --wan-vae-assets "${BENCH_ASSETS}/Wan2.1-VAE"
  --data-root "${BENCH_DATA}" --long-document-policy error
  --tp 1 --global-batch "${WORLD}" --sequence-length 8192
  --output "${BENCH_ROOT}/unused")

run() {
  local name="$1" threads="$2"
  shift 2
  local dir="${BENCH_ROOT}/${name}"
  mkdir -p "${dir}"
  echo "=== ${name}: torch_threads=${threads} $* ==="
  local pids=()
  for ((r = 0; r < RANKS; r++)); do
    OMP_NUM_THREADS="${OMP:-4}" "${PYTHON}" -m scripts.validation.bench_loader_host \
      --label "${name}" --rank "${r}" --world "${WORLD}" --packs "${PACKS}" \
      --torch-threads "${threads}" --output "${dir}/rank_${r}.json" \
      "${COMMON[@]}" "$@" >"${dir}/rank_${r}.log" 2>&1 &
    pids+=($!)
  done
  local bad=0
  for pid in "${pids[@]}"; do wait "${pid}" || bad=$((bad + 1)); done
  [[ ${bad} -gt 0 ]] && echo "  ${bad}/${RANKS} ranks failed; first log tail:" && tail -15 "${dir}/rank_0.log"
  "${PYTHON}" - "${dir}" <<'PY'
import json, statistics as st, sys
from pathlib import Path
rows = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("rank_*.json"))]
if not rows:
    print("  no reports"); raise SystemExit
def med(key):
    return round(st.median(r[key] for r in rows), 3)
total = sum(r["packs_per_second"] for r in rows)
print(f"  ranks_reporting   = {len(rows)}")
print(f"  packs/s per rank  = {med('packs_per_second')}  (node total {round(total, 2)})")
print(f"  seconds per pack  = {round(1 / max(1e-9, med('packs_per_second')), 2)}")
print(f"  images/s per rank = {med('images_per_second')}")
print(f"  image_wait_share  = {med('image_wait_share')}")
print(f"  occupied tok/s    = {med('occupied_tokens_per_second')}")
print(f"  torch_threads     = {rows[0]['settings']['torch_threads']}"
      f" (default was {rows[0]['settings']['torch_threads_before']})")
PY
  echo "=== ${name} done ==="
}

# A step at 8 packs per card must produce 8 packs in the compute time. Measured
# Ascend compute after varlen is ~29 s, so the target is >= 0.28 packs/s/rank.
PROD=(--data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24)
run baseline 0 "${PROD[@]}"
run threads1 1 "${PROD[@]}"
run threads1-wide 1 --data-workers 8 --image-workers 24 --image-prefetch 96 --data-prefetch 24
run wide-only 0 --data-workers 8 --image-workers 24 --image-prefetch 96 --data-prefetch 24
echo "sweep complete: ${BENCH_ROOT}"
