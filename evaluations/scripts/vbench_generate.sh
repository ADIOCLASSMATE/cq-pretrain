#!/usr/bin/env bash
set -euo pipefail
UMM_VBENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# UMM_DEVICE controls environment setup; it also supplies training.device.
UMM_DEVICE="${UMM_DEVICE:-npu}"
if [[ "$UMM_DEVICE" == npu ]]; then
  source "$UMM_VBENCH_ROOT/scripts/env_mindspeed.sh"
fi
export PYTHONPATH="$UMM_VBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python3}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-1}" \
  -m evaluations.benchmarks.vbench.generate \
  --set "training.device=$UMM_DEVICE" "$@"
