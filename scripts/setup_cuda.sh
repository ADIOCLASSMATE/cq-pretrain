#!/usr/bin/env bash
# Use CPU-downloaded wheels and the image's ABI-matched torch/Apex runtime.
set -euo pipefail
CQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${CQ_ROOT}"
CQ_PYTHON="${CQ_PYTHON:-/usr/bin/python3.12}"
CQ_VENV="${CQ_VENV:-${CQ_ROOT}/.venv-cuda}"
CQ_WHEELS="${CQ_WHEELS:-${CQ_ROOT}/.deps/cuda-wheels}"
"${CQ_PYTHON}" -m venv --system-site-packages "${CQ_VENV}"
"${CQ_VENV}/bin/python" -m pip install --no-index --no-deps --find-links "${CQ_WHEELS}" \
  -r "${CQ_ROOT}/environment.cuda.txt"
PRETRAIN_BACKEND=cuda "${CQ_VENV}/bin/python" - <<'PY'
import importlib.util
import json
import torch
import triton
import fused_weight_gradient_mlp_cuda as wgrad
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from megatron.core.optimizer import AdamOptimizerConfig
from utils.dependencies import installed_versions

assert torch.cuda.is_available(), 'CUDA device required for environment validation'
assert torch.__version__ == '2.7.0a0+ecf3bae40a.nv25.02', 'Use the pinned NGC 25.02 image'
assert triton.__version__ == '3.2.0', 'Reuse the image pytorch-triton; do not overlay triton'
assert importlib.util.find_spec('torch_npu') is None
assert callable(wgrad.wgrad_gemm_accum_fp32)
print(json.dumps(dict(cuda=torch.version.cuda, torch=torch.__version__, triton=triton.__version__,
    device=torch.cuda.get_device_name(), wgrad=wgrad.__file__,
    mcore_optimizer=AdamOptimizerConfig.__module__, installed=installed_versions()), indent=2))
PY
