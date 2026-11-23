#!/usr/bin/env bash
# One launcher for both accelerators. PRETRAIN_BACKEND selects which runtime is
# set up (cuda: .venv-cuda + NCCL; npu: the pinned CANN 9.0 / MCore 0.17 venv,
# the AscendC GDN vendor package and HCCL). Everything after the backend block
# is shared: torchrun topology, offline Hugging Face, and the training module.
set -euo pipefail
PRETRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND="${PRETRAIN_BACKEND:-cuda}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ "${PRETRAIN_BACKEND}" == npu ]]; then
  export PRETRAIN_STACK="${PRETRAIN_STACK:-core17}"
  if [[ "${PRETRAIN_STACK}" != core17 ]]; then
    echo 'The Ascend backend requires its pinned CANN 9.0 / Core 0.17 stack.' >&2
    exit 2
  fi
  PRETRAIN_PYTHON="${PRETRAIN_PYTHON:-${PRETRAIN_ROOT}/.venv-core17/bin/python}"
  PRETRAIN_CANN_ENV="${PRETRAIN_CANN_ENV:-${PRETRAIN_ROOT}/.deps/Ascend90/cann/set_env.sh}"
  PRETRAIN_ATB_ENV="${PRETRAIN_ATB_ENV:-${PRETRAIN_ROOT}/.deps/Ascend90/nnal/atb/set_env.sh}"
  export PATH="$(dirname "${PRETRAIN_PYTHON}"):${PATH}"
  set +u
  source "${PRETRAIN_CANN_ENV}"
  set -u
  for PRETRAIN_DRIVER_LIB in /usr/local/Ascend/driver/lib64/driver /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
    if [[ -d "${PRETRAIN_DRIVER_LIB}" ]]; then
      export LD_LIBRARY_PATH="${PRETRAIN_DRIVER_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
  done
  PRETRAIN_CXX_ABI="$("${PRETRAIN_PYTHON}" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')"
  set +u
  source "${PRETRAIN_ATB_ENV}" "--cxx_abi=${PRETRAIN_CXX_ABI}"
  set -u
  export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-300}"
  # 300 s is too tight for a preemptible job that can stall behind a restart.
  export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
  export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PRETRAIN_ROOT}/.deps/extensions-${PRETRAIN_STACK}}"
  unset CUDA_VISIBLE_DEVICES TORCH_DEVICE_BACKEND_AUTOLOAD
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTHONPATH="${PRETRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  # The CQ backbone always uses the validated A2 training backend.
  PRETRAIN_DELTA_SITE="${PRETRAIN_ROOT}/.deps/delta-ascendc/site-packages"
  export PYTHONPATH="${PRETRAIN_ROOT}/.deps/wan/site-packages:${PRETRAIN_ROOT}/.deps/moss/site-packages:${PRETRAIN_DELTA_SITE}:${PRETRAIN_ROOT}/.deps/delta64/site-packages:${PYTHONPATH}"
  export ASCEND_CUSTOM_OPP_PATH="${PRETRAIN_DELTA_SITE}/fla_npu/opp/vendors/fla_npu_transformer:${PRETRAIN_DELTA_SITE}/fla_npu/opp${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
  export FLA_NPU_STABLE_ABI=stable
  export TRITON_ALL_BLOCKS_PARALLEL=0
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PRETRAIN_ROOT}/.deps/delta64/triton-cache}"
  # Ascend kernels for every dispatched operator; a silent fallback to the
  # portable expression would be both slower and a different implementation
  # from the one these numbers were gated against.
  export PRETRAIN_REQUIRE_FUSED="${PRETRAIN_REQUIRE_FUSED:-1}"
  export PRETRAIN_NPU_FUSED_NORM="${PRETRAIN_NPU_FUSED_NORM:-1}"
  export PRETRAIN_NPU_FUSED_OPS="${PRETRAIN_NPU_FUSED_OPS:-1}"
  export PRETRAIN_NPU_FUSED_SWIGLU="${PRETRAIN_NPU_FUSED_SWIGLU:-1}"
  export PRETRAIN_NPU_FUSED_CE="${PRETRAIN_NPU_FUSED_CE:-1}"
  PRETRAIN_DEFAULT_RANKS=16
else
  export PRETRAIN_STACK="${PRETRAIN_STACK:-cuda}"
  PRETRAIN_PYTHON="${PRETRAIN_PYTHON:-${PRETRAIN_ROOT}/.venv-cuda/bin/python}"
  export PATH="$(dirname "${PRETRAIN_PYTHON}"):${PATH}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PRETRAIN_ROOT}/.deps/extensions-cuda}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PRETRAIN_ROOT}/.deps/triton-cache}"
  # Measured on L40S and gated against FP64/dense references; each falls back on
  # its own when it is not applicable (TP>1, another GPU, deterministic mode).
  # run.json records the effective value under kernel_options.
  export PRETRAIN_CUDA_FUSED_NORM="${PRETRAIN_CUDA_FUSED_NORM:-1}"
  export PRETRAIN_FLEX_ATTENTION="${PRETRAIN_FLEX_ATTENTION:-1}"
  export PRETRAIN_CUDA_FUSED_CE="${PRETRAIN_CUDA_FUSED_CE:-1}"
  export PRETRAIN_CUDA_FUSED_OPS="${PRETRAIN_CUDA_FUSED_OPS:-1}"
  export PRETRAIN_CUDA_FUSED_SWIGLU="${PRETRAIN_CUDA_FUSED_SWIGLU:-1}"
  export PYTHONPATH="${PRETRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  PRETRAIN_DEFAULT_RANKS=8
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NNODES="${NNODES:-${PET_NNODES:-1}}"
export NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
cd "${PRETRAIN_ROOT}"
PRETRAIN_MODULE="${PRETRAIN_MODULE:-training.train}"
PRETRAIN_RANKS="${NPROC_PER_NODE:-${PRETRAIN_DEFAULT_RANKS}}"
if [[ "${NNODES:-1}" == 1 ]]; then
  exec "${PRETRAIN_PYTHON}" -m torch.distributed.run --standalone \
    --nproc-per-node "${PRETRAIN_RANKS}" -m "${PRETRAIN_MODULE}" "$@"
else
  export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node launch}"
  : "${NODE_RANK:?Set NODE_RANK for multi-node launch}"
  if [[ "${PRETRAIN_BACKEND}" != npu && -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    PRETRAIN_INTERFACE="$("${PRETRAIN_PYTHON}" -c \
      'import os; from utils.network import rendezvous_interface; print(rendezvous_interface(os.environ["MASTER_ADDR"])[0])')"
    export NCCL_SOCKET_IFNAME="=${PRETRAIN_INTERFACE}"
  fi
  exec "${PRETRAIN_PYTHON}" -m torch.distributed.run --nnodes "${NNODES}" \
    --nproc-per-node "${PRETRAIN_RANKS}" --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-29500}" \
    -m "${PRETRAIN_MODULE}" "$@"
fi
