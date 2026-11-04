#!/usr/bin/env bash
set -euo pipefail
PRETRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_STACK="${PRETRAIN_STACK:-core17}"
if [[ "${PRETRAIN_STACK}" != core17 ]]; then
  echo 'This project requires its pinned CANN 9.0/Core 0.17 stack.' >&2
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-300}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-300}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PRETRAIN_ROOT}/.deps/extensions-${PRETRAIN_STACK}}"
unset CUDA_VISIBLE_DEVICES TORCH_DEVICE_BACKEND_AUTOLOAD
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${PRETRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# The active CQ backbone always uses the validated A2 training backend.
PRETRAIN_DELTA_SITE="${PRETRAIN_ROOT}/.deps/delta-ascendc/site-packages"
export PYTHONPATH="${PRETRAIN_ROOT}/.deps/wan/site-packages:${PRETRAIN_ROOT}/.deps/moss/site-packages:${PRETRAIN_DELTA_SITE}:${PRETRAIN_ROOT}/.deps/delta64/site-packages:${PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH="${PRETRAIN_DELTA_SITE}/fla_npu/opp/vendors/fla_npu_transformer:${PRETRAIN_DELTA_SITE}/fla_npu/opp${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export FLA_NPU_STABLE_ABI=stable
export TRITON_ALL_BLOCKS_PARALLEL=0
export TRITON_CACHE_DIR="${PRETRAIN_ROOT}/.deps/delta64/triton-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "${PRETRAIN_ROOT}"
PRETRAIN_MODULE="${PRETRAIN_MODULE:-training.train}"
if [[ "${NNODES:-1}" == 1 ]]; then
  exec "${PRETRAIN_PYTHON}" -m torch.distributed.run --standalone \
    --nproc-per-node "${NPROC_PER_NODE:-8}" -m "${PRETRAIN_MODULE}" "$@"
else
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node launch}"
  : "${NODE_RANK:?Set NODE_RANK for multi-node launch}"
  exec "${PRETRAIN_PYTHON}" -m torch.distributed.run --nnodes "${NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE:-16}" --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-29500}" \
    -m "${PRETRAIN_MODULE}" "$@"
fi
