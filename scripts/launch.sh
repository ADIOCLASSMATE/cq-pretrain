#!/usr/bin/env bash
set -euo pipefail
PRETRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_STACK="${PRETRAIN_STACK:-cuda}"
export PRETRAIN_BACKEND="${PRETRAIN_BACKEND:-cuda}"
PRETRAIN_PYTHON="${PRETRAIN_PYTHON:-${PRETRAIN_ROOT}/.venv-cuda/bin/python}"
export PATH="$(dirname "${PRETRAIN_PYTHON}"):${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NNODES="${NNODES:-${PET_NNODES:-1}}"
export NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
cd "${PRETRAIN_ROOT}"
PRETRAIN_MODULE="${PRETRAIN_MODULE:-training.train}"
if [[ "${NNODES:-1}" == 1 ]]; then
  exec "${PRETRAIN_PYTHON}" -m torch.distributed.run --standalone \
    --nproc-per-node "${NPROC_PER_NODE:-8}" -m "${PRETRAIN_MODULE}" "$@"
else
  export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node launch}"
  : "${NODE_RANK:?Set NODE_RANK for multi-node launch}"
  if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    PRETRAIN_INTERFACE="$("${PRETRAIN_PYTHON}" -c \
      'import os; from utils.network import rendezvous_interface; print(rendezvous_interface(os.environ["MASTER_ADDR"])[0])')"
    export NCCL_SOCKET_IFNAME="=${PRETRAIN_INTERFACE}"
  fi
  exec "${PRETRAIN_PYTHON}" -m torch.distributed.run --nnodes "${NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE:-8}" --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-29500}" \
    -m "${PRETRAIN_MODULE}" "$@"
fi
