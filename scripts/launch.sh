#!/usr/bin/env bash
set -euo pipefail
PRETRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_STACK="${PRETRAIN_STACK:-cuda}"
export PRETRAIN_BACKEND="${PRETRAIN_BACKEND:-cuda}"
PRETRAIN_PYTHON="${PRETRAIN_PYTHON:-${PRETRAIN_ROOT}/.venv/bin/python}"
export PATH="$(dirname "${PRETRAIN_PYTHON}"):${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PRETRAIN_ROOT}/.deps/extensions-cuda}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PRETRAIN_ROOT}/.deps/triton-cache}"
export PYTHONPATH="${PRETRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
cd "${PRETRAIN_ROOT}"
PRETRAIN_MODULE="${PRETRAIN_MODULE:-training.train}"
if [[ "${NNODES:-1}" == 1 ]]; then
  exec "${PRETRAIN_PYTHON}" -m torch.distributed.run --standalone \
    --nproc-per-node "${NPROC_PER_NODE:-8}" -m "${PRETRAIN_MODULE}" "$@"
else
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node launch}"
  : "${NODE_RANK:?Set NODE_RANK for multi-node launch}"
  exec "${PRETRAIN_PYTHON}" -m torch.distributed.run --nnodes "${NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE:-8}" --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-29500}" \
    -m "${PRETRAIN_MODULE}" "$@"
fi
