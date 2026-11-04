#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv-npu/bin/python -m data.preprocess.distributed "$@"
