#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# The frozen legacy module remains unchanged. STOP also guards direct calls.
for ((argument=1; argument<=$#; argument++)); do
    if [[ ${!argument} == --output ]]; then
        next_argument=$((argument + 1))
        if [[ -f ${!next_argument}/_distributed/plan.json ]]; then
            printf '%s\n' 'This dataset uses distributed.py; the legacy entry point is disabled.' >&2
            exit 2
        fi
    fi
done
exec .venv-npu/bin/python -m data.preprocess.openvidhd "$@"
