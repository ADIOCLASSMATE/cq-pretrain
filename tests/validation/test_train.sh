#!/usr/bin/env bash
# Real 910B validation: 4 ranks, two updates, same nine frames for both encoders.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-npu/bin/python}"
# Allocated 910B cards may span communication planes on the same host.
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
source "$ROOT/scripts/env_mindspeed.sh"
exec bash scripts/train.sh --config configs/tests/ascend_smoke.yaml "$@"
