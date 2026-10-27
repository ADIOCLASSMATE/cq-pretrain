#!/usr/bin/env bash
set -euo pipefail
UMM_VBENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$UMM_VBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Select a separate official VBench/CUDA environment; do not source the NPU environment.
exec "${VBENCH_PYTHON:-python3}" -m evaluations.benchmarks.vbench.score "$@"
