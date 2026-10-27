#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-npu/bin/python}"
if [[ $# -eq 0 || " $* " == *" --help "* || " $* " == *" -h "* ]]; then
  exec env TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" -m evaluations.eval "$@"
fi
UMM_SELECTED_BACKEND="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" -m evaluations.eval --print-environment "$@")"
if [[ "$UMM_SELECTED_BACKEND" == mindspeed && " $* " != *" --dry-run "* ]]; then
  source "$ROOT/scripts/env_mindspeed.sh"
else
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
fi
exec "$PYTHON_BIN" -m evaluations.eval "$@"
