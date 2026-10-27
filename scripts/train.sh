#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-npu/bin/python}"
UMM_SELECTED_BACKEND="$("$PYTHON_BIN" -m utils.run_logging --print-backend "$@")"
if [[ "$UMM_SELECTED_BACKEND" == mindspeed ]]; then
  source "$ROOT/scripts/env_mindspeed.sh"
fi
exec "$PYTHON_BIN" -m utils.run_logging "$@"
