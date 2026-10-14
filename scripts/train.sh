#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
  exec "$PYTHON_BIN" -m accelerate.commands.launch --config_file "$ACCELERATE_CONFIG" \
    -m training.train "$@"
fi
exec "$PYTHON_BIN" -m training.train "$@"
