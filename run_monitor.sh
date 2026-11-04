#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-npu/bin/python}"
exec "$PYTHON_BIN" "$ROOT/scripts/push_site_experiments.py" --watch "$@"
