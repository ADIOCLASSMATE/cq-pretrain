#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env_mindspeed.sh"

# Fail before loading models if the platform did not expose the requested node.
"$PYTHON_BIN" - <<'PY'
import torch
import torch_npu

count = torch.npu.device_count()
if count != 16:
    raise RuntimeError(f"This job requires 16 visible NPUs on one node; found {count}")
print(f"NPU preflight passed: {count} visible devices; global batch = 16 * 1 * 8 = 128", flush=True)
PY

exec bash "$ROOT/scripts/train.sh" \
  --config "$ROOT/configs/openvid81_832x480_npu16_trial.yaml" \
  --auto-resume "$@"
