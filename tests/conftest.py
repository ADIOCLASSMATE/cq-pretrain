"""Keep incidental Ascend compiler output out of the repository root."""

import os
from pathlib import Path
import socket


# CPU regressions can import torch_npu through optional library hooks too.
# Device tests use the same directory as scripts/env_mindspeed.sh.
_work = Path(__file__).resolve().parents[1] / ".cache/ascend-work" / (
    os.environ.get("HOSTNAME") or socket.gethostname()
)
os.environ.setdefault("ASCEND_WORK_PATH", str(_work))
Path(os.environ["ASCEND_WORK_PATH"]).mkdir(parents=True, exist_ok=True)
