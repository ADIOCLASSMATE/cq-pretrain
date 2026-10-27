"""Run explicitly selected GDN and CQ invariants after MindSpeed initialization.

References are used only by this verification command, never by training.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gdn-backend", choices=("native", "mindspeed"), default="mindspeed")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["UMM_VIDEO_DEVICE"] = f"npu:{local_rank}"
    # Capture reference and UMM native fallback decorators before fake FLA
    # registration, exactly as training preserves UMM's fallback functions.
    import tests.test_modeling_umm_video  # noqa: F401
    from configs.config import DEFAULTS
    from training.mcore import MCoreRuntime
    import pytest

    config = deepcopy(DEFAULTS)
    config["distributed"]["backend"] = "mindspeed"
    config["model"]["qwen_gdn_backend"] = args.gdn_backend
    runtime = MCoreRuntime(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = (
        ["tests/test_qwen35_gdn_npu.py", "tests/test_qwen35_mindspeed_integration.py"]
        if args.gdn_backend == "mindspeed"
        else ["tests/test_modeling_umm_video.py", "-k", "npu"]
    )
    code = pytest.main(["-v", "--tb=short", *selected, f"--junitxml={output / f'rank_{local_rank}.xml'}"])
    (output / f"rank_{local_rank}.json").write_text(
        json.dumps(dict(rank=local_rank, exit_code=int(code), runtime=runtime.runtime_contract()), indent=2) + "\n"
    )
    runtime.end_training()
    raise SystemExit(int(code))


if __name__ == "__main__":
    main()
