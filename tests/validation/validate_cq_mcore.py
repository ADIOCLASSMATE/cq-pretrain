"""Run explicitly selected GDN and CQ invariants after MindSpeed initialization.

References are used only by this verification command, never by training.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["UMM_VIDEO_DEVICE"] = f"npu:{local_rank}"
    # Capture native vision/reference functions before MindSpeed patches FLA.
    import tests.test_modeling_umm_video  # noqa: F401
    from distributed.megatron import prepare_npu
    import pytest
    prepare_npu()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = ["tests/test_qwen35_gdn_npu.py", "tests/test_qwen35_mindspeed_integration.py"]
    code = pytest.main(["-v", "--tb=short", *selected, f"--junitxml={output / f'rank_{local_rank}.xml'}"])
    (output / f"rank_{local_rank}.json").write_text(
        json.dumps(dict(rank=local_rank, exit_code=int(code), backends=["ascendc", "mindspeed"]), indent=2) + "\n"
    )
    raise SystemExit(int(code))


if __name__ == "__main__":
    main()
