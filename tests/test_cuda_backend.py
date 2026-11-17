"""CUDA runtime regressions that can be checked without a GPU."""
import io
import os
import re
import tokenize
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every one of these is a *use* of the Ascend runtime. Strings such as
# `is_torch_npu_available`, which only tell Transformers that NPU is absent,
# are deliberately not matched.
ASCEND = {
    "torch_npu import": re.compile(r"\bimport torch_npu\b"),
    "torch.npu attribute": re.compile(r"\btorch\.npu\."),
    "AscendC kernels": re.compile(r"[Aa]scendC|from fla_npu"),
    "MindSpeed": re.compile(r"[Mm]indspeed|MindSpeed|megatron_adaptor"),
    "TorchAIR": re.compile(r"\btorchair\b"),
    "HCCL": re.compile(r"\bHCCL_|hccl"),
    "MC2": re.compile(r"distributed\.mc2"),
}


def code_only(text):
    """Blank out comments and string literals, keeping every byte offset.

    Prose is allowed to say the Ascend backend was removed; what must not
    survive is a call into it. Scanning raw text cannot tell the two apart.
    """
    starts, pos = [], 0
    for line in text.splitlines(keepends=True):
        starts.append(pos)
        pos += len(line)
    masked = list(text)
    skip = {tokenize.COMMENT, tokenize.STRING}
    skip |= {getattr(tokenize, n) for n in ("FSTRING_MIDDLE",) if hasattr(tokenize, n)}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in skip:
                continue
            a = starts[tok.start[0] - 1] + tok.start[1]
            b = starts[tok.end[0] - 1] + tok.end[1]
            for i in range(a, min(b, len(masked))):
                if masked[i] != "\n":
                    masked[i] = " "
    except (tokenize.TokenError, IndentationError):
        return text
    return "".join(masked)


def test_no_ascend_runtime_remains_on_the_training_path():
    offenders = []
    for folder in ("ops", "distributed", "models", "data", "training", "utils", "configs"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            text = code_only(path.read_text())
            for label, rx in ASCEND.items():
                match = rx.search(text)
                if match:
                    line = text[: match.start()].count("\n") + 1
                    offenders.append(f"{path.relative_to(ROOT)}:{line} {label}")
    assert not offenders, "Ascend runtime still called: " + "; ".join(offenders)


def test_only_the_cuda_stack_is_accepted():
    from utils.dependencies import lock_path, name

    with patch.dict(os.environ):
        os.environ.pop("PRETRAIN_STACK", None)
        assert name() == "cuda"
        assert lock_path().name == "deps.cuda.lock.json"
    with patch.dict(os.environ, PRETRAIN_STACK="core17"), pytest.raises(ValueError):
        name()


def test_production_synthetic_cli_keeps_model_configuration():
    from training.train import parse_args
    from configs.config import RunConfig

    path = "configs/qwen35_2b_wan.yaml"
    before = RunConfig.load(path).to_dict()
    with patch.dict(os.environ):
        args = parse_args(['--config', path, '--synthetic', '--tp', '1', '--global-batch', '1',
                           '--log-all-ranks', '--output', '/tmp/cq-test-unused'])
    assert args.synthetic and args.log_all_ranks
    assert args.tp == 1 and args.global_batch == 1
    assert RunConfig.load(path).to_dict() == before
