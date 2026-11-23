"""The two accelerator backends stay separated, and the dispatch picks one.

Rules this file enforces, none of which need a GPU:

1. Vendor runtime calls (torch_npu, AscendC, MindSpeed, TorchAIR, HCCL, Apex,
   flash-attn, FLA, FlexAttention, torch.cuda) appear only in the per-backend
   modules and in the two places that must know the backend: the selector and
   the process-group/stack preparation. Shared model, data and training code
   asks ``ops.fused`` / ``ops.gated_delta`` / ``distributed.accel`` instead.
2. ``ops.fused`` resolves to the module of the selected backend.
3. Each backend accepts only its own pinned dependency stack.
"""

import importlib
import io
import os
import re
import sys
import tokenize
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

VENDOR = {
    "torch_npu": re.compile(r"\bimport torch_npu\b|\btorch_npu\.|\btorch\.npu\b"),
    "AscendC GDN": re.compile(r"\bfla_npu\b"),
    "MindSpeed": re.compile(r"\bmindspeed\b|\bmegatron_adaptor\b", re.IGNORECASE),
    "TorchAIR": re.compile(r"\btorchair\b"),
    "HCCL": re.compile(r"\bHCCL_|\bhccl\b"),
    "MC2": re.compile(r"distributed\.mc2"),
    "Apex wgrad": re.compile(r"\bfused_weight_gradient_mlp_cuda\b"),
    "flash-attn": re.compile(r"\bflash_attn\b"),
    "FLA": re.compile(r"\bfrom fla\.|\bimport fla\b"),
    "FlexAttention": re.compile(r"flex_attention\("),
    "torch.cuda": re.compile(r"\btorch\.cuda\b"),
}

# Modules that are allowed to name a vendor runtime, and why.
BACKEND_MODULES = {
    "distributed/accel.py",  # the selector itself
    "distributed/megatron.py",  # process group and per-backend stack preparation
    "distributed/mc2.py",  # Ascend MatmulAllReduce adapter
    "ops/cuda_fused.py",
    "ops/cuda_gated_delta.py",
    "ops/cuda_cross_entropy.py",
    "ops/cuda_compile.py",
    "ops/flex_cq.py",
    "ops/npu_fused.py",
    "ops/npu_gated_delta.py",
    "ops/npu_attention.py",
    "ops/npu_cross_entropy.py",
    "ops/npu_compile.py",
}
SHARED_FOLDERS = ("ops", "distributed", "models", "data", "training", "utils", "configs")


def code_only(text):
    """Blank out comments and string literals, keeping every byte offset.

    Prose may discuss a backend; what must not appear in shared code is a call
    into one. Scanning raw text cannot tell the two apart.
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


def _reset_backend_modules():
    for name in [m for m in sys.modules if m.startswith(("ops.", "distributed.accel"))]:
        sys.modules.pop(name, None)


class BackendDispatchTests(unittest.TestCase):
    def tearDown(self):
        # Drop the per-backend imports so the next import observes the real
        # environment rather than a backend this test forced.
        _reset_backend_modules()

    def test_vendor_runtimes_are_confined_to_their_backend_modules(self):
        offenders = []
        for folder in SHARED_FOLDERS:
            for path in sorted((ROOT / folder).rglob("*.py")):
                relative = str(path.relative_to(ROOT))
                if relative in BACKEND_MODULES:
                    continue
                text = code_only(path.read_text())
                for label, rx in VENDOR.items():
                    match = rx.search(text)
                    if match:
                        line = text[: match.start()].count("\n") + 1
                        offenders.append(f"{relative}:{line} {label}")
        self.assertFalse(offenders, "Vendor runtime called outside a backend module: " + "; ".join(offenders))

    def test_every_backend_module_is_listed(self):
        """A new ops/<backend>_*.py must be added to the allowlist deliberately."""
        present = {
            str(path.relative_to(ROOT))
            for path in (ROOT / "ops").glob("*.py")
            if path.name.startswith(("cuda_", "npu_"))
        }
        self.assertLessEqual(present, BACKEND_MODULES, f"Unlisted backend modules: {sorted(present - BACKEND_MODULES)}")

    def test_fused_dispatch_follows_the_selected_backend(self):
        for backend in ("cuda", "npu"):
            with self.subTest(backend=backend), patch.dict(os.environ, PRETRAIN_BACKEND=backend):
                _reset_backend_modules()
                accel = importlib.reload(importlib.import_module("distributed.accel"))
                self.assertEqual(accel.BACKEND, backend)
                self.assertEqual(accel.IS_NPU, backend == "npu")
                self.assertEqual(accel.dist_backend(), "hccl" if backend == "npu" else "nccl")
                fused = importlib.import_module("ops.fused")
                self.assertEqual(fused._backend.__name__, f"ops.{backend}_fused")

    def test_each_backend_accepts_only_its_pinned_stack(self):
        import utils.dependencies as dependencies

        for backend, stack in (("cuda", "cuda"), ("npu", "core17")):
            with self.subTest(backend=backend), patch.dict(os.environ, PRETRAIN_BACKEND=backend):
                os.environ.pop("PRETRAIN_STACK", None)
                accel = importlib.reload(importlib.import_module("distributed.accel"))
                self.assertEqual(accel.BACKEND, backend)
                self.assertEqual(dependencies.name(), stack)
                self.assertEqual(dependencies.lock_path().name, f"deps.{stack}.lock.json")
                self.assertTrue(dependencies.lock_path().exists())
                with patch.dict(os.environ, PRETRAIN_STACK="something-else"), self.assertRaises(ValueError):
                    dependencies.name()

    def test_production_synthetic_cli_keeps_model_configuration(self):
        from training.train import parse_args
        from configs.config import RunConfig

        path = "configs/qwen35_2b_wan.yaml"
        before = RunConfig.load(path).to_dict()
        with patch.dict(os.environ):
            args = parse_args(
                [
                    "--config",
                    path,
                    "--synthetic",
                    "--tp",
                    "1",
                    "--global-batch",
                    "1",
                    "--log-all-ranks",
                    "--output",
                    "/tmp/cq-test-unused",
                ]
            )
        self.assertTrue(args.synthetic and args.log_all_ranks)
        self.assertEqual((args.tp, args.global_batch), (1, 1))
        self.assertEqual(RunConfig.load(path).to_dict(), before)
