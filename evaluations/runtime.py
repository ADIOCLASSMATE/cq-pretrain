"""Shared lazy inference runtime, aligned with the training forward environment."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import torch

from training.train import load_model, load_processor, register_device
from training.loss import LazyVideoCodec
from utils.checkpoint import preserved_rng_state


def prepare_runtime(config):
    """Match MCore's forward environment without creating an optimizer or DDP."""
    if config["training"]["deterministic"]:
        os.environ["HCCL_DETERMINISTIC"] = "True"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        torch.use_deterministic_algorithms(True)
    device = config["training"]["device"]
    register_device(device)
    if config["distributed"]["backend"] != "mindspeed":
        if device in {"cuda", "npu"}:
            getattr(torch, device).set_device(int(os.environ.get("LOCAL_RANK", 0)))
            return torch.device(device, int(os.environ.get("LOCAL_RANK", 0))), None
        return torch.device(device), None

    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "configs/deps.core17.lock.json").read_text())
    cann_version = Path(os.environ.get("ASCEND_HOME_PATH", "/missing-cann")) / "opp/version.info"
    if (not cann_version.is_file()
            or f"Version={lock['runtime']['cann']}" not in cann_version.read_text().splitlines()):
        raise RuntimeError(
            "Use scripts/evaluate.sh or source scripts/env_mindspeed.sh for CQ-pinned CANN"
        )
    from importlib.metadata import version
    import platform

    if platform.python_version() != lock["runtime"]["python"] or any(
        version(package) != lock["runtime"][key]
        for package, key in (("torch", "torch"), ("torch-npu", "torch_npu"))
    ):
        raise RuntimeError("Use the CQ-pinned Python, torch and torch_npu in .venv-npu")
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    import models.vision_encoder  # capture native vision before MindSpeed patches
    from distributed.megatron import prepare_npu
    from models.backbone_qwen35 import resolve_gdn_backend_identity
    with redirect_stdout(sys.stderr):
        prepare_npu()
        identity = resolve_gdn_backend_identity(config["model"]["qwen_gdn_backend"])
    print(json.dumps({"qwen_gdn": identity, "model_dtype": "torch.bfloat16"}), file=sys.stderr)
    return torch.device("npu", torch.npu.current_device()), torch.bfloat16


class EvaluationVideoCodec(LazyVideoCodec):
    """VAE construction cannot change the next CFG/flow or sampling RNG draw."""

    def get(self):
        if self.codec is None:
            with preserved_rng_state():
                return super().get()
        return self.codec


class EvaluationRuntime:
    """One lazy UMM/processor/VAE per worker; score-only jobs need none of them."""

    def __init__(self, config, checkpoint=None):
        self.config = deepcopy(config)
        if checkpoint is not None:
            self.config["model"]["checkpoint"] = str(checkpoint)
        self._device = self._model_dtype = None
        self._processor = self._model = self._codec = None
        self.backend_identity = None

    def _prepare(self):
        if self._device is None:
            self._device, self._model_dtype = prepare_runtime(self.config)

    @property
    def device(self):
        self._prepare()
        return self._device

    @property
    def processor(self):
        self._prepare()
        if self._processor is None:
            self._processor = load_processor(self.config)
        return self._processor

    def get_model(self):
        if self._model is None:
            self._prepare()
            # Construction must not affect caller RNG or the next task's draws.
            with preserved_rng_state():
                torch.manual_seed(self.config["sampling"]["seed"])
                _ = self.processor
                model = load_model(self.config)
                compile_function = torch.compile
                try:
                    self.backend_identity = model.backbone.gdn_backend_identity()
                finally:
                    torch.compile = compile_function
                if self._model_dtype == torch.bfloat16:
                    from models.qwen35_layers import cast_for_training
                    model = cast_for_training(model, self.device)
                else:
                    model = model.to(device=self.device, dtype=self._model_dtype)
                self._model = model.eval()
        return self._model

    @property
    def model(self):
        return self.get_model()

    @property
    def codec(self):
        if self._codec is None:
            self._codec = EvaluationVideoCodec(self.config, self.device)
        return self._codec

    def get_codec(self):
        return self.codec.get()

    def autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config["training"]["mixed_precision"] == "bf16",
        )
