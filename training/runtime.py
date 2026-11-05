"""MindSpeed / Megatron Core DP runtime; ordinary UMM modules stay intact.

There is no Accelerate engine on this path. Its dataloader utility is reused
only for deterministic DP batch sharding, including weighted samplers.
"""

from contextlib import contextmanager, nullcontext
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

from distributed.communication import (
    OrderedParameterGathers,
    OrderedGradientBuckets,
)
from training.optimizer import parameter_groups, wsd_multiplier
import distributed.megatron as P

ROOT = Path(__file__).resolve().parents[1]


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def _optimizers(optimizer):
    return getattr(optimizer, "chained_optimizers", [optimizer])


def _gdn_contract(identity):
    # Relocating an identical checkout is harmless; code, backend and package
    # versions are strict numerical invariants. Keep the absolute file in logs.
    return {key: value for key, value in identity.items() if key != "file"}


class WSDScheduler:
    """Use UMM's exact step-based WSD convention for every optimizer role."""

    def __init__(self, optimizer, config):
        self.optimizer, self.config = optimizer, config
        self.base_lrs = [group["max_lr"] for group in optimizer.param_groups]
        self.last_epoch = 0
        self._apply()

    def _apply(self):
        opt = self.config["optimizer"]
        scale = wsd_multiplier(
            self.last_epoch,
            opt["warmup_steps"],
            opt["decay_steps"],
            self.config["training"]["max_train_steps"],
            opt["min_lr_ratio"],
        )
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * scale

    def step(self):
        self.last_epoch += 1
        self._apply()

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        if state["base_lrs"] != self.base_lrs:
            raise ValueError("Optimizer role learning rates changed on resume")
        self.last_epoch = state["last_epoch"]
        self._apply()


class Optimizer:
    """Expose the loop's update API while MCore owns masters, Adam and clipping."""

    def __init__(self, runtime, core):
        self.runtime, self.core = runtime, core

    @property
    def param_groups(self):
        return self.core.param_groups

    def zero_grad(self, set_to_none=True):
        self.core.zero_grad(set_to_none=set_to_none)
        self.runtime.model.zero_grad_buffer()
        self.runtime.gradients.begin()

    def step(self):
        runtime = self.runtime
        runtime.last_absent_gradients = runtime.gradients.finish_backward()
        P.finish_grad_sync(runtime.model)
        success, norm, _ = self.core.step()
        runtime.optimizer_step_was_skipped = not success
        runtime.last_grad_norm = float(norm) if norm is not None else None
        if norm is not None and not bool(torch.isfinite(torch.as_tensor(norm))):
            raise FloatingPointError("MCore produced a non-finite gradient norm")


class MCoreRuntime:
    is_mcore = True
    distributed_type = "MEGATRON"
    mixed_precision = "bf16"

    def __init__(self, config):
        import subprocess
        import torch_npu  # noqa: F401

        self.config = config
        cann_home = Path(os.environ.get("ASCEND_HOME_PATH", "/missing-cann"))
        cann_version = cann_home / "opp/version.info"
        if not cann_version.is_file() or "Version=9.0.0" not in cann_version.read_text().splitlines():
            raise RuntimeError("CANN 9.0.0 is required; source scripts/env_mindspeed.sh")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.npu.set_device(local_rank)
        self.device = torch.device("npu", local_rank)
        self.lock = json.loads((ROOT / "configs/deps.core17.lock.json").read_text())
        import platform
        if platform.python_version() != self.lock["runtime"]["python"]:
            raise RuntimeError("Use the CQ-pinned Python in .venv-npu")
        for name, source in self.lock["sources"].items():
            path = ROOT / ".deps/core17" / name
            revision = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != source["commit"]:
                raise RuntimeError(f"{name} differs from the CQ dependency lock")
            subprocess.run(
                ["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"], check=True
            )
            sys.path.insert(0, str(path))
        from importlib.metadata import version

        for package, key in (("torch", "torch"), ("torch-npu", "torch_npu")):
            if version(package) != self.lock["runtime"][key]:
                raise RuntimeError(f"Use .venv-npu matching the CQ runtime: {package}")
        if config["model"].get("qwen_gdn_backend", "ascendc") == "mindspeed":
            gdn_runtime = self.lock["gdn_runtime"]
            for package, expected in (
                ("triton-ascend", gdn_runtime["triton_ascend"]["version"]),
            ):
                if version(package) != expected:
                    raise RuntimeError(
                        f"MindSpeed CQ GDN requires the locked {package}=={expected}; "
                        "run scripts/setup_mindspeed.py"
                    )
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from models.backbone_qwen35 import layer_config, resolve_gdn_backend_identity
        import models.vision_encoder  # capture native vision before MindSpeed patches
        architecture = Qwen3_5Config.from_pretrained(config["model"]["qwen_path"], local_files_only=True)
        P.initialize(1, layer_config(architecture), timeout_minutes=config["distributed"]["timeout_minutes"])
        parallel_state = P.state()
        self.gdn_identity = resolve_gdn_backend_identity(config["model"]["qwen_gdn_backend"])
        logging.getLogger(__name__).info("CQ GDN implementation: %s", self.gdn_identity)
        self.dp_group = parallel_state.get_data_parallel_group()
        self.process_index = parallel_state.get_data_parallel_rank()
        self.num_processes = parallel_state.get_data_parallel_world_size()
        self.is_main_process = self.process_index == 0
        self.state = SimpleNamespace(deepspeed_plugin=None)
        self.sync_gradients, self.optimizer_step_was_skipped = True, False
        self.micro_step, self.last_grad_norm, self.last_absent_gradients = 0, None, 0
        self._models, self._optimizers, self._schedulers = [], [], []

    def prepare_model(self, model):
        from megatron.core.optimizer import AdamOptimizerConfig
        from megatron.core.optimizer.optimizer_config import ParamKey
        from models.qwen35_layers import cast_for_training

        self._audit_gdn_backend(model)
        cast_for_training(model, self.device)
        settings = self.config["distributed"]
        groups = parameter_groups(model, self.config)
        overrides = {}
        for group in groups:
            # DDP prefixes native names with module.; exact tuples avoid role ambiguity.
            overrides[
                ParamKey(name=tuple("module." + name for name in group["param_names"]))
            ] = {
                "lr": group["lr"],
                "max_lr": group["lr"],
                "min_lr": group["lr"] * self.config["optimizer"]["min_lr_ratio"],
                "weight_decay": group["weight_decay"],
                "role": group["role"],
            }
        opt = self.config["optimizer"]
        opt_config = AdamOptimizerConfig(
            optimizer="adam",
            lr=opt["backbone_lr"],
            min_lr=opt["backbone_lr"] * opt["min_lr_ratio"],
            weight_decay=opt["weight_decay"],
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            clip_grad=self.config["training"]["max_grad_norm"],
            adam_beta1=opt["betas"][0],
            adam_beta2=opt["betas"][1],
            adam_eps=opt["eps"],
        )
        self.model, core = P.make_optimizer(
            model, opt["backbone_lr"], bucket_size=settings["bucket_size"],
            overlap_param_gather=settings["overlap_param_gather"],
            overlap_grad_reduce=settings["overlap_grad_reduce"],
            optimizer_config=opt_config, config_overrides=overrides,
        )
        self.model.broadcast_params()
        self.gathers = OrderedParameterGathers(self.model)
        self.gradients = OrderedGradientBuckets(self.model)
        self.optimizer = Optimizer(self, core)
        self.scheduler = WSDScheduler(self.optimizer, self.config)
        self._models, self._optimizers, self._schedulers = (
            [self.model],
            [self.optimizer],
            [self.scheduler],
        )
        names = {id(p): name for name, p in model.named_parameters()}
        layout = [
            [
                (names[id(p)], list(p.shape), str(p.dtype))
                for bucket in group.buckets
                for p in bucket.params_list
            ]
            for group in self.gradients.groups
        ]
        self.layout_digest = hashlib.sha256(
            json.dumps(layout, sort_keys=True).encode()
        ).hexdigest()
        self._audit_gdn_backend(model)
        return self.model, self.optimizer, self.scheduler

    def _audit_gdn_backend(self, model):
        # The distributed-runtime smoke test uses a plain toy module. Real UMM
        # models must expose and audit every GDN layer's explicit backend.
        backbone = getattr(model, "backbone", None)
        audit = getattr(backbone, "gdn_backend_identity", None)
        if audit is None:
            if hasattr(getattr(model, "config", None), "qwen_config"):
                raise RuntimeError("UMM backbone does not expose its CQ GDN runtime identity")
            return
        actual = audit()
        if _gdn_contract(actual) != _gdn_contract(self.gdn_identity):
            raise RuntimeError(
                "CQ GDN backend differs from the recorded runtime contract: "
                + json.dumps({"expected": self.gdn_identity, "actual": actual})
            )

    def prepare_data_loader(self, loader, device_placement=False):
        from accelerate.data_loader import prepare_data_loader

        return prepare_data_loader(
            loader,
            device=self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            put_on_device=device_placement,
            split_batches=False,
            dispatch_batches=False,
            even_batches=True,
            rng_types=[],
        )

    @contextmanager
    def accumulate(self, model):
        self.micro_step += 1
        self.sync_gradients = (
            self.micro_step == self.config["training"]["gradient_accumulation_steps"]
        )
        with nullcontext() if self.sync_gradients else model.no_sync():
            yield
        if self.sync_gradients:
            self.micro_step = 0

    def backward(self, loss):
        # Finish optional trailing parameter gathers before backward/updates.
        # Most gathers have already overlapped the forward via module hooks.
        self.gathers.finish()
        loss.backward()

    def autocast(self):
        return torch.autocast("npu", dtype=torch.bfloat16)

    def reduce(self, tensor, reduction="sum"):
        result = tensor.clone()
        dist.all_reduce(result, group=self.dp_group)
        if reduction == "mean":
            result.div_(self.num_processes)
        elif reduction != "sum":
            raise ValueError(reduction)
        return result

    def unwrap_model(self, model):
        return model.module if model is self.model else model

    def synchronize_parameters(self):
        self.gathers.finish()

    @contextmanager
    def evaluation_parameters(self):
        self.synchronize_parameters()
        if self.model.ddp_config.overlap_param_gather:
            self.model.disable_forward_pre_hook(param_sync=False)
        try:
            yield
        finally:
            if self.model.ddp_config.overlap_param_gather:
                self.model.enable_forward_pre_hook()

    def runtime_contract(self):
        return dict(
            backend="MEGATRON",
            world_size=self.num_processes,
            device="npu",
            mixed_precision="bf16",
            topology={"tp": 1, "pp": 1, "cp": 1, "dp": self.num_processes},
            dependencies=self.lock,
            distributed=self.config["distributed"],
            loss_normalization="global_text_tokens_and_target_videos_v1",
            qwen_gdn=_gdn_contract(self.gdn_identity),
        )

    def save_state(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.synchronize_parameters()
        if self.is_main_process:
            torch.save(_cpu(self.model.module.state_dict()), directory / "model.pt")
        states = []
        for optimizer in _optimizers(self.optimizer.core):
            states.append(
                dict(
                    inner=optimizer.optimizer.state_dict(),
                    masters=[
                        [p.detach() for p in group["params"]]
                        for group in optimizer.optimizer.param_groups
                    ],
                )
            )
        payload = dict(
            schema="umm_mcore_rank_v2",
            layout=self.layout_digest,
            optimizer=states,
            scheduler=self.scheduler.state_dict(),
            # Restore the communication schedule too, so a resumed update does
            # not repeat MCore's non-overlapped golden-count warmup backward.
            readiness=[
                dict(
                    first=group.is_first_batch and not group.per_param_grad_ready_counts,
                    counts=[
                        (group.per_param_grad_ready_counts if group.is_first_batch
                         else group.golden_per_param_grad_ready_counts).get(p, 0)
                        for bucket in group.buckets
                        for p in bucket.params_list
                    ],
                )
                for group in self.gradients.groups
            ],
        )
        torch.save(
            _cpu(payload), directory / f"optimizer_rank_{self.process_index:05d}.pt"
        )

    def load_state(self, directory):
        directory = Path(directory)
        payload = torch.load(
            directory / f"optimizer_rank_{self.process_index:05d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if (
            payload["schema"] != "umm_mcore_rank_v2"
            or payload["layout"] != self.layout_digest
        ):
            raise ValueError("MCore optimizer bucket layout changed on resume")
        state = torch.load(
            directory / "model.pt", map_location="cpu", weights_only=True
        )
        self.model.module.load_state_dict(state, strict=True)
        del state
        optimizers = _optimizers(self.optimizer.core)
        if len(optimizers) != len(payload["optimizer"]):
            raise ValueError("MCore optimizer partition count changed")
        for optimizer, saved in zip(optimizers, payload["optimizer"]):
            groups = optimizer.optimizer.param_groups
            if len(groups) != len(saved["masters"]) or any(
                len(group["params"]) != len(values)
                for group, values in zip(groups, saved["masters"])
            ):
                raise ValueError("MCore master-weight grouping changed")
            optimizer.optimizer.load_state_dict(saved["inner"])
            with torch.no_grad():
                for group, values in zip(
                    optimizer.optimizer.param_groups, saved["masters"]
                ):
                    for parameter, value in zip(group["params"], values):
                        if (
                            parameter.shape != value.shape
                            or parameter.dtype != value.dtype
                        ):
                            raise ValueError("MCore master-weight shape/dtype changed")
                        parameter.copy_(value)
        self.scheduler.load_state_dict(payload["scheduler"])
        if len(payload["readiness"]) != len(self.gradients.groups):
            raise ValueError("MCore readiness group layout changed")
        for group, saved in zip(self.gradients.groups, payload["readiness"]):
            parameters = [p for bucket in group.buckets for p in bucket.params_list]
            if len(parameters) != len(saved["counts"]):
                raise ValueError("MCore readiness parameter layout changed")
            group.is_first_batch = saved["first"]
            group.golden_per_param_grad_ready_counts = {
                p: count for p, count in zip(parameters, saved["counts"]) if count
            }
            group.per_param_grad_ready_counts = {}

    def end_training(self):
        if dist.is_initialized():
            dist.destroy_process_group()
