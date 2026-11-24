"""One evaluation model loaded from committed weights, sampling with a fixed protocol."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from transformers.initialization import no_init_weights

from configs.config import RunConfig
from data.codec import DevicePreparer
from data.packing import token_contract
from data.vision import NativeImageProcessor
import distributed.accel as accel
import distributed.megatron as parallel
from evaluations.sampling import caption, generate, generate_batch_dit, generate_dit
from models.backbone_qwen35 import cast_for_training
from models.modeling_umm import JointModel

ACCEL = accel.accel_type()
DEV = accel.device_module()


def read_weights(path):
    """An evaluation snapshot (online backbone + EMA renderer) or a TP1 training checkpoint."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("kind") == "inference_weights":
        from utils.eval_snapshot import inspect_snapshot

        return "snapshot", inspect_snapshot(path)
    from utils.checkpoint_catalog import inspect_checkpoint

    return "checkpoint", inspect_checkpoint(path)


@torch.no_grad()
def install_checkpoint(model, path, meta):
    """Online backbone plus the renderer EMA gathered from every optimizer shard."""
    if meta["topology"]["tp"] != 1 or meta["ema"]["num_updates"] != meta["step"]:
        raise ValueError("Expected TP1 and renderer EMA for this exact update")
    model.load_state_dict(
        torch.load(path / "model_tp_000.pt", mmap=True, map_location="cpu", weights_only=False), strict=True
    )
    params = {n: p for n, p in model.named_parameters() if n.startswith("renderer.") and p.requires_grad}
    covered = {n: [] for n in params}
    for rank in range(meta["world"]):
        state = torch.load(path / f"rank_{rank:05d}.pt", mmap=True, map_location="cpu", weights_only=False)
        if state["step"] != meta["step"] or state["tp_rank"] != 0:
            raise ValueError("Checkpoint shard disagrees with its manifest")
        for span, value in zip(state["ema"]["layout"], state["ema"]["shadows"], strict=True):
            p = params[span["name"]]
            p.view(-1)[span["start"] : span["end"]].copy_(value.reshape(-1))
            covered[span["name"]].append((span["start"], span["end"]))
    for name, p in params.items():
        cursor = 0
        for lo, hi in sorted(covered[name]):
            if lo != cursor:
                raise ValueError(f"Missing or overlapping EMA values for {name}")
            cursor = hi
        if cursor != p.numel():
            raise ValueError(f"Incomplete EMA for {name}")


class EvaluationModel:
    def __init__(self, path, sampling):
        self.path = Path(path)
        self.kind, self.meta = read_weights(self.path)
        self.step = self.meta["step"]
        self.sampling = sampling
        self.cfg = RunConfig(**self.meta["config"])
        if self.meta["topology"]["tp"] != 1:
            raise ValueError("Evaluation loads TP1 weights")
        if not dist.is_initialized():
            parallel.initialize(1, self.cfg, initialize_weights=False)
        torch.manual_seed(42)
        DEV.manual_seed_all(42)
        with no_init_weights():
            self.model = cast_for_training(JointModel(self.cfg))
        if dist.get_rank() == 0:
            if self.kind == "snapshot":
                state = torch.load(self.path / "model_tp_000.pt", mmap=True, map_location="cpu", weights_only=False)
                self.model.load_state_dict(state, strict=True)
                del state
            else:
                install_checkpoint(self.model, self.path, self.meta)
        self.model = self.model.to(ACCEL).eval()
        for value in self.model.state_dict().values():
            if isinstance(value, torch.Tensor):
                dist.broadcast(value, src=0)
        self.weights = self.meta.get("weight_sources") or {"backbone": "online", "renderer": "ema"}
        self.tokens = token_contract(self.cfg.text_assets, self.cfg)
        self.tokenizer = Tokenizer.from_file(str(Path(self.cfg.text_assets) / "tokenizer.json"))
        self.processor = NativeImageProcessor(self.cfg, 2048)
        self.prep = None

    def describe(self):
        return dict(
            weights=str(self.path),
            kind=self.kind,
            step=self.step,
            weight_sources=self.weights,
            query_interaction=getattr(self.cfg, "query_interaction", "none"),
            null_condition=getattr(self.cfg, "null_condition", "masked"),
        )

    def answer(self, case):
        return caption(self.model, case, self.tokenizer, self.tokens, self.processor)

    def _preparer(self):
        if self.prep is None:
            self.prep = DevicePreparer(
                self.cfg,
                device=f"{ACCEL}:{DEV.current_device()}",
                vae_dtype=getattr(torch, self.sampling.get("vae_dtype", "float32")),
                encoder="wan",
            )
        return self.prep

    def case(self, name, prompt, seed):
        s = self.sampling
        return dict(
            id=name,
            prompt=prompt,
            task="T2I",
            height=s["resolution"],
            width=s["resolution"],
            steps=s["steps"],
            guidance=s["guidance"],
            shift=s["shift"],
            seed=seed,
        )

    def produce(self, prompt, seed, dest):
        """Write one image atomically; the file appears only when complete."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        writing = dest.with_name("." + dest.stem + ".writing.png")
        generate(
            self.model,
            self.case(writing.stem, prompt, seed),
            self.tokenizer,
            self.tokens,
            self.processor,
            self._preparer().vae,
            dest.parent,
        )
        os.replace(writing, dest)

    def produce_many(self, specs):
        """Batch the DiT over same-shape cases with the native per-image seeds."""
        batch = int(self.sampling.get("batch", 8))
        for start in range(0, len(specs), batch):
            group = specs[start : start + batch]
            if len(group) == 1 or len({dest.parent for _, _, dest in group}) != 1:
                for spec in group:
                    self.produce(*spec)
                continue
            directory = group[0][2].parent
            directory.mkdir(parents=True, exist_ok=True)
            cases = [self.case("." + dest.stem + ".writing", prompt, seed) for prompt, seed, dest in group]
            try:
                conditions = [generate_dit(self.model, c, self.tokenizer, self.tokens, self.processor) for c in cases]
                generate_batch_dit(self.model, conditions, cases, self._preparer().vae, directory)
            # torch.cuda.OutOfMemoryError is only an alias for this; the NPU
            # device module does not export one at all, so naming it through
            # DEV would raise AttributeError here and hide the real failure.
            except torch.OutOfMemoryError:
                conditions = None
                DEV.empty_cache()
                for spec in group:
                    self.produce(*spec)
                continue
            for _, _, dest in group:
                os.replace(dest.with_name("." + dest.stem + ".writing.png"), dest)
