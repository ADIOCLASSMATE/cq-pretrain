"""Qwen3.5 CQ + Wan2.1 composition; losses and data preparation stay external."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

@dataclass
class UMMConfig:
    qwen_config: Qwen3_5Config | dict
    mask_token_id: int
    video_mask_token_id: int
    wan_config: dict = field(default_factory=dict)
    mask_query_read: str = "before_update"
    freeze_vision: bool = True
    gradient_checkpointing: bool = False
    backbone_type: str = "qwen35"
    generator_type: str = "wan21"

    def __post_init__(self):
        if self.backbone_type != "qwen35" or self.generator_type != "wan21":
            raise ValueError("Video composition requires qwen35 + wan21")
        self.qwen_config = (Qwen3_5Config.from_dict(self.qwen_config) if isinstance(self.qwen_config, dict)
                            else deepcopy(self.qwen_config))
        self.wan_config = deepcopy(self.wan_config)
        if self.mask_query_read != "before_update":
            raise ValueError("UMM requires before_update CQ visibility")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in (self.mask_token_id, self.video_mask_token_id)):
            raise ValueError("Set nonnegative mask_token_id and video_mask_token_id from the tokenizer")
        text = self.qwen_config.text_config
        text.vocab_size = max(text.vocab_size, self.mask_token_id + 1, self.video_mask_token_id + 1)
        text.mask_token_id = self.mask_token_id
        # The existing backbone treats type 1 as a visual slot; this legacy
        # field selects its video query embedding without changing native code.
        text.image_mask_token_id = self.video_mask_token_id
        text.mask_query_read = self.mask_query_read
        text.image_latent_dim = None
        text.gradient_checkpointing = self.gradient_checkpointing
        if self.qwen_config.vision_config.out_hidden_size != text.hidden_size:
            raise ValueError("Native visual merger width must equal backbone hidden_size")
        if self.wan_config.get("condition_dim", text.hidden_size) != text.hidden_size:
            raise ValueError("Wan condition_dim must equal backbone hidden_size")

    @classmethod
    def from_dict(cls, values):
        return cls(**deepcopy(values))

    def to_dict(self):
        return {"qwen_config": self.qwen_config.to_dict(), "mask_token_id": self.mask_token_id,
                "video_mask_token_id": self.video_mask_token_id, "wan_config": deepcopy(self.wan_config),
                "mask_query_read": self.mask_query_read, "freeze_vision": self.freeze_vision,
                "gradient_checkpointing": self.gradient_checkpointing,
                "backbone_type": self.backbone_type, "generator_type": self.generator_type}



class UMM(nn.Module):
    def __init__(self, config: UMMConfig, *, backbone=None, vision_encoder=None, generator=None):
        super().__init__()
        self.config = UMMConfig.from_dict(config) if isinstance(config, dict) else deepcopy(config)
        config = self.config
        from .backbone_qwen35 import Qwen35CQBackbone, Qwen35VideoEncoder
        cfg = self.config.qwen_config
        self.backbone = Qwen35CQBackbone(cfg.text_config) if backbone is None else backbone
        if vision_encoder is not None:
            self.backbone.vision_encoder = vision_encoder
        elif not hasattr(self.backbone, "vision_encoder"):
            self.backbone.vision_encoder = Qwen35VideoEncoder(cfg)
        if generator is None:
            from .generator_wan21 import Wan21Generator
            settings = {**self.config.wan_config, "condition_dim": cfg.text_config.hidden_size}
            generator = Wan21Generator(**settings)
        self.generator = generator
        if (self.backbone.config.hidden_size != cfg.text_config.hidden_size
                or self.backbone.embed_tokens.num_embeddings != cfg.text_config.vocab_size
                or self.vision_encoder.visual.config.out_hidden_size != cfg.text_config.hidden_size
                or self.generator.condition_dim != cfg.text_config.hidden_size):
            raise ValueError("Injected components must match the configured vocabulary and hidden dimensions")
        for name, expected in (("mask_query_read", "before_update"), ("mask_token_id", config.mask_token_id),
                               ("image_mask_token_id", config.video_mask_token_id)):
            if getattr(self.backbone.config, name, None) != expected:
                raise ValueError(f"Injected backbone has incompatible {name}")
        self.backbone.gradient_checkpointing = config.gradient_checkpointing
        tied = cfg.text_config.tie_word_embeddings
        self.lm_head = nn.Linear(cfg.text_config.hidden_size, cfg.text_config.vocab_size, bias=False,
                                 device="meta" if tied else self.backbone.embed_tokens.weight.device,
                                 dtype=self.backbone.embed_tokens.weight.dtype)
        if tied:
            self.lm_head.weight = self.backbone.embed_tokens.weight
        else:
            nn.init.normal_(self.lm_head.weight, std=cfg.text_config.initializer_range)
        self.vision_encoder.requires_grad_(not config.freeze_vision)
        if config.freeze_vision:
            self.vision_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_vision:
            self.vision_encoder.eval()
        return self

    @classmethod
    def from_pretrained_components(cls, qwen_path, wan_path, *, mask_token_id, video_mask_token_id,
                                   freeze_vision=True, gradient_checkpointing=False,
                                   attn_implementation="sdpa", wan_attention_backend="auto",
                                   dtype=torch.float32, wan_loading_kwargs=None):
        """Load local native components, expanding only new tokenizer mask rows.

        Qwen3.5-2B ties its text head to the embedding. No second native LLM is
        allocated, and constructing this wrapper never reinitializes Wan/ViT.
        Processor/tokenizer creation and complete training resumes stay outside
        this stage-one API. Supplied IDs must agree with the caller's tokenizer.
        """
        from .generator_wan21 import Wan21Generator
        from .backbone_qwen35 import Qwen35CQBackbone, Qwen35VideoEncoder
        qwen = Qwen3_5Config.from_pretrained(qwen_path, local_files_only=True)
        if not qwen.text_config.tie_word_embeddings:
            raise ValueError("Component loading requires tied Qwen text embeddings, as in Qwen3.5-2B")
        backbone = Qwen35CQBackbone.from_pretrained(
            qwen_path, config=deepcopy(qwen.text_config), local_files_only=True, dtype=dtype,
            attn_implementation=attn_implementation, mask_token_id=mask_token_id,
            image_mask_token_id=video_mask_token_id, mask_query_read="before_update",
        )
        size = max(backbone.config.vocab_size, mask_token_id + 1, video_mask_token_id + 1)
        if size != backbone.config.vocab_size:
            backbone.resize_token_embeddings(size, mean_resizing=False)
        qwen.text_config = deepcopy(backbone.config)
        visual = Qwen35VideoEncoder.from_pretrained(
            qwen_path, config=qwen, dtype=dtype, attn_implementation=attn_implementation,
        )
        generator = Wan21Generator.from_wan_pretrained(
            wan_path, condition_dim=backbone.config.hidden_size, torch_dtype=dtype,
            attention_backend=wan_attention_backend, gradient_checkpointing=gradient_checkpointing,
            **(wan_loading_kwargs or {}),
        )
        config = UMMConfig(qwen, mask_token_id, video_mask_token_id,
                               wan_config={k: v for k, v in generator.config.items() if not k.startswith("_")},
                               freeze_vision=freeze_vision, gradient_checkpointing=gradient_checkpointing)
        return cls(config, backbone=backbone, vision_encoder=visual, generator=generator).eval()

    @property
    def vision_encoder(self):
        return self.backbone.vision_encoder

    def prepare_inputs(self, batch):
        return self.backbone.prepare_inputs(batch)

    def generate_text(self, batch, **kwargs):
        return generate_text(self, batch, **kwargs)

    @torch.no_grad()
    def generate_video(self, batch, *, latent_shapes=None, initial_noise=None, steps=20,
                       solver="euler", cfg=1.0, time_shift=1.0, timestep_scale=1000.0,
                       decode=None, rng=None):
        """Generate whole-video latents from layout-only targets, without backbone caches.

        latent_shapes use Wan VAE coordinates [C,T,H,W], independently of Qwen
        grids. An optional decode(latents) callable owns the external VAE contract.
        """
        if "pixel_video_ids" in batch:
            raise ValueError("pixel_video_ids is unsupported; use one Qwen representation per batch")
        if batch.get("video_features") is not None and batch.get("pixel_values_videos") is not None:
            raise ValueError("Supply video_features or pixel_values_videos for the whole batch, not both")
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            layout = self.backbone._video_layout(batch)
            grids = layout["video_grid_thw"]
            count = len(layout["video_token_indices"])
            targets = batch.get("target_video_ids", torch.arange(count, device=grids.device))
            if (not isinstance(targets, torch.Tensor) or targets.ndim != 1
                    or targets.dtype not in (torch.int32, torch.int64) or targets.numel() == 0
                    or targets.unique().numel() != targets.numel()
                    or bool(((targets < 0) | (targets >= count)).any())):
                raise ValueError("Generation needs unique valid target_video_ids")
            observed = self.backbone._video_flags(batch, "video_observed_mask", count, grids.device, True)
            remaining = observed.clone()
            remaining[targets] = False
            clean = {k: v for k, v in batch.items() if k not in ("labels", "video_uncond_mask")}
            if "video_features" in clean:
                clean["video_features"] = [x if remaining[i] else None for i, x in enumerate(clean["video_features"])]
            if "pixel_values_videos" in clean:
                sizes = grids[observed].prod(-1).tolist()
                chunks = clean["pixel_values_videos"].split(sizes)
                keep = [x for i, x in zip(torch.where(observed)[0].tolist(), chunks) if remaining[i]]
                if keep:
                    clean["pixel_values_videos"] = torch.cat(keep)
                else:
                    clean.pop("pixel_values_videos")
            clean.update(video_observed_mask=remaining, target_video_ids=targets)
            prepared = self.prepare_inputs(clean)
            if float(cfg) == 1.0:
                hidden = self.backbone(prepared)["query_hidden"]
                condition, mask = self._condition_from_hidden(hidden, prepared)
                unconditional = unconditional_mask = None
            else:
                # One batched backbone evaluation; CFG removes query reads only.
                doubled = dict(prepared)
                for key in ("input_ids", "inputs_embeds", "token_types", "segment_ids",
                            "visual_group_ids", "query_read_end"):
                    doubled[key] = torch.cat((prepared[key], prepared[key]), 0)
                doubled["position_ids"] = torch.cat((prepared["position_ids"], prepared["position_ids"]), 1)
                drop = torch.zeros_like(prepared["token_types"], dtype=torch.bool)
                for i in targets.tolist():
                    drop[prepared["video_batch_indices"][i], prepared["video_token_indices"][i]] = True
                doubled["visual_uncond_mask"] = torch.cat((torch.zeros_like(drop), drop), 0)
                conditional_hidden, unconditional_hidden = self.backbone(doubled)["query_hidden"].chunk(2)
                condition, mask = self._condition_from_hidden(conditional_hidden, prepared)
                unconditional, unconditional_mask = self._condition_from_hidden(unconditional_hidden, prepared)
            latents = self.generator.sample(
                condition, condition_mask=mask, latent_shapes=latent_shapes, initial_noise=initial_noise,
                steps=steps, solver=solver, cfg=cfg, unconditional=unconditional,
                unconditional_mask=unconditional_mask, time_shift=time_shift,
                timestep_scale=timestep_scale, rng=rng,
            )
            return latents if decode is None else decode(latents)
        finally:
            for module, mode in modes:
                module.training = mode

    @staticmethod
    def _condition_from_hidden(hidden, prepared):
        targets = prepared["target_video_ids"].tolist()
        if not targets:
            raise ValueError("Video generation requires at least one target video")
        rows, indices = prepared["video_batch_indices"], prepared["video_token_indices"]
        selected = [hidden[rows[i], indices[i]] for i in targets]
        condition = pad_sequence(selected, batch_first=True)
        lengths = torch.tensor([x.shape[0] for x in selected], device=hidden.device)
        mask = torch.arange(condition.shape[1], device=hidden.device)[None] < lengths[:, None]
        return condition, mask

    def encode_condition(self, batch):
        """Return (query conditions [M,N,D], valid mask [M,N]) for fixed Wan conditioning."""
        prepared = self.prepare_inputs(batch)
        hidden = self.backbone(prepared)["query_hidden"]
        return self._condition_from_hidden(hidden, prepared)

    def forward(self, batch, noisy_latents=None, timesteps=None, *, compute_text=None,
                return_hidden_states=False, flow_repeats=1):
        if type(flow_repeats) is not int or flow_repeats < 1:
            raise ValueError("flow_repeats must be a positive integer")
        if (noisy_latents is None) != (timesteps is None):
            raise ValueError("noisy_latents and native Wan timesteps must be supplied together")
        if compute_text is None:
            compute_text = noisy_latents is None or batch.get("labels") is not None
        prepared = self.prepare_inputs(batch)
        streams = self.backbone(prepared, return_content=return_hidden_states)
        query = streams["query_hidden"]
        output = dict(streams) if return_hidden_states else {}
        if noisy_latents is not None:
            condition, mask = self._condition_from_hidden(query, prepared)
            targets = prepared["target_video_ids"]
            # CQ conditions are computed once; only Wan receives repeated flow draws.
            # Repeat-major order: [video_0, ..., video_M-1] for each independent draw.
            if flow_repeats > 1:
                condition = condition.repeat(flow_repeats, 1, 1)
                mask = mask.repeat(flow_repeats, 1)
                targets = targets.repeat(flow_repeats)
            if not isinstance(noisy_latents, (list, tuple)) or len(noisy_latents) != condition.shape[0]:
                raise ValueError("Noisy latent list must match target_video_ids order repeated flow_repeats times")
            output.update(velocity=self.generator(noisy_latents, timesteps, condition, condition_mask=mask),
                          condition=condition, condition_mask=mask, target_video_ids=targets)
        if compute_text:
            labels = batch.get("labels")
            if labels is None:
                output.update(text_logits=self.lm_head(query), text_labels=None)
            else:
                if labels.shape != prepared["input_ids"].shape:
                    raise ValueError("labels must align with input_ids without a next-token shift")
                valid = labels.ne(-100) & prepared["token_types"].ne(1) & prepared["token_types"].ne(3)
                output.update(text_logits=self.lm_head(query[valid]), text_labels=labels[valid])
        return output



def _sample(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k:
        threshold = logits.topk(min(top_k, logits.shape[-1]), -1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1:
        sorted_logits, order = logits.sort(descending=True)
        remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.masked_fill(torch.zeros_like(remove).scatter(1, order, remove), -torch.inf)
    return torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)



@torch.no_grad()
def generate_text(model, batch, *, max_new_tokens=128, temperature=0.0, top_k=0,
                  top_p=1.0, eos_token_id=None, use_cache=False, return_logits=False):
    """Same-position CQ decoding. Qwen3.5 recomputes the full prefix each step."""
    if use_cache:
        raise NotImplementedError("Qwen3.5 CQ generation currently supports no-cache decoding only")
    if max_new_tokens < 0 or top_k < 0 or not 0 < top_p <= 1 or not math.isfinite(float(temperature)):
        raise ValueError("Invalid text generation settings")
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        prepared = model.prepare_inputs(batch)
        valid = prepared["token_types"].ne(3) & prepared["segment_ids"].ge(0)
        if bool((~valid.any(-1)).any()):
            raise ValueError("Each prompt needs nonempty content")
        for row in range(valid.shape[0]):
            if prepared["segment_ids"][row, valid[row]].unique().numel() != 1:
                raise ValueError("Unpack independent segments before generation")
        sequences = [row[keep].clone() for row, keep in zip(prepared["input_ids"], valid)]
        device = prepared["input_ids"].device
        text_config = model.config.qwen_config.text_config
        eos = text_config.eos_token_id if eos_token_id is None else eos_token_id
        eos_ids = [] if eos is None else (list(eos) if isinstance(eos, (list, tuple)) else [eos])
        pad = text_config.pad_token_id
        pad = 0 if pad is None else pad
        # Frozen input features are reusable inputs, not decoder KV/GDN caches.
        observed = model.backbone._video_flags(batch, "video_observed_mask",
                                              len(prepared["video_token_indices"]), device, True)
        features = [prepared["inputs_embeds"][row, slots] if observed[i] else None
                    for i, (row, slots) in enumerate(zip(prepared["video_batch_indices"],
                                                        prepared["video_token_indices"]))]
        fixed = dict(video_grid_thw=prepared["video_grid_thw"], video_features=features,
                     video_observed_mask=observed, target_video_ids=torch.empty(0, device=device, dtype=torch.long))
        generated, scores = [], []
        finished = torch.zeros(len(sequences), device=device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            lengths = torch.tensor([s.numel() for s in sequences], device=device)
            ids = prepared["input_ids"].new_full((len(sequences), int(lengths.max()) + 1), pad)
            attention_mask = torch.zeros_like(ids, dtype=torch.bool)
            for row, seq in enumerate(sequences):
                ids[row, :seq.numel()] = seq
                attention_mask[row, :seq.numel()] = True
                ids[row, seq.numel()] = model.config.mask_token_id
                attention_mask[row, seq.numel()] = ~finished[row]
            step = model.prepare_inputs({**fixed, "input_ids": ids, "attention_mask": attention_mask})
            hidden = model.backbone(step)["query_hidden"]
            logits = model.lm_head(hidden[torch.arange(len(sequences), device=device), lengths])
            sampling_logits = logits.clone()
            # Input placeholders are not output text. Keep raw logits for parity checks.
            reserved = {model.config.mask_token_id, model.config.video_mask_token_id,
                        model.config.qwen_config.video_token_id, model.config.qwen_config.image_token_id}
            sampling_logits[:, list(reserved)] = -torch.inf
            token = _sample(sampling_logits, temperature, top_k, top_p)
            token = torch.where(finished, pad, token)
            generated.append(token)
            if return_logits:
                scores.append(logits)
            for row in range(len(sequences)):
                if not finished[row]:
                    sequences[row] = torch.cat((sequences[row], token[row:row + 1]))
            for eos_id in eos_ids:
                finished |= token.eq(eos_id)
            if bool(finished.all()):
                break
        tokens = torch.stack(generated, 1) if generated else prepared["input_ids"].new_empty((len(sequences), 0))
        if return_logits:
            logits = torch.stack(scores, 1) if scores else model.lm_head.weight.new_empty((len(sequences), 0, text_config.vocab_size))
            return {"sequences": tokens, "logits": logits}
        return tokens
    finally:
        for module, mode in modes:
            module.training = mode
