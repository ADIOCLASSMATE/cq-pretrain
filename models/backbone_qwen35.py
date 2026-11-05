"""Video input adapter over the shared main-branch Qwen CQ layers."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

import distributed.megatron as P
from models.qwen35_layers import QwenBlock, RMSNorm, unfused_parameters, cast_for_training
from models.cq_layout import MainlinePlan
from ops.attention_layout import document_cq_mask, plan_attention
from ops.gated_delta_layout import build_delta_plan, rope_tables


def layer_config(config):
    text = getattr(config, "text_config", config)
    types = text.layer_types
    interval = next((i + 1 for i, kind in enumerate(types) if kind == "full_attention"), len(types) + 1)
    if any((kind == "full_attention") != ((i + 1) % interval == 0) for i, kind in enumerate(types)):
        raise ValueError("Qwen full-attention layers must follow a regular interval")
    if text.linear_key_head_dim != text.linear_value_head_dim:
        raise ValueError("Shared GDN requires equal key/value head dimensions")
    rope = text.rope_parameters
    return SimpleNamespace(
        width=text.hidden_size, layers=text.num_hidden_layers, ffn=text.intermediate_size,
        heads=text.num_attention_heads, kv_heads=text.num_key_value_heads, head_dim=text.head_dim,
        linear_heads=text.linear_num_key_heads, linear_value_heads=text.linear_num_value_heads,
        linear_head_dim=text.linear_key_head_dim, linear_conv_kernel=text.linear_conv_kernel_dim,
        full_attention_interval=interval, norm_eps=text.rms_norm_eps,
        rope_theta=rope["rope_theta"], partial_rotary_factor=rope["partial_rotary_factor"],
        mrope_section=rope["mrope_section"], fused=True, qwen_fused_projections=True,
        gdn_backend=getattr(text, "qwen_gdn_backend", "ascendc"),
        content_visibility=getattr(text, "content_visibility", "visual_block"),
    )


def resolve_gdn_backend_identity(backend):
    if backend not in {"ascendc", "mindspeed"}:
        raise ValueError("qwen_gdn_backend must be ascendc or mindspeed")
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        from models.qwen35_layers import GatedDelta
        path = Path(inspect.getfile(GatedDelta))
        provider, detail = "torch_reference", {}
    elif backend == "ascendc":
        from ops.gated_delta_net import provenance
        detail = provenance()
        detail = {k: v for k, v in detail.items() if k not in {"calls", "package"}}
        provider = "AscendC"
        from models.qwen35_layers import GatedDelta
        path = Path(inspect.getfile(GatedDelta))
    else:
        from mindspeed.core.ssm.npu_chunk_gated_delta_rule import chunk_gated_delta_rule
        path = Path(inspect.getfile(chunk_gated_delta_rule))
        provider, detail = "MindSpeed", {}
    return dict(backend=backend, provider=provider, file=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), kernels=detail)


def cpu_metadata(batch):
    names = ("input_ids", "attention_mask", "token_types", "segment_ids", "mm_token_type_ids",
             "video_grid_thw", "video_batch_indices", "visual_group_ids", "query_read_end", "position_ids",
             "video_observed_mask", "video_uncond_mask", "visual_uncond_mask", "target_video_ids", "labels")
    metadata = {k: batch[k].detach().cpu() for k in names if isinstance(batch.get(k), torch.Tensor)}
    if "video_token_indices" in batch:
        metadata["video_token_indices"] = tuple(x.cpu() for x in batch["video_token_indices"])
    return metadata


def cq_plan(batch, cfg, positions, *, query_stream=True):
    ids, types = batch["input_ids"], batch["token_types"]
    valid = types.ne(3)
    segments = batch.get("segment_ids", torch.where(valid, 0, -1))
    valid &= segments.ge(0)
    content_positions = valid.flatten().nonzero().flatten()
    if not len(content_positions):
        raise ValueError("CQ needs nonempty content")
    packed_segments = torch.full_like(segments, -1)
    sequence = 0
    for row in range(len(ids)):
        for segment in segments[row, valid[row]].unique().tolist():
            selected = valid[row] & segments[row].eq(segment)
            packed_segments[row, selected] = sequence
            sequence += 1
    packed_types = types.flatten()[content_positions]
    seg = packed_segments.flatten()[content_positions]
    sigma = torch.arange(len(content_positions))
    if cfg.content_visibility == "visual_block":
        start = 0
        while start < len(sigma):
            end = start + 1
            if packed_types[start] == 1:
                while end < len(sigma) and packed_types[end] == 1 and seg[end] == seg[start]:
                    end += 1
                sigma[start:end] = start
            start = end
    elif cfg.content_visibility != "causal":
        raise ValueError("content_visibility must be visual_block or causal")
    selected = valid.clone() if query_stream else torch.zeros_like(valid)
    labels = batch.get("labels")
    if labels is not None:
        selected &= labels.ne(-100) | types.eq(1)
    targets = batch.get("target_video_ids")
    if query_stream and targets is not None and "video_token_indices" in batch:
        selected[types.eq(1)] = False
        for video in targets.tolist():
            selected[batch["video_batch_indices"][video], batch["video_token_indices"][video]] = True
    outputs = selected.flatten().nonzero().flatten()
    inverse = torch.full((ids.numel(),), -1, dtype=torch.long)
    inverse[content_positions] = torch.arange(len(content_positions))
    read_end = batch.get("query_read_end")
    if read_end is None:
        read_end = torch.arange(ids.shape[1])[None].expand_as(ids).clone()
        visual = types.eq(1)
        groups = batch.get("visual_group_ids")
        if groups is None:
            starts = visual & ~F.pad(visual[:, :-1], (1, 0))
            groups = starts.long().cumsum(-1)
        for row in range(len(ids)):
            for group in groups[row, visual[row]].unique().tolist():
                slots = (visual[row] & groups[row].eq(group)).nonzero().flatten()
                read_end[row, slots] = slots[0]
    anchors = read_end + torch.arange(len(ids))[:, None] * ids.shape[1]
    query_positions = inverse[anchors.flatten()[outputs]]
    drop = batch.get("visual_uncond_mask", torch.zeros_like(valid)).flatten()[outputs]
    pos = positions.reshape(3, -1)
    empty = torch.empty(0, dtype=torch.long)
    plan = MainlinePlan(
        method="CQ_IT", prefix=0, query_positions=query_positions,
        query_image=types.flatten()[outputs].eq(1)[None], query_rope=pos[:, outputs, None],
        generation_positions=empty, generation_image=torch.empty(1, 0, dtype=torch.bool),
        generation_rope=torch.empty(3, 0, 1), output_positions=outputs,
        condition_indices=empty, generation_condition=empty,
        cq_mask=torch.empty(0), cq_valid=torch.empty(0), gen_mask=torch.empty(0),
        gen_valid=torch.empty(0), gen_suffix_mask=torch.empty(0), output_size=ids.numel(),
        query_unconditional=drop,
    )
    packed = SimpleNamespace(ids=ids.flatten()[content_positions][None], sigma=sigma[None],
        segments=seg[None], row_col=pos[:, content_positions, None], image_indices=empty.reshape(0, 1))
    plan.cq_mask, plan.cq_valid = document_cq_mask(packed, query_positions, drop)
    plan.cq_plan = plan_attention(plan.cq_mask, 1024, trim_key_start=True)
    plan.delta_plan = build_delta_plan(packed, plan, cfg)
    plan.rope_cos, plan.rope_sin = rope_tables(packed, plan, cfg)
    return plan, content_positions


class Qwen35CQBackbone(nn.Module):
    def __init__(self, config, *, mask_token_id=None, image_mask_token_id=None, mask_query_read=None):
        super().__init__()
        self.config = deepcopy(getattr(config, "text_config", config))
        for key, value in (("mask_token_id", mask_token_id), ("image_mask_token_id", image_mask_token_id),
                           ("mask_query_read", mask_query_read)):
            if value is not None:
                setattr(self.config, key, value)
        self.config.mask_query_read = getattr(self.config, "mask_query_read", "before_update")
        if self.config.mask_query_read != "before_update":
            raise ValueError("CQ requires before_update visibility")
        self.cfg = layer_config(self.config)
        self.embed_tokens = P.make_embedding(self.config.vocab_size, self.cfg.width)
        self.layers = nn.ModuleList(QwenBlock(self.cfg, i) for i in range(self.cfg.layers))
        self.norm = RMSNorm(self.cfg.width, self.cfg.norm_eps)
        self.gradient_checkpointing = bool(getattr(self.config, "gradient_checkpointing", False))
        self.configure_gdn_backend(self.cfg.gdn_backend)

    def configure_gdn_backend(self, backend):
        if backend not in {"ascendc", "mindspeed"}:
            raise ValueError("qwen_gdn_backend must be ascendc or mindspeed")
        self.config.qwen_gdn_backend = self.cfg.gdn_backend = backend

    def gdn_backend_identity(self):
        for layer in self.layers:
            if hasattr(layer, "linear_attn") and layer.linear_attn.cfg.gdn_backend != self.cfg.gdn_backend:
                raise RuntimeError("CQ GDN layer backend disagrees with backbone config")
        return resolve_gdn_backend_identity(self.cfg.gdn_backend)

    def native_parameters(self):
        return {key: view for name, parameter in self.named_parameters()
                if not name.startswith("vision_encoder.")
                for key, view in unfused_parameters(name, parameter, self.cfg).items()}

    @classmethod
    def from_pretrained(cls, path, *, config=None, dtype=torch.float32, output_loading_info=False,
                        local_files_only=True, attn_implementation="sdpa", **kwargs):
        from safetensors import safe_open
        config = config or Qwen3_5TextConfig.from_pretrained(path, local_files_only=True)
        model = cls(config, **kwargs)
        if dtype == torch.bfloat16:
            cast_for_training(model)
        else:
            model.to(dtype=dtype)
        targets = model.native_parameters()
        root = Path(path)
        index = root / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
        seen = set()
        for filename in files:
            with safe_open(str(root / filename), framework="pt", device="cpu") as source, torch.no_grad():
                for key in source.keys():
                    clean = key
                    for prefix in ("model.language_model.", "language_model.", "model."):
                        if clean.startswith(prefix):
                            clean = clean[len(prefix):]
                            break
                    if clean not in targets:
                        if clean.startswith(("visual.", "mtp.", "lm_head.")):
                            continue
                        raise ValueError(f"Unexpected Qwen text weight: {key}")
                    value = P.checkpoint_shard(source.get_tensor(key), targets[clean])
                    if value.shape != targets[clean].shape:
                        raise ValueError(f"Qwen shape mismatch: {key}")
                    targets[clean].copy_(value)
                    seen.add(clean)
        if seen != set(targets):
            raise ValueError(f"Missing Qwen text weights: {sorted(set(targets) - seen)}")
        info = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
        return (model, info) if output_loading_info else model

    def resize_token_embeddings(self, size, mean_resizing=False):
        previous = self.embed_tokens
        self.embed_tokens = P.make_embedding(size, self.cfg.width).to(previous.weight)
        with torch.no_grad():
            self.embed_tokens.weight[:previous.weight.shape[0]].copy_(previous.weight)
        self.config.vocab_size = size
        return self.embed_tokens

    def embed_content(self, batch):
        if batch.get("inputs_embeds") is not None:
            return batch["inputs_embeds"]
        if batch["token_types"].eq(1).any():
            raise ValueError("Visual slots require native visual inputs_embeds")
        return self.embed_tokens(batch["input_ids"])

    def _video_layout(self, batch):
        ids = batch["input_ids"]
        if ids.ndim != 2 or ids.shape[1] == 0 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a nonempty integer [B,L] tensor")
        valid = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool)).bool()
        if valid.shape != ids.shape:
            raise ValueError("attention_mask must align with input_ids")
        segments = batch.get("segment_ids", torch.where(valid, 0, -1))
        if segments.shape != ids.shape:
            raise ValueError("segment_ids must align with input_ids")
        valid = valid & segments.ge(0)
        segments = segments.masked_fill(~valid, -1)
        cfg = self.vision_encoder.config
        video = ids.eq(cfg.video_token_id) & valid
        mm_types = batch.get("mm_token_type_ids", video.long() * 2)
        if (mm_types.shape != ids.shape or not torch.equal(mm_types.eq(2) & valid, video)
                or bool(((mm_types != 0) & (mm_types != 2) & valid).any())
                or bool((ids.eq(cfg.image_token_id) & valid).any())):
            raise ValueError("This video entry expects native text/video mm_token_type_ids (0/2)")
        types = torch.zeros_like(ids)
        types[ids.eq(cfg.vision_start_token_id) | ids.eq(cfg.vision_end_token_id)] = 2
        types[video], types[~valid] = 1, 3
        if "token_types" in batch and not torch.equal(batch["token_types"], types):
            raise ValueError("CQ token_types differ from native video slots/padding; do not use mm_token_type_ids")
        grids = batch.get("video_grid_thw", ids.new_empty((0, 3))).to(ids.device)
        counts = self.vision_encoder.token_counts(grids).tolist()
        groups = torch.full_like(ids, -1)
        rows, indices = [], []
        video_id = 0
        merge = self.vision_encoder.visual.spatial_merge_size
        for row in range(ids.shape[0]):
            slots = torch.where(video[row])[0]
            offset = 0
            while offset < slots.numel():
                if video_id >= len(counts) or offset + counts[video_id] > slots.numel():
                    raise ValueError("Native video placeholder count does not match video_grid_thw")
                selected = slots[offset:offset + counts[video_id]]
                t, h, w = grids[video_id].tolist()
                frames = selected.view(t, h * w // merge**2)
                if (not bool((frames[:, 1:] - frames[:, :-1]).eq(1).all())
                        or bool((frames[:, 0] < 1).any()) or bool((frames[:, -1] + 1 >= ids.shape[1]).any())):
                    raise ValueError("Each native video frame must have contiguous visual placeholders")
                if (not bool(ids[row, frames[:, 0] - 1].eq(cfg.vision_start_token_id).all())
                        or not bool(ids[row, frames[:, -1] + 1].eq(cfg.vision_end_token_id).all())):
                    raise ValueError("Each video frame requires native vision_start/vision_end markers")
                start, end = int(selected[0]) - 1, int(selected[-1]) + 2
                if not bool(valid[row, start:end].all()) or segments[row, start:end].unique().numel() != 1:
                    raise ValueError("A complete video must remain inside one valid packed segment")
                groups[row, selected] = video_id
                rows.append(row)
                indices.append(selected)
                offset += counts[video_id]
                video_id += 1
        if video_id != len(counts):
            raise ValueError("Native video placeholder count does not match video_grid_thw")
        rows = ids.new_tensor(rows)
        for key, expected in (("visual_group_ids", groups), ("video_batch_indices", rows)):
            if key in batch and not torch.equal(batch[key], expected):
                raise ValueError(f"{key} does not match processor video order")
        if "video_token_indices" in batch:
            supplied = batch["video_token_indices"]
            if (not isinstance(supplied, (list, tuple)) or len(supplied) != len(indices)
                    or any(not torch.equal(a, b) for a, b in zip(supplied, indices))):
                raise ValueError("video_token_indices do not match native video placeholders")

        positions = ids.new_zeros((3, *ids.shape))
        read_end = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids).clone()
        for row, slots in zip(rows.tolist(), indices):
            read_end[row, slots] = slots[0]
        for row in range(ids.shape[0]):
            for segment in segments[row, valid[row]].unique():
                selected = torch.where(valid[row] & segments[row].eq(segment))[0]
                local_videos = [i for i, r in enumerate(rows.tolist())
                                if r == row and segments[row, indices[i][0]] == segment]
                native_positions, _ = self.vision_encoder.get_rope_index(
                    ids[row:row + 1, selected], mm_types[row:row + 1, selected],
                    video_grid_thw=grids[local_videos] if local_videos else None,
                )
                positions[:, row, selected] = native_positions[:, 0]
        if "query_read_end" in batch and not torch.equal(batch["query_read_end"], read_end):
            raise ValueError("query_read_end must use the physical video prefix/text position")
        if "position_ids" in batch and not torch.equal(batch["position_ids"], positions):
            raise ValueError("position_ids must match native Qwen M-RoPE for each segment")
        return dict(input_ids=ids, token_types=types, segment_ids=segments, query_read_end=read_end,
                    position_ids=positions, visual_group_ids=groups, video_grid_thw=grids,
                    video_batch_indices=rows, video_token_indices=tuple(indices))

    @staticmethod
    def _video_flags(batch, name, count, device, default):
        flags = batch.get(name, torch.full((count,), default, device=device, dtype=torch.bool))
        if not isinstance(flags, torch.Tensor) or flags.shape != (count,) or flags.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean [videos]")
        return flags.to(device)

    def prepare_inputs(self, batch):
        metadata = dict(batch.get("_cpu_layout_inputs") or cpu_metadata(batch))
        # CFG decisions are made by the existing task policy, once per target.
        for key in ("video_uncond_mask", "target_video_ids", "video_observed_mask"):
            if key in batch and (key not in metadata or key == "video_uncond_mask"):
                metadata[key] = batch[key].detach().cpu()
        prepared = self._video_layout(metadata)
        ids, grids = prepared["input_ids"], prepared["video_grid_thw"]
        rows, indices = prepared["video_batch_indices"], prepared["video_token_indices"]
        count = len(indices)
        observed = self._video_flags(metadata, "video_observed_mask", count, "cpu", True)
        uncond = self._video_flags(metadata, "video_uncond_mask", count, "cpu", False)
        if any(key in batch for key in ("image_uncond_mask", "image_uncond_rows", "visual_uncond_mask")):
            raise ValueError("UMMVideo CFG uses video_uncond_mask, one flag per complete video")
        targets = metadata.get("target_video_ids", torch.arange(count))
        if targets.ndim != 1 or targets.unique().numel() != targets.numel() or ((targets < 0) | (targets >= count)).any():
            raise ValueError("target_video_ids must be unique indices into processor-order videos")
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (row, slots) in enumerate(zip(rows.tolist(), indices)):
            mask[row, slots] = uncond[i]
        prepared.update(visual_uncond_mask=mask, target_video_ids=targets)
        if "labels" in metadata:
            prepared["labels"] = metadata["labels"]
        plan, content_positions = cq_plan(prepared, self.cfg, prepared["position_ids"])
        device = batch["input_ids"].device
        def move(value):
            if isinstance(value, torch.Tensor):return value.to(device, non_blocking=True)
            if isinstance(value, tuple):return tuple(move(x) for x in value)
            return value
        prepared = {key: move(value) for key, value in prepared.items()}
        prepared["_cq_plan"] = plan.to(device, non_blocking=True)
        prepared["_content_positions"] = content_positions.to(device, non_blocking=True)
        content = self.embed_tokens(prepared["input_ids"].masked_fill(prepared["token_types"].eq(1), self.config.image_mask_token_id))
        features, pixels = batch.get("video_features"), batch.get("pixel_values_videos")
        if "pixel_video_ids" in batch:
            raise ValueError("pixel_video_ids is unsupported; use one Qwen representation per batch")
        if features is not None and pixels is not None:
            raise ValueError("Supply video_features or pixel_values_videos for the whole batch, not both")
        if features is not None:
            if not isinstance(features, (list, tuple)) or len(features) != count:
                raise ValueError("video_features must contain one post-merger tensor/None per video")
            features = list(features)
        else:
            features = [None] * count
            if pixels is not None:
                encoded = self.vision_encoder(pixels, grids[observed].to(device))
                for i, values in zip(observed.nonzero().flatten().tolist(), encoded):features[i] = values
        visual_positions, visual_values = [], []
        for i, (row, slots) in enumerate(zip(rows.tolist(), prepared["video_token_indices"])):
            values = features[i]
            if observed[i]:
                if not isinstance(values, torch.Tensor) or values.shape != (len(slots), self.cfg.width) or values.device != device:
                    raise ValueError("Observed video features must match native token count, width and device")
                visual_positions.append(slots + row * ids.shape[1])
                visual_values.append(values.to(content.dtype))
            elif values is not None:
                raise ValueError("Unobserved target videos must not supply visual features")
        if visual_positions:
            # MCore embeddings can be custom-autograd views. Replace all visual
            # slots together without mutating that view or cloning per video.
            content = content.flatten(0, 1).index_copy(
                0, torch.cat(visual_positions), torch.cat(visual_values)).view_as(content)
        prepared["inputs_embeds"] = content
        return prepared

    def _positions(self, batch, positions):
        types = batch["token_types"]
        positions = batch.get("position_ids") if positions is None else positions
        if positions is None:
            positions = torch.zeros((3, *types.shape), device=types.device, dtype=torch.long)
            valid = types.ne(3)
            segments = batch.get("segment_ids", torch.where(valid, 0, -1))
            for row in range(types.shape[0]):
                for segment in segments[row, valid[row] & segments[row].ge(0)].unique():
                    indices = torch.where(valid[row] & segments[row].eq(segment))[0]
                    local = types[row, indices][None]
                    if bool(local.eq(1).any()):
                        raise ValueError("Visual input requires native position_ids from the multimodal backbone")
                    else:
                        local_positions = torch.arange(indices.numel(), device=types.device)[None, None].expand(3, 1, -1)
                    positions[:, row, indices] = local_positions[:, 0]
            return positions
        positions = positions.to(device=types.device)
        if positions.shape == types.shape:
            return positions[None].expand(3, -1, -1)
        if positions.ndim != 3 or positions.shape[1:] != types.shape or positions.shape[0] not in {2, 3, 4}:
            raise ValueError("position_ids must be [B,L], [2,B,L], [3,B,L] or native [4,B,L]")
        if positions.shape[0] == 4:
            return positions[1:]
        if positions.shape[0] == 2:
            # Legacy UMM images have H/W axes. Their T coordinate stays at the
            # image anchor; text uses the same scalar coordinate on all axes.
            image = types.eq(1)
            starts = image & ~F.pad(image[:, :-1], (1, 0), value=False)
            indices = torch.arange(types.shape[1], device=types.device)[None].expand_as(types)
            anchors = torch.where(starts, indices, 0).cummax(-1).values
            temporal = torch.where(image, positions[0].gather(1, anchors), positions[0])
            return torch.cat((temporal[None], positions), dim=0)
        return positions

    def forward(self, batch, *, return_content=False, query_stream=True, positions=None, masks=None):
        batch = {**batch, "token_types": batch.get("token_types", torch.zeros_like(batch["input_ids"]))}
        content = self.embed_content(batch)
        plan = batch.get("_cq_plan") if query_stream else None
        if plan is None:
            metadata = cpu_metadata(batch)
            pos = self._positions(metadata, None if positions is None else positions.cpu())
            plan, indices = cq_plan(metadata, self.cfg, pos, query_stream=query_stream)
            plan, indices = plan.to(content.device), indices.to(content.device)
        else:
            indices = batch["_content_positions"]
        if masks is not None:
            raise ValueError("Explicit visibility masks are replaced by prepared CQ layouts")
        n = len(indices)
        content = content.reshape(-1, self.cfg.width).index_select(0, indices)[:, None]
        query_ids = torch.where(plan.query_image, self.config.image_mask_token_id, self.config.mask_token_id)
        query = self.embed_tokens(query_ids).transpose(0, 1)
        hidden = torch.cat((content, query)).contiguous()
        rope = (plan.rope_cos.to(hidden.dtype), plan.rope_sin.to(hidden.dtype))
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, rope, plan, n, use_reentrant=True)
            else:
                hidden = layer(hidden, rope, plan, n)
        shape = (*batch["input_ids"].shape, self.cfg.width)
        output = dict(query_hidden=None)
        if query_stream:
            values = self.norm(hidden[n:], self.cfg.fused)[:, 0]
            output["query_hidden"] = values.new_zeros(plan.output_size, self.cfg.width).index_copy_(0, plan.output_positions, values).view(shape)
        if return_content or not query_stream:
            values = self.norm(hidden[:n], self.cfg.fused)[:, 0]
            output["content_hidden"] = values.new_zeros(plan.output_size, self.cfg.width).index_copy_(0, indices, values).view(shape)
        return output
