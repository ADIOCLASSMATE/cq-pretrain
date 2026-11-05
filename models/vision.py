"""Native Qwen vision weights; GEMM patch embedding and varlen NPU attention."""

from contextlib import nullcontext
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import infra.parallel as P
from utils.assets import asset_path, asset_manifest, vision_config


class PatchProjection(nn.Module):
    """Conv3d checkpoint layout, GEMM execution and normal module gather hooks."""

    def __init__(self, cfg):
        super().__init__()
        kernel = (cfg["temporal_patch_size"], cfg["patch_size"], cfg["patch_size"])
        self.weight = nn.Parameter(torch.empty(cfg["hidden_size"], cfg["in_channels"], *kernel))
        self.bias = nn.Parameter(torch.empty(cfg["hidden_size"]))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(cfg["in_channels"] * math.prod(kernel))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, pixels):
        return F.linear(pixels.to(self.weight.dtype), self.weight.flatten(1), self.bias)


class VisionPatchEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Avoid torch_npu's automatic Conv3d weight-format conversion. Keep
        # the exact public weight names and invoke proj to preserve DDP hooks.
        self.proj = PatchProjection(cfg)

    def forward(self, pixels):
        return self.proj(pixels)


def rotary(x, cos, sin):
    a, b = x.float().chunk(2, dim=-1)
    return (x.float() * cos[:, None].float() + torch.cat((-b, a), dim=-1) * sin[:, None].float()).to(x.dtype)


def column(cfg, in_features, out_features, stride=1):
    if cfg.get("_tensor_parallel") and P.tp_size() > 1:
        return P.ColumnLinear(in_features, out_features, bias=True, stride=stride)
    return nn.Linear(in_features, out_features)


def row(cfg, in_features, out_features):
    if cfg.get("_tensor_parallel") and P.tp_size() > 1:
        return P.RowLinear(in_features, out_features, bias=True)
    return nn.Linear(in_features, out_features)


class VisionAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["hidden_size"]
        self.dim = width // cfg["num_heads"]
        self.heads = cfg["num_heads"] // (P.tp_size() if cfg.get("_tensor_parallel") else 1)
        self.qkv = column(cfg, width, 3 * width, stride=3)
        self.proj = row(cfg, width, width)

    def forward(self, x, layout, fused):
        q, k, v = self.qkv(x).reshape(len(x), 3, self.heads, self.dim).unbind(1)
        q, k = rotary(q, layout.cos, layout.sin), rotary(k, layout.cos, layout.sin)
        if x.device.type == "npu" and fused:
            import torch_npu

            result = torch_npu.npu_fusion_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                self.heads,
                input_layout="TND",
                scale=self.dim**-0.5,
                keep_prob=1.0,
                pre_tockens=2147483647,
                next_tockens=2147483647,
                sparse_mode=0,
                actual_seq_qlen=layout.ends,
                actual_seq_kvlen=layout.ends,
            )[0]
        else:
            # Reference/CPU path. Never mix keys from different packed images.
            outputs = []
            for qi, ki, vi in zip(q.split(layout.lengths), k.split(layout.lengths), v.split(layout.lengths)):
                outputs.append(
                    F.scaled_dot_product_attention(
                        qi.transpose(0, 1)[None],
                        ki.transpose(0, 1)[None],
                        vi.transpose(0, 1)[None],
                        dropout_p=0.0,
                        is_causal=False,
                    )[0].transpose(0, 1)
                )
            result = torch.cat(outputs)
        return self.proj(result.reshape(len(x), -1))


class VisionMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear_fc1 = column(cfg, cfg["hidden_size"], cfg["intermediate_size"])
        self.linear_fc2 = row(cfg, cfg["intermediate_size"], cfg["hidden_size"])

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg["hidden_size"], eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg["hidden_size"], eps=1e-6)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def forward(self, x, layout, fused):
        x = x + self.attn(self.norm1(x), layout, fused)
        return x + self.mlp(self.norm2(x))


class VisionMerger(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width = cfg["hidden_size"] * cfg["spatial_merge_size"] ** 2
        self.norm = nn.LayerNorm(cfg["hidden_size"], eps=1e-6)
        self.linear_fc1 = column(cfg, self.width, self.width)
        self.linear_fc2 = row(cfg, self.width, cfg["out_hidden_size"])

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(self.norm(x).reshape(-1, self.width))))


class QwenVisionTower(nn.Module):
    def __init__(self, cfg, fused=True, recompute=False, freeze_encoder=False, freeze_merger=False):
        super().__init__()
        self.cfg, self.fused, self.recompute = cfg, fused, recompute and not freeze_encoder
        self.freeze_encoder, self.freeze_merger = freeze_encoder, freeze_merger
        self.patch_embed = VisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg["num_position_embeddings"], cfg["hidden_size"])
        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg["depth"])])
        self.merger = VisionMerger(cfg)
        if freeze_encoder:
            for module in (self.patch_embed, self.pos_embed, self.blocks):
                module.requires_grad_(False)
        if freeze_merger:
            self.merger.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            for module in (self.patch_embed, self.pos_embed, self.blocks):
                module.eval()
        if self.freeze_merger:
            self.merger.eval()
        return self

    def encode(self, layout):
        x = self.patch_embed(layout.pixel_values)
        values = self.pos_embed(layout.pos_indices) * layout.pos_weights.to(x.dtype)[..., None]
        # Preserve the official accumulation order, including BF16 rounding.
        x = x + (values[0] + values[1] + values[2] + values[3])
        for block in self.blocks:
            if self.recompute and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, layout, self.fused, use_reentrant=True)
            else:
                x = block(x, layout, self.fused)
        return x

    def forward(self, layout):
        # Frozen visual features are ordinary tensors, not inference tensors:
        # a trainable merger must be able to save them for its weight gradient.
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            x = self.encode(layout)
        return self.merger(x)


class QwenVisionInput(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vc = vision_config(cfg.vision_assets)
        vc["_tensor_parallel"] = cfg.vision_tensor_parallel
        self.visual = QwenVisionTower(
            vc, cfg.fused, cfg.checkpoint_vision, cfg.freeze_vision, cfg.freeze_vision_merger
        )
        self.bridge = (
            nn.Identity()
            if vc["out_hidden_size"] == cfg.width
            else nn.Linear(vc["out_hidden_size"], cfg.width, bias=False)
        )
        if cfg.freeze_vision and cfg.freeze_vision_merger:
            self.bridge.requires_grad_(False)

    def load_pretrained(self, path):
        from safetensors import safe_open

        path = asset_path(path)
        manifest = asset_manifest(path)
        targets = {name.replace(".impl.", "."): p for name, p in self.visual.named_parameters()}
        seen = set()
        files = (
            [path / "vision.safetensors"]
            if (path / "vision.safetensors").exists()
            else sorted(path.glob("model*.safetensors"))
        )
        for file in files:
            with safe_open(str(file), framework="pt", device="cpu") as values, torch.no_grad():
                for key in values.keys():
                    name = key.removeprefix("model.visual.")
                    if name not in targets:
                        continue
                    p = targets[name]
                    value = P.checkpoint_shard(values.get_tensor(key), p)
                    if value.shape != p.shape:
                        raise ValueError(f"Vision weight shape mismatch: {key}")
                    p.copy_(value)
                    seen.add(name)
        if seen != set(targets):
            raise ValueError(f"Missing vision weights: {set(targets) - seen}")
        return manifest

    def forward(self, layout):
        if layout.features is None:
            return self.bridge(self.visual(layout))
        if not (self.visual.freeze_encoder and self.visual.freeze_merger):
            raise ValueError("Cached final visual features require a frozen encoder and merger")
        features = layout.features.to(self.visual.patch_embed.proj.weight.dtype)
        if layout.online is None:
            return features
        return features.index_copy(0, layout.online.content_indices, self.bridge(self.visual(layout.online)))
