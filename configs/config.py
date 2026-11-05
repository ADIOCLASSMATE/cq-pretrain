"""Grouped run configuration, with direct attributes in the execution path.

Sections in YAML have distinct ownership; RunConfig is a plain dataclass so
batch preparation and numerical code need no proxy objects or dynamic lookup.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class RuntimeConfig:
    gdn_backend: str = "ascendc"
    ce_chunk: int = 256
    checkpoint_backbone: bool = False
    checkpoint_dit: bool = False
    checkpoint_backbone_layers: int = 0
    checkpoint_dit_layers: int = 0
    keep_attention_activations: int = 0
    fused: bool = True
    checkpoint_vision: bool = False


@dataclass
class ModelConfig(RuntimeConfig):
    method: str = "CQ_IT"
    content_visibility: str = "visual_block"
    image_condition_queries: int = 64
    visual_query_mode: str = "native"
    wan_assets: str = "/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Wan2.1-T2V-1.3B"
    wan_vae_assets: str = "/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Wan2.1-VAE"
    wan_freq_dim: int = 256
    text_mask_token_id: int = 248063
    image_mask_token_id: int = 248056
    architecture: str = "qwen35_cq"
    text_assets: str = "/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Qwen3.5-2B-Base"
    vocab_size: int = 248320
    width: int = 2048
    layers: int = 24
    heads: int = 8
    kv_heads: int = 2
    head_dim: int = 256
    ffn: int = 6144
    linear_value_heads: int = 0
    qwen_fused_projections: bool = True
    tie_word_embeddings: bool = True
    renderer: str = "wan"
    moss_assets: str = ""
    moss_fused_qkv: bool = False
    moss_cross_attention: bool = False
    linear_heads: int = 16
    linear_head_dim: int = 128
    linear_conv_kernel: int = 4
    full_attention_interval: int = 4
    partial_rotary_factor: float = 0.25
    mrope_section: tuple = (11, 11, 10)
    rope_theta: float = 10000000.0
    norm_eps: float = 1e-6
    image_size: int = 512
    patch_size: int = 16
    latent_channels: int = 16
    dit_width: int = 1536
    dit_heads: int = 12
    dit_layers: int = 30
    dit_ffn: int = 8960
    flow_draws: int = 1
    vision_encoder: str = "qwen35"
    vision_assets: str = "/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Qwen3.5-2B-Base"
    freeze_vision: bool = True
    freeze_vision_merger: bool = True
    vision_tensor_parallel: bool = False

    @property
    def image_tokens(self):
        return (self.image_size // self.patch_size) ** 2

    def validate_model(self, tp=1):
        import math

        if self.content_visibility not in ("causal", "visual_block"):
            raise ValueError("content_visibility must be causal or visual_block")
        if self.architecture != "qwen35_cq":
            raise ValueError(
                "The active backbone is Qwen3.5 CQ; legacy full-attention checkpoints require their archived source"
            )
        if tp not in (1, 2, 4):
            raise ValueError("Supported TP sizes: 1, 2, 4")
        if self.linear_heads % tp or (self.linear_value_heads or self.linear_heads) % tp:
            raise ValueError("GDN head counts must divide TP")
        if (self.linear_value_heads or self.linear_heads) % self.linear_heads:
            raise ValueError("GDN grouped heads must be integral")
        if self.renderer == "moss" and (self.flow_draws != 1 or self.latent_channels != 128):
            raise ValueError("MOSS uses RF1 and packed 128-channel latents")
        if self.full_attention_interval != 4:
            raise ValueError("Qwen3.5 CQ uses three GatedDeltaNet layers per gated-attention layer")
        if min(self.linear_heads, self.linear_head_dim, self.linear_conv_kernel) < 1:
            raise ValueError("Invalid linear attention dimensions")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim % 2 or sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError("MRoPE sections must cover the partial rotary dimensions")
        if self.method not in ("CQ_IT", "Bi_I+CQ_T"):
            raise ValueError("method must be CQ_IT or Bi_I+CQ_T")
        q = self.image_condition_queries
        if self.visual_query_mode not in ("fixed", "native"):
            raise ValueError("visual_query_mode must be fixed or native")
        if self.gdn_backend not in ("ascendc", "mindspeed"):
            raise ValueError("gdn_backend must be ascendc or mindspeed")
        if self.renderer == "wan" and (self.flow_draws != 1 or self.latent_channels != 16):
            raise ValueError("Wan uses one flow draw and 16-channel latents")
        if q < 1 or (self.renderer != "wan" and math.isqrt(q) ** 2 != q):
            raise ValueError("Image query count must be a positive square")
        if self.vision_encoder not in ("rgb_patch", "qwen35"):
            raise ValueError("vision_encoder must be rgb_patch or qwen35")
        if (self.freeze_vision or self.freeze_vision_merger) and self.vision_encoder != "qwen35":
            raise ValueError("Vision freezing is defined for the Qwen vision encoder and merger")
        if self.vision_encoder == "rgb_patch" and q > self.image_tokens:
            raise ValueError("Legacy RGB queries must fit the RGB grid")
        if any(not 0 <= t < self.vocab_size for t in (self.text_mask_token_id, self.image_mask_token_id)):
            raise ValueError("Mask embeddings must be valid vocabulary entries")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        for name in ("heads", "kv_heads", "ffn", "dit_heads", "dit_ffn", "vocab_size"):
            if getattr(self, name) % tp:
                raise ValueError(f"{name} must be divisible by TP={tp}")
        if self.heads % self.kv_heads or self.head_dim % 4:
            raise ValueError("GQA ratio must be integral and 2D RoPE head_dim divisible by four")
        if self.dit_width % self.dit_heads or (self.dit_width // self.dit_heads) % 4:
            raise ValueError("DiT head_dim must be divisible by four")
        if min(self.layers, self.dit_layers, self.flow_draws, self.ce_chunk) < 1:
            raise ValueError("layer, draw and chunk counts must be positive")
        if not 0 <= self.checkpoint_backbone_layers <= self.layers:
            raise ValueError("Invalid count of recomputed backbone layers")
        if not 0 <= self.checkpoint_dit_layers <= self.dit_layers:
            raise ValueError("Invalid count of recomputed DiT layers")
        if not 0 <= self.keep_attention_activations <= self.layers // self.full_attention_interval:
            raise ValueError("Invalid count of retained full-attention activations")


@dataclass
class DataConfig:
    visual_cache: str = ""
    generation_max_pixels: int = 1048576
    generation_token_budget: int = 8192
    preserve_documents: bool = False
    long_document_policy: str = "window"
    generation_microbatch_images: int = 0
    i2t_image_loss_probability: float = 1.0
    visual_uncond_probability: float = 0.1
    cfg_null_condition: str = "empty"
    vision_min_pixels: int = 65536
    vision_max_pixels: int = 1048576


@dataclass
class LossConfig:
    text_weight: float = 0.05
    image_weight: float = 1.0


@dataclass
class RunConfig(ModelConfig, DataConfig, LossConfig):
    def validate(self, tp=1):
        self.validate_model(tp)
        if self.generation_max_pixels and self.generation_max_pixels < 256:
            raise ValueError("Generation must fit one 16x16 cell")
        if self.generation_token_budget < 0 or self.generation_microbatch_images < 0:
            raise ValueError("Generation microbatch limits must be nonnegative")
        if not 0 <= self.i2t_image_loss_probability <= 1:
            raise ValueError("I2T image loss probability must be between 0 and 1")
        if self.i2t_image_loss_probability != 1 and self.renderer not in ("moss", "wan"):
            raise ValueError("I2T image loss sampling requires the MOSS renderer")
        if not 0 <= self.visual_uncond_probability <= 1:
            raise ValueError("Unconditional probability must be between 0 and 1")
        if self.cfg_null_condition not in ("empty", "empty_t2i", "i2t_prefix"):
            raise ValueError("CFG null condition must be empty, empty_t2i or i2t_prefix")
        if self.renderer == "wan" and self.cfg_null_condition == "i2t_prefix":
            raise ValueError("Wan understanding inputs have no generation queries")
        if self.renderer == "wan" and self.visual_uncond_probability and self.cfg_null_condition != "empty":
            raise ValueError("Wan unconditional training requires the empty query-read condition")
        if self.preserve_documents and (
            self.renderer not in ("moss", "wan") or self.vision_encoder != "qwen35"
        ):
            raise ValueError("Whole-document packing requires the native Qwen/MOSS path")
        if self.long_document_policy not in ("window", "error"):
            raise ValueError("Long documents require an explicit window or error policy")
        if self.visual_cache and not (self.freeze_vision and self.freeze_vision_merger):
            raise ValueError("Final visual cache requires frozen ViT and merger")
        if not 1024 <= self.vision_min_pixels <= self.vision_max_pixels:
            raise ValueError("Require 1024 <= vision_min_pixels <= vision_max_pixels")

    @classmethod
    def load(cls, path):
        value = Path(path).read_text()
        if Path(path).suffix in (".yaml", ".yml"):
            import yaml

            value = yaml.safe_load(value)
            if "model" not in value:
                return cls(**value)
            resolved = {}
            for section in ("model", "data", "loss", "runtime"):
                for name, setting in value.get(section, {}).items():
                    if name in resolved:
                        raise ValueError(f"Duplicate configuration field: {name}")
                    resolved[name] = setting
            return cls(**resolved)
        return cls(**json.loads(value))

    def to_dict(self):
        values = asdict(self)
        values["mrope_section"] = list(values["mrope_section"])
        return values


def tiny_config(**kwargs):
    values = dict(
        renderer="legacy",
        qwen_fused_projections=False,
        freeze_vision=False,
        freeze_vision_merger=False,
        vocab_size=128,
        width=64,
        layers=4,
        heads=4,
        kv_heads=2,
        head_dim=16,
        ffn=128,
        image_size=32,
        patch_size=16,
        dit_width=64,
        dit_heads=4,
        dit_layers=2,
        dit_ffn=128,
        flow_draws=2,
        ce_chunk=16,
        checkpoint_backbone=False,
        checkpoint_dit=False,
        image_condition_queries=4,
        visual_query_mode="fixed",
        visual_uncond_probability=0.0,
        cfg_null_condition="empty_t2i",
        linear_heads=2,
        linear_head_dim=8,
        linear_conv_kernel=4,
        partial_rotary_factor=0.5,
        mrope_section=(2, 1, 1),
        vision_encoder="rgb_patch",
        text_assets="",
        text_mask_token_id=4,
        image_mask_token_id=0,
    )
    values.update(kwargs)
    return RunConfig(**values)
