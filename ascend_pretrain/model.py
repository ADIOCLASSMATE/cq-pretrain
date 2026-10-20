"""Joint Qwen3.5 CQ backbone and flow renderer; MOSS is the active recipe."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import kernels as K
from . import parallel as P
from .loss import linear_cross_entropy


class RGBPatchLinear(nn.Module):
    """Patchify raw RGB then GEMM; preserve Conv2d's exact weight layout/init.

    A 16x16 stride-16 convolution is mathematically identical but the measured
    CANN path spends unnecessary time on convolution layout conversions.
    """
    def __init__(self, width, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.weight = nn.Parameter(torch.empty(width, 3, patch_size, patch_size))
        self.bias = nn.Parameter(torch.empty(width))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1/math.sqrt(3*patch_size**2), 1/math.sqrt(3*patch_size**2))

    def forward(self, pixels):
        n, channels, height, width = pixels.shape
        p = self.patch_size
        patches = pixels.reshape(n, channels, height//p, p, width//p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(n, -1, channels*p*p)
        projected = F.linear(patches, self.weight.flatten(1), self.bias)
        return projected.transpose(1, 2).reshape(n, self.weight.shape[0], height//p, width//p)


class DiTAttention(nn.Module):
    def __init__(self, cfg, cross=False):
        super().__init__()
        self.cross, self.cfg = cross, cfg
        self.heads = cfg.dit_heads // P.tp_size()
        self.dim = cfg.dit_width // cfg.dit_heads
        if cross:
            self.q = P.ColumnLinear(cfg.dit_width, cfg.dit_width, bias=True)
            self.kv = P.ColumnLinear(cfg.dit_width, 2 * cfg.dit_width, bias=True, stride=2)
        else:
            self.qkv = P.ColumnLinear(cfg.dit_width, 3 * cfg.dit_width, bias=True, stride=3)
        self.out = P.RowLinear(cfg.dit_width, cfg.dit_width, bias=True)

    def forward(self, hidden, rope=None, memory=None):
        length, batch, _ = hidden.shape
        if self.cross:
            q = self.q(hidden).reshape(length, batch, self.heads, self.dim)
            k, v = self.kv(memory).reshape(memory.shape[0], batch, 2, self.heads, self.dim).unbind(2)
        else:
            q, k, v = self.qkv(hidden).reshape(length, batch, 3, self.heads, self.dim).unbind(2)
            q, k = (K.rotary(t.contiguous(), *rope, self.cfg.fused) for t in (q, k))
        return self.out(K.attention(q, k, v, fused=self.cfg.fused))


class DiTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width = cfg.dit_width
        self.fused = cfg.fused
        self.self_attention = DiTAttention(cfg)
        self.cross_attention = DiTAttention(cfg, cross=True)
        self.ada = nn.Linear(cfg.dit_width, 6 * cfg.dit_width)
        self.up = P.ColumnLinear(cfg.dit_width, cfg.dit_ffn, bias=True)
        self.down = P.RowLinear(cfg.dit_ffn, cfg.dit_width, bias=True)

    def forward(self, hidden, time, memory, rope):
        sa, ca, ga, sm, cm, gm = self.ada(F.silu(time)).chunk(6, -1)
        norm = K.modulated_norm(hidden, ca, sa, self.fused)
        hidden = hidden + ga * self.self_attention(norm, rope=rope)
        hidden = hidden + self.cross_attention(F.layer_norm(hidden, (self.width,), eps=1e-6), memory=memory)
        norm = K.modulated_norm(hidden, cm, sm, self.fused)
        return hidden + gm * self.down(F.gelu(self.up(norm), approximate='tanh'))


class FlowRenderer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.condition_proj = nn.Linear(cfg.width, cfg.width)
        self.memory_proj = nn.Linear(cfg.width, cfg.dit_width)
        self.input_proj = nn.Linear(cfg.latent_channels, cfg.dit_width)
        self.time_embed = nn.Sequential(nn.Linear(256, cfg.dit_width), nn.SiLU(),
                                        nn.Linear(cfg.dit_width, cfg.dit_width))
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.dit_layers)])
        self.final_ada = nn.Linear(cfg.dit_width, 2 * cfg.dit_width)
        self.output = nn.Linear(cfg.dit_width, cfg.latent_channels)
        self.register_buffer('time_frequencies', torch.exp(-math.log(10000) * torch.arange(128).float() / 128), persistent=False)
        local = torch.arange(cfg.image_tokens)
        side = math.isqrt(cfg.image_tokens)
        positions = torch.stack((local // side, local % side))[:, :, None]
        cos, sin = K.rope_tables(positions, cfg.dit_width // cfg.dit_heads, 10000., torch.float32)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        for block in self.blocks:
            nn.init.zeros_(block.ada.weight)
            nn.init.zeros_(block.ada.bias)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, conditions, batch):
        cfg = self.cfg
        target = batch.latents.repeat(cfg.flow_draws, 1, 1).float()
        time = batch.times
        xt = ((1 - time[:, None, None]) * batch.noise.float() + time[:, None, None] * target)
        # The expensive backbone condition is computed once, shared by RF draws.
        memory = self.memory_proj(self.condition_proj(conditions))
        memory = memory.repeat(cfg.flow_draws, 1, 1).transpose(0, 1).contiguous()
        hidden = self.input_proj(xt.to(conditions.dtype)).transpose(0, 1).contiguous()
        frequencies = time[:, None].float() * 1000 * self.time_frequencies.float()[None]
        emb = torch.cat((frequencies.cos(), frequencies.sin()), -1).to(conditions.dtype)
        emb = self.time_embed(emb).unsqueeze(0)
        rope = (self.rope_cos.to(hidden.dtype), self.rope_sin.to(hidden.dtype))
        for index, block in enumerate(self.blocks):
            recompute = cfg.checkpoint_dit or index >= cfg.dit_layers - cfg.checkpoint_dit_layers
            if recompute and self.training and torch.is_grad_enabled():
                # Reentrant recomputation preserves Parameter.main_grad used by
                # MCore's fused weight-gradient accumulation. Non-reentrant
                # saved-tensor hooks reconstruct a plain Tensor without it.
                hidden = checkpoint(block, hidden, emb, memory, rope, use_reentrant=True)
            else:
                hidden = block(hidden, emb, memory, rope)
        shift, scale = self.final_ada(F.silu(emb)).chunk(2, -1)
        norm = K.modulated_norm(hidden, scale, shift, cfg.fused)
        prediction = self.output(norm).transpose(0, 1)
        return (prediction.float() - (target - batch.noise.float())).square().mean()


class OutputHead(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.weight=P.make_embedding(cfg.vocab_size,cfg.width).weight
        # This untied parameter is used only by CE, so its gradient can go
        # straight into MCore's persistent FP32 accumulation buffer.
        self.weight._ce_direct_accumulation=True

    def forward(self,hidden,targets,chunk):
        return linear_cross_entropy(hidden,self.weight,targets,chunk)


class JointModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate(P.tp_size())
        self.cfg = cfg
        self.config = P._config
        self.embedding = P.make_embedding(cfg.vocab_size, cfg.width)
        if cfg.vision_encoder == 'qwen35':
            from .vision import QwenVisionInput
            self.patch_embed = QwenVisionInput(cfg)
        else:
            self.patch_embed = RGBPatchLinear(cfg.width, cfg.patch_size)
        from .qwen35 import QwenBlock, RMSNorm
        self.layers = nn.ModuleList([QwenBlock(cfg,index) for index in range(cfg.layers)])
        self.final_norm = RMSNorm(cfg.width,cfg.norm_eps)
        if cfg.renderer=='moss':
            from .moss import MOSSRenderer
            self.renderer=MOSSRenderer(cfg)
        else:self.renderer = FlowRenderer(cfg)
        self.lm_head=None if cfg.tie_word_embeddings else OutputHead(cfg)

    @property
    def text_query(self):
        # CPU / TP1 inspection convenience. Forward uses vocab-parallel lookup.
        return self.embedding.weight[self.cfg.text_mask_token_id]

    @property
    def image_query(self):
        return self.embedding.weight[self.cfg.image_mask_token_id]

    def backbone(self, batch, reference=False):
        from .mainline import backbone
        return backbone(self,batch,reference)

    def forward(self, batch, reference=False, return_hidden=False, loss_scales=None, defer_renderer=False):
        hidden = self.backbone(batch, reference=reference)
        # Match module/parameter registration order: renderer then untied LM
        # head. Backward can then reduce the LM-head bucket first and overlap
        # later buckets with renderer/backbone work, without queueing ready
        # renderer gradients behind an uncomputed output-head gradient.
        image_loss=None
        image_condition=None
        if len(batch.image_indices):
            image_condition = hidden[batch.mainline.condition_indices]
            if not defer_renderer:image_loss = self.renderer(image_condition, batch)
        text_hidden = hidden[batch.text_indices]
        targets = batch.ids.flatten()[batch.text_indices]
        text_loss = ((self.lm_head(text_hidden,targets,self.cfg.ce_chunk) if self.lm_head is not None else
                      linear_cross_entropy(text_hidden, self.embedding.weight, targets, self.cfg.ce_chunk))
                     if targets.numel() else hidden.sum() * 0)
        if image_loss is None:image_loss = text_loss.detach() * 0
        text_scale,image_scale=loss_scales if loss_scales is not None else (1.,1.)
        result = dict(loss=self.cfg.text_weight * text_loss * text_scale + self.cfg.image_weight * image_loss * image_scale,
                      text_loss=text_loss, image_loss=image_loss)
        if return_hidden:
            result['hidden'] = hidden
        if defer_renderer:
            if self.cfg.renderer!='moss' or image_condition is None or not batch.generation.microbatches:
                raise ValueError('Deferred rendering needs CPU-planned MOSS microbatches')
            result['_image_conditions']=image_condition
            result['_image_scale']=image_scale
        return result
