"""Gemma 4 E2B decoder adapter for controlled CQ training benchmarks only.

Native PLE, KV sharing, MQA, wide MLPs and RoPE are retained. The existing
MCore optimizer and TP projections are shared with the Qwen training path.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import distributed.megatron as P
import ops.functional as K
from ops.attention_layout import plan_attention
from ops.attention import tiled_attention


class Norm(nn.Module):
    def __init__(self, width, eps, scale=True):
        super().__init__(); self.eps = eps
        if scale: self.weight = nn.Parameter(torch.ones(width))
        else: self.register_buffer('weight', torch.ones(width), persistent=False)

    def forward(self, x):
        return K.rms_norm(x, self.weight.float(), self.eps).to(x.dtype)


class MLP(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        width = cfg.intermediate_size * (2 if cfg.use_double_wide_mlp and index >= cfg.num_hidden_layers-cfg.num_kv_shared_layers else 1)
        self.gate_up_proj = P.ColumnLinear(cfg.hidden_size, 2*width, stride=2)
        self.down_proj = P.RowLinear(width, cfg.hidden_size)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, -1)
        return self.down_proj(F.gelu(gate, approximate='tanh') * up)


def attention_tiles(mask, chunk):
    return plan_attention(mask, chunk, trim_key_start=True)


class Attention(nn.Module):
    def __init__(self, cfg, index):
        super().__init__(); self.cfg = cfg
        self.kind = cfg.layer_types[index]
        self.dim = cfg.global_head_dim if self.kind == 'full_attention' else cfg.head_dim
        self.shared = index >= cfg.num_hidden_layers-cfg.num_kv_shared_layers
        self.q_proj = P.ColumnLinear(cfg.hidden_size, cfg.num_attention_heads*self.dim)
        self.o_proj = P.RowLinear(cfg.num_attention_heads*self.dim, cfg.hidden_size)
        self.q_norm = Norm(self.dim, cfg.rms_norm_eps)
        if not self.shared:
            # E2B has one KV head. Replicate it over TP and sum its partial grads.
            self.k_proj = nn.Linear(cfg.hidden_size, self.dim, bias=False)
            self.v_proj = nn.Linear(cfg.hidden_size, self.dim, bias=False)
            self.k_norm = Norm(self.dim, cfg.rms_norm_eps)
            self.v_norm = Norm(self.dim, cfg.rms_norm_eps, scale=False)

    def forward(self, x, rope, tiles, valid, content_length, shared_k, shared_v):
        n, batch, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(n, batch, -1, self.dim))
        q = K.rotary(q.contiguous(), *rope)
        if self.shared: k, v = shared_k, shared_v
        else:
            content = x[:content_length]
            if P.tp_size() > 1:
                from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
                content = copy_to_tensor_model_parallel_region(content)
            k = self.k_norm(self.k_proj(content).unsqueeze(2))
            v = self.v_norm(self.v_proj(content).unsqueeze(2))
            k = K.rotary(k.contiguous(), rope[0][:content_length], rope[1][:content_length])
        y = tiled_attention(q, k, v, tiles.owner, valid, tiles, scale=1.)
        return self.o_proj(y), k, v


class Layer(nn.Module):
    def __init__(self, cfg, index):
        super().__init__(); w = cfg.hidden_size; eps = cfg.rms_norm_eps
        self.self_attn = Attention(cfg, index); self.mlp = MLP(cfg, index)
        for name in ('input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm',
                     'post_feedforward_layernorm', 'post_per_layer_input_norm'):
            setattr(self, name, Norm(w, eps))
        self.per_layer_input_gate = nn.Linear(w, cfg.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(cfg.hidden_size_per_layer_input, w, bias=False)
        self.register_buffer('layer_scalar', torch.ones(1))

    def forward(self, x, ple, rope, tiles, valid, content_length, shared_k=None, shared_v=None):
        y, k, v = self.self_attn(self.input_layernorm(x), rope, tiles, valid,
                                content_length, shared_k, shared_v)
        x = x + self.post_attention_layernorm(y)
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        y = F.gelu(self.per_layer_input_gate(x), approximate='tanh') * ple
        x = x + self.post_per_layer_input_norm(self.per_layer_projection(y))
        return x * self.layer_scalar, k, v


class GemmaDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.embed_tokens = P.make_embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_tokens_per_layer = P.make_embedding(cfg.vocab_size_per_layer_input,
            cfg.num_hidden_layers * cfg.hidden_size_per_layer_input)
        self.per_layer_model_projection = nn.Linear(cfg.hidden_size,
            cfg.num_hidden_layers*cfg.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection_norm = Norm(cfg.hidden_size_per_layer_input, cfg.rms_norm_eps)
        self.layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = Norm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, ids, visual_indices, features, content_length, ropes, tiles, valid, recompute=True):
        cfg = self.cfg
        hidden = self.embed_tokens(ids).transpose(0, 1)
        hidden = torch.where(ids.T[..., None] == cfg.pad_token_id, hidden.detach(), hidden)
        # Native embedding scale is rounded to the activation dtype.
        hidden = hidden * hidden.new_tensor(math.sqrt(cfg.hidden_size))
        if visual_indices.numel():
            hidden = hidden.index_copy(0, visual_indices, features[:, None])
        ple = self.embed_tokens_per_layer(ids).transpose(0, 1)
        ple = torch.where(ids.T[..., None] == cfg.pad_token_id, ple.detach(), ple)
        ple = ple * ple.new_tensor(math.sqrt(cfg.hidden_size_per_layer_input))
        ple = ple.unflatten(-1, (cfg.num_hidden_layers, cfg.hidden_size_per_layer_input))
        projection = self.per_layer_model_projection(hidden) * cfg.hidden_size**-.5
        projection = projection.unflatten(-1, (cfg.num_hidden_layers, cfg.hidden_size_per_layer_input))
        ple = (ple + self.per_layer_projection_norm(projection)) * 2**-.5
        shared = {}
        for i, layer in enumerate(self.layers):
            kind = cfg.layer_types[i]
            kv = shared.get(kind, (None, None)) if layer.self_attn.shared else (None, None)
            args = (hidden, ple[:, :, i], ropes[kind], tiles[kind], valid, content_length, *kv)
            hidden, k, v = checkpoint(layer, *args, use_reentrant=True) if recompute else layer(*args)
            if not layer.self_attn.shared: shared[kind] = (k, v)
        return self.norm(hidden[content_length:])

    def checkpoint_targets(self):
        targets = {}
        for name, param in self.named_parameters():
            name = name.replace('.impl.', '.')
            if '.gate_up_proj.' in name:
                for label, view in zip(('gate_proj', 'up_proj'), param.chunk(2, 0)):
                    targets[name.replace('gate_up_proj', label)] = P.mark_sharded(view)
            else: targets[name] = param
        targets.update({name: value for name, value in self.named_buffers() if name.endswith('layer_scalar')})
        return targets


class SoftcapCE(torch.autograd.Function):
    """The project's recomputed vocab-parallel CE, with Gemma's native softcap."""
    @staticmethod
    def probabilities(h, w, labels, cap):
        logits = F.linear(h, w).float()
        tanh = (logits / cap).tanh(); logits = tanh * cap
        ps = P.state(); size = P.tp_size(); group = ps.get_tensor_model_parallel_group() if ps else None
        rank = ps.get_tensor_model_parallel_rank() if ps else 0
        maximum = logits.max(-1).values
        if size > 1: torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
        logits -= maximum[:, None]
        local = labels - rank*w.shape[0]; owned = (local >= 0) & (local < w.shape[0])
        index = local.clamp(0, w.shape[0]-1)
        selected = logits.gather(1, index[:, None]).flatten() * owned
        if size > 1: torch.distributed.all_reduce(selected, group=group)
        logits.exp_(); denominator = logits.sum(-1)
        if size > 1: torch.distributed.all_reduce(denominator, group=group)
        return denominator.log()-selected, logits/denominator[:, None], tanh, index, owned

    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk, cap):
        ctx.save_for_backward(hidden, weight, labels); ctx.chunk = chunk; ctx.cap = cap
        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, len(hidden), chunk):
            total += SoftcapCE.probabilities(hidden[i:i+chunk], weight, labels[i:i+chunk], cap)[0].sum()
        return total / len(hidden)

    @staticmethod
    def backward(ctx, grad):
        h, w, labels = ctx.saved_tensors
        dh = torch.empty_like(h); dw = torch.zeros_like(w, dtype=torch.float32)
        for i in range(0, len(h), ctx.chunk):
            x = h[i:i+ctx.chunk]
            _, prob, tanh, index, owned = SoftcapCE.probabilities(x, w, labels[i:i+ctx.chunk], ctx.cap)
            prob.scatter_add_(1, index[:, None], -owned[:, None].to(prob.dtype))
            dy = (prob * (1-tanh.square()) * (grad.float()/len(h))).to(h.dtype)
            dh[i:i+ctx.chunk] = dy @ w
            if h.device.type == 'npu': K.fused_wgrad(x, dy, dw)
            else: dw.addmm_(dy.float().T, x.float())
        if P.tp_size() > 1: torch.distributed.all_reduce(dh, group=P.state().get_tensor_model_parallel_group())
        return dh, dw.to(w.dtype), None, None, None
