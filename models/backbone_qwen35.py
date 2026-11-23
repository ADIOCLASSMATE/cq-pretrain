"""Qwen3.5 parameterization with CQ read-only streams and image blocks."""

import json
from utils.assets import asset_path, asset_manifest
import torch
from torch import nn
from torch.nn import functional as F
import ops.functional as K
import distributed.megatron as P
from ops.attention import attend


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x, fused=True):
        # Qwen3.5 checkpoints store the offset from one, including Q/K norms.
        if x.dtype == torch.float64:
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * (1 + self.weight)
        weight = 1 + self.weight.float()
        from ops.cuda_fused import enabled as _cuda_fused, rms_norm as _cuda_rms_norm

        if _cuda_fused(x, fused):
            # Published Triton RMSNorm; the offset weight convention is applied
            # here, exactly as in the explicit reference below.
            return _cuda_rms_norm(x, weight, self.eps)
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps) * weight).to(x.dtype)


def _cuda_swiglu_enabled(x):
    """True when the Inductor-fused SwiGLU should run instead of chunk+silu+mul."""
    if x.device.type != "cuda":
        return False
    from ops.cuda_fused import swiglu_enabled

    return swiglu_enabled(x)


def _cuda_ops_enabled(x):
    """True when the published fused SwiGLU / causal conv should run."""
    if x.device.type != "cuda":
        return False
    from ops.cuda_fused import ops_enabled

    return ops_enabled(x)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fused_projections = cfg.qwen_fused_projections
        if self.fused_projections:
            self.gate_up_proj = P.ColumnLinear(cfg.width, 2 * cfg.ffn, stride=2)
        else:
            self.gate_proj = P.ColumnLinear(cfg.width, cfg.ffn)
            self.up_proj = P.ColumnLinear(cfg.width, cfg.ffn)
        self.down_proj = P.RowLinear(cfg.ffn, cfg.width)

    def forward(self, x):
        if self.fused_projections:
            projected = self.gate_up_proj(x)
            if _cuda_swiglu_enabled(projected):
                from ops.cuda_fused import swiglu as _cuda_swiglu

                activated = _cuda_swiglu(projected)
            else:
                gate, up = projected.chunk(2, -1)
                activated = F.silu(gate) * up
            return self.down_proj(activated)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def recurrent_delta(q, k, v, g, beta, bounds, output_final_state=False):
    """Differentiable CPU oracle. NPU training must use the fused backend."""
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    outputs = []
    states = []
    for begin, end in zip(bounds, bounds[1:]):
        state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        for i in range(begin, end):
            state = state * g[:, i].exp()[..., None, None]
            residual = v[:, i] - (state * k[:, i, :, :, None]).sum(-2)
            state = state + k[:, i, :, :, None] * (residual * beta[:, i, :, None])[..., None, :]
            outputs.append((state * q[:, i, :, :, None]).sum(-2))
        if output_final_state:
            states.append(state)
    output = torch.stack(outputs, dim=1)
    return (output, torch.cat(states)) if output_final_state else output


class CausalConv(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel))
        nn.init.uniform_(self.weight, -0.5, 0.5)

    def forward(self, x, plan):
        if _cuda_ops_enabled(x):
            from ops.cuda_fused import causal_conv1d as _cuda_conv

            return _cuda_conv(x, self.weight[:, 0], plan.content_cu)
        results = []
        for a, b in zip(plan.content_bounds, plan.content_bounds[1:]):
            y = F.conv1d(x[:, a:b].transpose(1, 2), self.weight, padding=self.weight.shape[-1] - 1, groups=x.shape[-1])[
                ..., : b - a
            ]
            results.append(F.silu(y).transpose(1, 2))
        return torch.cat(results, dim=1)


class GatedNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, z):
        from ops.cuda_fused import enabled as _cuda_fused, gated_rms_norm as _cuda_gated_norm

        if _cuda_fused(x):
            # Same upstream FLA entry point the Ascend backend calls.
            return _cuda_gated_norm(x, z, self.weight, self.eps)
        compute = x if x.dtype == torch.float64 else x.float()
        normalized = compute * torch.rsqrt(compute.square().mean(-1, keepdim=True) + self.eps)
        return (normalized.to(x.dtype) * self.weight * F.silu(z.to(compute.dtype))).to(x.dtype)


class GatedDelta(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        tp = P.tp_size()
        key = cfg.linear_heads * cfg.linear_head_dim
        value = (cfg.linear_value_heads or cfg.linear_heads) * cfg.linear_head_dim
        size = value
        h = cfg.linear_value_heads or cfg.linear_heads
        if cfg.qwen_fused_projections:
            self.in_proj_all = P.ColumnLinear(cfg.width, 2 * key + 2 * value + 2 * h)
            P.mark_sharded(self.in_proj_all.weight, splits=(key, key, value, value, h, h))
        else:
            self.in_proj_qkv = P.ColumnLinear(cfg.width, 2 * key + value)
            P.mark_sharded(self.in_proj_qkv.weight, splits=(key, key, value))
            self.in_proj_z = P.ColumnLinear(cfg.width, size)
            self.in_proj_a = P.ColumnLinear(cfg.width, h)
            self.in_proj_b = P.ColumnLinear(cfg.width, h)
        self.conv1d = CausalConv((2 * key + value) // tp, cfg.linear_conv_kernel)
        P.mark_sharded(self.conv1d.weight, splits=(key, key, value))
        self.A_log = P.mark_sharded(
            nn.Parameter(torch.rand((cfg.linear_value_heads or cfg.linear_heads) // tp).mul(15.99).add(0.01).log())
        )
        self.dt_bias = P.mark_sharded(nn.Parameter(torch.ones((cfg.linear_value_heads or cfg.linear_heads) // tp)))
        self.norm = GatedNorm(cfg.linear_head_dim, cfg.norm_eps)
        self.out_proj = P.RowLinear(size, cfg.width)

    def project(self, flat):
        cfg = self.cfg
        size = cfg.linear_heads * cfg.linear_head_dim // P.tp_size()
        h = (cfg.linear_value_heads or cfg.linear_heads) // P.tp_size()
        if cfg.qwen_fused_projections:
            return self.in_proj_all(flat).split((2 * size + h * cfg.linear_head_dim, h * cfg.linear_head_dim, h, h), -1)
        return self.in_proj_qkv(flat), self.in_proj_z(flat), self.in_proj_a(flat), self.in_proj_b(flat)

    def forward(self, x, plan, cache=None):
        if x.shape[1] != 1:
            raise ValueError("CQ GDN currently requires microbatch 1")
        cfg = self.cfg
        tp = P.tp_size()
        hk = cfg.linear_heads // tp
        h = (cfg.linear_value_heads or cfg.linear_heads) // tp
        d = cfg.linear_head_dim
        size = hk * d
        raw, z, a, beta = self.project(x[:, 0])
        updates = raw.index_select(0, plan.content_order)
        convolved = self.conv1d(updates[None], plan)[0]
        q, k, v = convolved.split((size, size, h * d), -1)
        q, k, v = q.reshape(-1, hk, d), k.reshape(-1, hk, d), v.reshape(-1, h, d)
        # Query convolution reads the previous real-content cache; neither the
        # target nor other queries can influence this cache. Do not recover it
        # by subtracting the target from a rounded convolution output.
        raw_q = raw[:, :size].index_select(0, plan.query_source)
        weight = self.conv1d.weight[:size, 0]
        query = raw_q * weight[:, -1]
        history = torch.cat((updates[:, :size], updates.new_zeros(1, size)), dim=0)
        for j in range(cfg.linear_conv_kernel - 1):
            query = query + history.index_select(0, plan.query_history[:, j]) * weight[:, j]
        query = F.silu(query).reshape(-1, hk, d)
        # Bidirectional visual Q also write, each into a private continuation of
        # its own prefix. Their key and value take the same causal convolution
        # as their read vector: the last real C before the target, then Q itself.
        writers = plan.query_source[plan.query_write_start :] if plan.query_write_start >= 0 else plan.query_source[:0]
        if len(writers):
            if cache is not None:
                raise ValueError("Recurrent decoding has no bidirectional visual queries")
            weight = self.conv1d.weight[size:, 0]
            written = raw[:, size:].index_select(0, writers) * weight[:, -1]
            history = torch.cat((updates[:, size:], updates.new_zeros(1, updates.shape[1] - size)), dim=0)
            for j in range(cfg.linear_conv_kernel - 1):
                written = (
                    written + history.index_select(0, plan.query_history[plan.query_write_start :, j]) * weight[:, j]
                )
            qk, qv = F.silu(written).split((size, h * d), -1)
            k = torch.cat((k, qk.reshape(-1, hk, d).to(k.dtype)), dim=0)
            v = torch.cat((v, qv.reshape(-1, h, d).to(v.dtype)), dim=0)
        if h != hk:
            q, k, query = (t.repeat_interleave(h // hk, dim=1) for t in (q, k, query))
        q = torch.cat((q, query), dim=0).index_select(0, plan.event_q)[None]
        zeros = k.new_zeros(1, h, d)
        k = torch.cat((k, zeros), dim=0).index_select(0, plan.event_write)[None]
        v = torch.cat((v, zeros), dim=0).index_select(0, plan.event_write)[None]
        control_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        control_order = torch.cat((plan.content_order, writers)) if len(writers) else plan.content_order
        a = a.index_select(0, control_order).to(control_dtype)
        g = -self.A_log.to(control_dtype).exp() * F.softplus(a + self.dt_bias.to(control_dtype))
        beta = beta.index_select(0, control_order).sigmoid().to(control_dtype)
        control_zero = g.new_zeros(1, h)
        g = torch.cat((g, control_zero)).index_select(0, plan.event_write)[None]
        beta = torch.cat((beta, control_zero)).index_select(0, plan.event_write)[None]
        if cache is not None:
            if torch.is_grad_enabled():
                raise ValueError("CQ state caching is inference-only")
            if x.device.type == "cuda":
                # The fused chunk kernel returns every sequence's final state in
                # FP32; the Python recurrence is the CPU oracle only.
                from ops.cuda_gated_delta import chunk_gated_delta

                y, state = chunk_gated_delta(q, k, v, g=g, beta=beta, cu_seqlens=plan.event_cu, output_final_state=True)
            else:
                y, state = recurrent_delta(q, k, v, g, beta, plan.event_bounds, True)
            # Recurrent decode uses value-first state; only real C enters history.
            cache["state"] = state.transpose(-1, -2).contiguous()
            count = cfg.linear_conv_kernel - 1
            history = raw.new_zeros(1, count, raw.shape[-1])
            tail = updates[max(0, len(updates) - 1 - count) : -1]
            if len(tail):
                history[0, -len(tail) :].copy_(tail)
            cache["conv"] = history
        elif x.device.type == "cuda":
            from ops.cuda_gated_delta import chunk_gated_delta

            y = chunk_gated_delta(
                q,
                k,
                v,
                g=g,
                beta=beta,
                cu_seqlens=plan.event_cu,
            )[0]
        else:
            y = recurrent_delta(q, k, v, g, beta, plan.event_bounds)
        selected = y[0].index_select(0, plan.hidden_event).to(x.dtype)
        if plan.padding_hidden.numel():
            selected.index_fill_(0, plan.padding_hidden, 0)
        z = z.reshape(-1, h, d)
        return self.out_proj(self.norm(selected, z).flatten(-2))[:, None]


class GatedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_proj = P.ColumnLinear(cfg.width, 2 * cfg.heads * cfg.head_dim)
        if cfg.qwen_fused_projections:
            self.kv_proj = P.ColumnLinear(cfg.width, 2 * cfg.kv_heads * cfg.head_dim, stride=2)
        else:
            self.k_proj = P.ColumnLinear(cfg.width, cfg.kv_heads * cfg.head_dim)
            self.v_proj = P.ColumnLinear(cfg.width, cfg.kv_heads * cfg.head_dim)
        self.o_proj = P.RowLinear(cfg.heads * cfg.head_dim, cfg.width)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, rope, plan, content_length, reference=False, cache=None):
        cfg = self.cfg
        s = content_length
        n, b, _ = x.shape
        q, gate = self.q_proj(x).reshape(n, b, cfg.heads // P.tp_size(), 2 * cfg.head_dim).chunk(2, -1)
        q = self.q_norm(q, cfg.fused and not reference)
        # Bidirectional visual Q are the only queries that are also keys. Their
        # rows are the tail of the sequence, so one projection covers [C ; block Q].
        blocks = plan.query_blocks
        first = blocks[0][0] if blocks else n
        source = torch.cat((x[:s], x[first:])) if blocks else x[:s]
        k, v = (
            self.kv_proj(source).chunk(2, -1)
            if cfg.qwen_fused_projections
            else (self.k_proj(source), self.v_proj(source))
        )
        k = self.k_norm(
            k.reshape(len(source), b, cfg.kv_heads // P.tp_size(), cfg.head_dim), cfg.fused and not reference
        )
        v = v.reshape(len(source), b, cfg.kv_heads // P.tp_size(), cfg.head_dim)
        dim = int(cfg.head_dim * cfg.partial_rotary_factor)

        def rotate(t, cos, sin):
            return torch.cat(
                (K.rotary(t[..., :dim].contiguous(), cos, sin, cfg.fused and not reference), t[..., dim:]), -1
            )

        q = rotate(q, *rope)
        if blocks:
            k = rotate(k, torch.cat((rope[0][:s], rope[0][first:])), torch.cat((rope[1][:s], rope[1][first:])))
        else:
            k = rotate(k, rope[0][:s], rope[1][:s])
        if cache is not None:
            if torch.is_grad_enabled():
                raise ValueError("CQ state caching is inference-only")
            if blocks:
                raise ValueError("Recurrent decoding has no bidirectional visual queries")
            # The last C slot is the uncommitted target placeholder.
            cache["k"] = k.new_empty(cache["capacity"], *k.shape[1:])
            cache["v"] = v.new_empty(cache["capacity"], *v.shape[1:])
            cache["k"][: s - 1].copy_(k[:-1])
            cache["v"][: s - 1].copy_(v[:-1])
            cache["length"] = s - 1
        y = attend(
            q,
            k[:s],
            v[:s],
            plan.cq_mask,
            plan.cq_valid,
            None if reference else plan.cq_plan,
            cfg.fused and not reference,
            None if reference else plan.cq_predicate,
        )
        if blocks:
            # One softmax over [the C prefix this target may read ; every Q of
            # the same target]. Both ranges are contiguous, so no mask is built.
            pieces = [y[:first]]
            for row_start, row_stop, key_start, key_stop in blocks:
                own = slice(s + row_start - first, s + row_stop - first)
                pieces.append(
                    K.attention(
                        q[row_start:row_stop],
                        torch.cat((k[key_start:key_stop], k[own])),
                        torch.cat((v[key_start:key_stop], v[own])),
                        fused=cfg.fused and not reference,
                    )
                )
            if blocks[-1][1] != n or any(a[1] != b[0] for a, b in zip(blocks, blocks[1:])):
                raise ValueError("Bidirectional query blocks must tile the tail of the sequence")
            y = torch.cat(pieces)
        return self.o_proj(y * gate.flatten(-2).sigmoid())


class QwenBlock(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.cfg = cfg
        self.input_layernorm = RMSNorm(cfg.width, cfg.norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.width, cfg.norm_eps)
        if (index + 1) % cfg.full_attention_interval:
            self.linear_attn = GatedDelta(cfg)
        else:
            self.self_attn = GatedAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, hidden, rope, plan, content_length, reference=False, cache=None):
        norm = self.input_layernorm(hidden, self.cfg.fused and not reference)
        mixed = (
            self.linear_attn(norm, plan.delta_plan, cache)
            if hasattr(self, "linear_attn")
            else self.self_attn(norm, rope, plan, content_length, reference, cache)
        )
        hidden = hidden + mixed
        return hidden + self.mlp(self.post_attention_layernorm(hidden, self.cfg.fused and not reference))


def cast_for_training(model, device="cpu"):
    # Official Qwen checkpoint keeps these recurrence-sensitive parameters in
    # FP32. Preserve their values before casting the bulk weights to BF16.
    keep = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.endswith((".linear_attn.A_log", ".linear_attn.norm.weight"))
    }
    for name, module in model.named_modules():
        if getattr(module, "keep_fp32", False):
            keep.update({f"{name}.{key}": p.detach().clone() for key, p in module.named_parameters()})
    rope_bands = {name: b.detach().clone() for name, b in model.named_buffers() if name.endswith(".rope.bands")}
    model.to(device=device, dtype=torch.bfloat16)
    for name, p in model.named_parameters():
        if name in keep:
            p.data = keep[name].to(device=device, dtype=torch.float32)
    for name, b in model.named_buffers():
        if name in rope_bands:
            b.data = rope_bands[name].to(device=device, dtype=torch.float32)
    return model


def unfused_parameters(name, tensor, cfg):
    """Writable checkpoint/gradient views preserving official tensor names."""
    clean = name.replace(".impl.", ".")
    if clean.endswith(".mlp.gate_up_proj.weight"):
        return {
            clean.replace("gate_up_proj", label): P.mark_sharded(view)
            for label, view in zip(("gate_proj", "up_proj"), tensor.chunk(2, 0))
        }
    if clean.endswith(".linear_attn.in_proj_all.weight"):
        key = cfg.linear_heads * cfg.linear_head_dim
        heads = cfg.linear_value_heads or cfg.linear_heads
        value = heads * cfg.linear_head_dim
        views = tensor.split(tuple(x // P.tp_size() for x in (2 * key + value, value, heads, heads)), 0)
        result = {
            clean.replace("in_proj_all", label): P.mark_sharded(view)
            for label, view in zip(("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"), views)
        }
        P.mark_sharded(views[0], splits=(key, key, value))
        return result
    if clean.endswith(".self_attn.kv_proj.weight"):
        return {
            clean.replace("kv_proj", label): P.mark_sharded(view)
            for label, view in zip(("k_proj", "v_proj"), tensor.chunk(2, 0))
        }
    return {clean: tensor}


def load_pretrained(model, path):
    """Strictly load all official language and vision tensors into TP views."""
    from safetensors import safe_open

    root = asset_path(path)
    manifest = asset_manifest(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    target = {}
    parameters = {
        key: value
        for name, tensor in model.named_parameters()
        for key, value in unfused_parameters(name, tensor, model.cfg).items()
    }
    for clean, tensor in parameters.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if clean.startswith("embedding."):
            key = "model.language_model.embed_tokens." + clean.removeprefix("embedding.")
        elif clean == "lm_head.weight":
            key = "lm_head.weight"
        elif clean.startswith("layers."):
            key = "model.language_model." + clean
        elif clean.startswith("final_norm."):
            key = "model.language_model.norm." + clean.removeprefix("final_norm.")
        elif clean.startswith("patch_embed.visual."):
            key = "model.visual." + clean.removeprefix("patch_embed.visual.")
        else:
            continue
        target[key] = tensor
    expected = {
        k
        for k in index
        if k.startswith(("model.language_model.", "model.visual."))
        or (k == "lm_head.weight" and not model.cfg.tie_word_embeddings)
    }
    if expected != set(target):
        raise ValueError(
            f"Official checkpoint key mismatch: missing={sorted(expected - set(target))}, extra={sorted(set(target) - expected)}"
        )
    for shard in sorted({index[k] for k in target}):
        with safe_open(str(root / shard), framework="pt", device="cpu") as source, torch.no_grad():
            for key, tensor in target.items():
                if index[key] != shard:
                    continue
                value = P.checkpoint_shard(source.get_tensor(key), tensor)
                if value.shape != tensor.shape:
                    raise ValueError(f"Checkpoint shape mismatch: {key}")
                tensor.copy_(value)
    return dict(**manifest, loaded_tensors=len(target), loaded_parameters=sum(x.numel() for x in target.values()))
