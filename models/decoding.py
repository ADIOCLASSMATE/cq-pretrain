"""Request-local CQ text decoding after one image/video/text prefill.

Only sampled content commits KV, convolution history and recurrent state.
Each fresh Q reads that prefix and starts from the trained text-mask embedding.
"""

from dataclasses import replace
import torch
from torch.nn import functional as F
import ops.functional as K
import distributed.megatron as P
from models.cq_backbone import backbone
from ops.gated_delta_layout import position_rope


def delta_step(layer, x, cache):
    cfg = layer.cfg
    hk = cfg.linear_heads // P.tp_size()
    d = cfg.linear_head_dim
    h = (cfg.linear_value_heads or cfg.linear_heads) // P.tp_size()
    size = hk * d
    raw, z, a, b = layer.project(x[:, 0])
    weight = layer.conv1d.weight[:, 0]
    history = cache["conv"][0]
    if x.device.type == "npu":
        from fla_npu.ops import ascendc

        convolved = F.silu(
            ascendc.causal_conv1d_update(
                raw[:1].contiguous(),
                cache["conv"],
                cache["conv_weight"],
                activation=None,
                null_block_id=-1,
                out=cache["conv_out"],
            )[0]
        )
    else:
        convolved = raw[0] * weight[:, -1]
        for j in range(len(history)):
            convolved = convolved + history[j] * weight[:, j]
        convolved = F.silu(convolved)
        for j in range(len(history) - 1):
            history[j].copy_(history[j + 1])
        if len(history):
            history[-1].copy_(raw[0])
    cq, k, v = convolved.split((size, size, h * d), -1)
    query = raw[1, :size] * weight[:size, -1]
    for j in range(len(history)):
        query = query + history[j, :size] * weight[:size, j]
    q = torch.stack((cq, F.silu(query))).reshape(2, hk, d)
    k = k.reshape(1, hk, d)
    v = v.reshape(1, h, d)
    if h != hk:
        q, k = (t.repeat_interleave(h // hk, dim=1) for t in (q, k))
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    g = -layer.A_log.to(dtype).exp() * F.softplus(a[:1].to(dtype) + layer.dt_bias.to(dtype))
    beta = b[:1].sigmoid()
    state = cache["state"]
    if x.device.type == "npu":
        from ops.gated_delta_net import call, recurrent

        q = call("l2_fwd", q.contiguous())[0]
        k = call("l2_fwd", k.contiguous())[0]
        # Persistent two-event buffers: C writes, Q has exactly zero gates/K/V.
        cache["k"][0].copy_(k[0])
        cache["v"][0].copy_(v[0])
        cache["g"][0].copy_(g[0])
        cache["beta"][0].copy_(beta[0])
        y = recurrent(q, cache["k"], cache["v"], cache["g"], cache["beta"], state, cache["out"])
    else:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * d**-0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        state.mul_(g.exp()[..., None, None])
        residual = v - torch.matmul(state, k[..., None]).squeeze(-1)
        state.add_((residual * beta[..., None])[..., None] * k[..., None, :])
        y = torch.matmul(state[0], q[..., None]).squeeze(-1).to(x.dtype)
    return layer.out_proj(layer.norm(y, z.reshape(2, h, d)).flatten(-2))[:, None]


def attention_step(layer, x, rope, cache):
    cfg = layer.cfg
    dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    q, gate = layer.q_proj(x).reshape(2, 1, cfg.heads // P.tp_size(), 2 * cfg.head_dim).chunk(2, -1)
    q = layer.q_norm(q, cfg.fused)
    k, v = (
        layer.kv_proj(x[:1]).chunk(2, -1)
        if cfg.qwen_fused_projections
        else (layer.k_proj(x[:1]), layer.v_proj(x[:1]))
    )
    k = layer.k_norm(k.reshape(1, 1, cfg.kv_heads // P.tp_size(), cfg.head_dim), cfg.fused)
    q = torch.cat((K.rotary(q[..., :dim].contiguous(), *rope, cfg.fused), q[..., dim:]), -1)
    k = torch.cat(
        (K.rotary(k[..., :dim].contiguous(), rope[0][:1], rope[1][:1], cfg.fused), k[..., dim:]), -1
    )
    n = cache["length"]
    cache["k"][n : n + 1].copy_(k)
    cache["v"][n : n + 1].copy_(v.reshape_as(k))
    cache["length"] = n + 1
    # C sees itself; the next Q sees the same committed prefix. No dense mask.
    y = K.attention(q, cache["k"][: n + 1], cache["v"][: n + 1], fused=cfg.fused)
    return layer.o_proj(y * gate.flatten(-2).sigmoid())


class CQDecoder:
    @torch.no_grad()
    def __init__(self, model, batch, max_new_tokens, vision_features=None):
        cfg = model.cfg
        plan = batch.mainline
        s = batch.ids.shape[1]
        if (
            batch.ids.device.type != "cpu"
            or cfg.method != "CQ_IT"
            or plan.query_positions.tolist() != [s - 1]
            or plan.query_image.any()
            or len(batch.segments.unique()) != 1
            or batch.segments.min() < 0
        ):
            raise ValueError("CQ decoding needs one CPU document and its final text Q")
        self.model = model
        self.step = 0
        self.limit = max_new_tokens
        device = model.embedding.weight.device
        dtype = model.embedding.weight.dtype
        self.layers = [dict(capacity=s - 1 + max_new_tokens) for _ in model.layers]
        # Preserve the ordinary prefill plan, but do not commit its unknown target.
        delta = plan.delta_plan
        writes = delta.event_write.clone()
        writes[writes == len(delta.content_order) - 1] = len(delta.content_order)
        local = replace(batch, mainline=replace(plan, delta_plan=replace(delta, event_write=writes))).to(
            device
        )
        output = backbone(model, local, vision_features=vision_features, cache=self.layers)
        self.hidden = output.index_select(0, local.mainline.output_positions)
        positions = plan.query_rope + torch.arange(max_new_tokens + 1)[None, :, None]
        if positions.shape[0] == 2:
            positions = torch.cat((positions[:1], positions))
        self.rope = tuple(t.to(device=device, dtype=dtype) for t in position_rope(positions, cfg))
        self.ids = torch.tensor([[0, cfg.text_mask_token_id]], device=device)
        for layer, cache in zip(model.layers, self.layers):
            if hasattr(layer, "linear_attn") and device.type == "npu":
                attn = layer.linear_attn
                h = cache["state"].shape[1]
                d = cfg.linear_head_dim
                cache["conv_weight"] = attn.conv1d.weight[:, 0].T.contiguous()
                cache["conv_out"] = torch.empty(1, attn.conv1d.weight.shape[0], device=device, dtype=dtype)
                for key in ("k", "v"):
                    cache[key] = torch.zeros(2, h, d, device=device, dtype=dtype)
                cache["out"] = torch.empty(2, h, d, device=device, dtype=dtype)
                cache["g"] = torch.zeros(2, h, device=device, dtype=torch.float32)
                cache["beta"] = torch.zeros(2, h, device=device, dtype=dtype)

    @torch.no_grad()
    def advance(self, token):
        if self.step >= self.limit:
            raise ValueError("CQ decode capacity exceeded")
        self.ids[0, 0] = token
        hidden = self.model.embedding(self.ids).transpose(0, 1)
        rope = tuple(t[self.step : self.step + 2] for t in self.rope)
        for layer, cache in zip(self.model.layers, self.layers):
            norm = layer.input_layernorm(hidden, self.model.cfg.fused)
            mixed = (
                delta_step(layer.linear_attn, norm, cache)
                if hasattr(layer, "linear_attn")
                else attention_step(layer.self_attn, norm, rope, cache)
            )
            hidden = hidden + mixed
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden, self.model.cfg.fused))
        self.step += 1
        self.hidden = self.model.final_norm(hidden[1], self.model.cfg.fused)
        return self.hidden
