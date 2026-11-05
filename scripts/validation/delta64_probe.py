"""Bounded Qwen3.5-shaped GDN training infrastructure experiment.

This is a causal decoder probe, NOT a replacement for CQ_IT's content/query
semantics. No vision, vocabulary head, renderer, or production default changes.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import statistics
import time
import traceback
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


_NPU_KERNELS = {}


def capture_npu_kernels():
    """Keep explicit references before MCore's compatibility monkeypatches.

    MegatronAdaptor scans module attributes named chunk_gated_delta_rule and
    replaces even the implementation module's alias with a torch reference.
    A private dictionary preserves the actual FLA callables in this probe only.
    """
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
    from fla.modules.conv.causal_conv1d import causal_conv1d
    _NPU_KERNELS.update(delta=chunk_gated_delta_rule, conv=causal_conv1d)
    for fn in _NPU_KERNELS.values():
        assert inspect.unwrap(fn).__module__.startswith('fla.'), fn


def kernel_provenance(output, label):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as public_delta
    from fla.ops.backends import BackendRegistry
    def identity(fn):
        unwrapped = inspect.unwrap(fn)
        return dict(module=unwrapped.__module__, name=unwrapped.__qualname__,
                    source=inspect.getsourcefile(unwrapped))
    backends = {}
    for operation in ('modules', 'common', 'gated_delta_rule', 'utils'):
        BackendRegistry.ensure_initialized(operation)
        registry = BackendRegistry._registries[operation]
        backend = registry.get_active()
        assert backend and backend.backend_type == 'triton_ascend', operation
        backends[operation] = dict(active=backend.backend_type,
                                   dispatched=sorted(registry._logged))
    result = dict(label=label, selected={k: identity(v) for k, v in _NPU_KERNELS.items()},
                  public_delta=identity(public_delta),
                  public_delta_replaced=public_delta is not _NPU_KERNELS['delta'],
                  backends=backends)
    for fn in _NPU_KERNELS.values():
        assert inspect.unwrap(fn).__module__.startswith('fla.'), result
    if label == 'after_primitive':
        dispatched = backends['common']['dispatched']
        assert 'common:chunk_gated_delta_rule_fwd_h:triton_ascend' in dispatched, result
        assert 'common:chunk_gated_delta_rule_bwd_dhu:triton_ascend' in dispatched, result
    rank = int(os.getenv('RANK', '0'))
    write_json(Path(output) / f'kernels-{label}-rank{rank:02d}.json', result)
    log('kernel_provenance', **result)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def log(event, **fields):
    record = dict(time=datetime.now(timezone.utc).isoformat(),
                  rank=int(os.getenv('RANK', '0')), event=event)
    record.update(fields)
    print(json.dumps(record), flush=True)


def error_stats(actual, expected, tolerance=.035):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    actual_finite, expected_finite = bool(torch.isfinite(a).all()), bool(torch.isfinite(b).all())
    finite = actual_finite and expected_finite
    delta = a - b
    relative = float(delta.norm() / b.norm().clamp_min(1e-12))
    absolute = float(delta.abs().max())
    return dict(relative_l2=relative, max_abs=absolute, tolerance=tolerance,
                actual_finite=actual_finite, expected_finite=expected_finite,
                passed=finite and (relative <= tolerance or absolute <= 2e-7))


def recurrent_reference(q, k, v, g, beta, ends=None):
    """Independent FP64 recurrence; differentiable, resets at document boundaries."""
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q * q.shape[-1] ** -.5
    ends = ends or [q.shape[1]]
    outputs = []
    begin = 0
    for end in ends:
        state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        for t in range(begin, end):
            state = state * g[:, t].exp()[..., None, None]
            residual = v[:, t] - (state * k[:, t, :, :, None]).sum(-2)
            state = state + k[:, t, :, :, None] * (residual * beta[:, t, :, None])[..., None, :]
            outputs.append((state * q[:, t, :, :, None]).sum(-2))
        begin = end
    return torch.stack(outputs, dim=1)


def primitive_oracle(output):
    chunk_gated_delta_rule = _NPU_KERNELS['delta']
    causal_conv1d = _NPU_KERNELS['conv']
    results = {}
    # Actual Qwen3.5-0.8B GDN head dimensions, including a chunk boundary and
    # non-aligned packed documents. FP64 reference sees the same rounded input.
    for packed in (False, True):
        torch.manual_seed(301)
        length, heads, dim = 129, 16, 128
        values = [torch.randn(1, length, heads, dim) for _ in range(3)]
        values += [-torch.rand(1, length, heads) * .2,
                   torch.rand(1, length, heads) * .8 + .1]
        inputs = [x.to(torch.bfloat16 if i < 3 else torch.float32)
                  for i, x in enumerate(values)]
        refs = [x.double().requires_grad_() for x in inputs]
        xs = [x.to('npu').requires_grad_() for x in inputs]
        ends = [37, 129] if packed else [129]
        kwargs = {}
        if packed:
            kwargs['cu_seqlens'] = torch.tensor([0, *ends], device='npu', dtype=torch.long)
            kwargs['cu_seqlens_cpu'] = torch.tensor([0, *ends], dtype=torch.long)
        log('primitive_start', packed=packed)
        actual, _ = chunk_gated_delta_rule(*xs, use_qk_l2norm_in_kernel=True, **kwargs)
        upstream = torch.randn_like(inputs[2]) * .1
        actual.backward(upstream.to('npu'))
        torch.npu.synchronize()
        expected = recurrent_reference(*refs, ends=ends)
        expected.backward(upstream.double())
        checks = {'output': error_stats(actual, expected, .02)}
        checks.update({f'd_{n}': error_stats(x.grad, ref.grad)
                       for n, x, ref in zip(('q', 'k', 'v', 'g', 'beta'), xs, refs)})
        results['packed' if packed else 'dense'] = checks
        write_json(Path(output) / 'primitive.json', results)
        log('primitive_done', packed=packed, checks=checks)
        if not all(c['passed'] for c in checks.values()):
            torch.save(dict(inputs=inputs, actual=actual.detach().cpu(),
                            expected=expected.detach().cpu(),
                            actual_grads=[x.grad.detach().cpu() for x in xs],
                            expected_grads=[x.grad.detach().cpu() for x in refs]),
                       Path(output) / f'failed-primitive-packed{int(packed)}.pt')
        assert all(c['passed'] for c in checks.values()), checks
    # Short causal convolution must train too; checking delta-rule alone misses it.
    torch.manual_seed(302)
    x = torch.randn(1, 129, 6144).bfloat16()
    w = (torch.randn(6144, 4) * .1).bfloat16()
    nx, nw = x.to('npu').requires_grad_(), w.to('npu').requires_grad_()
    rx, rw = x.double().requires_grad_(), w.double().requires_grad_()
    actual, _ = causal_conv1d(nx, nw, activation='silu')
    expected = F.silu(F.conv1d(rx.transpose(1, 2), rw[:, None],
                             padding=3, groups=6144)[..., :129]).transpose(1, 2)
    upstream = torch.randn_like(x) * .1
    actual.backward(upstream.to('npu'))
    expected.backward(upstream.double())
    checks = dict(output=error_stats(actual, expected, .02),
                  dx=error_stats(nx.grad, rx.grad), dw=error_stats(nw.grad, rw.grad))
    results['causal_conv'] = checks
    write_json(Path(output) / 'primitive.json', results)
    log('conv_done', checks=checks)
    assert all(c['passed'] for c in checks.values()), checks
    return results


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        from infra.kernels import rms_norm
        return rms_norm(x, (1 + self.weight.float()).to(x.dtype), 1e-6)


class CausalConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(6144, 1, 4))
        nn.init.uniform_(self.weight, -.5, .5)

    def forward(self, x):
        return _NPU_KERNELS['conv'](x, self.weight.squeeze(1), activation='silu')[0]


class GatedNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(128))

    def forward(self, y, z):
        from infra.kernels import rms_norm
        return rms_norm(y, self.weight, 1e-6) * F.silu(z.float()).to(y.dtype)


class GatedDelta(nn.Module):
    """Qwen3.5 checkpoint-compatible parameter names and shapes, causal training."""
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(1024, 6144, bias=False)
        self.in_proj_z = nn.Linear(1024, 2048, bias=False)
        self.in_proj_a = nn.Linear(1024, 16, bias=False)
        self.in_proj_b = nn.Linear(1024, 16, bias=False)
        self.conv1d = CausalConv()
        self.A_log = nn.Parameter(torch.rand(16).mul(15.99).add(.01).log())
        self.dt_bias = nn.Parameter(torch.full((16,), -3.0))
        self.norm = GatedNorm()
        self.out_proj = nn.Linear(2048, 1024, bias=False)

    def forward(self, x):
        qkv = self.conv1d(self.in_proj_qkv(x))
        q, k, v = [t.reshape(*x.shape[:-1], 16, 128) for t in qkv.chunk(3, -1)]
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())
        y, _ = _NPU_KERNELS['delta'](q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        z = self.in_proj_z(x).reshape_as(y)
        y = self.norm(y, z)
        return self.out_proj(y.flatten(-2))


class GatedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(1024, 4096, bias=False)
        self.k_proj = nn.Linear(1024, 512, bias=False)
        self.v_proj = nn.Linear(1024, 512, bias=False)
        self.o_proj = nn.Linear(2048, 1024, bias=False)
        self.q_norm, self.k_norm = RMSNorm(256), RMSNorm(256)
        self.register_buffer('causal_mask', torch.triu(torch.ones(2048, 2048, dtype=torch.bool), 1), persistent=False)

    def forward(self, x):
        import torch_npu
        q, gate = self.q_proj(x).reshape(*x.shape[:-1], 8, 512).chunk(2, -1)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).reshape(*x.shape[:-1], 2, 256))
        v = self.v_proj(x).reshape(*x.shape[:-1], 2, 256)
        # Text-only positions: Qwen's interleaved MRoPE reduces to ordinary 1D.
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        inv = 1e7 ** (-torch.arange(0, 64, 2, device=x.device).float() / 64)
        angles = positions[:, None] * inv[None]
        cos, sin = [torch.cat((a, a), -1).to(x.dtype)[None, :, None] for a in (angles.cos(), angles.sin())]
        def rotate(t):
            r, tail = t[..., :64], t[..., 64:]
            a, b = r.chunk(2, -1)
            return torch.cat((r * cos + torch.cat((-b, a), -1) * sin, tail), -1)
        q, k = rotate(q), rotate(k)
        y = torch_npu.npu_fusion_attention(
            q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(), head_num=8, input_layout='BNSD',
            atten_mask=self.causal_mask, scale=256 ** -.5, keep_prob=1., sparse_mode=2)[0]
        y = y.transpose(1, 2).reshape(*x.shape[:-1], 2048) * gate.flatten(-2).sigmoid()
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(1024, 3584, bias=False)
        self.up_proj = nn.Linear(1024, 3584, bias=False)
        self.down_proj = nn.Linear(3584, 1024, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(1024), RMSNorm(1024)
        if linear:
            self.linear_attn = GatedDelta()
        else:
            self.self_attn = GatedAttention()
        self.mlp = MLP()

    def forward(self, x):
        mixer = self.linear_attn if hasattr(self, 'linear_attn') else self.self_attn
        x = x + mixer(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class DecoderProbe(nn.Module):
    def __init__(self, layers=24, hybrid=True, recompute=True):
        super().__init__()
        import infra.parallel as P
        self.config = P._config
        self.layers = nn.ModuleList(Block(hybrid and i % 4 != 3) for i in range(layers))
        self.norm = RMSNorm(1024)
        self.recompute = recompute
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False) if self.recompute else layer(x)
        return self.norm(x)


def data(seed, length):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(1, length, 1024, generator=generator).to(device='npu', dtype=torch.bfloat16)


def reduce_scalar(value, op=dist.ReduceOp.MAX):
    t = torch.tensor(float(value), device='npu', dtype=torch.float32)
    dist.all_reduce(t, op=op)
    return float(t)


def distributed_oracle(output):
    import infra.parallel as P
    from utils.checkpoint import _optimizers
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(410)
    model = DecoderProbe(4, recompute=False).to(device='npu', dtype=torch.bfloat16)
    x = data(411 + rank, 128)
    reference = {}
    # Use real per-rank BF16 backward, then explicitly average parameter .grad
    # with FP32 all-reduce. Analytically scaling an unscaled BF16 backward is
    # NOT an exact oracle: rounding enters every intermediate backward op.
    # This reference is independent of MCore's buckets/reduce-scatter/main_grad.
    y = model(x).float()
    ((y.square().mean() + .1 * y.mean()) * (1 + rank / world)).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        expected = p.grad.float().clone()
        dist.all_reduce(expected)
        expected.div_(world)
        if rank == 0:
            reference[name] = expected.cpu()
    model.zero_grad(set_to_none=True)
    dist.barrier()
    wrapped, optimizer = P.make_optimizer(model, lr=3e-4, overlap_param_gather=True)
    optimizer.zero_grad(); wrapped.zero_grad_buffer()
    y = wrapped(x).float()
    loss = (y.square().mean() + .1 * y.mean()) * (1 + rank / world)
    loss.backward()
    P.finish_grad_sync(wrapped)
    checks = {}
    for name, p in model.named_parameters():
        reduced = torch.zeros_like(p.main_grad)
        for opt in _optimizers(optimizer):
            try:
                owned = opt._get_model_param_range_map(p)['param']
            except KeyError:
                continue
            reduced.view(-1)[owned.start:owned.end] = p.main_grad.view(-1)[owned.start:owned.end]
        dist.all_reduce(reduced, group=P.state().get_data_parallel_group(with_context_parallel=True,
                                                                       partial_data_parallel=True))
        if rank == 0:
            checks[name] = error_stats(reduced, reference[name], .003)
    updated, grad_norm, _ = optimizer.step()
    assert updated and torch.isfinite(torch.tensor(float(grad_norm)))
    wrapped.start_param_sync(force_sync=True)
    # Compare every updated parameter against rank 0, not just a checksum.
    max_difference = torch.zeros((), device='npu')
    for p in model.parameters():
        root = p.detach().clone()
        dist.broadcast(root, src=0)
        max_difference = torch.maximum(max_difference, (p.detach() - root).float().abs().max())
    difference = reduce_scalar(float(max_difference))
    passed = all(c['passed'] for c in checks.values()) if rank == 0 else True
    passed = bool(reduce_scalar(int(passed), dist.ReduceOp.MIN)) and difference == 0
    result = dict(world_size=world, tp=1, dp=world, passed=passed,
                  oracle='explicit FP32 all-reduce of unwrapped per-rank BF16 parameter gradients',
                  optimizer='Megatron Core distributed AdamW', updated=bool(updated),
                  grad_norm=float(grad_norm), max_parameter_difference=difference, checks=checks)
    if rank == 0:
        write_json(Path(output) / 'distributed.json', result)
        log('distributed_oracle_done', passed=passed, max_parameter_difference=difference,
            max_gradient_relative_l2=max(c['relative_l2'] for c in checks.values()))
    wrapped.disable_forward_pre_hook()
    del wrapped, optimizer, model, reference, reduced, root, p, loss, y, expected
    gc.collect(); torch.npu.empty_cache()
    assert passed, 'Distributed gradient or optimizer synchronization failed'
    return result


def benchmark(output, lengths, layers, warmup, steps, ga):
    import infra.parallel as P
    rank, world = dist.get_rank(), dist.get_world_size()
    results = []
    for hybrid in (False, True):
        torch.manual_seed(500)
        model = DecoderProbe(layers, hybrid=hybrid).to(device='npu', dtype=torch.bfloat16)
        parameter_count = sum(p.numel() for p in model.parameters())
        wrapped, optimizer = P.make_optimizer(model, lr=1e-4)
        for length in lengths:
            x = data(600 + rank, length)
            times, losses, norms = [], [], []
            torch.npu.reset_peak_memory_stats()
            for step in range(warmup + steps):
                dist.barrier(); torch.npu.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(); wrapped.zero_grad_buffer()
                loss_value = 0.
                for micro in range(ga):
                    with wrapped.no_sync() if micro + 1 < ga else nullcontext():
                        y = wrapped(x)
                        # Linear term avoids the near-constant loss of pure RMS^2.
                        loss = (y.float().square().mean() + .1 * y.float().mean()) / ga
                        loss.backward()
                    loss_value += float(loss.detach())
                P.finish_grad_sync(wrapped)
                updated, grad_norm, _ = optimizer.step()
                wrapped.start_param_sync(force_sync=True)
                torch.npu.synchronize()
                seconds = time.perf_counter() - start
                assert updated and torch.isfinite(torch.tensor([loss_value, float(grad_norm)])).all()
                slowest = reduce_scalar(seconds)
                if step >= warmup:
                    times.append(slowest); losses.append(loss_value); norms.append(float(grad_norm))
                if rank == 0:
                    log('benchmark_step', hybrid=hybrid, length=length, step=step,
                        measured=step >= warmup, slowest_seconds=slowest, loss=loss_value)
            memory = reduce_scalar(torch.npu.max_memory_allocated() / 2**30)
            median = statistics.median(times)
            result = dict(kind='hybrid_3_1' if hybrid else 'gated_full_attention',
                          layers=layers, parameters=parameter_count, length=length,
                          world_size=world, tp=1, dp=world, micro_batch_size=1, grad_accumulation=ga,
                          activation_recompute=True, slowest_rank_seconds=times,
                          median_step_seconds=median, tokens_per_second=world * length * ga / median,
                          max_peak_memory_gib=memory, losses=losses, grad_norms=norms)
            results.append(result)
            write_json(Path(output) / f'benchmark-rank{rank:02d}.json', results)
            if rank == 0:
                write_json(Path(output) / 'benchmark.json', results)
            del x, y, loss
        wrapped.disable_forward_pre_hook()
        del wrapped, optimizer, model
        gc.collect(); torch.npu.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('primitive', 'distributed'), default='distributed')
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-world', type=int, default=64)
    parser.add_argument('--layers', type=int, default=24)
    parser.add_argument('--lengths', type=int, nargs='+', default=[2048, 8192])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--ga', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    import torch_npu
    torch.npu.set_device(int(os.getenv('LOCAL_RANK', '0')))
    from fla.utils import IS_NPU, device_platform
    assert IS_NPU and device_platform == 'npu'
    capture_npu_kernels()
    rank = int(os.getenv('RANK', '0'))
    environment = dict(hostname=socket.gethostname(), rank=rank,
                       device=torch.npu.get_device_name(), torch=torch.__version__,
                       torch_npu=torch_npu.__version__,
                       versions={n: importlib.metadata.version(n) for n in
                                 ('fla-core', 'flash-linear-attention', 'triton-ascend')},
                       npu_backend=IS_NPU)
    driver_info = Path('/usr/local/Ascend/driver/version.info')
    if driver_info.is_file():
        environment['driver_version_info'] = driver_info.read_text()[:4000]
    write_json(Path(args.output) / f'environment-rank{rank:02d}.json', environment)
    log('environment', **environment)
    if args.stage == 'primitive':
        primitive_oracle(args.output)
        kernel_provenance(args.output, 'after_primitive')
        log('primitive_passed')
        return
    import infra.parallel as P
    cfg = SimpleNamespace(layers=args.layers, width=1024, heads=8, kv_heads=2, head_dim=256, ffn=3584)
    P.initialize(1, cfg, timeout_minutes=20)
    assert dist.get_world_size() == args.expected_world
    kernel_provenance(args.output, 'after_initialize')
    dist.barrier()
    # Validate every device after all ranks finish framework/communicator init.
    primitive_output = Path(args.output) if rank == 0 else Path(args.output) / f'primitive-rank{rank:02d}'
    primitive_oracle(primitive_output)
    kernel_provenance(args.output, 'after_primitive')
    dist.barrier()
    # All ranks use the tested NPU kernels. Short correctness runs precede any
    # performance conclusion and the production training defaults stay intact.
    distributed_oracle(args.output)
    benchmark(args.output, args.lengths, args.layers, args.warmup, args.steps, args.ga)
    kernel_provenance(args.output, 'after_training')
    if rank == 0:
        write_json(Path(args.output) / 'completion.json', dict(passed=True, world_size=dist.get_world_size(),
                   scope='causal decoder blocks; no CQ, vision, vocab head, renderer, TP, or checkpoint-resume validation'))
        log('all_passed')
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        log('failed', traceback=traceback.format_exc())
        raise
