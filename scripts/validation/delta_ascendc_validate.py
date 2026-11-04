"""Boundary, packed-conv and long-sequence checks, independent of benchmark timing."""
import argparse
from pathlib import Path
import gc
import torch
import torch_npu
from torch.nn import functional as F
import models.delta_ascendc as b
from scripts.validation.delta64_probe import error_stats, recurrent_reference, write_json, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args, _ = parser.parse_known_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    b.initialize()
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as native
    results = []
    for length in (1, 63, 64, 65, 257, 2048, 8192):
        torch.manual_seed(900 + length)
        values = [torch.randn(1, length, 16, 128).bfloat16() for _ in range(3)]
        values += [-torch.rand(1, length, 16) * .2, torch.rand(1, length, 16) * .8 + .1]
        upstream = torch.randn_like(values[2]) * .1
        ends = [min(37, length), length] if length > 1 else [1]
        kwargs = dict(cu_seqlens=torch.tensor([0, *ends], device='npu', dtype=torch.long),
                      cu_seqlens_cpu=torch.tensor([0, *ends], dtype=torch.long)) if length < 2048 else {}
        # Short cases have a fully independent mathematical FP64 derivative.
        # Long cases use CPU FP64 forward plus the previously validated native
        # FLA backward, catching launch-grid/tail errors beyond smoke shapes.
        refs = [x.double().requires_grad_(length < 2048) for x in values]
        expected = recurrent_reference(*refs, ends=ends if kwargs else None)
        if length < 2048:
            expected.backward(upstream.double())
            expected_grads = [x.grad for x in refs]
            gradient_reference = 'CPU FP64 recurrence'
        else:
            nx = [x.npu().requires_grad_() for x in values]
            ny, _ = native(*nx, use_qk_l2norm_in_kernel=True)
            ny.backward(upstream.npu())
            expected_grads = [x.grad.cpu() for x in nx]
            assert error_stats(ny, expected, .02)['passed']
            gradient_reference = 'native FLA Triton-Ascend; forward independently checked against CPU FP64'
            del nx, ny
        for mode in ('fused', 'cached'):
            b.MODE = mode
            xs = [x.npu().requires_grad_() for x in values]
            y, _ = b.delta(*xs, use_qk_l2norm_in_kernel=True, **kwargs)
            y.backward(upstream.npu())
            checks = dict(output=error_stats(y, expected, .02))
            checks.update({f'd_{name}': error_stats(x.grad, ref, .035)
                           for name, x, ref in zip(('q','k','v','g','beta'),xs,expected_grads)})
            result = dict(mode=mode, length=length, packed=bool(kwargs), checks=checks,
                          gradient_reference=gradient_reference,
                          passed=all(c['passed'] for c in checks.values()))
            results.append(result)
            write_json(Path(args.output) / 'validation.json', results)
            log('extended_validation', **result)
            assert result['passed'], result
            del xs, y
        del expected, expected_grads, refs, values, upstream
        gc.collect(); torch.npu.empty_cache()
    # Packed convolution must reset both activations and gradients at document boundaries.
    for length in (1, 65, 257):
        torch.manual_seed(length + 1000)
        x = torch.randn(1, length, 6144).bfloat16()
        w = (torch.randn(6144, 4) * .1).bfloat16()
        bounds = [0, length] if length == 1 else [0, 1, 37, length]
        nx, nw = x.npu().requires_grad_(), w.npu().requires_grad_()
        rx, rw = x.double().requires_grad_(), w.double().requires_grad_()
        y, _ = b.conv(nx, nw, activation='silu',
                      cu_seqlens=torch.tensor(bounds, device='npu', dtype=torch.long),
                      cu_seqlens_cpu=bounds)
        pieces = [F.silu(F.conv1d(rx[:,s:e].transpose(1,2), rw[:,None], padding=3,
                                 groups=6144)[...,:e-s]).transpose(1,2)
                  for s,e in zip(bounds,bounds[1:])]
        expected = torch.cat(pieces, dim=1)
        grad = torch.randn_like(x) * .1
        y.backward(grad.npu()); expected.backward(grad.double())
        checks = dict(output=error_stats(y,expected,.02), dx=error_stats(nx.grad,rx.grad),
                      dw=error_stats(nw.grad,rw.grad))
        result = dict(kind='packed_conv', length=length, checks=checks,
                      passed=all(c['passed'] for c in checks.values()))
        results.append(result); write_json(Path(args.output) / 'validation.json',results)
        log('extended_validation', **result)
        assert result['passed'], result
    write_json(Path(args.output) / 'provenance.json',b.provenance())
    write_json(Path(args.output) / 'completion.json',dict(passed=True,cases=len(results)))


if __name__ == '__main__':
    main()
