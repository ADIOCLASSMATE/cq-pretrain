"""Validate current NPU fused primitives at actual CE/GEMM dimensions."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu
from torch.nn import functional as F


def elapsed(fn, repeats=10):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kind', choices=['wgrad', 'ce'], required=True)
    parser.add_argument('--rows',type=int,default=256)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(123)
    torch.npu.manual_seed_all(123)
    result = dict(torch=torch.__version__, torch_npu=torch_npu.__version__, kind=args.kind,rows=args.rows)
    if args.kind == 'wgrad':
        x = torch.randn(args.rows, 2048, device='npu', dtype=torch.bfloat16) * .02
        dy = torch.randn(args.rows, 151936, device='npu', dtype=torch.bfloat16) * .002
        out = torch.zeros(151936, 2048, device='npu', dtype=torch.float32)
        torch_npu._npu_matmul_add_fp32(dy, x, out)
        torch.npu.synchronize()
        expected = dy[:, ::313].float().T @ x.float()
        torch.testing.assert_close(out[::313], expected, atol=2e-6, rtol=2e-4)
        result['milliseconds'] = elapsed(lambda: torch_npu._npu_matmul_add_fp32(dy, x, out))
        result['max_error'] = (out[::313] / 14 - expected).abs().max().item()
    else:
        x = torch.randn(args.rows, 151936, device='npu', dtype=torch.float32) * 3
        targets = torch.randint(151936, (args.rows,), device='npu')
        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(x, targets, reduction='sum')
        expected = F.cross_entropy(x, targets, reduction='sum')
        torch.testing.assert_close(loss.reshape(()), expected, atol=2e-3, rtol=2e-6)
        grad = torch_npu.npu_cross_entropy_loss_backward(
            x.new_ones(1), log_prob, targets, reduction='sum')
        reference = x.softmax(-1)
        reference.scatter_add_(1, targets[:, None], -x.new_ones(args.rows, 1))
        torch.testing.assert_close(grad, reference, atol=3e-7, rtol=2e-4)
        result['fused_ms'] = elapsed(lambda: torch_npu.npu_cross_entropy_loss(x, targets, reduction='sum'))
        result['torch_ms'] = elapsed(lambda: F.cross_entropy(x, targets, reduction='sum'))
        result['gradient_max_abs_error'] = (grad-reference).abs().max().item()
    result['passed'] = True
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
