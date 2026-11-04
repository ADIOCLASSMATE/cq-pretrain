"""Numerical and timing comparison of training TorchAIR pointwise graphs."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu
from .compiled_kernels import compiled, modulated_norm_reference, gated_add_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(52)
    cases = dict(modulated_norm=(modulated_norm_reference, [(1024, 8, 864), (1, 8, 864), (1, 8, 864)]),
                 gated_add=(gated_add_reference, [(1024, 8, 864), (1024, 8, 864), (1, 8, 864)]))
    result = {}
    for name, (reference, shapes) in cases.items():
        inputs = [torch.randn(shape, device='npu', dtype=torch.bfloat16).requires_grad_() for shape in shapes]
        upstream = torch.randn(shapes[0], device='npu', dtype=torch.bfloat16)
        expected = reference(*inputs)
        grads = torch.autograd.grad(expected, inputs, upstream)
        candidate = compiled(name)
        actual = candidate(*inputs)
        actual_grads = torch.autograd.grad(actual, inputs, upstream)
        torch.testing.assert_close(actual, expected, atol=.04, rtol=.02)
        relative = [(a.float()-b.float()).norm().item()/max(b.float().norm().item(), 1e-6)
                    for a,b in zip(actual_grads,grads)]
        if max(relative) > .015:
            raise AssertionError(f'{name}: gradient relative L2 {relative}')
        times = {}
        for mode, function in [('eager', reference), ('compiled', candidate)]:
            for _ in range(4):
                torch.autograd.grad(function(*inputs), inputs, upstream)
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(30):
                torch.autograd.grad(function(*inputs), inputs, upstream)
            torch.npu.synchronize()
            times[mode] = (time.perf_counter()-start)*1000/30
        result[name] = dict(gradient_relative_l2=relative, training_ms=times)
    result['passed'] = True
    path = Path(args.output);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
