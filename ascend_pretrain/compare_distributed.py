"""CPU verification of NPU TP/DP oracle and restart outputs."""
import argparse
import json
from pathlib import Path
import torch


def compare_topologies(root, bf16_quantization_bound=False):
    reference = torch.load(root/'tp1_dp1.pt', weights_only=False, map_location='cpu')
    report = {}
    expected = ['tp1_dp1', 'tp2_dp1', 'tp4_dp1', 'tp2_dp2']
    if (root/'tp1_dp4.pt').exists():
        expected.append('tp1_dp4')
    expected.extend(p.stem for p in sorted(root.glob('tp1_dp4_opt*.pt')))
    expected.extend(p.stem for p in sorted(root.glob('tp1_dp4_tile*.pt')))
    for name in expected:
        result = torch.load(root/f'{name}.pt', weights_only=False, map_location='cpu')
        if 'initial' in reference:
            for key, value in reference['initial'].items():
                assert torch.equal(value, result['initial'][key]), f'Initial weights differ: {name}/{key}'
        torch.testing.assert_close(result['losses'], reference['losses'], atol=1e-4, rtol=1e-3)
        tensors = {}
        for key, target in reference['gradients'].items():
            actual = result['gradients'][key]
            difference = actual-target
            # Near-zero attention derivatives need an absolute tolerance.
            # Significant gradients must ALSO pass a per-parameter L2 check.
            relative = float(difference.norm()/target.norm().clamp_min(1e-12))
            maximum = float(difference.abs().max())
            # BF16 reductions can cancel around individual near-zero entries.
            # Bound those absolutely and require <2% L2 error for each
            # significant parameter, instead of relying on elementwise rtol.
            # Different TP reductions perturb BF16 activations before the final
            # weight-gradient GEMM. New tied-mask mainlines use a dtype-scaled
            # cancellation bound (two BF16 ULPs at parameter-gradient RMS),
            # while retaining the independent <2% per-parameter L2 gate.
            atol=1e-4
            if bf16_quantization_bound:
                atol=max(atol,2*torch.finfo(torch.bfloat16).eps*float(target.square().mean().sqrt()))
            torch.testing.assert_close(actual, target, atol=atol, rtol=.06)
            if target.norm() > 1e-5:
                assert relative < .02, (name, key, relative)
            tensors[key] = dict(max_abs=maximum, relative_l2=relative,absolute_tolerance=atol,
                               reference_l2=float(target.norm()))
        max_weight_error = 0.
        for key, target in reference['weights'].items():
            actual = result['weights'][key]
            torch.testing.assert_close(actual.float(), target.float(), atol=1e-3, rtol=.01)
            max_weight_error = max(max_weight_error, float((actual-target).abs().max()))
        report[name] = dict(losses=result['losses'].tolist(), grad_norm=float(result['grad_norm']),
                            max_weight_error=max_weight_error, gradients=tensors)
    return report


def assert_identical(a, b, path='', optimizer_differences=None):
    if isinstance(a, torch.Tensor):
        if optimizer_differences is not None and a.is_floating_point():
            torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-4)
            if not torch.equal(a, b):
                difference = a.float()-b.float()
                optimizer_differences.append(dict(path=path, max_abs=float(difference.abs().max()),
                    relative_l2=float(difference.norm()/a.float().norm().clamp_min(1e-30))))
        else:
            assert torch.equal(a, b), f'Restart tensor differs: {path}'
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_identical(a[key], b[key], path+'/'+str(key), optimizer_differences)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_identical(x, y, path+'/'+str(i), optimizer_differences)
    else:
        assert a == b, path


def compare_restart(root):
    count = 0
    optimizer_differences = []
    for file in sorted((root/'continuous/step_0000007').glob('rank_*.pt')):
        reference = torch.load(file, weights_only=False)
        actual = torch.load(root/'resumed/step_0000007'/file.name, weights_only=False)
        model_file = f"model_tp_{reference['tp_rank']:03d}.pt"
        reference['model'] = torch.load(root/'continuous/step_0000007'/model_file, weights_only=False)
        actual['model'] = torch.load(root/'resumed/step_0000007'/model_file, weights_only=False)
        for key in ('model', 'optimizer', 'cpu_rng', 'npu_rng', 'step'):
            assert_identical(reference[key], actual[key], file.name+'/'+key,
                             optimizer_differences if key == 'optimizer' else None)
        count += 1
    assert count == 4, 'Expected all four rank shards'
    return dict(ranks=count, status='model_and_rng_bitwise_identical', steps=7,
                topology=reference['topology'],
                split='4 + restart + 3', includes='BF16 model, FP32 master/Adam, RNG, cosine LR',
                optimizer_tolerance=dict(atol=1e-8, rtol=1e-4),
                optimizer_differences=optimizer_differences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology-root', type=Path, default=Path('ascend_pretrain/artifacts/distributed'))
    parser.add_argument('--restart-root', type=Path)
    parser.add_argument('--bf16-quantization-bound',action='store_true',
                        help='Use the documented mainline BF16 cancellation bound; retain the 2%% L2 gate')
    parser.add_argument('--output', type=Path, default=Path('ascend_pretrain/artifacts/distributed_correctness.json'))
    args = parser.parse_args()
    report = dict(topologies=compare_topologies(args.topology_root,args.bf16_quantization_bound))
    if args.restart_root:
        report['restart'] = compare_restart(args.restart_root)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status='passed', report=str(args.output))))
