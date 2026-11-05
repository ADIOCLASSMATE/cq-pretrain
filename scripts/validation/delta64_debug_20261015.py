"""Small single-device diagnostic of FLA before/after framework initialization."""
import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace
import traceback

import torch
import torch.distributed as dist

import scripts.validation.delta64_probe as p
import distributed.megatron as P


def describe(x):
    if isinstance(x, torch.Tensor):
        cpu = x.detach().cpu().double()
        return dict(shape=list(x.shape), dtype=str(x.dtype), device=str(x.device),
                    finite=bool(torch.isfinite(cpu).all()),
                    nan=int(torch.isnan(cpu).sum()),
                    max_abs=float(cpu.abs().max()) if cpu.numel() else 0.)
    if isinstance(x, (list, tuple)):
        return [describe(t) for t in x]
    return str(x)


def instrument():
    chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    names = ['l2norm_fwd', 'chunk_local_cumsum', 'chunk_gated_delta_rule_fwd_intra',
             'chunk_gated_delta_rule_fwd_h', 'chunk_fwd_o', 'recompute_w_u_fwd',
             'chunk_gated_delta_rule_bwd_dhu', 'chunk_bwd_dqkwg', 'prepare_wy_repr_bwd']
    for name in names:
        original = getattr(chunk, name)
        def wrapper(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            p.log('intermediate', function=_name, result=describe(result))
            return result
        setattr(chunk, name, wrapper)
    original_stats = p.error_stats
    def stats(actual, expected, tolerance=.035):
        result = original_stats(actual, expected, tolerance)
        if not result['passed']:
            p.log('comparison_failed', actual=describe(actual), expected=describe(expected))
        return result
    p.error_stats = stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-world', type=int, default=1)
    parser.add_argument('--phase', choices=('before', 'after'), default='after')
    args = parser.parse_args()
    import torch_npu
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    p.capture_npu_kernels()
    results = {}
    for stage in (args.phase,):
        if stage == 'after':
            P.initialize(1, SimpleNamespace(layers=4, width=1024, heads=8,
                                          kv_heads=2, head_dim=256, ffn=3584))
        instrument()
        try:
            p.log('diagnostic_stage', stage=stage)
            results[stage] = p.primitive_oracle(Path(args.output) / stage)
        except Exception:
            results[stage] = dict(error=traceback.format_exc())
            p.log('diagnostic_failure', stage=stage, error=traceback.format_exc())
        p.kernel_provenance(args.output, stage)
        p.write_json(Path(args.output) / 'diagnosis.json', results)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
