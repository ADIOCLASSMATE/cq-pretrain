"""Opt-in AscendC backend for the identical delta64 training workload."""
import inspect
import os
from pathlib import Path
import traceback

import scripts.validation.delta64_probe as probe
import ops.gated_delta_net as backend
from scripts.validation.delta_ascendc_distributed_check import check as distributed_check


_original_decoder = probe.DecoderProbe
_original_benchmark = probe.benchmark
_original_primitive = probe.primitive_oracle
_recompute = os.getenv('DELTA_ASCENDC_RECOMPUTE', '0') == '1'


class GatedNorm(probe.GatedNorm):
    def forward(self, y, z):
        return backend.gated_norm(y, z, self.weight)


class Decoder(_original_decoder):
    def __init__(self, layers=24, hybrid=True, recompute=None):
        super().__init__(layers, hybrid, _recompute if recompute is None else recompute)


def primitive(output):
    previous = backend.MODE
    modes = ('fused', 'cached') if os.getenv('DELTA_ASCENDC_SWEEP', '0') == '1' else (previous,)
    result = {}
    for mode in modes:
        backend.MODE = mode
        result[mode] = _original_primitive(Path(output) / mode)
    backend.MODE = previous
    return result


def benchmark(output, lengths, layers, warmup, steps, ga):
    global _recompute
    import torch.distributed as dist
    configurations = [(backend.MODE, _recompute, backend.POINTWISE)]
    if os.getenv('DELTA_ASCENDC_SWEEP', '0') == '1':
        configurations = [('fused', True, backend.POINTWISE), ('fused', False, backend.POINTWISE),
                          ('cached', False, backend.POINTWISE)]
    pointwise_sweep = os.getenv('DELTA_ASCENDC_POINTWISE_SWEEP', '0') == '1'
    if pointwise_sweep:
        configurations = [('fused', False, False), ('fused', False, True)]
    results = []
    for mode, recompute, pointwise in configurations:
        backend.MODE, _recompute, backend.POINTWISE = mode, recompute, pointwise
        label = f'{mode}-recompute{int(recompute)}'
        if pointwise_sweep:
            label += f'-pointwise{int(pointwise)}'
        run = Path(output) / label
        current = _original_benchmark(run, lengths, layers, warmup, steps, ga)
        for row in current:
            row.update(backend=mode, activation_recompute=recompute, configuration=label, pointwise=pointwise)
        results.extend(current)
        probe.write_json(run / f'benchmark-rank{dist.get_rank():02d}.json', current)
        if dist.get_rank() == 0:
            probe.write_json(run / 'benchmark.json', current)
            probe.write_json(Path(output) / 'benchmark.json', results)
        provenance(output, label)
    if os.getenv('DELTA_ASCENDC_PROFILE', '0') == '1':
        best = min((r for r in results if r['kind'] == 'hybrid_3_1' and r['length'] == max(lengths)),
                   key=lambda r: r['median_step_seconds'])
        backend.MODE, _recompute = best['backend'], best['activation_recompute']
        backend.POINTWISE = best['pointwise']
        from scripts.validation.delta64_profile_20261015 import profile_training
        profile_training(Path(output) / best['configuration'], layers, max(lengths), ga)
    return results


def capture():
    backend.initialize()
    probe._NPU_KERNELS.update(delta=backend.delta, conv=backend.conv)


def provenance(output, label):
    assert probe._NPU_KERNELS['delta'] is backend.delta
    assert probe._NPU_KERNELS['conv'] is backend.conv
    result = backend.provenance()
    result['label'] = label
    result['selected'] = {name: inspect.getsourcefile(fn) for name, fn in probe._NPU_KERNELS.items()}
    if label in ('after_primitive', 'after_training'):
        required = ('chunk_bwd_dqkwg', 'chunk_gated_delta_rule_bwd_dhu', 'causal_conv1d_bwd')
        assert all(result['calls'].get(k, 0) > 0 for k in required), result
    rank = int(os.getenv('RANK', '0'))
    probe.write_json(Path(output) / f'kernels-{label}-rank{rank:02d}.json', result)
    probe.log('kernel_provenance', **result)


probe.capture_npu_kernels = capture
probe.kernel_provenance = provenance
probe.DecoderProbe = Decoder
probe.GatedNorm = GatedNorm
probe.benchmark = benchmark
probe.distributed_oracle = distributed_check
probe.primitive_oracle = primitive


if __name__ == '__main__':
    try:
        probe.main()
    except Exception:
        probe.log('failed', traceback=traceback.format_exc())
        raise
