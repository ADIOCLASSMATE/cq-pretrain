"""Check DP reduction against the same autograd gradients, before MCore consumes them.

Re-running BF16 backward in an unwrapped model changes allocator alignment and
can change a few rounding decisions. It is not an exact reduction oracle. Here
parameter hooks independently accumulate the real pre-reduction gradients in
FP32 across two microbatches, then a separate all-reduce supplies the reference.
The GDN mathematical derivative is checked separately against CPU FP64.
"""
from contextlib import nullcontext
import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist


def check(output):
    import scripts.validation.delta64_probe as p
    import infra.parallel as P
    from utils.checkpoint import _optimizers
    rank, world = dist.get_rank(), dist.get_world_size()
    length = int(os.getenv('DELTA_ASCENDC_ORACLE_LENGTH', '128'))
    torch.manual_seed(410)
    model = p.DecoderProbe(4, recompute=False).to(device='npu', dtype=torch.bfloat16)
    wrapped, optimizer = P.make_optimizer(model, lr=3e-4, overlap_param_gather=True)
    accumulated, counts, handles = {}, {}, []
    def capture(name):
        def hook(grad):
            if name not in accumulated:
                accumulated[name] = grad.detach().float().clone()
            else:
                accumulated[name].add_(grad.detach().float())
            counts[name] = counts.get(name, 0) + 1
            return grad
        return hook
    for name, param in model.named_parameters():
        handles.append(param.register_hook(capture(name)))
    optimizer.zero_grad(); wrapped.zero_grad_buffer()
    for micro in range(2):
        x = p.data(411 + rank + micro * world, length)
        with wrapped.no_sync() if micro == 0 else nullcontext():
            y = wrapped(x).float()
            loss = (y.square().mean() + .1 * y.mean()) * (1 + rank / world) / 2
            loss.backward()
    P.finish_grad_sync(wrapped)
    checks = {}
    group = P.state().get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True)
    for name, param in model.named_parameters():
        assert counts.get(name) == 2, (name, counts.get(name))
        expected = accumulated[name]
        dist.all_reduce(expected)
        expected.div_(world)
        actual = torch.zeros_like(param.main_grad)
        for opt in _optimizers(optimizer):
            try:
                owned = opt._get_model_param_range_map(param)['param']
            except KeyError:
                continue
            actual.view(-1)[owned.start:owned.end] = param.main_grad.view(-1)[owned.start:owned.end]
        dist.all_reduce(actual, group=group)
        if rank == 0:
            checks[name] = p.error_stats(actual, expected, .003)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    updated, grad_norm, _ = optimizer.step()
    assert updated and torch.isfinite(torch.tensor(float(grad_norm)))
    wrapped.start_param_sync(force_sync=True)
    changed = sum(int(not torch.equal(param, before[name])) for name, param in model.named_parameters())
    assert changed > 0, 'Optimizer did not change any model parameter'
    difference = 0.
    for param in model.parameters():
        root = param.detach().clone()
        dist.broadcast(root, src=0)
        difference = max(difference, float((param.detach() - root).float().abs().max()))
    difference = p.reduce_scalar(difference)
    passed = all(c['passed'] for c in checks.values()) if rank == 0 else True
    passed = bool(p.reduce_scalar(int(passed), dist.ReduceOp.MIN)) and difference == 0
    result = dict(world_size=world, tp=1, dp=world, length=length, grad_accumulation=2, passed=passed,
                  oracle='independent FP32 accumulation/all-reduce of same-backward parameter hooks before MCore reduction',
                  optimizer='Megatron Core distributed AdamW', updated=bool(updated),
                  grad_norm=float(grad_norm), changed_parameters=changed,
                  max_parameter_difference=difference, checks=checks)
    if rank == 0:
        p.write_json(Path(output) / 'distributed.json', result)
        p.log('distributed_oracle_done', passed=passed, max_parameter_difference=difference,
              max_gradient_relative_l2=max(c['relative_l2'] for c in checks.values()))
    for handle in handles:
        handle.remove()
    wrapped.disable_forward_pre_hook()
    del wrapped, optimizer, model, accumulated, expected, actual, param, root, loss, y, x, handles, before
    gc.collect(); torch.npu.empty_cache()
    assert passed, 'Distributed gradient or optimizer synchronization failed'
    return result
