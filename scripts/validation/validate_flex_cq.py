"""Gate the FlexAttention CQ path against the dense-mask path it replaces.

Both paths must evaluate the same visibility rule, so the comparison is made at
the attention call itself (same Q/K/V, same scale, same valid_rows) in FP32,
where any disagreement is a semantic difference rather than BF16 rounding.
A BF16 pass is reported separately at training precision.
"""
import argparse
import json
from pathlib import Path

import torch

from configs.config import tiny_config
from data.batch import fake_batch
from ops import flex_cq
from ops.functional import attention as dense_attention

ASSETS = Path(__file__).resolve().parents[2] / '.assets/Qwen3.5-2B-Base'


def relerr(a, b):
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-8))


def cfg_for(images):
    cfg = tiny_config(linear_heads=2, linear_head_dim=8)
    cfg.renderer = 'wan'
    cfg.latent_channels = 16
    cfg.flow_draws = 1
    cfg.vision_encoder = 'qwen35'
    cfg.vision_assets = str(ASSETS)
    cfg.text_assets = str(ASSETS)
    return cfg


def one_case(length, images, heads, kv_heads, dim, dtype, seed):
    cfg = cfg_for(images)
    batch = fake_batch(cfg, length, 1, images, seed=seed)
    plan = batch.mainline
    pred = plan.cq_predicate
    assert pred is not None, 'plan carries no CQ predicate'
    rows, keys = len(pred.row_sigma), len(pred.key_sigma)

    dev = 'cuda'
    g = torch.Generator(device='cpu').manual_seed(seed)
    q = torch.randn(rows, 1, heads, dim, generator=g).to(dev).to(dtype).requires_grad_()
    k = torch.randn(keys, 1, kv_heads, dim, generator=g).to(dev).to(dtype).requires_grad_()
    v = torch.randn(keys, 1, kv_heads, dim, generator=g).to(dev).to(dtype).requires_grad_()
    q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))

    mask = plan.cq_mask.to(dev)
    valid = plan.cq_valid.to(dev)
    dpred = pred.to(dev)

    # dense path: exactly what the port runs today
    ref = dense_attention(q, k, v, mask, valid, fused=True)
    # flex path: same rule, block-sparse
    out = flex_cq.attention(q2, k2, v2, dpred, valid)

    u = torch.randn(ref.shape, generator=g).to(dev).to(ref.dtype)
    (ref.float() * u.float()).sum().backward()
    (out.float() * u.float()).sum().backward()

    return dict(
        length=length, images=images, rows=rows, keys=keys,
        heads=heads, kv_heads=kv_heads, dim=dim, dtype=str(dtype).replace('torch.', ''),
        output=relerr(out, ref), dq=relerr(q2.grad, q.grad),
        dk=relerr(k2.grad, k.grad), dv=relerr(v2.grad, v.grad),
        block_sparsity=flex_cq.sparsity(dpred, rows, keys),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--tol-fp32', type=float, default=2e-5)
    ap.add_argument('--tol-bf16', type=float, default=0.02)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA device required')
    flex_cq._ENABLED = True

    cases = []
    # GQA ratio 4 and head_dim 256 match the production backbone.
    for dtype, tol in ((torch.float32, a.tol_fp32), (torch.bfloat16, a.tol_bf16)):
        for length, images in ((1024, 1), (1536, 2), (2048, 3)):
            for heads, kv_heads, dim in ((8, 2, 256), (8, 8, 256), (4, 2, 64)):
                c = one_case(length, images, heads, kv_heads, dim, dtype, seed=100 + length)
                c['tolerance'] = tol
                c['passed'] = all(c[k] <= tol for k in ('output', 'dq', 'dk', 'dv'))
                cases.append(c)
                flag = 'ok  ' if c['passed'] else 'FAIL'
                print(f"  {flag} {c['dtype']:8s} len={length:5d} img={images} "
                      f"h={heads}/{kv_heads} d={dim:3d} out={c['output']:.2e} "
                      f"dq={c['dq']:.2e} dk={c['dk']:.2e} dv={c['dv']:.2e} "
                      f"sparsity={c['block_sparsity']}")

    result = dict(passed=all(c['passed'] for c in cases),
                  device=torch.cuda.get_device_name(0), cases=cases)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print('FLEX CQ GATE', 'PASSED' if result['passed'] else 'FAILED')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
