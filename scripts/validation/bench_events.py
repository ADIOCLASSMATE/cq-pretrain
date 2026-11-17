"""Measure the CQ event gathers using a real plan, not synthetic random indices.

A first attempt used torch.randint for event_write and concluded the block ran
at ~83 GB/s on an 864 GB/s card. That was an artifact: the real plan writes
content in sigma order, so its indices are nearly monotonic and the gather has
far better locality than a random permutation. Measure the plan the model
actually builds.

Variants, all producing bitwise-identical values and gradients:
  current  cat a zero row onto k, v, g and beta separately, then four gathers
  merged   one cat and one gather for k|v (adjacent slices of `convolved`),
           one for g|beta
  masked   clamp the index and zero the read-only rows instead of padding
"""
import argparse
import json
from pathlib import Path

import torch


def current(convolved, control, event_write):
    size = convolved.shape[-1] // 3
    _, k, v = convolved.split((size, size, size), -1)
    zkv = k.new_zeros(1, size)
    k = torch.cat((k, zkv), 0).index_select(0, event_write)
    v = torch.cat((v, zkv), 0).index_select(0, event_write)
    g, beta = control.chunk(2, -1)
    zc = g.new_zeros(1, g.shape[-1])
    g = torch.cat((g, zc), 0).index_select(0, event_write)
    beta = torch.cat((beta, zc), 0).index_select(0, event_write)
    return k, v, g, beta


def merged(convolved, control, event_write):
    """k and v are adjacent in `convolved`, and share one event_write index."""
    size = convolved.shape[-1] // 3
    kv = convolved[:, size:]
    kv = torch.cat((kv, kv.new_zeros(1, 2 * size)), 0).index_select(0, event_write)
    gb = torch.cat((control, control.new_zeros(1, control.shape[-1])), 0)
    gb = gb.index_select(0, event_write)
    k, v = kv.split((size, size), -1)
    g, beta = gb.chunk(2, -1)
    return k, v, g, beta


def masked(convolved, control, event_write):
    size = convolved.shape[-1] // 3
    n = convolved.shape[0]
    index = event_write.clamp(max=n - 1)
    keep = (event_write < n).unsqueeze(-1)
    kv = convolved[:, size:].index_select(0, index) * keep
    gb = control.index_select(0, index) * keep
    k, v = kv.split((size, size), -1)
    g, beta = gb.chunk(2, -1)
    return k, v, g, beta


def scattered(convolved, control, event_write, write_slots, n_events):
    """Writes are arange(nc) in order, so scatter content instead of gathering.

    Verified against `current` bitwise, values and gradients.
    """
    size = convolved.shape[-1] // 3
    kv_src = convolved[:, size:]
    kv = kv_src.new_zeros(n_events, 2 * size).index_copy_(0, write_slots, kv_src)
    gb = control.new_zeros(n_events, control.shape[-1]).index_copy_(0, write_slots, control)
    k, v = kv.split((size, size), -1)
    g, beta = gb.chunk(2, -1)
    return k, v, g, beta


def timed(fn, args, iters=20):
    def run():
        out = fn(*args)
        sum(t.float().square().sum() for t in out).backward()
        for a in args:
            if torch.is_tensor(a) and a.dtype.is_floating_point and a.grad is not None:
                a.grad = None
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--images", type=int, default=2)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    from configs.config import RunConfig
    from data.batch import fake_batch

    cfg = RunConfig.load("configs/qwen35_2b_wan.yaml")
    cfg.text_assets = ".assets/Qwen3.5-2B-Base"
    cfg.vision_assets = ".assets/Qwen3.5-2B-Base"
    cfg.wan_assets = ".assets/Wan2.1-T2V-1.3B"
    batch = fake_batch(cfg, a.length, 1, a.images, seed=3)
    plan = batch.mainline.delta_plan
    ew = plan.event_write.cuda()
    content = len(plan.content_order)
    events = len(plan.event_q)

    # How ordered is the real index, versus the random one I first assumed?
    diffs = (ew[1:].long() - ew[:-1].long())
    monotonic = float((diffs >= 0).float().mean())
    print(f"  real plan: content={content} events={events} "
          f"nondecreasing_steps={monotonic:.1%}")
    rnd = torch.randint(0, content, (events,), device="cuda")
    rdiff = (rnd[1:].long() - rnd[:-1].long())
    print(f"  random index for comparison: nondecreasing_steps="
          f"{float((rdiff >= 0).float().mean()):.1%}")

    size = cfg.linear_heads * cfg.linear_head_dim // 1
    heads = cfg.linear_value_heads or cfg.linear_heads
    torch.manual_seed(11)
    conv = torch.randn(content, 3 * size, device="cuda", dtype=torch.bfloat16)
    ctrl = torch.randn(content, 2 * heads, device="cuda", dtype=torch.float32)

    out = {}
    base = bgrad = None
    bms = 0.0
    # Derived from the existing plan: which event slots carry a real write.
    write_slots = (ew < content).nonzero().flatten()
    src = ew.index_select(0, write_slots)
    identity = bool(torch.equal(src, torch.arange(content, device=src.device, dtype=src.dtype)))
    print(f"  write events are arange(nc) in order: {identity} "
          f"({len(write_slots)} writes of {events} events)")

    variants = [("current", current), ("merged", merged), ("masked", masked)]
    if identity:
        variants.append(("scattered", scattered))
    for name, fn in variants:
        args = [conv.clone().requires_grad_(), ctrl.clone().requires_grad_(), ew]
        if name == "scattered":
            args = args[:2] + [ew, write_slots, events]
        with torch.no_grad():
            values = fn(*args)
        o = fn(*args)
        sum(t.float().square().sum() for t in o).backward()
        grads = [args[0].grad.clone(), args[1].grad.clone()]
        args[0].grad = args[1].grad = None
        ms = timed(fn, args)
        if base is None:
            base, bgrad, bms = values, grads, ms
            verr = gerr = 0.0
        else:
            verr = max(float((x - y).abs().max()) for x, y in zip(values, base))
            gerr = max(float((x - y).abs().max()) for x, y in zip(grads, bgrad))
        out[name] = dict(ms=ms, value_max_abs=verr, grad_max_abs=gerr,
                         speedup=bms / ms)
        print(f"  {name:9s} {ms:7.3f} ms  value_err={verr:.1e} grad_err={gerr:.1e} "
              f"-> {bms/ms:.2f}x")

    # Is this block bandwidth-bound? Bytes actually moved by `current`.
    fwd = (content * size * 2 * 2          # k, v padded copies
           + events * size * 2 * 2 * 2)    # k, v gather read+write
    gbytes = fwd * 2 / 2**30               # forward + a comparable backward
    best = min(v["ms"] for v in out.values())
    print(f"\n  ~{gbytes:.2f} GiB moved per layer; best {best:.3f} ms "
          f"-> {gbytes/(best/1000):.0f} GiB/s of ~864 GB/s peak")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(content=content, events=events,
                                        nondecreasing_fraction=monotonic,
                                        variants=out), indent=2))


if __name__ == "__main__":
    main()
