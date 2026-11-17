# CUDA / L40S runtime

Use the official `docker.sii.shaipower.online/base/ngc-pytorch:25.02-cuda12.8.0-py3`
image. Its PyTorch `2.7.0a0+ecf3bae40a.nv25.02`, NVIDIA Triton `3.2.0`, Apex
FP32 weight-gradient extension and TransformerEngine are an ABI-matched set.
`environment.cuda.txt` pins only the additional wheels. Do not replace the
image's `pytorch-triton` with a separate newer `triton` distribution.

Prepare the wheelhouse on the CPU host, then install once in the shared checkout
from a GPU instance of that image:

```bash
uvx --from pip pip download --no-deps --only-binary=:all: --python-version 3.12 \
  --dest .deps/cuda-wheels -r environment.cuda.txt
bash scripts/setup_cuda.sh
```

All nodes of a job use the same image and shared `.venv-cuda`; no network install
or concurrent environment mutation is needed at job startup. The launcher shares
the upstream Triton compilation cache under `.deps/triton-cache`.

Upstream [FLA 0.4.0](https://github.com/fla-org/flash-linear-attention/tree/v0.4.0)
accepts packed `[1, events, heads, dim]` tensors and device `cu_seqlens`.
`ops/cuda_gated_delta.py` passes the existing plan's boundaries directly, retains
FP32 log-decay `g`, casts the precomputed sigmoid `beta` to BF16, and enables
upstream Q/K L2 normalization. No `head_first` or CPU-boundary keyword is needed.
No kernel, CPU CQ plan or indexing algorithm is replaced.

GPU gates (FP64 CPU oracles; the GDN tolerance remains 0.035):

```bash
PRETRAIN_BACKEND=cuda PRETRAIN_STACK=cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-cuda/bin/python -m scripts.validation.validate_cq_cuda --output .runs/gdn-tp1
PRETRAIN_BACKEND=cuda PRETRAIN_STACK=cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-cuda/bin/python -m scripts.validation.validate_cq_cuda \
  --kernel-head-shards 2 --output .runs/gdn-tp2
PRETRAIN_BACKEND=cuda PRETRAIN_STACK=cuda OMP_NUM_THREADS=1 \
  .venv-cuda/bin/python -m scripts.validation.validate_cuda_ops --output .runs/cuda-ops.json
```

The second gate keeps the same full layer and FP64 oracle and splits each FLA
call into two 8-head calls. It tests the TP-local kernel shapes on one GPU;
distributed collectives require the separate two-GPU training run. An additional
independently initialized 8-head stress fixture (`--key-heads 8 --value-heads 8`)
exceeds the 0.035 gradient tolerance in one case, including in a pure CPU BF16
recurrence. Preserve that failure in the evidence; do not loosen the tolerance
or treat the finite-loss training smoke as proof that it passed.

Use the original `configs/qwen35_2b_wan.yaml` with `--synthetic` and explicit local
asset paths. Synthetic batches exercise the full model with fake pixels/latents;
they do not exercise VAE decoding or corpus I/O. TP1 on a single L40S needs upstream
MCore CPU optimizer offload because the unsharded Adam training state exceeds its
45 GiB memory. Report offload separately from GPU-only TP/DP throughput.

Every development Notebook and Job must use **priority 3**. See `INSPIRE.md` for
the explicit workspace, project and compute group. Multi-node jobs use complete
8-GPU nodes, `--shm-size 128`, platform automatic fault tolerance and training
`--auto-resume` with a shared checkpoint directory. `scripts/launch.sh` maps
Inspire's `PET_*` rendezvous variables into torchrun. TP groups must fit in a node.
Successful updates are recorded in `steps_rank_*.jsonl`; run
`python -m scripts.validation.summarize_cuda_run RUN --nodes N --steps K` to
require finite updates from every rank before accepting the run.

## L40S parallelism and tuning (measured 2026-11-17)

Numbers below are from `evaluations/benchmarks/cuda_l40s_tuning_20261117/tuning.json`,
full Qwen3.5-2B-Base + Wan2.1-1.3B weights, synthetic batches, 5 steps after warmup.
They are device-time measurements without corpus I/O, so they bound the compute
path only.

### Sequence length 8192 needs BF16 gradient reduction, not tensor parallelism

| config | peak GiB | s/update | throughput |
|---|---:|---:|---:|
| TP1 DP2, FP32 grad reduce | OOM | - | - |
| TP1 DP2, `--bf16-grad-reduce` | 38.41 | 4.051 | 4044 |
| TP2 DP1, FP32 grad reduce | 33.06 | 2.898 | 2826 |
| TP2 DP1, GA=2 | 33.06 | 5.627 | 2912 |

TP1 is **43% faster than TP2 at equal card count**, because L40S has no NVLink and
every tensor-parallel collective crosses PCIe. Prefer TP=1 and scale with DP.
Raise TP only when a configuration cannot be made to fit by the memory levers below.

Peak memory is dominated by **distributed-optimizer state, which shrinks with DP
width**, not by activations: with full recompute the activation working set is
roughly 5 GiB at sequence 2048 and 7 GiB at 8192, while FP32 master weights plus
Adam moments are ~36.9 GiB divided by DP. This is why the same shape OOMs at DP2
and fits at DP16. Do not size a production run from a two-GPU probe.

`--bf16-grad-reduce` sets MCore `grad_reduce_in_fp32=False`. It saves 5.05 GiB per
rank (42.57 -> 37.52 at sequence 2048) and halves the inter-node gradient all-reduce
from 13.2 GB to 6.6 GB for the 3.3B trainable parameters. Six-step same-seed A/B:
maximum absolute loss drift 1.2e-3, non-monotonic, with grad norms tracking to three
digits — BF16 round-off, not a systematic bias. Re-check this on a long run before
trusting it for a full pretraining schedule.

### Gradient accumulation is exercised

GA=2 at sequence 8192 keeps peak memory identical and throughput within noise,
confirming `OrderedGradientBuckets`, the MCore `no_sync` nesting and the deferred
DiT chunked backward behave on CUDA. Earlier single-GA runs did not cover this path.

### DiT pointwise fusion: measured, rejected

Compiling `modulated_norm` and `gated_add` through Inductor (`PRETRAIN_COMPILE_MODULATION=1`,
`ops/cuda_compile.py`) is **1% slower** at sequence 8192 (4.090 vs 4.050 s) with
identical peak memory. Keep it at `0`. The wiring is retained because the measurement
should be repeated with DiT recompute disabled: under `use_reentrant=True`
checkpointing the compiled region is re-entered rather than fused with its neighbours,
which plausibly explains the absent win. This mirrors the Ascend result where
fusing `affine`/`gated_add` was also slower.

### Attention memory

`ops/functional.attention` no longer materializes GQA key/value copies
(`expand` instead of `repeat_interleave`, 4x for 8 heads over 2 KV heads) and caches
the inverted CQ mask on the plan instead of allocating a fresh `~mask` per layer —
128 MB per call at sequence 8192, 6 full-attention layers per step. Both are exact;
the CPU reference suites are unchanged.

The remaining attention inefficiency is structural: the CQ mask reaches the kernel
as a dense `[1,1,2S,S]` boolean, which prevents SDPA from selecting its flash
backend. Removing it requires `sigma`/`segments` on device at the attention call
site so a FlexAttention `mask_mod` can evaluate the predicate blockwise. That means
extending `MainlinePlan` and the CPU plan builders, which are deliberately frozen —
raise it as a design change rather than patching it in.

### Where the step time actually goes

Torch profiler, rank 0, sequence 8192, TP1 DP2, full recompute
(`.runs/prof-1117`). Phase timing on the same shape: forward 941 ms (23%),
backward 2916 ms (72%), optimizer 155 ms (4%) — backward is 3.1x forward, which
is the signature of recomputing every layer.

| CUDA kernel category | share |
|---|---:|
| elementwise | 36.1% |
| GEMM | 24.0% |
| attention | 11.8% |
| NCCL collective | 11.2% |
| norm/reduce | 5.5% |
| other | 4.8% |
| copy/index | 3.6% |
| GDN (FLA Triton) | 3.0% |

Three things follow, and two of them contradict the obvious guesses:

- **The GDN kernel is not a bottleneck at 3%.** Effort belongs in the surrounding
  projections and normalization, not in the recurrence.
- **Elementwise work, not GEMM, dominated.** That was unfused normalization; see
  `ops/cuda_fused.py`.
- **Attention runs on the memory-efficient backend, not flash**
  (`fmha_cutlassB_bf16` 775 ms backward, `fmha_cutlassF_bf16` 219 ms forward),
  because the CQ mask arrives as a dense `[1,1,2S,S]` boolean. Flash cannot take
  an arbitrary mask, so this is structural; see the FlexAttention note above.
- NCCL is 11.2% here only because this probe used GA=1. At the production GA=8 it
  amortizes, which is why per-card throughput *rises* from 2 to 16 cards.

### Recompute depth is a throughput knob, not a binary

At production shape the full-recompute configuration leaves 22 GiB of each card
unused. Backbone recompute depth converts that into speed (16xL40S, sequence
8192, TP1 DP16, GA8, BF16 grad reduce, all 16 ranks verified):

| backbone / DiT layers recomputed | s/update | peak GiB | vs full |
|---|---:|---:|---:|
| 24 / 30 (full) | 30.374 | 22.32 | baseline |
| 22 / 30 | 29.717 | 27.84 | +2.2% |
| 18 / 30 | 27.238 | 36.70 | +11.5% |
| 24 / 0 | 30.163 | 28.41 | +0.7% |
| 0 / 30, and 0 / 0 | OOM | | |

Roughly 2.4 GiB and 1.9% per skipped backbone layer. Recomputing the whole
30-layer DiT costs only 0.7%, so the DiT is a small share of *this* shape — one
image per pack. Packs with many generation targets will shift that, so re-measure
before carrying the DiT conclusion to real data.

Choose depth by the margin you want: 18/30 leaves about 7.7 GiB of a 44.39 GiB
card, which is thin once real packs vary in image count. 20/30 or 22/30 are the
conservative picks. Treat these as upper bounds on achievable speed, not as
settings to adopt blind.

## L40S tuning (measured 2026-11-17)

Production shape is 16x L40S (2 full nodes), `--sequence-length 8192`,
`--global-batch 128` (GA 8 at DP16) and `--bf16-grad-reduce`.

**Parallelism.** TP1 beats TP2 by 43% on two cards at seq 8192 (4044 vs 2826
tok/s): without NVLink a tensor-parallel all-reduce rides PCIe every layer, so
scale with DP and keep TP at 1. Per-card throughput *rises* from 2 cards to 16
(2023 -> 2158 tok/s) because GA 8 amortizes the gradient all-reduce, so the
multi-node collective is not the limit.

**Gradient reduction dtype.** `--bf16-grad-reduce` is what makes seq 8192 fit:
FP32 reduction OOMs on two cards where BF16 peaks at 38.41 GiB, and it halves
the inter-node all-reduce from 13.2 GB to 6.6 GB. Six same-seed steps drift at
most 1.2e-3 in loss, non-monotonically -- BF16 rounding, not a bias.

**Peak memory is optimizer state, not activations.** At seq 2048 the same
configuration peaks at 42.56 GiB on DP2 and 26.47 GiB on DP16; activations under
full recompute are only ~5 GiB there and ~7 GiB at seq 8192. Memory therefore
falls as DP widens, and a single-card measurement does not predict production.

**Recompute depth is not a useful knob here.** Measured from one frozen source
tree at production shape: full recompute 24.599s / 22.19 GiB, `bb24-dit22`
24.600s / 23.83 GiB (-0.0%), `bb22-dit26` 24.355s / 26.40 GiB (+1.0%). Keep
full recompute: it is tied-fastest *and* cheapest, which leaves headroom for
real-data packs whose image counts vary. The Ascend conclusion that full
recompute is optimal survives the port, for a different reason (there it was a
hard memory bound; here the activations are simply small).

**Fused normalization is the large win.** `PRETRAIN_CUDA_FUSED_NORM=1` (default)
routes RMSNorm to flash-attn's Triton kernel and the gated RMSNorm to the same
upstream `fla.modules.fused_norm_gate` the Ascend path already used: 7.6x and
7.3x at production shapes, **1.212x end to end on two cards and 1.235x at 16**.

**Pointwise fusion of the DiT is not worth enabling.** `torch.compile` on
`modulated_norm`/`gated_add` measured 1% *slower*; keep
`PRETRAIN_COMPILE_MODULATION=0`. The profile explains why -- recomputing all 30
DiT layers costs only 0.7%, so the DiT is a small share of this step.

### Where the time goes

Kernel time at production shape, after fused norms:

| category | share |
|---|---:|
| GEMM | 34.0% |
| elementwise | 25.4% |
| attention | 17.2% |
| norm/reduce | 7.0% |
| copy/index | 5.1% |
| GDN (FLA Triton) | 4.1% |
| NCCL | 3.7% |

Fusing the norms moved elementwise from 36.1% to 25.4%. The GDN recurrence is
**not** a bottleneck at 4.1%, and NCCL is not one either at 3.7%. The remaining
addressable item is attention: the mem-efficient backward kernel alone is 12.9%
because the CQ mask reaches SDPA as a dense tensor.

### Experiment hygiene

Jobs read source from the shared checkout, so editing the tree while a sweep
runs silently changes what later configurations execute. One sweep was
invalidated this way; its recompute "speedups" were the fused norms landing
mid-run. Freeze first and submit the snapshot:

```bash
bash scripts/freeze_source.sh .runs/<run>/source
```

`run.json` records `source_sha256` and `kernel_options`; compare both before
comparing timings, and treat runs from different trees as incomparable.

## CQ attention: predicate + FlexAttention

The CQ visibility rule is structured, not arbitrary, so it does not need a dense
mask. `ops/attention_layout.py` exposes the same rule as small int/bool vectors
(`cq_mask_vectors` -> `CQPredicate`) and `ops/flex_cq.py` evaluates it as a
FlexAttention `mask_mod`:

```
allowed(q, k) = row_segment[q] == key_segment[k] and row_segment[q] >= 0
                and not row_unconditional[q]
                and (key_sigma[k] <  row_sigma[q]  if q is a query row
                     key_sigma[k] <= row_sigma[q]  if q is a content row)
             or (row_invalid[q] and k == 0)          # same finite sentinel
```

`tests/test_cq_predicate.py` requires the predicate to rebuild the dense mask
**bitwise**, over single and multi-document packs, padding rows, tied-sigma
image blocks (bidirectional inside the group), unconditional queries, real plan
builder layouts and randomized layouts. The dense path stays the reference:
`reference=True` never takes this route, and an unrecorded GPU falls back.

Two gates run on device: `validate_flex_cq.py` compares forward and dq/dk/dv
against the dense path *through the real `attend()` dispatch*, and
`validate_flex_unconditional.py` forces every visual query unconditional and
checks behaviourally that perturbing all content moves those rows by exactly
zero while conditional rows do move. CFG dropout is the case where a wrong mask
stays silent -- every loss still looks finite.

### Tiles are probed offline, never searched at runtime

FlexAttention's default tiles assume far more shared memory than Ada provides.
L40S (sm89) has ~99 KiB per SM; at this backbone's `head_dim=256` the default
forward kernel asks for 213 KiB and cannot compile. Measured costs on this
kernel: `BLOCK_N` ~2 KiB per unit, `BLOCK_M` ~0.7 KiB per unit.

| BLOCK_M | BLOCK_N | stages | shared memory | result |
|---|---|---|---|---|
| 128 | 64 | default | 213120 B | fails |
| 128 | 32 | default | 147584 B | fails |
| 64 | 64 | default | 172032 B | fails |
| 64 | 32 | default | 102400 B | fails by 1024 B |
| **64** | **32** | **2** | fits | **10.53x fwd+bwd** |
| 32 | 32 | default | fits | 10.38x fwd+bwd |
| 16 | 16 | default | fits | 6.45x fwd+bwd |

Bigger tiles are not uniformly better: `(128,32)` at one stage compiles but
collapses to 1.00x. The configuration is measured, not derived, and recorded in
`configs/flex_tiles.json` keyed by capability and head size.

**Do not search for a tile inside the training process.** Inductor's async
compile worker pool does not recover after a kernel raises `OutOfResources`: a
retry ladder deadlocks (observed as a live process at ~1% CPU, idle GPU and no
new compile artifacts) instead of falling back. Probe candidates offline with
`scripts/validation/probe_flex_tile.py`, one isolated process per candidate with
a hard timeout, and record the winner:

```bash
.venv-cuda/bin/python scripts/validation/probe_flex_tile.py --block-m 64 --block-n 32 --num-stages 2
```

Enable with `PRETRAIN_FLEX_ATTENTION=1`; it is opt-in and recorded in
`run.json`'s `kernel_options`.

## Text cross entropy

The CUDA path had no fused branch: `fused_ce` was gated on `device.type == "npu"`,
so every chunk built an FP32 `[chunk, vocab]` logits tensor and made eight
PyTorch passes over it, in both forward and the backward recompute. With
vocabulary 248320 that is ~1.0 GiB per chunk of pure bandwidth, twice.

`ops/cuda_cross_entropy.py` keeps the chunking, the backward recompute and the
MCore direct-accumulation protocol and only swaps the per-chunk softmax and
gradient for flash-attn's Triton cross entropy, which also writes its gradient
back into the logits buffer. This is numerically the same computation:
`F.linear` on BF16 inputs already rounds the logits to BF16 before the old path
upcast them.

Measured on one L40S at vocabulary 248320, width 2048, chunk 1024:

| rows | old ms | fused ms | speedup | old peak | fused peak |
|---:|---:|---:|---:|---:|---:|
| 4096 | 224.9 | 105.4 | 2.13x | 5.72 GiB | 3.82 GiB |
| 6144 | 332.2 | 155.2 | 2.14x | 5.73 GiB | 3.84 GiB |
| 8192 | 441.1 | 217.2 | 2.03x | 5.75 GiB | 3.85 GiB |

liger-kernel 0.8.5's `fused_linear_cross_entropy` was measured on the same
shapes and is **slower** here (0.54x at 4096 and 6144 rows, 1.04x at 8192)
although it uses less memory. It is installed in `.venv-cuda` but the training
path does not reference it; do not adopt it without re-measuring.

`PRETRAIN_CUDA_FUSED_CE=1` enables the fused path. It refuses itself on NPU,
on CPU, under `torch.use_deterministic_algorithms`, and whenever TP > 1 --
vocabulary parallelism needs the original path, which all-reduces the maxima,
the selected logit and the partition function across the TP group. All six
refusals are asserted by `scripts/validation/validate_cuda_ce.py`, together with
an FP64 oracle at the real vocabulary and agreement with the old path at
production shapes.

## FP8 on Ada, measured before committing to it

Adopting FP8 means replacing the tensor-parallel Linears with TransformerEngine
ones, which changes parameter structure, the checkpoint schema and the gradient
protocol. `scripts/validation/bench_fp8.py` measures whether the hardware pays
for that first, using `torch._scaled_mm` -- the same cuBLASLt FP8 path TE
reaches -- at this model's shapes. TransformerEngine 2.0.0 reports FP8
available on sm89.

| GEMM | MxKxN | BF16 ms | FP8 ms | ratio |
|---|---|---:|---:|---:|
| lm_head / CE vocab | 8192x2048x248320 | 41.732 | 27.438 | 1.52 |
| backbone mlp gate_up | 8192x2048x12288 | 2.003 | 1.222 | 1.64 |
| backbone mlp down | 8192x6144x2048 | 1.022 | 0.584 | 1.75 |
| backbone attn q | 8192x2048x4096 | 0.664 | 0.374 | 1.77 |
| backbone attn kv | 8192x2048x1024 | 0.157 | 0.097 | 1.62 |
| backbone attn o | 8192x2048x2048 | 0.303 | 0.186 | 1.62 |
| gdn in_proj_all | 8192x2048x8224 | 1.197 | 0.801 | 1.49 |
| gdn out_proj | 8192x2048x2048 | 0.343 | 0.199 | 1.72 |
| dit ffn up | 4096x1536x8960 | 0.473 | 0.301 | 1.57 |
| dit ffn down | 4096x8960x1536 | 0.568 | 0.255 | 2.23 |
| dit attn qkv | 4096x1536x4608 | 0.308 | 0.154 | 1.99 |

GEMM-time-weighted ceiling: **1.54x**. This is an upper
bound with pre-quantized inputs and fixed scales; real FP8 training adds a cast
of every activation and weight, amax reduction and scale updates, and changes
numerics enough to need a long loss-curve comparison rather than a six-step A/B.

## Where the time goes, with every optimization enabled

Profiled at production shape (16 x L40S, seq 8192, TP1 DP16, GA8, BF16 gradient
reduction) with fused norms, FlexAttention and fused CE all on. Shares are of
total GPU kernel time; kernels overlap, so they do not sum to step time.

| category | before any of this | after fused norms | all on |
|---|---:|---:|---:|
| GEMM | 24.0% | 34.0% | **41.8%** |
| elementwise | 36.1% | 25.4% | 15.8% |
| copy/index | 3.6% | 5.1% | 12.0% |
| other | 4.8% | 3.5% | 9.2% |
| attention | 11.8% | 17.2% | 8.4% |
| GDN recurrence | 3.0% | 4.1% | 4.7% |
| NCCL | 11.2% | 3.7% | 4.0% |
| norm/reduce | 5.5% | 7.0% | 2.6% |

Read the attention row carefully. FlexAttention compiles to Inductor *template*
kernels named `triton_tem_fused_*`; a parser that does not know this reports
attention at 0.6% and inflates "other". The A/B is unambiguous: with
`PRETRAIN_FLEX_ATTENTION=0` those kernels are absent and `fmha_cutlass*` appear
at the same share. `scripts/validation/profile_summary.py` classifies them, and
also attributes each kernel to its launching aten op, which is what exposed the
two fallbacks below.

Each profile so far has overturned the previous priority list, so re-profile
after a change rather than working down a stale ranking. Current reads:

- **GEMM now dominates at 41.8%.** Arithmetic, not framework overhead. This is
  the only remaining category large enough for a step-level win, and FP8 is the
  one lever on it; see the ceiling measurement above.
- **NCCL fell to 4.0%.** Multi-node data parallelism is not a bottleneck at
  GA=8 with BF16 gradient reduction. Per-card throughput *rose* from 2 cards to
  16 (2023 -> 2158 tok/s), because GA amortizes the all-reduce.
- **copy/index reached 12%**: `index_select` / `index_add` / `cat` over the CQ
  event plan. This is the architecture's own layout machinery, not a port
  artifact. Reducing it means restructuring the GDN event gather, which sits on
  the protected CPU-plan boundary.

## Two more CUDA reference fallbacks

Both dispatched on `device.type == "npu"`, so CUDA silently took the explicit
numerical reference. This is the same bug shape as the cross entropy, and worth
grepping for after any port: a reference path that is correct and slow produces
no error, only a bad profile.

**Wired: the GDN causal convolution.** CUDA ran a Python loop calling
`F.conv1d` once per packed document, with a transpose into and out of each call
and a concatenation at the end. `fla.modules.convolution.causal_conv1d` takes
the document boundaries the plan already carries (`plan.content_cu`), which is
exactly what the Ascend path hands its own fused convolution. **6.41x** on the
operator ([1,8192,6144], 4 documents, forward+backward: 8.709 ms -> 1.360 ms).

`scripts/validation/validate_fused_ops.py` checks it against the reference loop
and, separately, that the properties the loop existed to guarantee survive:
perturbing every later document leaves earlier outputs **bitwise** unchanged,
and perturbing a token leaves outputs before it **bitwise** unchanged, while
that token's own output moves. Document independence and causality are verified,
not assumed.

**Rejected: the fused SwiGLU.** Upstream ships one, and it is 1.77x over the
reference pointwise pair -- but it reads its two inputs as contiguous, while
this MLP's gate and up are strided views of one packed `gate_up` projection.
Passing them directly returns **silently wrong values**: relative error 1.414
against the reference, no exception raised. Copying first is correct but moves
about 400 MB per call, leaving 1.15x, roughly 1% of a step. It is not wired, and
`ops/cuda_fused.py` records why. If anything here ever uses that kernel, prove
the inputs are contiguous first; the failure mode is wrong numbers, not a crash.
