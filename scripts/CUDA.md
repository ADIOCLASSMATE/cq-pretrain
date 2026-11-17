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
