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
  --key-heads 8 --value-heads 8 --output .runs/gdn-tp2
PRETRAIN_BACKEND=cuda PRETRAIN_STACK=cuda OMP_NUM_THREADS=1 \
  .venv-cuda/bin/python -m scripts.validation.validate_cuda_ops --output .runs/cuda-ops.json
```

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
