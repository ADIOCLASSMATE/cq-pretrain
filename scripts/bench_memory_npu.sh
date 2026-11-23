#!/usr/bin/env bash
# Can the 910B's spare 28 GiB be spent on speed?
#
# With the loader fed (scripts/bench_dataloader_npu.sh, then bench_final_npu.sh)
# the 8-pack configuration settles at 37.1 s per update and peaks at 32.5 GiB of
# 60.96. The earlier suite only tested the extreme -- `--no-checkpoint`, which
# recomputes nothing and needs 58.8 GiB -- and that OOMed. The whole middle of
# the range was never measured, and that middle is exactly what the extra memory
# buys: recompute fewer layers, or give the Wan DiT a larger token chunk.
#
# `--checkpoint-{backbone,dit}-layers N` recomputes the LAST N layers (see
# models/cq_backbone.py:33 and models/generator_wan21.py:236), so N below the
# layer count keeps the first layers' activations resident. Passing either one
# turns the all-or-nothing booleans off, so full backbone recompute must be
# requested explicitly as 24.
#
# `--micro-batch` is deliberately absent: training/configure.py rejects anything
# but 1 for the real variable-layout adapter, so it is not an available axis.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-memory-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"
echo "cpus=$(nproc) ranks=${NPROC_PER_NODE}"

failed=()
run() {
  local name="$1"
  shift
  echo "=== ${name}: $* ==="
  BENCH_NAME="${name}" PACKS_PER_CARD=8 BENCH_STEPS=14 BENCH_WARMUP=4 \
    bash "${ROOT}/scripts/bench_accel.sh" "$@" 2>&1 || failed+=("${name}")
  python3 - "${BENCH_ROOT}/${name}" <<'PY' || true
import json, statistics as st, sys
from pathlib import Path
path = Path(sys.argv[1]) / "steps.jsonl"
if not path.exists():
    print("  no steps.jsonl")
    raise SystemExit(0)
rows = [json.loads(line) for line in path.open()]
measured = [r for r in rows if not r.get("warmup")]
if not measured:
    print("  no measured steps")
    raise SystemExit(0)
seconds = sorted(r["seconds"] for r in measured)
waits = [r.get("data_wait_seconds", 0.0) for r in measured]
peaks = [r["peak_memory_gib"] for r in measured]
occupied = st.median(r["occupied_tokens"] for r in measured) / 16
print(f"  n={len(measured)} mean={st.mean(seconds):.2f} median={st.median(seconds):.2f} max={max(seconds):.2f}")
print(f"  data_wait median={st.median(waits):.2f} max={max(waits):.2f}")
print(f"  peak_gib median={st.median(peaks):.2f} max={max(peaks):.2f}")
print(f"  fwd/bwd median ms = {st.median(r['forward_ms'] for r in measured):.0f}"
      f" / {st.median(r['backward_ms'] for r in measured):.0f}")
print(f"  occupied_tokens/s/card = {occupied / st.median(seconds):.1f}")
PY
  echo "=== ${name} done ==="
}

LOADER=(--data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24)

# The measured baseline, repeated short, so this suite carries its own control
# instead of comparing against another node's numbers.
run base-full --checkpoint --keep-attention-activations 6 "${LOADER[@]}"

# Spend memory on the DiT first: it holds 30 of the 54 recomputed layers and
# backward is 25 s of the 37 s step.
run dit-keep6 --checkpoint-backbone-layers 24 --checkpoint-dit-layers 24 --keep-attention-activations 6 "${LOADER[@]}"
run dit-keep12 --checkpoint-backbone-layers 24 --checkpoint-dit-layers 18 --keep-attention-activations 6 "${LOADER[@]}"
run dit-keep18 --checkpoint-backbone-layers 24 --checkpoint-dit-layers 12 --keep-attention-activations 6 "${LOADER[@]}"

# Then the backbone, which also carries the GDN recurrence and attention.
run bb-keep6 --checkpoint-backbone-layers 18 --checkpoint-dit-layers 30 --keep-attention-activations 6 "${LOADER[@]}"

# A different use of the same memory: fewer, larger Wan chunks per update.
run ditchunk16k --checkpoint --keep-attention-activations 6 --dit-microbatch-tokens 16384 "${LOADER[@]}"

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED configurations: ${failed[*]}"
  exit 1
fi
echo "suite complete: ${BENCH_ROOT}"
