#!/usr/bin/env bash
# Usage: scripts/evaluate.sh TASK WEIGHTS OUTPUT_DIR
# Runs one benchmark task (qa, vqav2, geneval or fid) on all local GPUs. FID
# features are extracted on the CPU while images are sampled; GenEval images are
# scored by the official detector once sampling has released the GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task="$1"; weights="$2"; out="$3"
cd "$ROOT"
read_setting() { "${ROOT}/.venv-cuda/bin/python" -c "from evaluations.suites import settings;print(settings()['assets']['$1'])"; }
benchmarks="$(read_setting benchmarks)"; geneval="$(read_setting geneval)"
mkdir -p "$out"
export PRETRAIN_MODULE=evaluations.suites.run
export NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
export OMP_NUM_THREADS=1 PRETRAIN_BACKEND=cuda PRETRAIN_STACK=cuda PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$out/../.triton}"
{ echo "host=$(hostname) task=$task weights=$weights gpus=$NPROC_PER_NODE commit=$(cat SOURCE_COMMIT 2>/dev/null || echo unknown) $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c; } >> "$out/job.log"
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  # Smoke test: first N items only, no FID scoring or GenEval detection.
  bash scripts/launch.sh "$task" "$weights" "$out" --limit "$EVAL_LIMIT" >> "$out/job.log" 2>&1
  [[ -f "$out/SMOKE.json" ]] || { echo "No SMOKE.json for $task" >> "$out/job.log"; exit 3; }
  exit 0
fi
if [[ "$task" == fid ]]; then
  env PYTHONPATH="$benchmarks/fid/tf-env/lib/python3.12/site-packages" CUDA_VISIBLE_DEVICES="" TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2 OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12 \
    /usr/bin/python3.12 -S "$ROOT/evaluations/suites/fid_score.py" "$out" "$benchmarks" >> "$out/fid-score.log" 2>&1 &
  scorer=$!
  bash scripts/launch.sh "$task" "$weights" "$out" >> "$out/job.log" 2>&1 &
  sampler=$!
  trap 'kill "$scorer" "$sampler" 2>/dev/null || true' EXIT
  wait -n -p finished "$scorer" "$sampler"
  if [[ "$finished" == "$sampler" ]]; then wait "$sampler"; wait "$scorer"; else wait "$scorer"; wait "$sampler"; fi
  trap - EXIT
else
  bash scripts/launch.sh "$task" "$weights" "$out" >> "$out/job.log" 2>&1
  if [[ "$task" == geneval ]]; then
    python3 "$ROOT/evaluations/suites/geneval_detect.py" "$out" "$geneval" >> "$out/detector.log" 2>&1
  fi
fi
[[ -f "$out/RESULTS.json" ]] || { echo "No RESULTS.json for $task" >> "$out/job.log"; exit 3; }
