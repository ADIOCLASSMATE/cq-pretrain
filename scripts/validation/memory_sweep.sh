#!/usr/bin/env bash
# Production-shape boundary sweep on 2x L40S (45 GiB usable each).
#
# Peak memory here is dominated by distributed-optimizer state, not activations:
# measured 42.56 GiB at DP2 vs 26.47 GiB at DP16 for the same 2048 sequence.
# Per-rank floor is params(bf16) 6.1 + main_grad(fp32) 12.3 + (adam+master)/DP,
# i.e. 55.3 GiB at DP1 -- above a single L40S before any activation. So always
# occupy both GPUs and let TP choose between DP2 (TP1) and DP1 (TP2).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT=.runs/memsweep2-$(date +%m%d)
mkdir -p "$ROOT"
ASSETS=(--text-assets .assets/Qwen3.5-2B-Base --vision-assets .assets/Qwen3.5-2B-Base
        --wan-assets .assets/Wan2.1-T2V-1.3B)
export NPROC_PER_NODE=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
run() {
  TP=$1; SEQ=$2; GA=$3; EXTRA=${4:-}
  DP=$(( 2 / TP )); GB=$(( DP * GA ))
  NAME="tp${TP}-dp${DP}-seq${SEQ}-ga${GA}${EXTRA:+-bf16grad}"
  printf '=== %-28s ' "$NAME"
  timeout 1200 bash scripts/launch.sh --config configs/qwen35_2b_wan.yaml --synthetic "${ASSETS[@]}" \
    --tp "$TP" --sequence-length "$SEQ" --global-batch "$GB" --images 1 \
    --steps 4 --warmup 1 --renderer-ema-decay 0 --eval-snapshot-every 0 --log-all-ranks \
    $EXTRA --output "$ROOT/$NAME" > "$ROOT/$NAME.log" 2>&1
  CODE=$?
  if [ $CODE -eq 0 ] && [ -f "$ROOT/$NAME/summary.json" ]; then
    python3 - "$ROOT/$NAME/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("OK   peak=%5.2f GiB  mean=%6.3f s  slots/s=%7.0f" % (
    d.get("peak_memory_gib",0), d.get("mean_seconds",0),
    d.get("tokens_per_second") or d.get("capacity_slots_per_second") or 0))
PY
  elif grep -qiE "out of memory|OutOfMemoryError" "$ROOT/$NAME.log"; then
    echo "OOM"
  else
    echo "FAIL(exit $CODE): $(grep -ioE 'error[^\"]{0,90}' "$ROOT/$NAME.log" | tail -1)"
  fi
}
# Production sequence length, and the untested grad-accumulation path.
run 1 2048 1 ; run 1 8192 1 ; run 2 8192 1
run 1 8192 2 ; run 2 8192 2
# Does BF16 gradient reduction unlock TP1 at production length?
run 1 8192 1 --bf16-grad-reduce
echo "=== sweep root: $ROOT ==="
