#!/usr/bin/env bash
# Which lines make the 910B step host bound?
#
# The 2026-11-23 profile established *that* the device idles about a third of
# every update while the host issues ~351k launches and ~1.9M torch ops, but it
# ran with with_stack=False, so not one of those could be traced to a call
# site. Nothing here measures throughput; a census step is far slower than a
# training step by construction, and its seconds must never be quoted.
#
# Two independent attributions, so neither is taken on trust:
#   census   every dispatch counted against the repository line that issued it,
#            separating calls written on a line from ones the autograd engine
#            issued beneath it, and listing device-to-host syncs on their own.
#   profile  the vendor profiler with Python stacks, which attributes *device*
#            time rather than call counts.
#
# The census runs first: it is the decisive one, its report is small and it is
# written the moment its window closes, so a later failure cannot lose it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_BACKEND=npu
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export BENCH_DATA="${BENCH_DATA:-/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/training_corpus_stage1_512_20261121_v1/release}"
export BENCH_ASSETS="${BENCH_ASSETS:-/inspire/sj-ssd3/project/high-dimensionaldata/public/models}"
export BENCH_ROOT="${BENCH_ROOT:-${ROOT}/.runs/bench-attribution-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${BENCH_ROOT}"
echo "suite root ${BENCH_ROOT}"

# The production configuration, unchanged: attribution of a different shape
# would send us to optimize lines the real run does not execute.
PRODUCTION=(--checkpoint --keep-attention-activations 6
            --data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24)
failed=()

echo "=== census: operators per repository line (2 updates, rank 0) ==="
BENCH_NAME=census PACKS_PER_CARD=8 BENCH_STEPS=7 BENCH_WARMUP=4 \
  bash "${ROOT}/scripts/bench_accel.sh" "${PRODUCTION[@]}" \
  --operator-census 2 --profile-ranks 0 2>&1 || failed+=(census)

python3 - "${BENCH_ROOT}/census/census_rank_00000.json" <<'PY' || failed+=(census-report)
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("no census report; read census/log.txt"); raise SystemExit(1)
report = json.loads(path.read_text())
print(f"steps={report['steps']}  dispatches/step={report['dispatches_per_step']:,}"
      f"  device->host syncs/step={report['device_to_host_syncs_per_step']:,}")
print("\n--- lines by dispatches per step (written = issued on that line itself) ---")
print(f"  {'line':<52}{'per_step':>10}{'written':>10}{'beneath':>10}")
for row in report["by_site"][:30]:
    print(f"  {row['site']:<52}{row['per_step']:>10,}{row['written_on_this_line']:>10,}{row['issued_beneath_it']:>10,}")
print("\n--- operators by dispatches per step ---")
for row in report["by_operator"][:25]:
    print(f"  {row['operator']:<52}{row['per_step']:>10,}")
print("\n--- device->host syncs (each drains the launch queue) ---")
for row in report["device_to_host_syncs"][:25]:
    print(f"  {row['site']:<52}{row['per_step']:>10,}  {row['operator']}")
print("\n--- line x operator ---")
for row in report["by_site_and_operator"][:30]:
    print(f"  {row['site']:<52}{row['per_step']:>10,}  {row['operator']}")
PY
echo "=== census done ==="

echo "=== profile: device time with Python stacks (1 active update, rank 0) ==="
PRETRAIN_PROFILE_STACK=1 BENCH_NAME=profile-stack PACKS_PER_CARD=8 BENCH_STEPS=7 BENCH_WARMUP=4 \
  bash "${ROOT}/scripts/bench_accel.sh" "${PRODUCTION[@]}" \
  --profile --profile-ranks 0 2>&1 || failed+=(profile-stack)

python3 - "${BENCH_ROOT}/profile-stack" <<'PY' || failed+=(profile-report)
import csv, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
details = list(root.rglob("ASCEND_PROFILER_OUTPUT/operator_details.csv"))
if not details:
    print("no operator_details.csv; read profile-stack/log.txt"); raise SystemExit(1)
calls, host, device, stacks = Counter(), Counter(), Counter(), Counter()
seen = rows = 0
with details[0].open() as handle:
    for row in csv.DictReader(handle):
        rows += 1
        name = row.get("Name", "")
        calls[name] += 1
        host[name] += float(row.get("Host Self Duration(us)") or 0)
        device[name] += float(row.get("Device Self Duration(us)") or 0)
        stack = row.get("Call Stack") or ""
        if stack:
            seen += 1
            # Keep the innermost frame of ours; the vendor stacks live under
            # the repository root too, so a bare prefix test is not enough.
            frames = [f for f in stack.split(";") if "ascend-pretrain/" in f
                      and not any(p in f for p in ("/.venv", "/site-packages/", "/.deps/", "/.python/"))]
            if frames:
                stacks[frames[-1].strip()] += 1
print(f"rows={rows:,}  rows carrying a call stack={seen:,}")
print(f"\n--- torch ops by host self time (us) ---")
for name, value in host.most_common(20):
    print(f"  {name:<46}{value:>14,.0f}{calls[name]:>10,}")
if stacks:
    print(f"\n--- innermost repository frame, by rows ---")
    for frame, count in stacks.most_common(30):
        print(f"  {count:>9,}  {frame}")
else:
    print("\nNo repository frame in any call stack: stacks were not captured.")
PY
echo "=== profile done ==="

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "suite complete: ${BENCH_ROOT}"
