#!/usr/bin/env bash
# Quick integration test for competitive job submission.
# Tests node discovery, dry-run, and validates the output format.

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

echo "=== Testing competitive job submission ===" >&2

# Test 1: Node discovery and dry-run
echo "[1/3] Testing node discovery (dry-run)..." >&2
result=$("$PYTHON" "$REPO/scripts/competitive_submit.py" \
    --job-name test-discovery-$RANDOM \
    --nodes 1 \
    --command "echo test" \
    --workspace "昇腾卡公共空间" \
    --project "公共科研项目" \
    --group "910B资源" \
    --dry-run \
    --verbose 2>&1)

if echo "$result" | grep -q "Discovery took"; then
    echo "✓ Node discovery working" >&2
    # Extract timing
    timing=$(echo "$result" | grep "Discovery took" | sed 's/.*Discovery took //; s/ ms.*//')
    echo "  Latency: ${timing}ms" >&2
else
    echo "❌ Node discovery failed" >&2
    echo "$result" >&2
    exit 1
fi

# Test 2: Submit wrapper script
echo "[2/3] Testing submit wrapper script..." >&2
bash -n "$REPO/scripts/submit_competitive.sh" && echo "✓ Wrapper syntax OK" >&2 || {
    echo "❌ Wrapper has syntax errors" >&2
    exit 1
}

# Test 3: Eval job wrapper
echo "[3/3] Testing eval job wrapper script..." >&2
bash -n "$REPO/scripts/submit_eval_job.sh" && echo "✓ Eval wrapper syntax OK" >&2 || {
    echo "❌ Eval wrapper has syntax errors" >&2
    exit 1
}

echo "=== All tests passed ===" >&2
echo "Ready to integrate competitive submission into controllers." >&2
