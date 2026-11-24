#!/usr/bin/env bash
# Integration layer for inspire-wjx skill.
#
# Usage from agent prompts:
#   inspire-wjx submit-job \
#     --job-name my-job \
#     --command "bash train.sh" \
#     --job-type training \
#     --required-cards 32 \
#     [--priority 3|4] [--cluster ascend|l40s] [--dry-run]
#
# Agents ONLY need:
#   - job_name
#   - command
#   - job_type (the controller decides priority/cluster based on this)
#
# Everything else is automatic or agent-optional.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# Forward all args to the resource controller
exec "$PYTHON" "$REPO/scripts/resource_controller.py" "$@"
