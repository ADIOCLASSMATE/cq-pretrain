#!/usr/bin/env bash
# Source in Bash or zsh before importing torch_npu. System CANN stays intact.
if [[ -n "${ZSH_VERSION:-}" ]]; then
  UMM_ENV_SCRIPT="${(%):-%x}"
elif [[ -n "${BASH_VERSION:-}" ]]; then
  UMM_ENV_SCRIPT="${BASH_SOURCE[0]}"
else
  echo "env_mindspeed.sh must be sourced from Bash or zsh" >&2
  return 1
fi
UMM_ENV_ROOT="$(cd -- "$(dirname -- "$UMM_ENV_SCRIPT")/.." && pwd)" || return 1
UMM_CANN_ENV="$UMM_ENV_ROOT/.deps/Ascend90/cann/set_env.sh"
if [[ ! -f "$UMM_CANN_ENV" ]]; then
  echo "Missing CQ-pinned CANN: run scripts/setup_mindspeed.py --sources --python --cann" >&2
  return 1
fi
# Vendor environment scripts reference optional unset variables.
UMM_RESTORE_NOUNSET=0
[[ $- == *u* ]] && UMM_RESTORE_NOUNSET=1
set +u
UMM_ENV_STATUS=0
source "$UMM_CANN_ENV" || UMM_ENV_STATUS=$?
if [[ "$UMM_ENV_STATUS" == 0 && -f "$UMM_ENV_ROOT/.deps/Ascend90/nnal/atb/set_env.sh" ]]; then
  source "$UMM_ENV_ROOT/.deps/Ascend90/nnal/atb/set_env.sh" || UMM_ENV_STATUS=$?
fi
[[ "$UMM_RESTORE_NOUNSET" == 1 ]] && set -u
[[ "$UMM_ENV_STATUS" == 0 ]] || return "$UMM_ENV_STATUS"
export PYTHON_BIN="${PYTHON_BIN:-$UMM_ENV_ROOT/.venv-npu/bin/python}"
export PATH="$UMM_ENV_ROOT/.venv-npu/bin:$PATH"
# MindSpeed's package initializer imports MegatronAdaptor even for a standalone
# GDN call. Inference needs the same pinned Megatron sources as MCore training.
export PYTHONPATH="$UMM_ENV_ROOT/.deps/core17/Megatron-LM:$UMM_ENV_ROOT/.deps/core17/MegatronAdaptor:$UMM_ENV_ROOT/.deps/core17/TransformerEngineNPU:$UMM_ENV_ROOT/.deps/core17/MindSpeed${PYTHONPATH:+:$PYTHONPATH}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# CANN otherwise writes temporary kernel_meta/debug files in the caller's cwd.
# Separate hosts' work directories on the shared checkout; honor explicit paths.
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$UMM_ENV_ROOT/.cache/ascend-work/${HOSTNAME:-$(hostname)}}"
mkdir -p "$ASCEND_WORK_PATH" || return $?
# Metadata versions alone miss a community reinstall overwriting NPU files.
# This uses only stdlib and verifies the Ascend RECORD before loading kernels.
"$PYTHON_BIN" "$UMM_ENV_ROOT/utils/triton_integrity.py" --quiet || return $?
