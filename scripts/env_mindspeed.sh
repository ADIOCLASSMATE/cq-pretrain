#!/usr/bin/env bash
# Source in Bash or zsh; use the same CANN/ATB/AscendC stack as main.
if [[ -n "${ZSH_VERSION:-}" ]]; then
  UMM_ENV_SCRIPT="${(%):-%x}"
else
  UMM_ENV_SCRIPT="${BASH_SOURCE[0]}"
fi
UMM_ENV_ROOT="$(cd -- "$(dirname -- "$UMM_ENV_SCRIPT")/.." && pwd)" || return 1
export PYTHON_BIN="${PYTHON_BIN:-$UMM_ENV_ROOT/.venv-npu/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
UMM_RESTORE_NOUNSET=0
[[ $- == *u* ]] && UMM_RESTORE_NOUNSET=1
set +u
UMM_ENV_STATUS=0
source "$UMM_ENV_ROOT/.deps/Ascend90/cann/set_env.sh" || UMM_ENV_STATUS=$?
for UMM_DRIVER_LIB in /usr/local/Ascend/driver/lib64/driver /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
  [[ ! -d "$UMM_DRIVER_LIB" ]] || export LD_LIBRARY_PATH="$UMM_DRIVER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
if [[ "$UMM_ENV_STATUS" == 0 ]]; then
  UMM_CXX_ABI="$("$PYTHON_BIN" -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')" || UMM_ENV_STATUS=$?
fi
if [[ "$UMM_ENV_STATUS" == 0 ]]; then
  source "$UMM_ENV_ROOT/.deps/Ascend90/nnal/atb/set_env.sh" "--cxx_abi=$UMM_CXX_ABI" || UMM_ENV_STATUS=$?
fi
[[ "$UMM_RESTORE_NOUNSET" == 1 ]] && set -u
[[ "$UMM_ENV_STATUS" == 0 ]] || return "$UMM_ENV_STATUS"
UMM_DELTA_SITE="$UMM_ENV_ROOT/.deps/delta-ascendc/site-packages"
export PYTHONPATH="$UMM_ENV_ROOT:$UMM_DELTA_SITE:$UMM_ENV_ROOT/.deps/delta64/site-packages:$UMM_ENV_ROOT/.deps/core17/Megatron-LM:$UMM_ENV_ROOT/.deps/core17/MegatronAdaptor:$UMM_ENV_ROOT/.deps/core17/TransformerEngineNPU:$UMM_ENV_ROOT/.deps/core17/MindSpeed${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH="$UMM_DELTA_SITE/fla_npu/opp/vendors/fla_npu_transformer:$UMM_DELTA_SITE/fla_npu/opp${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export FLA_NPU_STABLE_ABI=stable TRITON_ALL_BLOCKS_PARALLEL=0
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$UMM_ENV_ROOT/.deps/delta64/triton-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$UMM_ENV_ROOT/.deps/extensions-core17}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset CUDA_VISIBLE_DEVICES TORCH_DEVICE_BACKEND_AUTOLOAD
export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-$UMM_ENV_ROOT/.cache/ascend-work/${HOSTNAME:-$(hostname)}}"
mkdir -p "$ASCEND_WORK_PATH" || return $?
"$PYTHON_BIN" "$UMM_ENV_ROOT/utils/triton_integrity.py" --quiet || return $?
