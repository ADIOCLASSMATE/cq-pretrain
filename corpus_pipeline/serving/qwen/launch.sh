#!/bin/bash
set -euo pipefail
umask 077
TASK_ROOT=/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/serving/qwen
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
TASK_NIC=$(python3 -c 'from pathlib import Path; print(next(x.split()[0] for x in Path("/proc/net/route").read_text().splitlines()[1:] if x.split()[1]=="00000000"))')
export GLOO_SOCKET_IFNAME="$TASK_NIC" TP_SOCKET_IFNAME="$TASK_NIC" HCCL_SOCKET_IFNAME="$TASK_NIC"
export HCCL_IF_IP=$(python3 -c 'import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(("1.1.1.1",80)); print(s.getsockname()[0]); s.close()')
export VLLM_HOST_IP="$HCCL_IF_IP"
export HCCL_EXEC_TIMEOUT=600 HCCL_CONNECT_TIMEOUT=600 HCCL_TRANSFER_TIMEOUT=600
export HCCL_BUFFSIZE=1024 OMP_PROC_BIND=false OMP_NUM_THREADS=2 ACL_OP_INIT_MODE=1 TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_RPC_TIMEOUT=360000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
exec python3 "$TASK_ROOT/serve.py"
