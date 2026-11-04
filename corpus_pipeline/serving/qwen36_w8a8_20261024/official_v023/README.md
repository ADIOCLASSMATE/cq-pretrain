# Official Ascend Qwen3.6 W8A8 TP2 deployment

This scope runs the unmodified fixed-tag Ascend recipe through native Inspire Serving. The server command directly executes upstream `vllm serve`; no project `serve.py`, gateway, response-header injection, custom kernel, graph-capture patch, or model wrapper runs in the allocation. External SDK code only manages allocation identity, readiness leases, basic health, recovery and stop.

## Pinned upstream contract

- [Qwen3.6-35B-A3B A2/A3 guide, v0.23.0rc1](https://docs.vllm.ai/projects/ascend/en/v0.23.0rc1/tutorials/models/Qwen3.6-35B-A3B.html)
- [Exact tagged Markdown](https://raw.githubusercontent.com/vllm-project/vllm-ascend/v0.23.0rc1/docs/source/tutorials/models/Qwen3.6-35B-A3B.md)
- [Matching installation guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0rc1/installation.html)
- Image: `vllm-ascend:v0.23.0rc1`, platform image ID `image-03380f97-1e02-4ed9-8c04-1e1797828ce9`.
- Fixed guide SHA-256: `3f8f063ef8d7861f46fa31d9289f833d2a5c3454dac4afb2f4e8d48f7cd1848f`.
- Model revision: `1a118d717bcbd59480f4a110fe22181d21711b4d`; verified full asset receipt: `../weights/WEIGHTS_READY.json`.
- The only model-command substitution is the remote repository argument → the exact verified local model directory. HIGH/LOW service names, platform IDs, priority and boot identity are outside the shared launch-contract hash.

```sh
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
vllm serve /inspire/sj-ssd3/project/high-dimensionaldata/public/models/Qwen3.6-35B-A3B-W8A8-1a118d71 --host 0.0.0.0 --port 8000 --data-parallel-size 1 --tensor-parallel-size 2 --enable-expert-parallel --seed 1024 --quantization ascend --served-model-name qwen3.6 --max-num-seqs 128 --max-model-len 262144 --max-num-batched-tokens 16384 --trust-remote-code --gpu-memory-utilization 0.90 --enable-prefix-caching --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"enable_cpu_binding":true, "enable_flashcomm1":true, "multistream_overlap_shared_expert": true}' --async-scheduling
```

The command preserves TP2, DP1, expert parallelism, Ascend W8A8, context 262144, sequences 128, batched tokens 16384, FULL_DECODE_ONLY, FlashComm1 and async scheduling. No explicit dtype or speculative/MTP options are added. Host-level governor/sysctl tuning from the example was not executed: these require shared-node administrator ownership.

## Observed image and startup

Installed package metadata was read from the exact current native instance without importing torch or opening a device context:

| Package | Observed version |
| --- | --- |
| vllm | `0.23.0+empty` |
| vllm-ascend | `0.23.0rc1` |
| torch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post2` |
| transformers | `5.5.4` |

This proves the observed package-version pair and binds it to the platform incarnation. It does not claim a binary digest comparison between the platform mirror and the upstream Quay image.

The official LD_PRELOAD path produced a missing-jemalloc loader warning in this platform image. It was not patched. Model loading, the 18-shape FULL decode graph capture, `/health`, `/v1/models` and the real image request all completed successfully despite that warning. Its possible throughput effect has not been isolated.

## Current-boot basic validation

- HIGH service: `sv-1dae334f-45a6-44b4-b1ea-276213c4d473`, two cards / 16 CPUs / 128 GiB, priority 6.
- Platform incarnation: `b28615e92f04521a3bff59e609408d8bee5b6d0514759af8fbfce7928d425c80`.
- API model returned by `/v1/models`: `qwen3.6`.
- One real image standard OpenAI SSE request completed with `[DONE]`, finish reason `stop`, 1051 prompt tokens and 19 output tokens; first-request latency 46.994 seconds.
- Receipt: `high/smoke/b28615e92f04521a3bff59e609408d8bee5b6d0514759af8fbfce7928d425c80/RESULT.json`.
- Independent fake-SDK checks: 8 adapter tests plus 11 budget-pool tests passed.

This is basic functional and current-instance validation, not the earlier 13-image quality canary or a detector-accuracy benchmark. CPU throughput is measured separately against the real pending annotation queue.

Registry `high/registry.json` uses `transport_mode=native_openai`, standard SSE, a 30-second ready lease renewed by the external observer, and `attestation_scope=platform_instance_and_registry`. `response_boot_attested=false`: there is no injected response-level boot proof. Clients check the platform instance, incarnation and shared launch contract before and after each full response.

The prior experimental custom-runtime HIGH has been stopped and freshly verified released; see `../high/USER_OFFICIAL_FLOW_RELEASED.json`. It is not part of this official runtime. The official HIGH completed the authorized CPU concurrency test and was drained before stopping: CPU confirmed it was absent from the active pool and had zero in-flight requests at epoch 1792794918.127. Fresh final verification found STOPPED plus INSTANCE_SUCCEEDED, two cards released, the observer exited, and registry ready=false. `high/RELEASED.json` records this final proof. LOW replicas remain owned by the separate shared-budget controller; the HIGH stayed reserved inside its 128-card cap until this verified release.

## External integration

`backend.py:Backend` implements prepare/create/observe/publish/restart/stop/released for `budget_pool.py`. Each slot owns its state and registry. Create and restart have persistent single-send fences; ambiguous results reserve capacity until reconciliation. Source identity and the command/asset hash are common to HIGH and LOW; native instance identity is separate.

`smoke.py` is an external CPU API client only. `high_control.py` is an external SDK lifecycle entry. Neither runs inside the model allocation.

## Read-only device observations during the real queue

These are six discrete `npu-smi` samples per window, not device busy-time integrals or a profiling trace. Each remote read was bracketed by matching current platform instance signatures. No extra inference requests or parameter changes were made.

| Stage | Epoch window | Per-card HBM, MB / 65536 | Sampled AI Core |
| --- | --- | --- | --- |
| Initial c16 | 1792794554–1792794566 | 60262–60270 | 0–27% |
| c32 after warmup | 1792794821–1792794833 | 60302–60304 | 0–28% |

Receipts are under `high/observations/c16-initial-1792794566.json` and `high/observations/c32-1792794833.json`. Completed-response throughput and token throughput belong to the CPU queue stage reports; these sparse utilization readings cannot replace those metrics or establish the cause of idle gaps. One transient SDK observation TransportError around epoch 1792794730 temporarily withdrew the ready lease; the following normal observation restored the same incarnation by epoch 1792794746, without restart or another smoke request.
