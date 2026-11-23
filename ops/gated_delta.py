"""One entry point for the packed gated-delta recurrence, per backend.

CUDA calls upstream flash-linear-attention; Ascend calls the pinned AscendC
backend (or MindSpeed's kernel, with ``PRETRAIN_NPU_GDN=mindspeed``). Both take
the CPU planner's packed write/read events, so the recurrence itself is the
only thing that differs. The CPU oracle lives in
``models.backbone_qwen35.recurrent_delta`` and is selected by the caller.
"""

import distributed.accel as accel
import ops.audit as audit


def chunk(q, k, v, g, beta, plan):
    """One pack's recurrence; returns the packed output in [1, T, H, D]."""
    if accel.IS_NPU:
        import ops.npu_gated_delta as npu

        audit.record("gated_delta", npu.backend_name())
        return npu.chunk(q, k, v, g, beta, plan)
    from ops.cuda_gated_delta import chunk_gated_delta

    audit.record("gated_delta", "fla_chunk_gated_delta_rule")
    return chunk_gated_delta(q, k, v, g=g, beta=beta, cu_seqlens=plan.event_cu)[0]


def chunk_with_state(q, k, v, g, beta, plan):
    """Inference prefill: the same recurrence, plus each sequence's final state.

    The fused kernels return the state in their own working precision (FP32);
    decoding continues in that precision and casts on the way out.
    """
    if accel.IS_NPU:
        import ops.npu_gated_delta as npu

        audit.record("gated_delta", "ascendc_prefill")
        return npu.chunk_with_state(q, k, v, g, beta, plan)
    from ops.cuda_gated_delta import chunk_gated_delta

    audit.record("gated_delta", "fla_chunk_gated_delta_rule_state")
    return chunk_gated_delta(q, k, v, g=g, beta=beta, cu_seqlens=plan.event_cu, output_final_state=True)
