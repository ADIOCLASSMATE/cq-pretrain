"""Read-only environment and fused-operator probe; execute on the NPU notebook."""
import importlib.metadata
import json
import torch
import torch_npu

result = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
          "devices": torch.npu.device_count()}
result["packages"] = {}
for name in ("megatron-core", "mindspeed", "transformers", "einops"):
    try:
        result["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["packages"][name] = None
result["fused_ops"] = {name: hasattr(torch_npu, name) for name in (
    "npu_rms_norm", "npu_add_rms_norm", "npu_rotary_mul", "npu_swiglu",
    "npu_fusion_attention", "npu_cross_entropy_loss", "npu_linear_cross_entropy_loss",
    "npu_fused_linear_cross_entropy_loss", "npu_fused_adamw", "npu_gelu")}
if torch.npu.is_available():
    result["device_name"] = torch.npu.get_device_name(0)
    result["memory_bytes"] = torch.npu.get_device_properties(0).total_memory
print(json.dumps(result, indent=2))
for name in ('npu_rms_norm', 'npu_rms_norm_backward', 'npu_add_rms_norm',
             'npu_add_rms_norm_backward', 'npu_rotary_mul', 'npu_rotary_mul_backward',
             'npu_cross_entropy_loss', 'npu_matmul_add_fp32'):
    op = getattr(torch.ops.npu, name, None)
    print(name, None if op is None else op._schemas)
