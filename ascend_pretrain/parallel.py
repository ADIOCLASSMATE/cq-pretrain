"""Megatron Core TP and distributed optimizer, adapted by MindSpeed for HCCL."""
from datetime import timedelta
import os
from pathlib import Path
import sys
import logging
import torch
from torch import nn
import torch.distributed as dist
from . import stack

_active = False
_config = None
_optimizer_instances = 1


# Allow enough time for every process to load weights before rendezvous.
def initialize(tp, model_config, timeout_minutes=60, optimizer_instances=1, initialize_weights=True):
    global _active, _config, _optimizer_instances
    _optimizer_instances = optimizer_instances
    import torch_npu  # noqa: F401
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    local_heads=(model_config.linear_value_heads or model_config.linear_heads)//tp
    if model_config.linear_head_dim!=128 or local_heads not in (4,8,16,32):
        raise ValueError('AscendC requires 128-dimensional GDN heads with a supported local head count')
    # Capture the native AscendC/Triton callables before MindSpeed patches FLA.
    from . import delta_ascendc
    delta_ascendc.initialize()
    for path in stack.source_directories():
        if not path.is_dir():
            raise RuntimeError('Run ascend_pretrain/setup_deps.py on the CPU host first')
        sys.path.insert(0, str(path))
    # Limit global patches; model kernels below call public torch_npu APIs explicitly.
    previous = sys.argv
    compile_function = torch.compile
    sys.argv = [previous[0], '--optimization-level', '0', '--optimizer-selection', 'fused_torch_adamw']
    try:
        from transformers.utils import logging as transformers_logging
        transformers_logging.set_verbosity_error()
        import megatron_adaptor  # noqa: F401
        import mindspeed.megatron_adaptor  # noqa: F401
    finally:
        sys.argv = previous
    # MA suppresses CUDA compile decorators during imports. Restore PyTorch's
    # real entry point for explicit TorchAIR compilation of our own model.
    torch.compile = compile_function
    # Native ATB accumulates directly into FP32, validated at vocabulary size.
    import fused_weight_gradient_mlp_cuda
    from .kernels import fused_wgrad
    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32 = fused_wgrad
    logging.getLogger('megatron.core.optimizer').setLevel(logging.WARNING)
    from megatron.core import parallel_state as ps
    from megatron.core.transformer.transformer_config import TransformerConfig
    dist.init_process_group('hccl', timeout=timedelta(minutes=timeout_minutes))
    ps.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1,
                                 create_gloo_process_groups=False,
                                 num_distributed_optimizer_instances=optimizer_instances)
    _config = TransformerConfig(num_layers=model_config.layers,
        hidden_size=model_config.width, num_attention_heads=model_config.heads,
        num_query_groups=model_config.kv_heads, kv_channels=model_config.head_dim,
        ffn_hidden_size=model_config.ffn, tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1, sequence_parallel=False,
        params_dtype=torch.bfloat16, bf16=True, use_cpu_initialization=True,
        gradient_accumulation_fusion=True, perform_initialization=initialize_weights)
    _active = True
    return _config


def state():
    if not _active:
        return None
    from megatron.core import parallel_state
    return parallel_state


def tp_size():
    return state().get_tensor_model_parallel_world_size() if _active else 1


def dp_size():
    return state().get_data_parallel_world_size() if _active else 1


def dp_rank():
    return state().get_data_parallel_rank() if _active else 0


def normal_init(tensor):
    return nn.init.normal_(tensor, std=.02)


class ColumnLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, stride=1):
        super().__init__()
        if _active:
            from megatron.core.tensor_parallel import ColumnParallelLinear
            self.impl = ColumnParallelLinear(in_features, out_features, config=_config,
                init_method=normal_init, bias=bias, gather_output=False, stride=stride)
            # MCore attaches these attributes inside random initialization.
            # Pretrained construction skips that work but still needs shards.
            mark_sharded(self.impl.weight, dim=0, stride=stride)
        else:
            self.impl = nn.Linear(in_features, out_features, bias=bias)
            normal_init(self.impl.weight)
            if bias:
                nn.init.zeros_(self.impl.bias)

    @property
    def weight(self):
        return self.impl.weight

    def forward(self, x):
        output = self.impl(x)
        return output[0] if isinstance(output, tuple) else output


class RowLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        if _active:
            from megatron.core.tensor_parallel import RowParallelLinear
            self.impl = RowParallelLinear(in_features, out_features, config=_config,
                init_method=normal_init, bias=bias, input_is_parallel=True, skip_bias_add=False)
            mark_sharded(self.impl.weight, dim=1)
            if (os.environ.get('PRETRAIN_MC2_ROW','0')=='1' and
                    self.impl.weight.shape[1]%32==0 and out_features%32==0):
                if tp_size() not in (2,4,8):raise ValueError('MC2 rows require TP2/4/8')
                from types import MethodType
                from .mc2 import row_forward
                # Invoke the original submodule so DDP parameter pre-hooks
                # still gather its current weight before the fused operation.
                self.impl.forward=MethodType(row_forward,self.impl)
        else:
            self.impl = nn.Linear(in_features, out_features, bias=bias)
            normal_init(self.impl.weight)
            if bias:
                nn.init.zeros_(self.impl.bias)

    def forward(self, x):
        output = self.impl(x)
        return output[0] if isinstance(output, tuple) else output


def make_embedding(vocab, width):
    if _active:
        from megatron.core.tensor_parallel import VocabParallelEmbedding
        embedding = VocabParallelEmbedding(vocab, width, config=_config, init_method=normal_init)
        mark_sharded(embedding.weight)
        return embedding
    embedding = nn.Embedding(vocab, width)
    normal_init(embedding.weight)
    return embedding


def make_optimizer(model, lr, weight_decay=.1, bucket_size=40_000_000, overlap_param_gather=True):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import get_megatron_optimizer, AdamOptimizerConfig as OptimizerConfig
    ddp_cfg = DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=True,
        overlap_param_gather=overlap_param_gather, use_distributed_optimizer=True, bucket_size=bucket_size,
        num_distributed_optimizer_instances=_optimizer_instances)
    wrapped = DistributedDataParallel(_config, ddp_cfg, model)
    optimizer_cfg = OptimizerConfig(optimizer='adam', lr=lr, min_lr=lr * .1,
        weight_decay=weight_decay, bf16=True, params_dtype=torch.bfloat16,
        use_distributed_optimizer=True, clip_grad=1., adam_beta1=.9, adam_beta2=.95)
    # Our checkpoint protocol writes local shards and uses HCCL broadcasts;
    # MCore's optional CPU/Gloo checkpoint groups are unnecessary.
    optimizer = get_megatron_optimizer(optimizer_cfg, [wrapped], use_gloo_process_groups=False)
    return wrapped, optimizer


def finish_grad_sync(wrapped):
    """DP reduction followed by a coalesced TP reduction of shared Q/K norms.

    Q/K normalization is applied to disjoint heads on TP ranks. Its scale is
    shared across heads, so those partial derivatives must be summed. Other
    replicated parameters already receive complete gradients through the
    ColumnParallelLinear input-gradient reduction.
    """
    wrapped.finish_grad_sync()
    if tp_size() > 1:
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        grads = [p.main_grad for name, p in wrapped.module.named_parameters()
                 if p.requires_grad and name.endswith(('.q_norm.weight', '.k_norm.weight', '.linear_attn.norm.weight'))]
        if not grads:return
        merged = _flatten_dense_tensors(grads)
        dist.all_reduce(merged, group=state().get_tensor_model_parallel_group())
        for grad, reduced in zip(grads, _unflatten_dense_tensors(merged, grads)):
            grad.copy_(reduced)


def mark_sharded(parameter, dim=0, stride=1, splits=None):
    parameter.tensor_model_parallel=True
    parameter.partition_dim=dim
    parameter.partition_stride=stride
    if splits is not None:parameter._checkpoint_splits=tuple(splits)
    return parameter


def checkpoint_shard(value, parameter):
    if tp_size()==1 or not getattr(parameter,'tensor_model_parallel',False):return value
    dim=parameter.partition_dim;rank=state().get_tensor_model_parallel_rank();size=tp_size()
    splits=getattr(parameter,'_checkpoint_splits',None)
    if splits is not None:
        return torch.cat([part.chunk(size,dim=dim)[rank] for part in value.split(splits,dim=dim)],dim=dim)
    stride=getattr(parameter,'partition_stride',1)
    chunks=value.split(value.shape[dim]//size//stride,dim=dim)
    return torch.cat(chunks[rank::size],dim=dim)
