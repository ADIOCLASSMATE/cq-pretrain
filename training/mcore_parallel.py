"""CQ-style ordered MCore communication for optional UMM branches (TP=1).

Based on cq-pretrain/ascend_pretrain/input_pipeline/parallel.py. Missing final
microbatch gradients are readiness events, never writes to the gradient buffer.
"""

import types


class OrderedParameterGathers:
    def __init__(self, wrapped):
        self.groups = list(
            reversed(wrapped.bucket_groups + wrapped.expert_parallel_bucket_groups)
        )
        self.original = [group.start_param_sync for group in self.groups]
        for index, group in enumerate(self.groups):
            if group.ddp_config.overlap_param_gather:

                def start(this, force_sync=False, index=index):
                    return self.request(index, force_sync)

                group.start_param_sync = types.MethodType(start, group)

    def request(self, index, force_sync):
        if force_sync:
            return self.original[index](force_sync=True)
        for i in range(index + 1):
            if not self.groups[i].param_gather_dispatched:
                self.original[i](force_sync=False)

    def finish(self):
        for group in self.groups:
            if group.ddp_config.overlap_param_gather:
                group.finish_param_sync()


class OrderedGradientBuckets:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.groups = wrapped.bucket_groups + wrapped.expert_parallel_bucket_groups
        self.original = [group.start_grad_sync for group in self.groups]
        for index, group in enumerate(self.groups):

            def start(this, force_all_reduce=False, index=index):
                return self.request(index, force_all_reduce)

            group.start_grad_sync = types.MethodType(start, group)
        self.begin()

    def begin(self):
        self.ready, self.next = {}, 0

    def request(self, index, force):
        if index < self.next:
            return
        self.ready[index] = force
        while self.next in self.ready:
            index = self.next
            self.next += 1
            self.original[index](force_all_reduce=self.ready.pop(index))

    def finish_backward(self):
        """Called after the final backward, before waiting for reductions.

        An unused parameter can still hold gradients from earlier microbatches.
        Register only missing readiness events. An entirely unused window has a
        zero gradient and follows MCore's normal Adam momentum/decay semantics.
        Non-reentrant checkpointing is required, so AccumulateGrad fires once.
        """
        missing = 0
        for group in self.groups:
            if group.ddp_config.overlap_grad_reduce:
                for parameter in group.params:
                    count = group.per_param_grad_ready_counts.get(parameter, 0)
                    if count == 0:
                        group.register_grad_ready(
                            parameter, self.wrapped.force_all_reduce
                        )
                        missing += 1
                    elif count != 1:
                        raise RuntimeError(
                            "UMM MCore requires one AccumulateGrad event per parameter"
                        )
        # Queue every group before waiting, including MCore's golden-count batch.
        for index in range(len(self.groups)):
            self.request(index, self.wrapped.force_all_reduce)
        return missing


def install_functional_parameter_hooks(wrapped):
    """Protect Qwen's functional conv1d without changing backbone source.

    HF's force_accelerate_hooks only calls Accelerate offload hooks, not the
    torch forward hooks used by MCore. The GDN parent runs before any convolution
    weight read, including recomputation. qkv is already called as a module by
    the content path before the query path slices the same projection weight.
    """
    handles = []
    if not wrapped.ddp_config.overlap_param_gather:
        return handles
    for name, module in wrapped.module.named_modules():
        if name.endswith(".linear_attn") and hasattr(module, "conv1d"):
            parameters = tuple(module.conv1d.parameters(recurse=False))

            def before_gdn(module, args, parameters=parameters):
                for parameter in parameters:
                    group = wrapped.param_to_bucket_group.get(parameter)
                    if group is not None:
                        group.finish_param_sync()

            handles.append(module.register_forward_pre_hook(before_gdn))
    return handles
