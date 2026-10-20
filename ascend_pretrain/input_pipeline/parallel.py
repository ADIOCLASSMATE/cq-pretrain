"""MCore 0.17 adapter for variable image counts without disabling overlap.

Text-only batches in A have no renderer/patch gradients. Explicitly mark their
already-zero main_grad buffers ready; preserve accumulated gradients when GA>1.
Dispatch ready buckets in one fixed order on all DP ranks, even if different
branches finish in a different order. No fake images or full zero-gradient
autograd tensors are needed.
"""
import types


class OrderedParameterGathers:
    """Preserve all-gather order when a rank skips optional vision/text modules.

    MCore's forward hook may request a later bucket before an unused earlier
    bucket. Dispatch all intervening buckets first, retaining asynchronous
    overlap and MCore's normal wait/copy hooks.
    """
    def __init__(self, wrapped):
        self.groups = list(reversed(wrapped.bucket_groups + wrapped.expert_parallel_bucket_groups))
        self.original = [g.start_param_sync for g in self.groups]
        for index, group in enumerate(self.groups):
            if not group.ddp_config.overlap_param_gather:continue
            def start(this, force_sync=False, index=index):
                return self.request(index, force_sync)
            group.start_param_sync = types.MethodType(start, group)

    def request(self, index, force_sync):
        if force_sync:
            # Explicit checkpoint/optimizer synchronization gathers the newly
            # updated shards even if this step's forward already gathered the
            # previous weights. Its caller visits every group on every rank.
            return self.original[index](force_sync=True)
        for i in range(index+1):
            group=self.groups[i]
            if not group.param_gather_dispatched:
                self.original[i](force_sync=False)


class OrderedGradientBuckets:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.groups = wrapped.bucket_groups + wrapped.expert_parallel_bucket_groups
        self.original = [g.start_grad_sync for g in self.groups]
        for index, group in enumerate(self.groups):
            def start(this, force_all_reduce=False, index=index):
                return self.request(index, force_all_reduce)
            group.start_grad_sync = types.MethodType(start, group)
        self.begin()

    def begin(self):
        self.ready = {}; self.next = 0

    def request(self, index, force):
        if index < self.next: return
        self.ready[index] = force
        while self.next in self.ready:
            i = self.next
            self.next += 1
            self.original[i](force_all_reduce=self.ready.pop(i))

    def absent_images(self):
        for module in (self.wrapped.module.patch_embed, self.wrapped.module.renderer):
            for parameter in module.parameters():
                group = self.wrapped.param_to_bucket_group.get(parameter)
                if group is not None:
                    group.register_grad_ready(parameter, self.wrapped.force_all_reduce)

    def absent_text(self):
        module=getattr(self.wrapped.module,'lm_head',None)
        if module is not None:
            for parameter in module.parameters():
                group=self.wrapped.param_to_bucket_group.get(parameter)
                if group is not None:group.register_grad_ready(parameter,self.wrapped.force_all_reduce)

    def finish(self):
        # The first batch learns golden hook counts. MCore normally dispatches
        # these inside finish_grad_sync; queue every group before waiting so
        # variable branch order cannot create an inter-rank collective mismatch.
        for index, group in enumerate(self.groups):
            counts = group.per_param_grad_ready_counts
            if len(counts) != len(group.params):
                raise RuntimeError('A trainable parameter has no real or explicitly absent gradient')
            self.request(index, self.wrapped.force_all_reduce)


def finish_parameter_gathers(wrapped):
    """Complete the normal forward-order gather chain for optional last modules."""
    # MCore bucket groups are stored in reverse parameter/forward order.
    for group in reversed(wrapped.bucket_groups + wrapped.expert_parallel_bucket_groups):
        if group.ddp_config.overlap_param_gather:
            group.finish_param_sync()
