"""FP32 renderer EMA on Megatron distributed-optimizer shards.

Updates require no collectives and never duplicate a full DiT on every DP rank.
Evaluation temporarily installs EMA master shards and uses the existing MOSS
parameter buckets to gather them. Online masters (including FP32 aliases into
model buffers) are restored in a finally block before training can continue.
"""
from contextlib import contextmanager
import math
import torch
import torch.distributed as dist

from .checkpoint import _optimizers
from . import parallel as P


class RendererEMA:
    schema='fp32_optimizer_shards_v1'

    @torch.no_grad()
    def __init__(self,model,optimizer,wrapped,decay=.9999):
        if not math.isfinite(decay) or not 0<=decay<1:
            raise ValueError('EMA decay must be finite and in [0, 1)')
        self.decay=float(decay);self.num_updates=0;self.active=False
        self.optimizer=optimizer;self.wrapped=wrapped
        names={p:name for name,p in model.named_parameters()
               if name.startswith('renderer.') and p.requires_grad}
        if not names:raise ValueError('Renderer EMA requires trainable renderer parameters')
        self.layout=[];self.masters=[];self.shadows=[];self.model_shards=[]
        for opt_index,opt in enumerate(_optimizers(optimizer)):
            for model_attr,master_attr in (
                ('model_float16_groups','shard_fp32_from_float16_groups'),
                ('model_fp32_groups','shard_fp32_groups')):
                if not hasattr(opt,model_attr) or not hasattr(opt,master_attr):
                    raise TypeError('EMA requires explicit Megatron FP32 optimizer shards')
                for params,masters in zip(getattr(opt,model_attr),getattr(opt,master_attr),strict=True):
                    for param,master in zip(params,masters,strict=True):
                        if param not in names:continue
                        if master is None or master.dtype!=torch.float32:
                            raise TypeError('EMA must accumulate FP32 master weights')
                        span=opt._get_model_param_range_map(param)['param']
                        if master.numel()!=span.end-span.start:raise ValueError('EMA shard range mismatch')
                        self.layout.append(dict(optimizer=opt_index,name=names[param],
                            start=span.start,end=span.end,model_shape=list(param.shape),
                            master_shape=list(master.shape)))
                        self.masters.append(master)
                        self.shadows.append(master.detach().clone())
                        self.model_shards.append(param.detach().view(-1)[span.start:span.end])
        self.groups=[group for group in wrapped.bucket_groups+wrapped.expert_parallel_bucket_groups
                     if any(param in names for param in group.params)]
        if not self.groups:raise ValueError('Renderer is absent from parameter buckets')
        # Every trainable renderer element must have exactly one DP owner.
        if dist.is_initialized():
            count=torch.tensor(sum(x.numel() for x in self.masters),dtype=torch.int64,
                               device=next(iter(names)).device)
            dist.all_reduce(count,group=P.state().get_data_parallel_group())
            if count.item()!=sum(p.numel() for p in names):
                raise ValueError('EMA optimizer shards do not cover the renderer exactly once')

    @property
    def local_bytes(self):return sum(x.numel()*x.element_size() for x in self.shadows)

    @torch.no_grad()
    def update(self):
        if self.active:raise RuntimeError('Cannot update EMA while EMA weights are installed')
        if self.shadows:torch._foreach_lerp_(self.shadows,self.masters,1.-self.decay)
        self.num_updates+=1

    def metadata(self):
        return dict(schema=self.schema,module='renderer',decay=self.decay,
                    num_updates=self.num_updates,dtype='float32')

    def state_dict(self):
        if self.active:raise RuntimeError('Checkpointing during an EMA swap is forbidden')
        return dict(**self.metadata(),layout=self.layout,shadows=self.shadows)

    @torch.no_grad()
    def load_state_dict(self,state):
        expected=dict(self.metadata());expected.pop('num_updates')
        if any(state.get(k)!=v for k,v in expected.items()) or state.get('layout')!=self.layout:
            raise ValueError('EMA decay, module, dtype or optimizer shard layout differs')
        if not isinstance(state.get('num_updates'),int) or state['num_updates']<0:
            raise ValueError('Invalid EMA update counter')
        values=state.get('shadows',[])
        if len(values)!=len(self.shadows):raise ValueError('EMA shard count differs')
        for value,target in zip(values,self.shadows,strict=True):
            if value.dtype!=torch.float32 or value.shape!=target.shape:
                raise ValueError('EMA tensor dtype/shape differs')
            target.copy_(value)
        self.num_updates=state['num_updates']

    @torch.no_grad()
    def _publish(self):
        for target,master in zip(self.model_shards,self.masters,strict=True):target.copy_(master)
        # The ordered-gather adapter forwards force_sync directly. All ranks
        # visit the same MOSS-containing bucket list, even with no local shards.
        for group in self.groups:group.start_param_sync(force_sync=True)

    @contextmanager
    def average_parameters(self):
        if self.active:raise RuntimeError('Nested EMA swaps are forbidden')
        # use_forward_hook is a configuration flag, not the current registration
        # state: Megatron leaves it true after disable_forward_pre_hook().
        if self.wrapped.remove_forward_pre_hook_handles:
            raise RuntimeError('Disable DDP forward pre-hooks before installing EMA')
        # Drain any pending online gathers before changing their source shards.
        self.wrapped.start_param_sync(force_sync=True)
        with torch.no_grad():online=[x.detach().clone() for x in self.masters]
        self.active=True
        try:
            with torch.no_grad():
                for master,shadow in zip(self.masters,self.shadows,strict=True):master.copy_(shadow)
                self._publish()
            yield self
        finally:
            with torch.no_grad():
                for master,value in zip(self.masters,online,strict=True):master.copy_(value)
                self._publish()
            self.active=False
