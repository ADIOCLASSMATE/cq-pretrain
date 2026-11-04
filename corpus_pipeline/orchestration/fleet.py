#!/usr/bin/env python3
"""Single-card LOW batch fleet using the durable shared resource manager."""
from __future__ import annotations
import argparse, hashlib, importlib, json, signal, threading
from pathlib import Path
from corpus_pipeline.resource_manager.engine import Controller, atomic_json
from corpus_pipeline.resource_manager.models import WorkloadSpec

POLICY={'schema':'genund.native_single_card_allocation.v1','quota':{'cards':1,'cpu':8,'memory':64},
        'priority':1,'max_cards':512,'max_creates_per_poll':8,'placement':'native_scheduler',
        'consolidation_enabled':False}

def remote_transport_unconfirmed(error):
    # This exact adapter error means authenticated command transport failed.
    # Ownership / schema / platform-state validation errors do not qualify.
    return ((type(error) is RuntimeError and str(error)=='Remote worker operation was not confirmed')
            or (type(error).__name__=='TransportError' and type(error).__module__=='inspire.platform.errors'))

class CompletionTransportFence:
    def __init__(self,inner,pending,owner):self.inner,self.pending,self.owner=inner,pending,owner
    def _shared(self,identifier):
        from corpus_pipeline.unified.worker_health import health
        rows=[x for x in (self.owner.state or {}).get('resources',{}).values() if x.get('resource_id')==identifier]
        if len(rows)!=1:return None
        return health(self.inner,rows[0])
    def health(self,identifier,readiness_command=None):
        if readiness_command is None:
            result=self._shared(identifier)
            if result is not None:return result
        return self.inner.health(identifier,readiness_command)
    def __getattr__(self,name):return getattr(self.inner,name)
    def completion(self,identifier,command):
        if identifier in self.pending or self._shared(identifier) is not None:return False
        try:return self.inner.completion(identifier,command)
        except Exception as error:
            if not remote_transport_unconfirmed(error):raise
            observed=self.inner.observe(identifier) # Revalidates full owned platform binding.
            if observed.get('state')!='RUNNING':raise
            self.pending.add(identifier)
            return False # No completion or worker-health claim.

class SingleCardController(Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.creates_this_poll=0;self.transport_pending_ids=set()
        if self.adapter is not None:self.adapter=CompletionTransportFence(self.adapter,self.transport_pending_ids,self)
    def _inputs(self,snapshot):
        nodes,quotas,reservations,remaining=super()._inputs(snapshot)
        quotas=[q for q in quotas if (q.cards,q.cpu,q.memory)==(1,8,64)]
        if not quotas:raise ValueError('Live catalog lacks the authorized 1,8,64 quota')
        if self.creates_this_poll>=POLICY['max_creates_per_poll']:quotas=[]
        return nodes,quotas,reservations,remaining
    def _create(self,placement,**kwargs):
        if (placement.quota.cards,placement.quota.cpu,placement.quota.memory)!=(1,8,64):
            raise ValueError('Only a real live 1,8,64 quota may be newly allocated')
        self.creates_this_poll+=1
        return super()._create(placement,**kwargs)
    def _observe(self,row):
        try:
            result=super()._observe(row)
        except Exception as error:
            if not remote_transport_unconfirmed(error) or not row.get('resource_id'):raise
            observed=self.adapter.observe(row['resource_id'])
            if observed.get('state')!='RUNNING':raise
            self.transport_pending_ids.add(row['resource_id'])
            row.update(state='RUNNING',ready=False,observed_at=self.clock(),
                worker_transport_pending=True,worker_transport_error_type=type(error).__name__,
                worker_transport_last_failure=self.clock())
            return
        row.pop('worker_transport_pending',None)
        self.transport_pending_ids.discard(row.get('resource_id'))
        return result
    def _migration(self,snapshot):return False
    def poll(self,**kwargs):
        self.creates_this_poll=0
        return super().poll(**kwargs)

def build(path):
    cfg=json.loads(Path(path).read_text());inner=cfg['adapter_config'].get('inner_config',cfg['adapter_config'])
    if inner.get('priority')!=1 or inner.get('notebook_placement')!='native_scheduler' or inner.get('backend')!='notebook':
        raise ValueError('Overlay is restricted to LOW native-scheduler Notebooks')
    spec=WorkloadSpec(**cfg['workload'])
    if spec.workload_kind!='batch' or tuple(spec.allowed_cards)!=(1,):
        raise ValueError('Unexpected workload or budget')
    module,factory=cfg['adapter'].split(':');adapter=getattr(importlib.import_module(module),factory)(cfg['adapter_config'])
    if adapter.capabilities.get('node_placement'):raise ValueError('Native scheduler must not claim physical placement')
    controller=SingleCardController(adapter,spec,cfg['state_dir'])
    proof={**POLICY,'max_cards':spec.max_cards,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'terminal_exec_sha256':hashlib.sha256((Path(__file__).parents[1]/'cluster/notebook_bound_exec.py').read_bytes()).hexdigest(),
        'shared_health_sha256':hashlib.sha256((Path(__file__).parents[1]/'unified/worker_health.py').read_bytes()).hexdigest()}
    p=controller.directory/'ALLOCATION_POLICY.json'
    if p.exists() and json.loads(p.read_text())!=proof:raise ValueError('Allocation overlay changed')
    if not p.exists():atomic_json(p,proof)
    return controller

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','run'));p.add_argument('--config',required=True);p.add_argument('--once',action='store_true');args=p.parse_args()
    c=build(args.config)
    if args.action=='plan':value=c.plan()
    else:
        stop=threading.Event()
        signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
        value=c.run(should_stop=stop.is_set,once=args.once)
    print(json.dumps(value,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
