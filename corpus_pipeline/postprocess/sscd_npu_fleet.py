"""Deadline-aware SSCD policy around the existing generic resource manager.

All resource ownership, create intents, live quota checks, placement,
consolidation and command lifecycle remain in resource_manager. This module
adds a bounded target, stage qualification gate, and stale-placement recovery.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

from corpus_pipeline.resource_manager.engine import Controller, atomic_json, summary,request_stop
from corpus_pipeline.resource_manager.inspire_adapter import InspireAdapter
from corpus_pipeline.resource_manager.models import WorkloadSpec
from corpus_pipeline.resource_manager import command_runner
from corpus_pipeline.postprocess.sscd_npu_pool import qualified,complete


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready_to_finalize(progress):
    return (type(progress.get('total_tasks')) is int and progress['total_tasks']>0
            and progress.get('completed_tasks')==progress['total_tasks']
            and type(progress.get('total_pairs')) is int and progress['total_pairs']>=0
            and progress.get('completed_pairs')==progress['total_pairs'])


def target_from_progress(samples,current,*,now,deadline,target,maximum,last_change=0):
    result={'target_cards':target,'reason':'observe','pairs_per_second':None,'run_eta_epoch':None}
    if not current or now-current.get('updated_at',0)>180:return dict(result,reason='stale_or_missing_business_progress')
    done=current.get('completed_pairs');total=current.get('total_pairs')
    if type(done) is not int or type(total) is not int or not 0<=done<=total:return dict(result,reason='invalid_business_counters')
    if done==total:return dict(result,reason='all_pairs_processed',run_eta_epoch=now)
    anchors=[v for v in samples if v.get('total_pairs')==total and type(v.get('completed_pairs')) is int
             and 0<=v['completed_pairs']<=done and 300<=now-v.get('observed_at',now+1)<=900]
    if not anchors:return dict(result,reason='need_five_minute_measured_rate')
    first=min(anchors,key=lambda v:v['observed_at']);rate=(done-first['completed_pairs'])/(now-first['observed_at'])
    if rate<=0:return dict(result,reason='no_progress_diagnose_before_expanding',pairs_per_second=rate)
    seconds=(total-done)/rate;result.update(pairs_per_second=rate,run_eta_epoch=now+seconds,remaining_seconds=seconds)
    if now-last_change<300:return dict(result,reason='post_scale_observation')
    budget=max(1,(deadline-now)*.7)
    if seconds<=budget:return dict(result,reason='measured_rate_fits_deadline')
    required=math.ceil(target*seconds/budget)
    new=min(maximum,target+16,max(target,required))
    return dict(result,target_cards=new,reason='deadline_requires_capacity' if new>target else 'maximum_capacity_reached',
                linear_scaling_assumption=True)


class PlacementRecoveringController(Controller):
    def __init__(self,*args,pending_timeout=300,**kwargs):
        super().__init__(*args,**kwargs);self.pending_timeout=pending_timeout
    def _observe(self,row):
        super()._observe(row)
        if (row.get('state')=='PENDING' and row.get('resource_id') and not row.get('stop')
                and self.clock()-row.get('accepted_at',self.clock())>=self.pending_timeout):
            row.update(retirement_reason='placement_wait_timeout',placement_retired_at=self.clock())
            self._register_replacement(row)
            self._stop_row(row)
    def _snapshot(self):
        value=super()._snapshot()
        # A raced node can be retried later, but not selected immediately again
        # while the same occupied placement is being retired.
        cooling={row['intent']['placement']['node'] for row in (self.state or {}).get('resources',{}).values()
                 if row.get('retirement_reason')=='placement_wait_timeout'
                 and self.clock()-row.get('placement_retired_at',0)<600}
        value['nodes']=[dict(row,eligible=False) if row['name'] in cooling else row for row in value['nodes']]
        return value


class Fleet:
    def __init__(self,config):
        self.cp=Path(config).absolute();self.c=read(self.cp);self.policy=self.c['sscd_policy']
        self.spec=WorkloadSpec(**self.c['workload']);self.output=Path(self.policy['output']);self.root=Path(self.c['state_dir'])
        self.maximum=self.policy['max_cards'];initial=self.policy['initial_cards']
        if type(self.maximum) is not int or type(initial) is not int or not 1<=initial<=self.maximum<=128:
            raise ValueError('Expected initial <= maximum <= 128 NPU cards')
        if self.c['adapter_config']['priority']!=3 or self.c['adapter_config']['backend']!='notebook':
            raise ValueError('This production fleet requires LOW priority Notebook allocation')
        self.state_path=self.root/'FLEET.json';self.stop=threading.Event()
        self.state=read(self.state_path) if self.state_path.exists() else {'target_cards':initial,'samples':[],'last_change':0}
        self.config_sha256=hashlib.sha256(self.cp.read_bytes()).hexdigest()
        if self.state.get('config_sha256',self.config_sha256)!=self.config_sha256:raise ValueError('Fleet policy binding changed')
        if not initial<=self.state['target_cards']<=self.maximum:raise ValueError('Persisted target escaped budget')
        self.state['config_sha256']=self.config_sha256
        self.adapter=InspireAdapter(self.c['adapter_config'])
        self.controller=PlacementRecoveringController(self.adapter,replace(self.spec,max_cards=self.state['target_cards']),self.root,
                                                      pending_timeout=self.policy.get('pending_timeout_seconds',300))
    def persist(self):atomic_json(self.state_path,self.state)
    def business_status(self):
        argv=self.policy['status_command']
        result=subprocess.run(argv,capture_output=True,text=True,timeout=120,cwd=self.c['adapter_config']['repository'])
        if result.returncode:raise RuntimeError('Business status command failed')
        value=json.loads(result.stdout)
        return value
    def finalize_tick(self):
        """One durable local argv supervisor; an uncertain launch is reconciled by ID."""
        runtime=Path(self.c['adapter_config']['repository'])
        python=self.c['adapter_config']['remote_python']
        argv=self.policy.get('finalize_command',[python,'-m','corpus_pipeline.postprocess.sscd_exact_npu',
                                                  'finalize','--output',str(self.output)])
        directory=self.root/'finalizer_runner';key='sscd-finalizer'
        binding={'config_sha256':self.config_sha256,'plan_sha256':sha(self.output/'PLAN.json'),
                 'argv':argv,'cwd':str(runtime),
                 'finalizer_code_sha256':sha(runtime/'corpus_pipeline/postprocess/sscd_exact_npu.py')}
        marker=directory/'INPUT.json'
        if marker.exists():
            if read(marker)!=binding:raise ValueError('Finalizer input or frozen runtime changed')
        else:atomic_json(marker,binding)
        record=self.state.get('finalizer')
        if record is None:
            record={'operation_id':uuid.uuid4().hex,'phase':'intent','at':time.time()}
            self.state['finalizer']=record;self.persist()
        health=command_runner.status(directory,key)
        if health.get('launch_operation_id') not in (None,record['operation_id']):
            raise ValueError('Finalizer operation identity changed')
        if health.get('completed'):
            record.update(phase='succeeded' if health.get('exit_code')==0 else 'await_finalizer_repair',health=health)
            if health.get('exit_code')!=0:raise RuntimeError('Finalizer failed; explicit repair required')
            if not complete(self.output):raise RuntimeError('Finalizer exited zero without valid full coverage receipt')
            atomic_json(self.output/'fleet'/'FINALIZER_SUCCESS.json',dict(binding,completed_at=time.time(),
                                                                       complete_sha256=sha(self.output/'COMPLETE.json')))
            self.persist();return True
        if health.get('running'):
            record.update(phase='running',health=health);self.persist();return False
        if health.get('process_gone'):
            record.update(phase='await_finalizer_repair',health=health);self.persist()
            raise RuntimeError('Finalizer process disappeared; refusing an ambiguous duplicate')
        if health.get('launch_operation_id'):
            record.update(phase='await_launch_reconciliation',health=health);self.persist();return False
        # The helper itself is frozen and its launch operation is idempotent.
        # Retries use the same operation ID, never a new speculative finalizer.
        command=[python,'-m','corpus_pipeline.resource_manager.command_runner','--root',str(directory),
                 '--resource-key',key,'launch','--operation-id',record['operation_id'],'--',*argv]
        result=subprocess.run(command,capture_output=True,text=True,timeout=30,cwd=runtime,
                              env=dict(os.environ,PYTHONPATH=str(runtime)))
        record.update(phase='submitted' if result.returncode==0 else 'await_launch_reconciliation',
                      launcher_returncode=result.returncode)
        self.persist();return False
    def run(self):
        if not qualified(self.output):raise ValueError('Full-run calibration is not qualified; no production resources started')
        signal.signal(signal.SIGTERM,lambda *_:self.stop.set());signal.signal(signal.SIGINT,lambda *_:self.stop.set())
        with self.controller.locked():
            while not self.stop.is_set():
                now=time.time()
                try:
                    if (self.output/'STOP').exists():
                        alarm={'at':now,'reason':'business_stop_present','stop_path':str(self.output/'STOP'),
                               'automatic_restart_allowed':False,'complete':False}
                        self.state.update(updated_at=now,phase='await_repair',alarm=alarm)
                        self.persist();atomic_json(self.root/'ALARM.json',alarm)
                        request_stop(self.root,scope=self.spec.scope,profile=self.spec.profile)
                        result=self.controller.poll(should_stop=self.stop.is_set)
                        self.state['resource_summary']=result;self.persist()
                        if result['phase']=='completed':return self.state
                        self.stop.wait(self.spec.poll_seconds);continue
                    progress=self.business_status();progress['observed_at']=now
                    decision=target_from_progress(self.state['samples'],progress,now=now,deadline=self.policy['deadline_epoch'],
                        target=self.state['target_cards'],maximum=self.maximum,last_change=self.state['last_change'])
                    if decision['target_cards']!=self.state['target_cards']:
                        self.state['target_cards']=decision['target_cards'];self.state['last_change']=now
                    self.state['samples']=[v for v in self.state['samples'] if now-v.get('observed_at',0)<=900]+[progress]
                    self.state.update(updated_at=now,decision=decision,business_progress=progress)
                    self.persist()
                    if ready_to_finalize(progress):
                        self.state['phase']='finalizing';self.persist()
                        if not self.finalize_tick():
                            # All tasks are done: do not request more NPUs or
                            # repeatedly launch idle pools during global hashes.
                            self.stop.wait(self.spec.poll_seconds);continue
                        self.state['phase']='releasing_after_verified_finalization';self.persist()
                        # Completion is already proved locally. Do not depend on
                        # an extant RUNNING notebook to execute a completion
                        # probe: all final workers may have been preempted while
                        # the CPU finalizer checked artifact hashes.
                        request_stop(self.root,scope=self.spec.scope,profile=self.spec.profile)
                    self.controller.spec=replace(self.spec,max_cards=self.state['target_cards'])
                    result=self.controller.poll(should_stop=self.stop.is_set)
                    self.state['resource_summary']=result;self.persist()
                    if result['phase']=='completed':return self.state
                except Exception as exc:
                    self.state.update(updated_at=now,alarm={'at':now,'error_type':type(exc).__name__})
                    self.persist()
                self.stop.wait(self.spec.poll_seconds)
            return self.state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plan','run','status'));p.add_argument('--config',required=True)
    a=p.parse_args();fleet=Fleet(a.config)
    value=fleet.controller.plan() if a.action=='plan' else fleet.run() if a.action=='run' else fleet.state
    print(json.dumps(value,ensure_ascii=False,sort_keys=True))
    return 75 if a.action=='run' and value.get('phase')=='await_repair' else 0


if __name__=='__main__':raise SystemExit(main())
