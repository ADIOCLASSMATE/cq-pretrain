"""Restartable multi-container SSCD coordinator and bounded CPU worker fleet.

The resource list is an explicit ownership ledger. Only those notebooks may be
restarted or released. Workers share global catalog/index plans, never separate
per-source similarity universes. Completion means candidate discovery only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

MODULE='corpus_pipeline.postprocess.sscd_cluster_runner'
CAT='corpus_pipeline.postprocess.sscd_parallel_catalog'
SEARCH='corpus_pipeline.postprocess.sscd_parallel_search'
SEARCH_POOL='corpus_pipeline.postprocess.sscd_search_pool'
INDEX='corpus_pipeline.postprocess.sscd_index'


def read(path): return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temp.open('x') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config(path):
    c=read(path)
    c.setdefault('search_processes',1)
    c.setdefault('search_threads',c['threads'])
    if (type(c['search_processes']) is not int or not 1<=c['search_processes']<=8
            or type(c['search_threads']) is not int or not 1<=c['search_threads']<=64
            or c['search_processes']*c['search_threads']>c['threads']):
        raise ValueError('Search process/thread allocation exceeds CPU thread budget')
    for k in ('root','output','python','runtime','sdk_python','input','adopt'):
        if not Path(c[k]).is_absolute(): raise ValueError('Absolute path required: '+k)
    names=c['notebooks']
    if len(names)!=len(set(names)) or not 1<=len(names)<=c['max_instances']<=32:
        raise ValueError('Invalid bounded fleet')
    ledger=read(Path(c['root'])/'resources'/'PLAN.json')
    owned={v['name'] for v in ledger['instances'] if v.get('owned')}
    if not set(names)<=owned: raise ValueError('Notebook is not owned by this run')
    for rel,h in c['runtime_sha256'].items():
        if sha(Path(c['runtime'])/rel)!=h: raise ValueError('Frozen runtime changed: '+rel)
    return c

def environment(c):
    e=dict(os.environ); e.update(PYTHONPATH=c['runtime'],OMP_NUM_THREADS=str(c['threads']),
        OPENBLAS_NUM_THREADS=str(c['threads']),MKL_NUM_THREADS=str(c['threads']),PYTHONUNBUFFERED='1')
    return e

def command(c,module,*args): return [c['python'],'-m',module,*map(str,args)]
def identity():
    return {'pid':os.getpid(),'host':socket.gethostname(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def phase_observation(phase, stats, now):
    """Counters are observational; a completed receipt remains the authority."""
    unit = 'queries' if phase == 'search' else 'vectors'
    completed, total = stats.get('completed_' + unit), stats.get('total_' + unit)
    if phase not in ('catalog', 'build', 'search') or type(completed) is not int or type(total) is not int:
        return None
    if not 0 <= completed <= total:
        raise ValueError('Invalid observed phase counters')
    return {'at': now, 'phase': phase, 'unit': unit, 'completed': completed, 'total': total}


def throughput_estimate(samples, phase, stats, now, deadline):
    current = phase_observation(phase, stats, now)
    result = {'deadline_overdue': now > deadline, 'deadline_is_stop_condition': False,
              'seconds_until_deadline': deadline - now, 'phase_eta_epoch': None,
              'phase_remaining_seconds': None, 'throughput_units_per_second': None,
              'eta_scope': 'current_phase_only; future training/calibration/finalization time is not estimated'}
    if current is None:
        return result
    remaining = current['total'] - current['completed']
    result.update(unit=current['unit'], phase_remaining=remaining,
                  required_speed_to_meet_deadline=remaining / (deadline - now) if now < deadline else None,
                  required_speed_to_finish_within_one_hour=remaining / 3600 if now >= deadline else None)
    # At least five minutes and two distinct checks; never infer from a startup burst.
    eligible = [x for x in samples if x.get('phase') == phase and x.get('total') == current['total']
                and 0 <= x.get('completed', -1) <= current['completed'] and 0 <= now - x.get('at', now + 1) <= 900]
    anchors = [x for x in eligible if now - x['at'] >= 300]
    if anchors and len({x['at'] for x in eligible} | {now}) >= 2:
        first = min(anchors, key=lambda x: x['at'])
        speed = (current['completed'] - first['completed']) / (now - first['at'])
        result.update(throughput_units_per_second=speed, measurement_seconds=now - first['at'],
                      measurement_observations=len(eligible) + 1)
        if speed > 0:
            seconds = remaining / speed
            result.update(phase_remaining_seconds=seconds, phase_eta_epoch=now + seconds,
                          current_phase_alone_misses_deadline=now + seconds > deadline)
    if remaining == 0:
        result.update(phase_remaining_seconds=0, phase_eta_epoch=now)
    return result

def spawn(c,cp,kind,worker=None):
    root=Path(c['root']); log=root/'controller'/('coordinator.log' if kind=='run' else worker+'.daemon.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    argv=command(c,MODULE,kind,'--config',cp)
    if worker: argv+=['--worker-id',worker]
    with log.open('ab',buffering=0) as f:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=f,
            cwd=c['runtime'],env=environment(c),start_new_session=True,close_fds=True)
    return {'submitted_pid':child.pid,'worker_id':worker,'log':str(log)}


def worker_command(c,stage,name,lock_fd):
    if stage=='catalog':
        return command(c,CAT,'worker','--output',c['output'],'--workers',c['io_threads'])
    if stage=='search':
        return command(c,SEARCH_POOL,'--output',c['output'],'--worker-id',name,
            '--processes',c['search_processes'],'--threads',c['search_threads'],'--inherited-lock-fd',lock_fd)
    return command(c,SEARCH,'build-worker','--output',c['output'],'--worker-id',name)


def worker(cp,name):
    c=config(cp); root=Path(c['root']); d=root/'controller'/'workers'; d.mkdir(parents=True,exist_ok=True)
    if name not in c['notebooks']+['coordinator-local']: raise ValueError('Unowned worker')
    with (d/(name+'.lock')).open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return
        current=None; process=None; errors={}; stop=False
        def stopping(*_):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,stopping); signal.signal(signal.SIGINT,stopping)
        while True:
            phase=read(root/'controller'/'PHASE.json') if (root/'controller'/'PHASE.json').exists() else {'phase':'waiting'}
            stage=phase['phase']; stopping_now=stop or (root/'controller'/'STOP').exists() or stage in ('complete','failed','stopped')
            if stopping_now and process and process.poll() is None:
                try: os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=60)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
            if process and process.poll() is not None:
                rc=process.returncode; process=None
                if rc and not stopping_now: errors[current]=errors.get(current,0)+1
                else: errors[current]=0
                save(d/(name+'.last_exit.json'),{'phase':current,'returncode':rc,'at':time.time()})
            fatal=any(v>=3 for v in errors.values())
            save(d/(name+'.json'),dict(identity(),worker_id=name,phase=current if process else stage,
                child_pid=process.pid if process else None,updated_at=time.time(),errors=errors,fatal=fatal,
                status='stopped' if stopping_now else 'failed' if fatal else 'running' if process else 'idle'))
            if stopping_now or fatal: return
            if process is None and stage in ('catalog','build','search'):
                current=stage
                argv=worker_command(c,stage,name,lock.fileno())
                with (d/(name+'.'+stage+'.log')).open('ab',buffering=0) as f:
                    process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=f,
                        cwd=c['runtime'],env=environment(c),start_new_session=True,pass_fds=(lock.fileno(),))
            time.sleep(10)


class Coordinator:
    def __init__(self,cp):
        self.cp=str(Path(cp).absolute()); self.c=config(cp); self.root=Path(self.c['root']); self.d=self.root/'controller'
        self.d.mkdir(parents=True,exist_ok=True); self.stop=False; self.phase='starting'; self.stats={}; self.child=None
        self.started=time.time(); self.lock=None; self.monitor_stop=threading.Event(); self.cleanup_done=False
        self.history=[]; self.monitor_errors={}; self.worker_warnings=[]; self.remaining_work={}
        history=self.d/'history.jsonl'
        if history.exists():
            # A crash may leave only the final append incomplete; old valid observations remain usable.
            for line in history.read_text().splitlines():
                try: self.history.append(json.loads(line))
                except json.JSONDecodeError: continue
    def observe(self):
        value=phase_observation(self.phase,self.stats,time.time())
        if value is None: return
        value['config_sha256']=sha(self.cp)
        stages=('catalog','build','search')
        for number,name in enumerate(stages):
            self.remaining_work[name]={'unit':'queries' if name=='search' else 'vectors',
                'remaining':0 if number<stages.index(self.phase) else
                            value['total']-value['completed'] if name==self.phase else value['total'],
                'basis':'current_catalog_population_estimate' if number>stages.index(self.phase) else 'observed_stage_counter'}
        self.history.append(value)
        with (self.d/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(value,sort_keys=True)+'\n'); stream.flush(); os.fsync(stream.fileno())
        self.history=[x for x in self.history if value['at']-x.get('at',0)<=3600]
    def publish(self,phase=None,**kw):
        if phase is not None:
            self.phase=phase; save(self.d/'PHASE.json',{'phase':phase,'at':time.time(),'config_sha256':sha(self.cp)})
        now=time.time()
        forecast=throughput_estimate(self.history,self.phase,self.stats,now,self.c['deadline_epoch'])
        save(self.d/'PROGRESS.json',dict(identity(),phase=self.phase,updated_at=now,started_at=self.started,
            deadline_epoch=self.c['deadline_epoch'],seconds_until_deadline=self.c['deadline_epoch']-time.time(),
            max_cpu_instances=self.c['max_instances'],configured_cpu_instances=len(self.c['notebooks']),
            stats=self.stats,forecast=forecast,monitor_errors=dict(self.monitor_errors),worker_warnings=self.worker_warnings,
            run_remaining_work=self.remaining_work,
            run_eta_epoch=forecast['phase_eta_epoch'] if self.phase=='search' else None,
            run_eta_excludes_finalization=True,
            candidate_edges_only=True,automatic_merge=False,training_records_deleted=0,**kw))
    def check(self):
        if self.stop or (self.d/'STOP').exists(): raise InterruptedError('Stop requested')
        self.worker_warnings=[]
        if (self.d/'workers').exists():
            for p in (self.d/'workers').glob('*.json'):
                value=read(p)
                if value.get('fatal'):
                    self.worker_warnings.append({'worker':p.name,'reason':'await_worker_repair; retries exhausted, automatic relaunch disabled'})
                elif value.get('worker_id') and time.time()-value.get('updated_at',0)>300:
                    self.worker_warnings.append({'worker':p.name,'reason':'heartbeat_stale; surviving stage child may still hold its lock'})
    def run_command(self,label,module,*args):
        self.publish(label); log=self.d/(label+'.log')
        with log.open('ab',buffering=0) as f:
            self.child=subprocess.Popen(command(self.c,module,*args),stdin=subprocess.DEVNULL,stdout=f,stderr=f,
                cwd=self.c['runtime'],env=environment(self.c),start_new_session=True,pass_fds=(self.lock.fileno(),))
            while self.child.poll() is None:
                self.check(); self.publish(child_pid=self.child.pid,log=str(log)); time.sleep(10)
            rc=self.child.returncode; self.child=None
        if rc: raise RuntimeError(label+' exited '+str(rc)+'; '+str(log))
    def remote(self,name):
        if self.monitor_stop.is_set(): return
        c=self.c; hb=self.d/'workers'/(name+'.json')
        if hb.exists():
            value=read(hb)
            # A new worker process resets its in-memory retry counter. Relaunching
            # fatal workers here would turn the three-attempt bound into an
            # unbounded loop over the same deterministic failure.
            if value.get('fatal'): return
            if time.time()-value['updated_at']<90 and not value.get('fatal') and value.get('status') not in ('failed','stopped'): return
        argv=command(c,MODULE,'launch-worker','--config',self.cp,'--worker-id',name)
        shell='cd '+shlex.quote(c['runtime'])+' && PYTHONPATH='+shlex.quote(c['runtime'])+' '+shlex.join(argv)
        call=[c['sdk_python'],'-m','corpus_pipeline.cluster.notebook_exec','--notebook',name,'--account',c['account'],
            '--workspace',c['workspace'],'--timeout','45','--max-output-bytes','8192','--command',shell]
        try:
            result=subprocess.run(call,capture_output=True,text=True,timeout=100,cwd=c['runtime'],env=environment(c))
            receipt=json.loads(result.stdout)
        except Exception as exc: receipt={'error_type':type(exc).__name__,'completed':False}
        # The authenticated SDK strips connection details from this receipt.
        save(self.d/'connections'/(name+'.json'),dict(receipt,observed_at=time.time()))
        if not self.monitor_stop.is_set() and receipt.get('notebook_status') in ('STOPPED','FAILED'):
            cmd=['inspire','--json','notebook','start',name,'--account',c['account'],'--workspace',c['workspace'],'--no-wait']
            try:
                v=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
                save(self.d/'connections'/(name+'.restart.json'),{'at':time.time(),'returncode':v.returncode})
            except subprocess.TimeoutExpired: pass
    def monitor(self):
        def guarded_remote(name):
            if self.monitor_stop.is_set(): return
            try:
                self.remote(name)
                self.monitor_errors.pop(name,None)
            except Exception as exc:
                # Recovery telemetry must not terminate an otherwise productive stage.
                self.monitor_errors[name]={'at':time.time(),'error_type':type(exc).__name__,'error':str(exc)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            while not self.monitor_stop.is_set():
                try:
                    list(pool.map(guarded_remote,self.c['notebooks']))
                    hb=self.d/'workers'/'coordinator-local.json'
                    value=read(hb) if hb.exists() else {}
                    if not self.monitor_stop.is_set() and not value.get('fatal') and (not value or time.time()-value['updated_at']>90):
                        spawn(self.c,self.cp,'worker','coordinator-local')
                except Exception as exc:
                    self.monitor_errors['monitor']={'at':time.time(),'error_type':type(exc).__name__,'error':str(exc)}
                self.monitor_stop.wait(90)
    def parallel(self,phase):
        self.publish(phase)
        if phase=='catalog':
            from corpus_pipeline.postprocess.sscd_parallel_catalog import status
            get=lambda:status(Path(self.c['output']))
        else:
            from corpus_pipeline.postprocess.sscd_parallel_search import status
            get=lambda:status(Path(self.c['output']),phase)
        while True:
            self.check(); self.stats=get(); self.observe(); self.publish()
            if self.stats.get('complete') or self.stats.get('ready_to_finalize'): return
            time.sleep(15)
    def release(self):
        # Resource ownership was checked against the creation ledger in config().
        for name in self.c['notebooks']:
            try:
                v=subprocess.run(['inspire','--json','notebook','stop',name,'--account',self.c['account'],'--workspace',self.c['workspace']],capture_output=True,text=True,timeout=90)
                save(self.d/'released'/(name+'.json'),{'at':time.time(),'returncode':v.returncode})
            except subprocess.TimeoutExpired: save(self.d/'released'/(name+'.json'),{'at':time.time(),'uncertain':True})
        self.cleanup_done=True
    def run(self):
        if self.c.get('coordinator_host') and self.c['coordinator_host']!=socket.gethostname():
            raise ValueError('Coordinator may only run on its configured host')
        with (self.d/'COORDINATOR.lock').open('a') as lock:
            self.lock=lock
            # Cross-host lockf was qualified on this shared filesystem. Unlike
            # worker flock, this lock is owned only by the coordinator process.
            try: fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: return
            def stopping(*_): self.stop=True
            signal.signal(signal.SIGTERM,stopping); signal.signal(signal.SIGINT,stopping)
            bound=self.d/'INPUT.json'; binding={'config_sha256':sha(self.cp),'config':self.c}
            if bound.exists() and read(bound)!=binding: raise ValueError('Coordinator config changed')
            if not bound.exists(): save(bound,binding)
            monitor=threading.Thread(target=self.monitor,daemon=True); monitor.start()
            c=self.c; out=c['output']
            try:
                if not (Path(out)/'catalog'/'COMPLETE.json').exists():
                    self.run_command('prepare_catalog',CAT,'prepare','--input',c['input'],'--output',out,'--adopt',c['adopt'])
                    self.parallel('catalog'); self.run_command('finalize_catalog',CAT,'finalize','--output',out)
                if not (Path(out)/'train'/'COMPLETE.json').exists():
                    self.run_command('train',INDEX,'train','--output',out,'--threads',c['threads'],'--nlist',c['nlist'],
                        '--pq-m',c['pq_m'],'--pq-bits',c['pq_bits'],'--coarse-hnsw',c['coarse_hnsw'],
                        '--niter',c['niter'],'--train-count',c['train_count'])
                if not (Path(out)/'shards'/'COMPLETE.json').exists():
                    self.run_command('prepare_build',SEARCH,'prepare-build','--output',out,'--threads',c['threads'])
                    self.parallel('build'); self.run_command('finalize_build',SEARCH,'finalize-build','--output',out)
                if not (Path(out)/'merged'/'COMPLETE.json').exists():
                    self.run_command('merge',INDEX,'merge','--output',out,'--checkpoint-shards',c['checkpoint_shards'])
                calibration=None
                for trial in c['calibration_candidates']:
                    self.run_command('calibrate_'+str(trial['nprobe'])+'_'+str(trial['candidate_k']),SEARCH,'calibrate','--output',out,
                        '--threshold',c['threshold'],'--query-count',c['query_count'],'--top-k',c['top_k'],
                        '--min-recall',c['min_recall'],'--min-positive-recall',c['min_positive_recall'],
                        '--threads',c['threads'],'--nprobe',trial['nprobe'],'--candidate-k',trial['candidate_k'],'--ef-search',trial['ef_search'])
                    latest=read(Path(out)/'calibration'/'LATEST.json'); p=Path(out)/latest['receipt_path']
                    if sha(p)!=latest['receipt_sha256']: raise ValueError('Calibration pointer hash changed')
                    if read(p).get('qualified_for_full_run'): calibration=p; break
                if calibration is None: raise RuntimeError('Full catalog recall gate not met; no unqualified search released')
                self.run_command('prepare_search',SEARCH,'prepare-search','--output',out,'--calibration',calibration,
                    '--queries-per-unit',c['queries_per_unit'],'--batch-size',c['search_batch_size'],'--threads',c['search_threads'])
                self.parallel('search'); self.run_command('finalize_search',SEARCH,'finalize-search','--output',out)
                self.publish('complete',full_query_completed=True)
                save(self.d/'COMPLETE.json',read(self.d/'PROGRESS.json'))
            except Exception as exc:
                if self.child and self.child.poll() is None:
                    os.killpg(self.child.pid,signal.SIGTERM)
                    try: self.child.wait(timeout=60)
                    except subprocess.TimeoutExpired: os.killpg(self.child.pid,signal.SIGKILL); self.child.wait()
                self.publish('stopped' if isinstance(exc,InterruptedError) else 'failed',error_type=type(exc).__name__,error=str(exc))
                raise
            finally:
                self.monitor_stop.set(); monitor.join(timeout=300)
            if self.phase=='complete': self.release()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=('launch','run','launch-worker','worker','status'))
    p.add_argument('--config',required=True); p.add_argument('--worker-id'); a=p.parse_args(); c=config(a.config)
    if a.action=='launch': result=spawn(c,str(Path(a.config).absolute()),'run')
    elif a.action=='launch-worker': result=spawn(c,str(Path(a.config).absolute()),'worker',a.worker_id)
    elif a.action=='worker': worker(a.config,a.worker_id); result={'exited':True}
    elif a.action=='run': Coordinator(a.config).run(); result={'exited':True}
    else: result=read(Path(c['root'])/'controller'/'PROGRESS.json')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
