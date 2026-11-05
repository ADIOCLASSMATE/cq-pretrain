"""Isolated, bounded 64-NPU tuning; never manages another platform job."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from utils.file_sync import atomic_json, wait_files

ROOT=Path(__file__).resolve().parent


def best(root,names,limit=55.5):
    candidates=[]
    for name in names:
        path=root/name/'summary.json'
        exits=list((root/name).glob('exit*.json'))
        if not path.exists() or len(exits)!=4 or any(json.loads(p.read_text())['returncode'] for p in exits):continue
        summary=json.loads(path.read_text())
        if summary['peak_memory_gib']>limit:continue
        score=summary['documents_per_second']
        candidates.append((score,name))
    if not candidates:raise RuntimeError(f'No successful infrastructure candidate within the {limit} GiB guard')
    return max(candidates)[1]


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True,type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());node=int(os.environ['PET_NODE_RANK'])
    if int(os.environ['PET_NNODES'])!=4:raise ValueError('This matrix requires 4 x 16 Ascend NPUs')
    root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        RAYON_NUM_THREADS='1',PRETRAIN_STACK='core17',PRETRAIN_MODULE='training.train',
        PRETRAIN_ATTENTION_TILE='2048',PRETRAIN_ATTENTION_LAYOUT='SBH',PRETRAIN_FUSED_CE='1',
        PRETRAIN_COMPILE_MODULATION='0',PRETRAIN_MOSS_FUSED_POINTWISE='situ',PRETRAIN_MC2_ROW='1',
        PYTHONDONTWRITEBYTECODE='1')
    outcomes=[]
    for stage in plan['stages']:
        # The owner may request a clean boundary between experiments. Never
        # interrupt an update or touch a different running platform job.
        if (root/'stop_after_current_stage').exists():break
        dependencies=stage.get('requires_success',[])
        unavailable=[name for name in dependencies if not any(
            row['name']==name and row.get('passed') for row in outcomes)]
        if unavailable:
            outcomes.append(dict(name=stage['name'],passed=False,skipped=True,requires=unavailable))
            if node==0:atomic_json(root/'progress.json',dict(completed=outcomes))
            continue
        folder=root/stage['name'];folder.mkdir(parents=True,exist_ok=True)
        if node==0:
            knobs=dict(plan['defaults'])
            if stage.get('select_from'):
                chosen=best(root,stage['select_from'],plan.get('memory_guard_gib',55.5))
                knobs.update(json.loads((root/chosen/'resolved.json').read_text())['knobs'])
            else:chosen=None
            knobs.update(stage.get('knobs',{}))
            fitted=None
            argv=list(stage.get('args',plan['common_args']))
            if stage.get('module'):knobs={}
            for name,value in knobs.items():
                option='--'+name.replace('_','-')
                if value is True:argv.append(option)
                elif value is not False and value is not None:
                    argv.append(option);argv.extend(map(str,value if isinstance(value,list) else [value]))
            argv+=stage.get('extra',[])
            stage_environment=dict(stage.get('environment',{}))
            if knobs.get('tp')==1:stage_environment['PRETRAIN_MC2_ROW']='0'
            atomic_json(folder/'resolved.json',dict(knobs=knobs,args=argv,selected_from=chosen,cost_fit=fitted,
                module=stage.get('module','training.train'),environment=stage_environment))
        wait_files([folder/'resolved.json'],600)
        resolved=json.loads((folder/'resolved.json').read_text())
        command=[sys.executable,str(ROOT/'launch_job.py'),'--rendezvous',str(folder/'rendezvous'),
            '--nodes','4','--nproc-per-node','16','--',*resolved['args'],'--output',str(folder)]
        stage_environment=dict(environment)
        stage_environment.update(resolved.get('environment',{}))
        stage_environment['PRETRAIN_MODULE']=resolved['module']
        with (folder/f'node{node}.log').open('w') as log:
            process=subprocess.Popen(command,env=stage_environment,
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+stage.get('timeout_seconds',5400)
            memory_rejection=None
            while process.poll() is None:
                peer_failed=any(json.loads(p.read_text())['returncode'] for p in folder.glob('exit*.json'))
                metrics=folder/'steps.jsonl'
                if metrics.exists():
                    complete=metrics.read_text().splitlines(keepends=True)
                    complete=[line for line in complete if line.endswith('\n')]
                    if complete:
                        last=json.loads(complete[-1]);limit=plan.get('memory_guard_gib',55.5)
                        if last['peak_memory_gib']>limit:
                            memory_rejection=dict(step=last['step'],peak_gib=last['peak_memory_gib'],limit_gib=limit)
                if peer_failed or memory_rejection or time.monotonic()>deadline:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                time.sleep(2)
        atomic_json(folder/f'exit{node}.json',dict(returncode=process.returncode,memory_rejection=memory_rejection))
        files=[folder/f'exit{i}.json' for i in range(4)];wait_files(files,600)
        passed=not any(json.loads(p.read_text())['returncode'] for p in files)
        memory_failure=not passed and (any(json.loads(p.read_text()).get('memory_rejection') for p in files) or any(
            'out of memory' in p.read_text(errors='replace').lower() for p in folder.glob('node*.log')))
        outcomes.append(dict(name=stage['name'],passed=passed,optional=stage.get('optional',False),
                             memory_failure=bool(memory_failure)))
        if node==0:atomic_json(root/'progress.json',dict(completed=outcomes))
        if not passed and not stage.get('optional',False):raise RuntimeError(f"Failed required stage {stage['name']}")
        if not passed and not memory_failure:
            raise RuntimeError(f"Unexpected failure in {stage['name']}; stop the matrix for diagnosis instead of repeating a software failure")
    atomic_json(root/f'complete{node}.json',dict(stages=outcomes,planned_stages=len(plan['stages']),
        stopped_at_boundary=(root/'stop_after_current_stage').exists()))


if __name__=='__main__':run()
