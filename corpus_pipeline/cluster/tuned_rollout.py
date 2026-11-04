"""Upgrade accepted running LOW Notebook workers after real-data qualification.

Uses verified PID ownership to stop workers, then retries only their live token
leases. Does not stop Notebook resources, restart distributed Jobs or delete data.
All selected old workers must have verified exit evidence before any active
requeue or replacement launch. Failed runs retain phase and stop evidence;
existing output directories cannot be blindly reused for another rollout.
"""
import argparse,json,os,shlex,time
from pathlib import Path
from .submit_jobs import ACCOUNT,WORKSPACE,REPO
from .fragment_controller import load_submissions,merge_managed,Platform,validate_live
from .fragment_allocator import Allocator
from .notebook_exec import execute
from .text_worker import verify_control_host
from .tuned_configuration import configuration
from corpus_pipeline.production.common import atomic_json,file_sha


def requeue_stopped_leases(root, stopped, control_server_id):
    """Lease fencing wins every race; expired/superseded tokens are never failed."""
    from . import queue
    if stopped.get('all_original_processes_exited') is not True or stopped.get('root') != str(root):
        raise RuntimeError('Verified worker stop evidence is required before requeue')
    outcomes = []
    for lease in stopped['retry_candidates']:
        state = json.loads((root/'states'/(lease['task_id']+'.json')).read_text())
        row = {'task_id': lease['task_id'], 'token': lease['token']}
        if not (state['status'] == 'running' and state.get('token') == lease['token']
                and state.get('worker_id') == lease['worker_id']
                and state.get('lease_expires_at', 0) > time.time()):
            row['status'] = 'skipped_not_current_live_lease'
        else:
            if verify_control_host(root) != control_server_id:
                raise RuntimeError('Queue control ownership changed during rollout')
            try:
                updated = queue.fail(root, lease['task_id'], lease['token'],
                    'planned_tuned_rollout_after_verified_worker_exit', retryable=True)
                row['status'] = updated['status']
            except queue.LeaseLost:
                # A controller/worker may reclaim or complete after the read.
                # The atomic queue transition rejects the stale token safely.
                row['status'] = 'skipped_lease_fenced_or_expired'
        outcomes.append(row)
    return outcomes


def run(root,output):
    root=root.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    qualified=configuration(root)
    if qualified is None:raise RuntimeError('Tuned qualification required')
    control_server_id=verify_control_host(root)
    for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(k,None)
    from inspire.sdk import InspireClient
    execution=root/'fragment_notebooks/execution-r02'
    rows=[];stage='stopping_all_selected_workers'
    qualification_sha256=file_sha(root/'TUNED_QUALIFIED.json')
    def save():
        atomic_json(output/'ROLLOUT.json',dict(root=str(root),stage=stage,rows=rows,at=time.time(),
            qualification_sha256=qualification_sha256))
    def qualification_unchanged():
        if file_sha(root/'TUNED_QUALIFIED.json')!=qualification_sha256:
            raise RuntimeError('Qualification changed during rollout; inspect saved evidence before recovery')
    try:
        with InspireClient(account=ACCOUNT) as client:
            allocator=Allocator(client,root,execution)
            items=merge_managed(load_submissions(execution),allocator.load_managed())
            atomic_json(output/'PLAN.json',dict(root=str(root),items=items,
                qualification_sha256=qualification_sha256,created_at=time.time()))
            platform=Platform(client,items,allocator)
            save()
            # Phase 1: never actively requeue or launch while an old selected
            # worker could still claim a task requeued from an earlier Notebook.
            # The controller may independently start qualified new workers.
            for item in items:
                live=platform.observe(item);validate_live(item,live)
                if live['status']!='RUNNING':continue
                status=platform.worker(root,item,'status')
                if status['status']!='running':continue
                live=platform.observe(item);validate_live(item,live)
                if live['status']!='RUNNING':continue
                qualification_unchanged()
                row=dict(item=item,before=live['status'],worker_before=status,phase='stopping_worker');rows.append(row);save()
                command=shlex.join(['bash',str(REPO/'corpus_pipeline/cluster/npu_env.sh'),'-m','corpus_pipeline.cluster.worker_maintenance','--root',str(root),'--name',item['name'],'--output',str(output/item['name'])])
                receipt=execute(command,notebook=item['name'],notebook_id=item['notebook_id'],workspace_id=item['workspace_id'],timeout=140,max_output_bytes=32000)
                atomic_json(output/(item['name']+'-transport.json'),receipt)
                if not receipt.get('completed') or receipt.get('returncode')!=0:raise RuntimeError('Worker maintenance did not exit successfully: '+item['name'])
                stop_path=output/item['name']/'STOPPED.json'
                stopped=json.loads(stop_path.read_text())
                if stopped.get('all_original_processes_exited') is not True or stopped.get('name')!=item['name'] or stopped.get('root')!=str(root):raise RuntimeError('Worker stop evidence inconsistent')
                row.update(phase='old_workers_exit_verified',stop_receipt=str(stop_path),
                    stop_receipt_sha256=file_sha(stop_path),stopped=stopped)
                save()
            qualification_unchanged()
            # Durable barrier also makes a failure after the last stop
            # distinguishable from an incomplete/uncertain first phase.
            stage='all_selected_old_workers_exit_verified';save()
            atomic_json(output/'STOPS_VERIFIED.json',dict(root=str(root),rows=rows,
                qualification_sha256=qualification_sha256,verified_at=time.time()))
            # Phase 2: lease transitions stay fenced even when the controller
            # has already restarted workers or another worker completed a task.
            stage='requeue_and_launch';save()
            for row in rows:
                item=row['item'];qualification_unchanged()
                stop_path=Path(row['stop_receipt'])
                if file_sha(stop_path)!=row['stop_receipt_sha256']:
                    raise RuntimeError('Verified worker stop evidence changed before requeue')
                row['lease_outcomes']=requeue_stopped_leases(root,row['stopped'],control_server_id)
                row['requeued_tasks']=[entry['task_id'] for entry in row['lease_outcomes'] if entry['status']=='pending']
                row['phase']='leases_reconciled';save()
                if any(entry['status']=='failed' for entry in row['lease_outcomes']):
                    raise RuntimeError('Rollout encountered a task at its terminal retry limit; inspect lease_outcomes')
                again=platform.observe(item);validate_live(item,again)
                if again['status']=='RUNNING':
                    qualification_unchanged()
                    row['replacement']=platform.worker(root,item,'launch');row['phase']='replacement_launch_observed'
                else:row['phase']='preempted_during_rollout_controller_will_resume'
                save()
        stage='complete';save()
        atomic_json(output/'COMPLETE.json',dict(success=True,rows=rows,finished_at=time.time(),qualification_sha256=qualification_sha256))
    except BaseException as error:
        save()
        atomic_json(output/'FAILURE.json',dict(success=False,stage=stage,error_type=type(error).__name__,
            at=time.time(),qualification_sha256=qualification_sha256,
            recovery='Inspect PLAN, ROLLOUT, transport and STOPPED evidence; do not blindly repeat maintenance. '
                     'Only verified stopped workers permit fenced lease reconciliation.'))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.root,a.output)
