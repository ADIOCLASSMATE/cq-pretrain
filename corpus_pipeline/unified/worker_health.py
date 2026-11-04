"""Fresh worker-written pool heartbeats bound to the accepted launch operation."""
from pathlib import Path
import hashlib,json,time

def read(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None

def health(stage_adapter,row,*,now=None):
    if not getattr(stage_adapter,'contract',None):return None
    if row.get('state')!='RUNNING' or not row.get('launch') or not row.get('runtime_id'):return None
    contract=stage_adapter.contract;inner=stage_adapter.inner;intent=row['intent'];allocation=intent['operation_id']
    root=Path(contract['config_path']).parent
    pool_path=root/'fleet/pools'/contract['stage']/(allocation+'.json');pool=read(pool_path)
    if pool is None:return None
    now=time.time()if now is None else now
    if type(pool.get('updated_at')) not in (int,float) or not 0<=now-pool['updated_at']<=15:return None
    if pool.get('status')!='running' or pool.get('fatal') is not False:return None
    expected={'schema':'corpus.unified.worker_pool.v1','allocation':allocation,
        'cards':intent['placement']['quota']['cards'],'config_path':contract['config_path'],
        'config_sha256':contract['config_sha256'],'execution_sha256':hashlib.sha256((root/'EXECUTION.json').read_bytes()).hexdigest(),
        'stage':contract['stage']}
    if pool.get('binding')!=expected:raise ValueError('Shared pool binding differs from accepted workload')
    if pool.get('runtime_id')!=row['runtime_id'] or pool.get('host')!=row['runtime_id'].split(':')[0]:return None
    if pool.get('allocation')!=allocation or pool.get('cards')!=expected['cards'] or pool.get('stage')!=contract['stage']:
        raise ValueError('Shared pool identity differs')
    active=pool.get('active_children',{})
    if set(active)!={str(i)for i in range(expected['cards'])} or not all(type(x)is int and x>0 for x in active.values()):return None
    directory=Path(inner.config['worker_state_dir'])/row['resource_id'];current=read(directory/'CURRENT_LAUNCH.json')
    if current is None:return None
    operation_id=row['launch']['operation_id']
    if current.get('operation_id')!=operation_id:raise ValueError('Shared launch operation differs')
    operation=read(directory/'operations'/(operation_id+'.json'))
    if operation is None or operation.get('phase')!='running':return None
    if (operation.get('operation_id')!=operation_id or operation.get('kind')!='launch'
            or operation.get('command')!=inner._workload_command(intent)):
        raise ValueError('Shared launch argv or operation binding differs')
    if (operation.get('runtime_id')!=row['runtime_id'] or (operation.get('child')or{}).get('runtime_id')!=row['runtime_id']
            or (operation.get('child')or{}).get('pid')!=pool.get('pid')):return None
    row['health_evidence']={'kind':'fresh_bound_pool_heartbeat','observed_at':now,'heartbeat_at':pool['updated_at'],
        'allocation':allocation,'launch_operation_id':operation_id,'runtime_id':row['runtime_id']}
    return {'running':True,'ready':True,'completed':False,'exit_code':None,
        'launch_operation_id':operation_id,'runtime_id':row['runtime_id'],'process_gone':False,
        'drained':False,'drain_operation_id':None}
