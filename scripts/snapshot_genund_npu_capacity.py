"""Read-only live capacity and owned one-card placement snapshot."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corpus_pipeline.resource_manager.inspire_adapter import InspireAdapter
from inspire.platform.web.browser_api.availability.api import list_node_dimension
from inspire.platform.web.browser_api.notebooks import get_notebook_detail,list_notebook_events


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run);start=time.time();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=run/'ops/capacity_snapshots'/stamp;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((run/'sources/BLIP3o-Pretrain-Long-Caption/fleet/frontpass.json').read_text())
    config=cfg['adapter_config']['inner_config'];a=InspireAdapter(config)
    image,quotas=a._catalog();quota=next(q for q in quotas if q=={'cards':1,'cpu':8,'memory':64})
    plan=a.service.plan(a._spec({'name':'genund-capacity-readonly-plan','placement':{'node':None,'quota':quota},'command':['true']},image))
    payload=plan.create_kwargs
    with a.client._transport.scope(timeout=30):
        rows=list_node_dimension(payload['logic_compute_group_id'],workspace_id=payload['workspace_id'],session=a.client._transport.session)
    excluded=set(config.get('excluded_nodes',[]));nodes=[]
    for row in rows:
        kind=row.get('gpu_type')or(row.get('gpu_info')or{}).get('gpu_type')
        eligible=row.get('status')=='Ready' and row.get('cordon_type')in('',None) and row['name']not in excluded and kind==config['gpu_type']
        values={k:max(0,float(row.get(k,{}).get('available',0)))for k in ('gpu','cpu','memory')}
        slots=min(math.floor(values['gpu']),math.floor(values['cpu']/8),math.floor(values['memory']/64)) if eligible else 0
        nodes.append({'name':row['name'],'status':row.get('status'),'cordon_type':row.get('cordon_type'),
                      'gpu_type':kind,'eligible':eligible,'free':values,'one_card_slots':slots})
    statepath=Path(cfg['state_dir'])/'STATE.json';statebytes=statepath.read_bytes();state=json.loads(statebytes)
    owned=[r for r in state['resources'].values()if r.get('resource_id')and r['intent']['placement']['quota']['cards']==1]
    def observe(row):
        with a.client._transport.scope(timeout=30):
            detail=get_notebook_detail(row['resource_id'],session=a.client._transport.session)
            events=list_notebook_events(row['resource_id'],session=a.client._transport.session)if detail['status']in('PENDING','CREATING','RUNNING')else []
        if detail['name']!=row['intent']['name']:raise ValueError('owned notebook identity differs')
        price=detail.get('resource_spec_price')or{};startcfg=detail.get('start_config')or{}
        return {'name':detail['name'],'resource_id':row['resource_id'],'status':detail['status'],
                'created_at':detail.get('created_at'),'updated_at':detail.get('updated_at'),
                'node':(detail.get('node')or{}).get('name'),'gpu_count':price.get('gpu_count'),
                'cpu_count':price.get('cpu_count'),'memory_gib':price.get('memory_size_gib'),
                'submitted_profile_priority':config['priority'],'live_project_priority':startcfg.get('project_priority'),
                'native_node_binding':startcfg.get('node_id'),
                'latest_events':[{'message':e.get('message')or e.get('content'),'time_ms':e.get('last_timestamp')or e.get('created_at')}for e in events[-5:]]}
    # The official SDK binds each Client to its creating thread.
    books=[observe(row)for row in owned]
    env={k:v for k,v in os.environ.items()if k.lower()not in{'https_proxy','http_proxy','all_proxy'}}
    command=['/root/.local/share/uv/tools/inspire-skill/bin/inspire','--json','resources','availability','--account',config['account'],'--workspace',config['workspace'],'--group',config['group'],'--all']
    cli=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60,check=True)
    availability=json.loads(cli.stdout)
    result={'observed_start':start,'observed_end':time.time(),'time_utc':stamp,'platform_mutations':False,
            'scope':{'account':config['account'],'workspace':config['workspace'],'group':config['group'],'project':config['project']},
            'quota':quota,'excluded_nodes':sorted(excluded),'node_count':len(nodes),
            'eligible_node_count':sum(n['eligible']for n in nodes),
            'eligible_free_gpu_cards':sum(n['free']['gpu']for n in nodes if n['eligible']),
            'immediate_one_card_slots_with_cpu_memory':sum(n['one_card_slots']for n in nodes),
            'nodes_with_one_card_slot':sum(n['one_card_slots']>0 for n in nodes),
            'owned_one_card_counts':dict(Counter(b['status']for b in books)),
            'owned_running_nodes':dict(Counter(b['node']for b in books if b['status']=='RUNNING')),
            'state_sha256':hashlib.sha256(statebytes).hexdigest(),'state_changed_during_snapshot':statepath.read_bytes()!=statebytes,
            'availability':availability,'nodes':nodes,'owned_one_card_notebooks':books,
            'limitation':'Resource views are sequential snapshots; scheduler reservations, affinity and other workloads can reduce actually schedulable slots. The owned set is captured from the run ledger before per-instance reads.'}
    (out/'SNAPSHOT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'path':str(out/'SNAPSHOT.json'),**{k:result[k]for k in('observed_start','observed_end','eligible_node_count','eligible_free_gpu_cards','immediate_one_card_slots_with_cpu_memory','nodes_with_one_card_slot','owned_one_card_counts','owned_running_nodes','state_changed_during_snapshot')}},ensure_ascii=False))


if __name__=='__main__':main()
