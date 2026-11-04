"""LOW native Serving recovery ladder; no node-placement fields or Job creation.

Only called after the monitor has persisted a fresh, explicit victim event.
A smaller precision gets independent image and production-quality gates.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import time
from inspire.sdk.models_serving import ServingCreateSpec

ROOT=Path(__file__).resolve().parent
WORKSPACE='昇腾卡公共空间'
PROJECT='多模态大模型新架构评测探索与scaling-law'
SKILL=Path('/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/.codex/skills/inspire-wjx-ascend/scripts')
sys.path.insert(0,str(SKILL))
from fast_job_control import node_fits


from corpus_pipeline.serving.account_budget import observable_project_budget


def capacity_choice(samples, current_cards):
    if not samples:
        return 'wait'
    newest=samples[-1]
    if newest['complete_free_nodes'] >= current_cards//16:
        return 'restart_same'
    if current_cards!=32 or newest['complete_free_nodes']!=1:
        return 'wait'
    if len(samples)<2 or newest['time']-samples[-2]['time']<15 or samples[-2]['complete_free_nodes']!=1:
        return 'wait'
    return 'w8a8_16'


def node_snapshot(client,state):
    from inspire.platform.web.browser_api.core import _request_json,_v2_result
    with client._transport.scope(timeout=20):
        rows={};total=None
        for page in range(1,41):
            data=_v2_result(_request_json(client._transport.session,'POST','/api/v2/workspace?Action=ListNodeDimension',
                body={'filter':{'workspace_id':state['payload']['workspace_id'],'logic_compute_group_id':state['payload']['logic_compute_group_id']},'PageNumber':page,'page_size':5000},timeout=10))
            items=data.get('node_dimensions'); total=data.get('total')
            if not isinstance(items,list) or total is None:
                raise ValueError('Incomplete native recovery node inventory')
            before=len(rows)
            for row in items:
                rows[row['name']]=row
            if len(rows)==int(total):
                break
            if len(rows)<=before:
                raise ValueError('Node pagination incomplete')
        else:
            raise ValueError('Node pagination budget exceeded')
    free=[row['name'] for row in rows.values() if node_fits(row,{'gpu_count':16,'cpu':128,'mem_gi':1024})]
    return {'time':time.time(),'complete_free_nodes':len(free),'free_nodes':free,'inventory_count':len(rows),
            'placement_control':'platform_only_no_exclusions_available'}


def prepare_spec(state):
    cfg=json.loads((ROOT/'qwen/config-w8a8.json').read_text())
    name='corpus-qwen-serving-low-w8a8-20261023-r02'
    deployment='qwen-serving-low-w8a8-20261023-r02'
    spec=ServingCreateSpec(name=name,model='corpus-qwen397b-w8a8-ref-20261023',model_version=1,
        command=f'env TZ=Asia/Shanghai CORPUS_DEPLOYMENT_ID={deployment} CORPUS_CONFIG={ROOT}/qwen/config-w8a8.json bash {ROOT}/qwen/launch.sh',
        port=8000,workspace='昇腾卡公共空间',project='多模态大模型新架构评测探索与scaling-law',group='910B资源',quota='16,128,1024',
        image='vllm-ascend:v0.23.0rc1',replicas=1,nodes_per_replica=1,shm_gib=128,priority=3,
        auto_scaling=False,public_path_readonly=True,description='Owned LOW16 W8A8 recovery after BF16 LOW victim preemption; independent image and production quality gates required')
    return spec,cfg,deployment


def recover_low(client,state,path,save,assert_owned):
    assert state['tier']=='low' and state['priority']==3 and state.get('recovery_pending'), 'No confirmed LOW victim recovery intent'
    assert state['recovery_pending']['service_id']==state['service_id']
    live=client.servings.get(state['name'],workspace='昇腾卡公共空间')
    assert_owned(state,live)
    transition=state.get('downsize_transition')
    uncertain=bool(transition and transition.get('phase')=='CREATE_INTENT')
    if live.status!='STOPPED' and not uncertain:
        state['recovery_phase']='PLATFORM_RESUMED_OR_NONTERMINAL_OBSERVE'
        state['recovery_pending']=None
        return False
    if state['role']!='generator':
        # Gemma already uses the smallest approved 4-card TP4 profile.
        save(ROOT/'runtime/recovery-observable-budget.json',observable_project_budget(client,4))
        client.servings.start(live.ref,workspace='昇腾卡公共空间')
        state.update(recovery_pending=None,recovery_phase='RESTARTED_SAME_LOW4',restart_confirmed_at=time.time())
        return False
    if transition and transition.get('phase')=='CREATE_INTENT':
        matches=[x for x in client.servings.list(workspace='昇腾卡公共空间',keyword=transition['name'],limit=100).items if x.name==transition['name']]
        if len(matches)!=1:
            state['recovery_phase']='CREATE_UNCERTAIN_FAIL_CLOSED'
            return False
        created=client.servings.get(matches[0].ref,workspace='昇腾卡公共空间')
        assert created.raw.get('user_id') and created.raw['user_id']==live.raw.get('user_id'), 'Candidate owner differs from owned original'
        proposed={**state,'name':transition['name'],'service_id':created.ref.key,'payload':transition['payload'],'expected_nodes':1,'cards':16}
        assert_owned(proposed,created)
        service_id=created.ref.key
        # Always resolve accepted-but-unacknowledged Creates before handling the
        # old service. If the platform revived the old one, retire only the
        # newly confirmed owned candidate; never leave an orphan allocation.
        old_again=client.servings.get(live.ref,workspace='昇腾卡公共空间')
        if old_again.status!='STOPPED':
            state['recovery_phase']='CONFLICT_OLD_RESUMED_RETIRE_OWNED_CANDIDATE'
            transition['service_id']=service_id
            save(path,state)
            if created.status!='STOPPED':
                client.servings.stop(created.ref,workspace='昇腾卡公共空间')
                return False
            transition['phase']='ROLLED_BACK_OLD_RESUMED'
            state['recovery_pending']=None
            return False
    else:
        samples=state.setdefault('capacity_samples',[])
        samples[:]=[x for x in samples if x['time']>=state['recovery_pending']['confirmed_at']]
        if samples and time.time()-samples[-1]['time']<15:
            state['recovery_phase']='WAIT_TWO_DISTINCT_CAPACITY_OBSERVATIONS'
            return False
        snapshot=node_snapshot(client,state)
        save(ROOT/f'runtime/recovery-nodes-{state["service_id"]}-{int(snapshot["time"])}.json',snapshot)
        samples.append(snapshot);del samples[:-2]
        choice=capacity_choice(samples,state['cards'])
        if choice=='wait':
            state['recovery_phase']='WAIT_COMPLETE_IDLE_NODE_CAPACITY'
            return False
        # Capacity observations are not placement promises. The scheduler still
        # chooses the actual node; no unsupported exclude_nodes is submitted.
        if choice=='restart_same':
            save(ROOT/'runtime/recovery-observable-budget.json',observable_project_budget(client,state['cards']))
            client.servings.start(live.ref,workspace='昇腾卡公共空间')
            state.update(recovery_pending=None,recovery_phase='RESTARTED_SAME_PRECISION',restart_confirmed_at=time.time())
            return False
        spec,cfg,deployment=prepare_spec(state)
        plan=client.servings.plan(spec)
        assert plan.priority==3 and plan.create_kwargs['node_num_per_replica']==1
        assert not any(k in plan.create_kwargs for k in ('exclude_nodes','specified_nodes'))
        if client.servings.get(live.ref,workspace='昇腾卡公共空间').status!='STOPPED':
            return False
        old_instances=client.servings.instances(live.ref,workspace='昇腾卡公共空间')
        if any(x.status not in ('INSTANCE_STOPPED','INSTANCE_FAILED','INSTANCE_SUCCEEDED','INSTANCE_DELETED','INSTANCE_TERMINATED','INSTANCE_COMPLETED') for x in old_instances):
            state['recovery_phase']='WAIT_OLD_INSTANCES_RELEASE'
            return False
        # Exact owned-ID budget is refreshed before every Create. Platform
        # project quota is an additional upper bound, not an exclusion feature.
        active_owned=0;budget=[]
        for source in (ROOT/'runtime').glob('*-*.json'):
            if source.name not in ('qwen-high.json','qwen-low.json','gemma-high.json','gemma-low.json'):
                continue
            own=json.loads(source.read_text());item=client.servings.get(own['name'],workspace='昇腾卡公共空间')
            assert_owned(own,item)
            active=item.status not in ('STOPPED','FAILED','DELETED')
            budget.append({'service_id':item.ref.key,'status':item.status,'cards':own['cards'],'active':active})
            active_owned+=own['cards'] if active else 0
        assert active_owned+16<=72, 'Native owned-card budget would be exceeded'
        observable=observable_project_budget(client,16)
        save(ROOT/'runtime/downsize-observable-budget.json',observable)
        save(ROOT/'runtime/downsize-budget-check.json',{'time':time.time(),'owned_services':budget,'proposed_cards':16,'new_owned_upper_bound':active_owned+16,'user_project_ceiling':256,'scope':'observable_exact_owned_services; platform separately enforces project quota'})
        transition={'phase':'CREATE_INTENT','name':spec.name,'operation_id':spec.name,'payload':plan.create_kwargs,
                    'old_service_id':state['service_id'],'cfg':cfg,'deployment_id':deployment,'time':time.time()}
        state['downsize_transition']=transition
        save(ROOT/'runtime/qwen-low-w8a8-plan.json',plan.to_dict())
        save(ROOT/'runtime/qwen-low-w8a8-payload.json',plan.create_kwargs)
        save(path,state)
        handle=client.servings.create(spec,operation_id=spec.name)
        service_id=handle.ref.key
    transition=state['downsize_transition'];cfg=transition['cfg']
    save(ROOT/f'runtime/retired-{state["service_id"]}.json',copy.deepcopy(state))
    history=state.get('owned_retired_services',[])+[state['service_id']]
    state.update(service_id=service_id,name=transition['name'],service_name=transition['name'],payload=transition['payload'],
        cards=16,expected_nodes=1,precision=cfg['precision'],model=cfg['model_name'],revision=cfg['revision'],processor_revision=cfg['processor_revision'],
        deployment_id=transition['deployment_id'],runtime_root=str(ROOT/'qwen/runtime'/cfg['epoch']),
        recovery_pending=None,recovery_phase='SMALLER_PRECISION_PENDING_QUALIFICATION',canary_passed=False,
        qualification_receipt=None,incarnation=None,probe_attempt=1,owned_retired_services=history,
        quality_gate_required=True,quality_gate_passed=False,was_running=False,last_running_at=None,
        downsize_transition={**transition,'phase':'CREATED','service_id':service_id})
    save(path,state)
    return True
