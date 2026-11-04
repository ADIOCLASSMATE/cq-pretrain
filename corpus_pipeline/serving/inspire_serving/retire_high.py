"""Planned release of exactly the two owned HIGH services after acceptance.

Dry-run by default; --execute is used only after the root confirms the frozen
BF16 baseline. Native LOW services retain their existing identities/queues.
"""
import argparse
import hashlib
import time
from inspire.sdk import InspireClient
from monitor import ROOT,RUNTIME,REGISTRY,load,save,assert_owned


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    baseline=RUNTIME/'BF16_BASELINE_READY.json'
    frozen=load(baseline)
    assert frozen['schema_version']=='corpus_pipeline.quantization_baseline.v1'
    assert len(frozen['cases'])>=5 and all(value>=1 for value in frozen['coverage'].values())
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='cbe0154e2414f4b3b1765bd3d85dcf56fe2467db323bda18a6de023403211720'
    production=ROOT.parents[1]/'production/runs/dcvlm_canary_v2_20261022'
    evidence_paths=[production/'COMPLETE.json',production/'final_reader_acceptance.json',
                    RUNTIME/'benchmark_16_32_20261023T020158/summary.json',baseline,RUNTIME/'BF16_BASELINE_READBACK.json']
    assert load(production/'COMPLETE.json')['canary_completed'] is True
    assert load(production/'final_reader_acceptance.json')['validation']['success'] is True
    assert all(x['passed'] for x in load(evidence_paths[2])['results'])
    evidence=[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in evidence_paths]
    states=[load(RUNTIME/f'{name}-high.json') for name in ('qwen','gemma')]
    lows=[load(RUNTIME/f'{name}-low.json') for name in ('qwen','gemma')]
    assert sum(s['cards'] for s in states)==36
    assert all(s['tier']=='high' and s['priority']==6 and not s.get('auto_recovery_enabled') for s in states)
    receipt={'purpose':'planned HIGH to LOW handoff after completed BF16 acceptance; not preemption',
             'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),'started_at':time.time(),
             'high_services':[],'low_services':[],'dry_run':not args.execute,'cards_to_release':36,'acceptance_evidence':evidence}
    with InspireClient(account='wjx-ascend') as client:
        for state in states+lows:
            detail=client.servings.get(state['name'],workspace='昇腾卡公共空间')
            assert_owned(state,detail)
            receipt['high_services' if state['tier']=='high' else 'low_services'].append({'id':detail.ref.key,'name':detail.name,'status_before':detail.status,'cards':state['cards']})
        save(RUNTIME/'planned-high-release.json',receipt)
        if not args.execute:
            print({'dry_run':True,'cards_to_release':36,'owned_high_ids':[s['service_id'] for s in states]})
            return
        save(RUNTIME/'HIGH_ACCEPTANCE_COMPLETE',{'time':time.time(),'meaning':'completed native BF16 pipeline acceptance; prefer qualified LOW','evidence':evidence})
        deadline=time.time()+30
        while time.time()<deadline:
            registry=load(REGISTRY)
            selected=[x for x in registry['endpoints'] if x['role']=='verifier']
            if len(selected)==1 and selected[0]['service_id']==lows[1]['service_id'] and selected[0]['ready'] and selected[0]['ready_expires_at']>time.time() and selected[0]['expected_boots']:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Qualified Gemma LOW was not selected; no HIGH service has been stopped')
        receipt['gemma_low_selected_before_stop']=selected[0];save(RUNTIME/'planned-high-release.json',receipt)
        for index,state in enumerate(states):
            detail=client.servings.get(state['name'],workspace='昇腾卡公共空间');assert_owned(state,detail)
            receipt['high_services'][index]['stop_intent_at']=time.time();save(RUNTIME/'planned-high-release.json',receipt)
            if detail.status!='STOPPED':
                client.servings.stop(detail.ref,workspace='昇腾卡公共空间')
            receipt['high_services'][index]['stop_submitted_at']=time.time();save(RUNTIME/'planned-high-release.json',receipt)
        deadline=time.time()+180
        while time.time()<deadline:
            stopped=True
            for index,state in enumerate(states):
                detail=client.servings.get(state['name'],workspace='昇腾卡公共空间');assert_owned(state,detail)
                instances=client.servings.instances(detail.ref,workspace='昇腾卡公共空间')
                row=receipt['high_services'][index]
                row.update(status_after=detail.status,instances_after=[vars(x) for x in instances])
                stopped=stopped and detail.status=='STOPPED' and all(x.status in ('INSTANCE_STOPPED','INSTANCE_FAILED','INSTANCE_SUCCEEDED','INSTANCE_DELETED','INSTANCE_TERMINATED','INSTANCE_COMPLETED') for x in instances)
            save(RUNTIME/'planned-high-release.json',receipt)
            if stopped:
                break
            time.sleep(2)
        assert stopped,'HIGH stop not fully observed; inspect exact owned services before any further action'
        for index,state in enumerate(lows):
            detail=client.servings.get(state['name'],workspace='昇腾卡公共空间');assert_owned(state,detail)
            receipt['low_services'][index]['status_after']=detail.status
        deadline=time.time()+20
        while time.time()<deadline and any(x['tier']=='high' and x['ready'] for x in load(REGISTRY)['endpoints']):
            time.sleep(1)
        assert not any(x['tier']=='high' and x['ready'] for x in load(REGISTRY)['endpoints']), 'Retired HIGH still published ready'
        receipt.update(finished_at=time.time(),high_stopped_confirmed=True,low_ids_unchanged=True,
                       registry_after=load(REGISTRY),qwen_waiting_window_possible=True)
        save(RUNTIME/'planned-high-release.json',receipt)
        print({'high_stopped_confirmed':True,'released_cards':36,'low_ids_unchanged':True})


if __name__=='__main__':
    main()
