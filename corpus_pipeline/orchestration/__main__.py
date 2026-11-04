"""One supported entry for preparing, checking, running and observing new corpora."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from .config import SOURCES, check_config, environment, read
from .prepare import prepare, verify_runtime


def preflight(cfg):
    verify_runtime(cfg['runtime'])
    # The qualified review backend is bound to this account/project. Do not
    # accept a CPU/fleet scope which would silently send review elsewhere.
    if (cfg['account'],cfg['workspace'],cfg['platform']['project'],cfg['platform']['group']) != (
            'wjx-ascend','昇腾卡公共空间','多模态大模型新架构评测探索与scaling-law','910B资源'):
        raise ValueError('Configuration differs from the qualified review backend scope')
    missing=[]
    for name in ('python','sdk_python','frontpass_qualification','sscd_qualification','api_key_file'):
        if not Path(cfg[name]).is_file():missing.append(name)
    for name in ('baseline_index','baseline_reconciliation'):
        marker=Path(cfg[name])/'COMPLETE.json'
        if not marker.is_file() or read(marker).get('complete') is not True:missing.append(name)
    from .normalization import ingest_integrity_verified
    if not ingest_integrity_verified(cfg):missing.append('registered_source_plans')
    from .cpu import validate_workers
    # Validate roles without mutating lifecycle or retirement state.
    validate_workers(cfg)
    dependencies={name:importlib.metadata.version(name) for name in ('aiohttp','httpx','certifi','pyarrow','duckdb')}
    return {'ready':not missing,'missing':missing,'dependencies':dependencies,
            'max_cards':cfg['max_cards'],'priority':cfg['priority'],'cpu_priority':cfg['cpu_priority'],
            'frontpass_cards_per_instance':1,'review_cards_per_replica':2,
            'resources_started':False,'mount_validation':'worker validates actual dataset paths and byte bindings before processing'}


def status(cfg):
    run=Path(cfg['run']);sources={}
    for name in SOURCES:
        root=run/'sources'/name
        normalize=read(run/'normalized_sources'/name/'COMPLETE.json',{})
        review=read(root/'api_review/PROGRESS.json',{})
        pool=read(root/'review_pool/STATUS.json',{})
        failure=read(root/'api_review/HEALTH_FAILURE.json')
        if failure and failure.get('worker_instance')!=review.get('worker_instance'):failure=None
        sources[name]={'normalized':normalize.get('complete',False),
            'records':normalize.get('statistics',{}).get('samples',0),
            'actual_review_responses':review.get('requests',{}).get('actual_responses',0),
            'review_updated_at':review.get('updated_at_epoch'),
            'review_failure':failure,
            'reserved_cards':pool.get('slot_reserved_cards'),
            'complete':read(root/'orchestrator/COMPLETE.json',{}).get('complete',False)}
    from .cpu import release_complete
    return {'complete':release_complete(run),'sources':sources,
            'controller':read(run/'ops/stage_control/STATE.json',{}),
            'cpu':read(run/'ops/CPU_POOL_STATUS.json',{})}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','preflight','run','status','stop','register-sources'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.action=='prepare':
        value=prepare(args.config)
    else:
        cfg=check_config(args.config)
        if args.action=='register-sources':
            from .ingestion import register
            value=register(cfg)
        elif args.action=='preflight':value=preflight(cfg)
        elif args.action=='status':value=status(cfg)
        elif args.action=='stop':
            run=Path(cfg['run']);run.mkdir(parents=True,exist_ok=True)
            (run/'STOP').touch();(run/'ops/STOP_INGEST').touch()
            for name in SOURCES:
                root=run/'sources'/name
                if root.exists():(root/'STOP').touch()
            value={'stop_requested':True,'released':False,'note':'Owning controllers drain and verify release'}
        else:
            result=preflight(cfg)
            if not result['ready']:raise ValueError('Preflight failed: '+','.join(result['missing']))
            if not args.frozen:
                os.chdir(cfg['runtime'])
                os.execve(cfg['python'],[cfg['python'],'-m','corpus_pipeline.orchestration',
                    'run','--config',cfg['config_path'],'--frozen'],environment(cfg))
            if not Path(__file__).resolve().is_relative_to(Path(cfg['runtime']).resolve()):
                raise ValueError('Run must execute its own verified frozen runtime')
            from .controller import controller
            return controller(cfg)
    print(json.dumps(value,ensure_ascii=False,sort_keys=True))
    return 2 if args.action=='preflight' and not value['ready'] else 0


if __name__=='__main__':raise SystemExit(main())
