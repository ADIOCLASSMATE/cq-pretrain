"""Versioned execution configuration; secrets and platform accounts stay external."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from . import SCHEMA

STAGES = ('normalize','frontpass','reconcile','screening','sscd_catalog','sscd_search',
          'assets','graph','records','source_view')
DEPENDENCIES = {
    'normalize':(), 'frontpass':('normalize',), 'reconcile':('frontpass',),
    'screening':('reconcile',), 'sscd_catalog':('reconcile',),
    'sscd_search':('sscd_catalog',), 'assets':('reconcile','screening'),
    'graph':('sscd_search','assets'), 'records':('frontpass','assets','graph'),
    'source_view':('records',),
}


def require(value, message):
    if not value: raise ValueError(message)


def collection_scope(value):
    require(isinstance(value,list) and bool(value) and
            all(isinstance(name,str) and name in {'dcvlm','genund'} for name in value),
            'required_collections must be a nonempty list of supported collections')
    require(len(set(value))==len(value),'required_collections cannot contain duplicates')
    return sorted(value)


def checked(value):
    value=deepcopy(value)
    require(value.get('schema')==SCHEMA,'Unsupported joint pipeline configuration')
    require(value.get('scope') in {'full','validation'},'Explicit full/validation scope required')
    require(isinstance(value.get('inputs'),list) and value['inputs'],'Explicit inputs required')
    require(isinstance(value.get('output'),str) and Path(value['output']).is_absolute(),'Absolute output required')
    sources=value['inputs'];out=Path(value['output']).resolve()
    for item in sources:
        require(item.get('collection') in {'dcvlm','genund'},'Only declared DCVLM/gen-und collections supported')
        require(item.get('adapter') in {'dcvlm_native','genund_native','dcvlm_balanced_wds','dcvlm_baseline_wds','genund_caption_tar'},'Explicit source adapter required')
        require(isinstance(item.get('root'),str) and Path(item['root']).is_absolute(),'Absolute dataset root required')
        if item['adapter']=='genund_native':
            require(item['collection']=='genund','genund_native requires collection=genund')
        if item['adapter']=='dcvlm_baseline_wds':
            require(item['collection']=='dcvlm' and len(sources)==1,
                    'Baseline manifest ingest currently requires a single DCVLM input')
            require(isinstance(item.get('download_control'),str) and Path(item['download_control']).is_absolute(),
                    'Baseline requires an absolute verified download control directory')
            require(value['scope']=='full' and not item.get('paths') and not item.get('source') and not item.get('max_records'),
                    'Baseline manifest ingest requires the full upstream inventory')
            if item.get('index_output') is not None:
                index=Path(item['index_output'])
                require(index.is_absolute() and index.resolve().is_relative_to(out)
                        and index.resolve()!=out and index.resolve()!=out/'native',
                        'Separate baseline index must be inside this run, outside the compact native snapshot')
        source=Path(item['root']).resolve()
        require(not out.is_relative_to(source) and not source.is_relative_to(out),'Output must not overlap original dataset')
    actual_collections={x['collection'] for x in sources}
    default_collections=['dcvlm','genund'] if value['scope']=='full' else sorted(actual_collections)
    value['required_collections']=collection_scope(value.get('required_collections',default_collections))
    require(actual_collections==set(value['required_collections']),
            'Input collections differ from required_collections; full defaults to both dcvlm and genund')
    if value['scope']=='full':
        require(not value.get('max_records'),'Full run cannot impose record sampling')
        require(all(not x.get('max_records') for x in sources),'Full input cannot impose record sampling')
        require(all(not x.get('paths') and not x.get('source') for x in sources),
                'Full collection enumeration cannot select only some shards or sources')
    value.setdefault('policy',{})
    policy=value['policy']
    task_aware=policy.get('profile')=='task_aware_v1'
    if 'profile' in policy:
        require(task_aware,'Unsupported policy profile')
    defaults={'minimum_short_edge':128,'uncertain_visual_review':'retain_annotated',
        'ai_generated_images':'exclude','ai_edits_from_verified_real_parent':'retain_annotated',
        'ai_generated_text':'retain_annotated','sscd_automatic_delete':False,
        'recaption':'disabled','layout_generation':'disabled','potential_min_pixels':1048576,
        'preserve_native_supervision':True}
    if task_aware:
        defaults.update(profile='task_aware_v1',ai_generated_images='exclude_t2i_only',
                        i2t_image_loss=False,i2t_screening='dedup_only',
                        preserve_all_captions=True)
        if 't2i_uncertain' in policy:
            require(policy['t2i_uncertain'] in ('retain_annotated','exclude'),
                    'Unsupported t2i_uncertain policy')
        for item in sources:
            if item['adapter']=='genund_caption_tar':
                directions=item.get('source_directions')
                require(('direction' in item)!=('source_directions' in item),
                        'Task-aware genund inputs require an explicit i2t/t2i direction or source_directions')
                if directions is None:
                    require(item.get('direction') in ('i2t','t2i'),'Invalid explicit i2t/t2i direction')
                else:
                    require(isinstance(directions,dict) and bool(directions) and
                            all(isinstance(k,str) and k and v in ('i2t','t2i') for k,v in directions.items()),
                            'source_directions requires explicit source-to-i2t/t2i mappings')
                item.setdefault('preserve_all_captions',True)
                require(item['preserve_all_captions'] is True,
                        'Task-aware genund must preserve all distinct captions')
    for key,default in defaults.items():
        policy.setdefault(key,default)
        require(policy[key]==default,'Unsupported policy change requires a new implementation: '+key)
    require(set(policy)-set(defaults) <= ({'t2i_uncertain'} if task_aware else set()),
            'Unknown early policy setting')
    compute=value.setdefault('compute',{})
    for key,default in {'threads':3,'memory_limit':'128GB','rows_per_task':8192,
                        'batch_size':128,'dtype':'float32','io_workers':8,
                        'inference_engine':'optimized','optimized_qualification':None}.items():compute.setdefault(key,default)
    for key in ('threads','rows_per_task','batch_size','io_workers'):
        require(type(compute[key]) is int and compute[key]>0,'Positive compute limit required: '+key)
    require(compute['threads']<=16,'CPU threads must not exceed validated reconciliation limit')
    require(compute['inference_engine'] in {'optimized','reference'},'Unsupported inference engine')
    for key in ('models_dir','sscd_model'):
        if compute.get(key) is not None:
            require(isinstance(compute[key],str) and Path(compute[key]).is_absolute(),
                    'Explicit model location must be absolute: '+key)
    if 'index_rows_per_file' in compute:
        require(type(compute['index_rows_per_file']) is int and compute['index_rows_per_file']>0,
                'Positive index_rows_per_file required')
    resources=value.setdefault('resources',{})
    for key,default in {'max_cards':128,'frontpass_max_cards':48,'review_max_cards':128,
                        'sscd_max_cards':48,'prefer_occupied_nodes':True}.items():resources.setdefault(key,default)
    for key in ('max_cards','frontpass_max_cards','review_max_cards','sscd_max_cards'):
        require(type(resources[key]) is int and resources[key]>0,'Positive resource budget required')
    require(max(resources[k] for k in ('frontpass_max_cards','review_max_cards','sscd_max_cards'))<=resources['max_cards'],
            'Stage allocation exceeds total accelerator budget')
    value.setdefault('cache',{'directory':str(out.parent/'shared_image_cache'),'histories':[]})
    value['cache'].setdefault('directory',str(out.parent/'shared_image_cache'))
    value['cache'].setdefault('histories',[])
    value['cache'].setdefault('enabled',True)
    require(type(value['cache']['enabled']) is bool,'cache.enabled must be boolean')
    require(isinstance(value['cache']['histories'],list),'cache.histories must be a list')
    require(value['cache']['enabled'] or not value['cache']['histories'],
            'Disabled image cache requires empty histories')
    require(Path(value['cache']['directory']).is_absolute(),'Absolute image cache directory required')
    for item in sources:
        source=Path(item['root']).resolve();cache=Path(value['cache']['directory']).resolve()
        require(not cache.is_relative_to(source) and not source.is_relative_to(cache),
                'Cache must not overlap original dataset')
    screen=value.setdefault('screening',{})
    for key,default in {'backend':'serving','primary':'qwen3.6-35b-a3b','concurrency':128,
                        'history_decision_manifests':[],'quality_decisions':[],
                        'prompt':str(Path(__file__).resolve().parents[1]/'production/image_screening_prompt_v4.txt'),
                        'timeout':600,'max_tokens':1600}.items():screen.setdefault(key,default)
    require(screen['backend'] in {'serving','api','history_only'},'Unsupported review backend')
    require(type(screen['concurrency']) is int and 1<=screen['concurrency']<=2048,'Invalid review concurrency')
    search=value.setdefault('sscd',{})
    for key,default in {'threshold':.75,'block_rows':250000,'query_tile':8192,
        'database_tile':65536,'candidate_batch':32768,'query_count':128,
        'reference_top_k':10,'min_free_gib':50,'backend_qualification':None}.items():search.setdefault(key,default)
    require(search['threshold']==.75,'Candidate threshold change requires separate calibration')
    require(search['query_count']>=1 and search['reference_top_k']>=1,'Invalid SSCD reference dimensions')
    return value


def load(path):
    return checked(json.loads(Path(path).read_text()))
