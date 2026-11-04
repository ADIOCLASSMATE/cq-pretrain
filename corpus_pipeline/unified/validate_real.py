"""Bounded validation on ten existing DCVLM records and their original TAR bytes.

No artificial images/scores. Metadata is copied into a tiny read-only validation
input. Two previously observed SSCD positives exercise the actual search path.
"""
import argparse
import json
import hashlib
from pathlib import Path
import time
import pyarrow.parquet as pq
from corpus_pipeline.production.common import atomic_json,file_sha,digest
from corpus_pipeline.production.records import snapshot,read_asset_bytes
from .config import require
from . import workflow,release


def prepare(native,output,pair_evidence,history_decisions):
    native=Path(native).resolve();output=Path(output).resolve();source=output/'selected_original_metadata'
    require(not output.exists(),'Choose a new validation output directory')
    source.mkdir(parents=True);snap=snapshot(native);bindings=[];catalog=[]
    selection=[('data/000000.parquet',[0,1]),('data/007299.parquet',[0,1]),
               ('data/006987.parquet',[0,1]),('data/004391.parquet',[0,1])]
    pair=json.loads(Path(pair_evidence).read_text())
    require(file_sha(pair['edge_parquet'])==pair['edge_parquet_sha256'],'Original SSCD positive evidence changed')
    for record in pair['records']:
        selection.append((str(Path(record['task']['parquet_path']).relative_to(native)),[record['record']['row_index']]))
    for number,(relative,indices) in enumerate(selection):
        original=native/relative;entry=snap['catalog_index'][str(original)]
        table=pq.read_table(original).take(indices);target=source/f'data/{number:06d}.parquet';target.parent.mkdir(exist_ok=True)
        pq.write_table(table,target)
        catalog.append({**entry,'path':str(target.relative_to(source)),'bytes':target.stat().st_size,'rows':len(table)})
        bindings.append({'path':str(original),'sha256':file_sha(original),'row_indices':indices,
                         'source_record_sha256s':[digest(row) for row in table.to_pylist()]})
    atomic_json(source/'catalog.json',catalog)
    atomic_json(source/'manifest.json',{'schema_version':'dcvlm.pretrain.v1','phase':'complete',
        'source_pool':snap['manifest']['source_pool'],'scope':'ten_selected_real_records_only',
        'statistics':{'samples':sum(x['rows'] for x in catalog)}})
    atomic_json(source/'COMPLETE.json',{'validation_passed':True,'catalog_sha256':file_sha(source/'catalog.json')})
    atomic_json(output/'SELECTION.json',{'source_snapshot':snap['fingerprint'],'bindings':bindings,
        'positive_pair_evidence':{'path':str(Path(pair_evidence).resolve()),'sha256':file_sha(pair_evidence)},
        'synthetic_data':False,'original_images_modified':False})
    cfg={'schema':'corpus.unified.pipeline.v1','scope':'validation','inputs':[
        {'adapter':'dcvlm_native','collection':'dcvlm','root':str(source)}],
        'output':str(output/'run'),'cache':{'directory':str(output/'cache'),'histories':[]},
        'compute':{'threads':3,'batch_size':4,'io_workers':3,'rows_per_task':4,'memory_limit':'8GB'},
        'screening':{'backend':'history_only','history_decision_manifests':[str(Path(history_decisions).resolve())]},
        'sscd':{'block_rows':4,'query_tile':4,'database_tile':4,'candidate_batch':16,'min_free_gib':0}}
    atomic_json(output/'config.json',cfg)
    return output/'config.json'


def validate(config):
    started=time.monotonic();result=workflow.run(config,local=True)
    require(result.get('complete') is True,'Real validation is incomplete: '+json.dumps(result))
    cfg=workflow.load(config);root=Path(cfg['output']);manifest=release.read_manifest(root)
    tasks=json.loads((root/'frontpass/frontpass_index.json').read_text())
    images={};counters={}
    for part in tasks['parts']:
        files={Path(a['path']).name:(root/'frontpass'/a['path']) for a in part['artifacts']}
        for row in pq.read_table(files['images.parquet']).to_pylist():
            ref=json.loads(row['original_ref_json']);blob=read_asset_bytes(ref)
            require(hashlib.sha256(blob).hexdigest()==row['asset_id'],'Original image bytes differ from asset identity')
            images[row['asset_id']]={'width':row['original_width'] if 'original_width' in row else row['width'],
                                    'height':row['original_height'] if 'original_height' in row else row['height'],
                                    'status':row['status'],'cf384_prob':row['cf384_prob'],'unifd_prob':row['unifd_prob'],
                                    'sscd_present':row['sscd_byte_offset'] is not None,'encoded_bytes':len(blob)}
        summary=json.loads(files['summary.json'].read_text())
        counters[part['task']['task_id']]=summary['performance']
    read_count=sum(len(batch) for batch in release.iter_batches(root,batch_size=4))
    require(read_count==manifest['counts']['keep'],'Published reader count differs')
    selection=json.loads((Path(config).parent/'SELECTION.json').read_text())
    expected=sorted(sha for binding in selection['bindings'] for sha in binding['source_record_sha256s'])
    actual=sorted(digest(json.loads(row['original_supervision_json']))
        for item in json.loads((root/'native/catalog.json').read_text())
        for row in pq.read_table(root/'native'/item['path']).to_pylist())
    require(actual==expected,'Normalized records changed original supervision')
    report={'schema':'corpus.unified.real_validation.v1','complete':True,'scope':'ten_selected_real_dcvlm_records',
        'models_ran_on':'cpu','model_identities':sorted({p['model_identity_sha256'] for p in tasks['parts']}),
        'records':manifest['records'],'counts':manifest['counts'],'reader_kept_records':read_count,
        'images':images,'performance':counters,'wall_seconds':time.monotonic()-started,
        'stage_receipts':{name:file_sha(root/'receipts'/f'{name}.json') for name in workflow.STAGES},
        'search':json.loads((root/'sscd_search/COMPLETE.json').read_text()),
        'release':result,'original_images_modified':False,'synthetic_inference':False,
        'all_original_supervision_hashes_verified':True,
        'balanced_live_mount_validated':False,'genund_live_mount_validated':False,
        'throughput_extrapolation_to_full_corpus':False}
    atomic_json(Path(config).parent/'REPORT.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',required=True);p.add_argument('--output',required=True)
    p.add_argument('--pair-evidence',required=True);p.add_argument('--history-decisions',required=True)
    a=p.parse_args();config=prepare(a.native,a.output,a.pair_evidence,a.history_decisions)
    result=validate(config);print(json.dumps({k:result[k] for k in ('complete','records','counts','wall_seconds')}))


if __name__=='__main__':main()
