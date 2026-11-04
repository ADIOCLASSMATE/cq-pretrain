"""Seal all 9,454 original FineVision shards only after every source slot binds."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corpus_pipeline.production.common import atomic_json,digest,file_sha
from corpus_pipeline.production.genund_collections import stamp


def finalize(run):
    run=Path(run); area=run/'finevision_recovery'
    if (run/'ingest_registry/finevision.json').exists(): return {'already_registered':True}
    binding_path=area/'BINDING_PLAN.json'
    if not binding_path.exists(): return {'waiting':'source_inventory'}
    plan=json.loads(binding_path.read_text()); source=json.loads((area/'PLAN.json').read_text())
    ready=sum((area/'recovered'/task['task_id']/'COMPLETE.json').exists() for task in plan['tasks'])
    if ready!=len(plan['tasks']): return {'waiting':'source_image_binding','recovered_files':ready,'source_files':len(plan['tasks'])}
    asset=json.loads((area/'RECOVERED_ASSET_BINDING.json').read_text())
    base=json.loads((run/'DATASET_BINDING.json').read_text()); units=[]; restored=restored_bytes=records=slots=0
    for task in plan['tasks']:
        receipt_path=area/'recovered'/task['task_id']/'COMPLETE.json'
        if not receipt_path.exists(): return {'waiting':'source_image_binding','recovered_prefix':len(units),'source_files':len(plan['tasks'])}
        receipt=json.loads(receipt_path.read_text())
        if not receipt.get('complete') or receipt['task']!=task or not receipt.get('every_image_sha_verified') or not receipt.get('source_parquet_sha_verified'):
            raise ValueError('incomplete or different source binding receipt')
        path=Path(receipt['path'])
        if path.stat().st_size!=receipt['size_bytes'] or file_sha(path)!=receipt['sha256']:
            raise ValueError('recovered original annotation changed')
        copied=receipt['recovered_missing_image_slots']>0
        unit={'adapter':'genund_finevision','collection':'genund','source':'FineVision','subset':task['subset'],
              'root':source['root'],'dataset_id':base['dataset_id'],'recovered_root':asset['root'],
              'recovered_dataset_id':asset['dataset_id'],'portable_datasets':[base['dataset_id'],asset['dataset_id']],
              'image_payloads_copied':copied,'image_payload_copy_scope':'only_missing_original_images_from_sha_verified_upstream_parquet',
              'directions':['i2t'],'path':str(path),'fingerprint':stamp(path),'sha256':receipt['sha256'],
              'annotation_path':str(path),'annotation_sha256':receipt['sha256'],'annotation_fingerprint':stamp(path),
              'source_shard':'FineVision/'+task['path'],'relative_path':'FineVision/'+task['path'],
              'row_start':0,'row_end':receipt['records'],'scope':'full_declared_original_parquet',
              'record_layout':'original_jsonl','original_binding_receipt':str(receipt_path),
              'original_binding_receipt_sha256':file_sha(receipt_path)}
        unit['unit_id']=digest(unit); units.append(unit)
        records+=receipt['records']; slots+=receipt['image_slots']; restored+=receipt['recovered_missing_image_slots']
        restored_bytes+=receipt['recovered_missing_image_slot_bytes']
    if len(units)!=source['task_count'] or records!=sum(source['expected_rows'].values()) or slots!=plan['image_slots']:
        raise ValueError('full FineVision source coverage differs')
    result={'complete':True,'source':'FineVision','units':units,'unit_count':len(units),'records':records,
            'image_slots':slots,'source_files_modified':False,'image_payloads_persisted':bool(restored),
            'recovered_missing_image_slots':restored,'recovered_missing_image_slot_bytes':restored_bytes,
            'recovered_asset_binding':asset,'source_binding_plan_sha256':file_sha(binding_path)}
    target=run/'ingest_units/finevision/PLAN.json'; atomic_json(target,result)
    registration={'complete':True,'source':'FineVision','plan':str(target),'plan_sha256':file_sha(target),
                  'image_payloads_persisted':bool(restored),'recovered_missing_image_slots':restored,
                  'recovered_missing_image_slot_bytes':restored_bytes}
    atomic_json(run/'ingest_registry/finevision.json',registration)
    return registration


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); args=parser.parse_args()
    print(json.dumps(finalize(args.run)))
