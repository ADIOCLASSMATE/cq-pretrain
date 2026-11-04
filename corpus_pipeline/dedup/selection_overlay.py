"""Auditable, immutable keep-to-exclude overlays over a completed source view.

Only exact original encoded-image IDs match a rule. Canonical identities,
SSCD neighbours, other labels and original native files are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pyarrow.parquet as pq

from corpus_pipeline.postprocess.reconcile import (
    atomic_json, artifact, bind_identity, digest, file_sha, locked, quoted,
    require, stamp, validate_artifacts,
)
from .source_view import CODES, verify_mask
from .training_release import connection, parquet_sql

SCHEMA = 'corpus.training_selection_overlay.v1'
REASON = 'confirmed_ecommerce_placeholder_asset'


def completed_input(directory):
    directory = Path(directory).resolve(strict=True)
    complete = json.loads((directory/'COMPLETE.json').read_text())
    require(complete.get('complete') is True, 'Base input is not complete')
    manifest = json.loads((directory/'manifest.json').read_text())
    require(complete['manifest_sha256'] == file_sha(directory/'manifest.json'), 'Base manifest SHA mismatch')
    require(manifest.get('complete') is True and manifest['records'] == complete['records'],
            'Base input coverage mismatch')
    binding = {'path': str(directory), 'complete_sha256': file_sha(directory/'COMPLETE.json'),
               'manifest_sha256': complete['manifest_sha256']}
    return directory, manifest, binding


def evidence_bindings(paths):
    result = []
    for path in sorted({str(Path(p).resolve(strict=True)) for p in paths}):
        before = stamp(path)
        result.append({'path': path, 'sha256': file_sha(path), 'size_bytes': before['size'], 'stamp': before})
        require(stamp(path) == before, 'Evidence changed while hashing')
    require(result, 'At least one confirmed quality evidence file is required')
    return result


def scan_bucket(con, candidate, bucket, output, rule, rule_sha, evidence_sha):
    name = bucket['bucket']
    require(re.fullmatch('[0-9a-f]{2}', name) is not None, 'Invalid signature bucket')
    directory = output/'matched_parts'/name
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'COMPLETE.json'
    validate_artifacts(candidate, bucket)
    if marker.exists():
        value = json.loads(marker.read_text()); validate_artifacts(output, value)
        require(value['base_bucket_sha256'] == digest(bucket), 'Resumed bucket binding changed')
        return value
    paths = [candidate/x['path'] for x in bucket['artifacts']]
    require({p.name for p in paths} == {'keep.parquet','exclude.parquet','quarantine.parquet'},
            'Base bucket must contain every decision view')
    source = parquet_sql(paths)
    require(con.execute('SELECT count(*) FROM '+source).fetchone()[0] == bucket['records'],
            'Base bucket row count mismatch')
    assets = ','.join(quoted(x) for x in rule['asset_ids'])
    matched = f'list_distinct(list_filter(image_ids,x->x IN ({assets})))'
    target = directory/'records.parquet'; partial = directory/'records.partial.parquet'
    # Preserve full pointer/identity evidence. Selection deliberately never uses canonical_image_ids.
    con.execute(f'''COPY (SELECT record_id,task_id,source,source_shard,sample_id,
          original_parquet_path,row_index,category,native_format,text_hash,source_record_sha256,
          image_ids,canonical_image_ids,duplicate_of_record_id,
          decision AS base_decision,decision_reason AS base_decision_reason,
          CASE WHEN decision='keep' THEN 'exclude' ELSE decision END AS overlay_decision,
          CASE WHEN decision='keep' THEN {quoted(REASON)} ELSE decision_reason END AS overlay_reason,
          decision='keep' AS overlay_changed,{matched} AS matched_asset_ids,
          {quoted(rule_sha)} AS rule_sha256,{quoted(evidence_sha)} AS evidence_sha256
        FROM {source} WHERE len({matched})>0 ORDER BY original_parquet_path,row_index)
        TO {quoted(partial)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
    os.replace(partial,target)
    value = {'complete': True, 'bucket': name, 'base_bucket_sha256': digest(bucket),
             'matched_records': pq.ParquetFile(target).metadata.num_rows,
             'artifacts': [artifact(target, output)]}
    atomic_json(marker,value); return value


def combine_matches(con, output, receipts, base_manifest):
    source = parquet_sql([output/x['artifacts'][0]['path'] for x in receipts])
    total, ids, locators = con.execute(f'''SELECT count(*),count(DISTINCT record_id),
         count(DISTINCT (original_parquet_path,row_index)) FROM {source}''').fetchone()
    require(total == ids == locators, 'Matched records have duplicate identities or native locators')
    for name, where in [('matched_records',''), ('changed_records','WHERE overlay_changed')]:
        temporary=output/(name+'.partial.parquet'); target=output/(name+'.parquet')
        con.execute(f'''COPY (SELECT * FROM {source} {where} ORDER BY original_parquet_path,row_index)
             TO {quoted(temporary)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        os.replace(temporary,target)
    changed = parquet_sql([output/'changed_records.parquet'])
    changed_count = con.execute('SELECT count(*) FROM '+changed).fetchone()[0]
    deltas = [dict(source=s,category=c,changed_records=n,from_decision='keep',to_decision='exclude')
              for s,c,n in con.execute(f'SELECT source,category,count(*) FROM {changed} GROUP BY source,category ORDER BY source,category').fetchall()]
    categories = Counter({(x['source'],x['category'],x['decision']):x['records'] for x in base_manifest['source_category_counts']})
    for item in deltas:
        for decision,sign in [('keep',-1),('exclude',1)]:
            key=(item['source'],item['category'],decision)
            categories[key] += sign*item['changed_records']
            require(categories[key]>=0,'Source/category overlay exceeds base count')
    reasons=Counter(base_manifest['reasons'])
    for reason,count in con.execute(f'SELECT base_decision_reason,count(*) FROM {changed} GROUP BY base_decision_reason').fetchall():
        reasons[reason]-=count;require(reasons[reason]>=0,'Reason overlay exceeds base count')
    reasons[REASON]+=changed_count
    final_counts={key:base_manifest['counts'].get(key,0) for key in ('keep','exclude','quarantine')}
    final_counts['keep']-=changed_count;final_counts['exclude']+=changed_count
    require(final_counts['keep']>=0 and sum(final_counts.values())==base_manifest['records'],'Overlay record coverage mismatch')
    return {'matched_records': total,'changed_records': changed_count,'final_counts':final_counts,
            'source_category_deltas':deltas,
            'source_category_counts':[dict(source=k[0],category=k[1],decision=k[2],records=v)
                                      for k,v in sorted(categories.items()) if v],
            'reason_counts':{k:v for k,v in sorted(reasons.items()) if v}}


def matched_groups(path):
    """Yield one native file's matches at a time; do not retain the full match set."""
    current=None;rows=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['original_parquet_path','row_index','base_decision','overlay_changed']):
        for row in batch.to_pylist():
            if row['original_parquet_path']!=current:
                if current is not None:yield current,rows
                current=row['original_parquet_path'];rows=[]
            rows.append(row)
    if current is not None:yield current,rows


def clone_source_view(base_view, base_manifest, output, identity, statistics):
    view=output/'source_view';view.mkdir(parents=True,exist_ok=True)
    bind_identity(view,{'schema':SCHEMA,'overlay_identity_sha256':file_sha(output/'IDENTITY.json'),
                       'base_source_view':identity['base_source_view'],'rule_sha256':identity['rule_sha256']})
    groups=iter(matched_groups(output/'matched_records.parquet'));pending=next(groups,None)
    counts=Counter({key:0 for key in CODES});files=[];changed_total=0;changed_files=0
    require(len({x['source_path'] for x in base_manifest['files']})==len(base_manifest['files']), 'Duplicate base source file')
    for original in sorted(base_manifest['files'],key=lambda x:x['source_path']):
        source=original['source_path'];mask=verify_mask(base_view,original)
        require(stamp(source)==original['source_stamp'],'Original native source changed')
        require(pending is None or pending[0]>=source,'Matched locator not covered by base view')
        changed=[]
        if pending is not None and pending[0]==source:
            previous=-1
            for row in pending[1]:
                index=row['row_index']
                require(previous<index<len(mask),'Duplicate/out-of-range matched locator')
                previous=index
                require(row['base_decision'] in CODES and int(mask[index])==CODES[row['base_decision']],
                        'Base candidate decision differs from source mask')
                require(row['overlay_changed']==(row['base_decision']=='keep'),'Invalid overlay action')
                if row['overlay_changed']:changed.append(index)
            pending=next(groups,None)
        require(not Path(original['mask_path']).is_absolute(), 'Base mask path must be relative')
        target=view/original['mask_path']
        require(target.resolve().is_relative_to(view), 'Overlay mask path escapes its view')
        target.parent.mkdir(parents=True,exist_ok=True)
        item=dict(original)
        if changed:
            # Copy before modification even if the prior output path is hard linked.
            mask=mask.copy();mask[np.asarray(changed,dtype=np.int64)]=0
            counted=np.bincount(mask,minlength=256)
            item['counts']={name:int(counted[code]) for name,code in CODES.items()}
            item['mask_sha256']=hashlib.sha256(mask.tobytes()).hexdigest()
            changed_total+=len(changed);changed_files+=1
        if target.exists():
            require(target.stat().st_size==item['mask_bytes'] and file_sha(target)==item['mask_sha256'],
                    'Existing overlay mask is corrupt or has a different rule')
        else:
            partial=target.with_suffix('.partial');partial.unlink(missing_ok=True)
            if changed:
                with partial.open('wb') as stream:
                    stream.write(mask.tobytes());stream.flush();os.fsync(stream.fileno())
            else:
                try:os.link(base_view/original['mask_path'],partial)
                except OSError:shutil.copyfile(base_view/original['mask_path'],partial)
            os.replace(partial,target)
        verify_mask(view,item);counts.update(item['counts']);files.append(item)
    require(pending is None,'Uncovered matched source file')
    require(changed_total==statistics['changed_records'],'Mask changes do not equal keep hits')
    require(all(counts[key]==statistics['final_counts'][key] for key in ('keep','exclude','quarantine')),
            'Final masks differ from exact overlay counts')
    require(counts['not_indexed']==base_manifest['counts'].get('not_indexed',0),'Overlay changed unindexed coverage')
    result=dict(base_manifest)
    result.update(files=files,counts=dict(counts),identity_sha256=file_sha(view/'IDENTITY.json'),
        selection_overlay=True,selection_overlay_schema=SCHEMA,selection_overlay_rule_sha256=identity['rule_sha256'],
        base_candidate=identity['base_candidate'],base_source_view=identity['base_source_view'],
        input_candidate_counts_are_base=True,changed_records=changed_total,changed_source_files=changed_files,
        producer_code_sha256=identity['code_sha256'],original_records_modified=False,
        published_training=False,source_certification=False)
    require(sum(counts[k] for k in ('keep','exclude','quarantine'))==result['records'],'Final mask full coverage mismatch')
    atomic_json(view/'manifest.json',result)
    complete={key:result[key] for key in ('schema','complete','scope','records','native_rows','source_files','mask_bytes','counts','published_training')}
    complete.update(manifest_sha256=file_sha(view/'manifest.json'),selection_overlay=True,changed_records=changed_total,completed_at=time.time())
    atomic_json(view/'COMPLETE.json',complete)
    return {'source_view':str(view),'source_view_manifest_sha256':complete['manifest_sha256'],
            'source_view_complete_sha256':file_sha(view/'COMPLETE.json'),'changed_source_files':changed_files}


def verify_complete(output):
    complete=json.loads((output/'COMPLETE.json').read_text())
    require(complete.get('complete') is True and complete['manifest_sha256']==file_sha(output/'manifest.json'),
            'Overlay completion or manifest changed')
    manifest=json.loads((output/'manifest.json').read_text())
    for name in ('base_candidate','base_source_view'):
        _,base,binding=completed_input(manifest[name]['path'])
        require(binding==manifest[name], 'Overlay base binding changed')
        if name=='base_candidate':
            for bucket in base['buckets']:validate_artifacts(Path(binding['path']),bucket)
        else:
            for item in base['files']:verify_mask(Path(binding['path']),item)
    require(evidence_bindings([x['path'] for x in manifest['evidence']])==manifest['evidence'], 'Overlay evidence changed')
    validate_artifacts(output,{'artifacts':manifest['files']})
    view=Path(manifest['source_view'])
    require(file_sha(view/'COMPLETE.json')==manifest['source_view_complete_sha256'] and
            file_sha(view/'manifest.json')==manifest['source_view_manifest_sha256'],'Overlay source-view binding changed')
    for item in json.loads((view/'manifest.json').read_text())['files']:
        verify_mask(view,item);require(stamp(item['source_path'])==item['source_stamp'],'Original native source changed')
    return complete


def build(candidate,source_view,output,*,asset_ids,evidence,threads=2,memory_limit='32GB',stop_after_buckets=None):
    candidate,base,base_binding=completed_input(candidate)
    source_view,view,view_binding=completed_input(source_view)
    require(view['input_candidate_manifest_sha256']==base_binding['manifest_sha256'],'Source view belongs to another candidate')
    require(view.get('selection_overlay') is not True,'Base must be the original completed source view')
    require(base['records']==view['records'] and all(base['counts'].get(k,0)==view['counts'].get(k,0)
            for k in ('keep','exclude','quarantine')),'Base candidate/view counts disagree')
    ids=sorted(set(asset_ids));require(ids and all(re.fullmatch('[0-9a-f]{64}',x) for x in ids),'Expected exact lowercase encoded-image SHA256 IDs')
    rule={'schema':SCHEMA,'match':'exact_original_image_ids','asset_ids':ids,'action':'keep_to_exclude',
          'reason':REASON,'propagate_canonical_identity':False,'propagate_sscd_similarity':False}
    evidence_rows=evidence_bindings(evidence);output=Path(output).resolve()
    require(not output.is_relative_to(candidate) and not candidate.is_relative_to(output) and
            not output.is_relative_to(source_view) and not source_view.is_relative_to(output),
            'Overlay output must not modify either base directory')
    identity={'schema':SCHEMA,'code_sha256':file_sha(__file__),'base_candidate':base_binding,'base_source_view':view_binding,
              'rule_sha256':digest(rule),'evidence':evidence_rows,'evidence_sha256':digest(evidence_rows)}
    with locked(output):
        bind_identity(output,identity)
        if (output/'COMPLETE.json').exists():return verify_complete(output)
        atomic_json(output/'asset_rule.json',rule);atomic_json(output/'evidence.json',evidence_rows)
        con=connection(output,threads,memory_limit)
        try:
            receipts=[]
            for bucket in base['buckets']:
                receipts.append(scan_bucket(con,candidate,bucket,output,rule,identity['rule_sha256'],identity['evidence_sha256']))
                atomic_json(output/'PROGRESS.json',{'phase':'scanning_exact_asset_ids','completed_buckets':len(receipts),
                            'total_buckets':len(base['buckets']),'matched_records':sum(x['matched_records'] for x in receipts),'updated_at':time.time()})
                if stop_after_buckets and len(receipts)>=stop_after_buckets:return {'complete':False,'completed_buckets':len(receipts)}
            statistics=combine_matches(con,output,receipts,base)
        finally:con.close()
        atomic_json(output/'PROGRESS.json',{'phase':'cloning_selection_masks','changed_records':statistics['changed_records'],'updated_at':time.time()})
        new_view=clone_source_view(source_view,view,output,identity,statistics)
        require(evidence_bindings(evidence)==evidence_rows,'Evidence changed during overlay')
        require(completed_input(candidate)[2]==base_binding and completed_input(source_view)[2]==view_binding,'Base completion changed during overlay')
        manifest={'schema':SCHEMA,'complete':True,'scope':base['scope'],'records':base['records'],
                  **identity,**statistics,**new_view,'base_counts':base['counts'],
                  'files':[artifact(output/name,output) for name in ('matched_records.parquet','changed_records.parquet','asset_rule.json','evidence.json')],
                  'source_records_deleted':0,'original_records_modified':False,'published_training':False,'source_certification':False}
        atomic_json(output/'manifest.json',manifest)
        complete={'schema':SCHEMA,'complete':True,'records':base['records'],'counts':statistics['final_counts'],
                  'changed_records':statistics['changed_records'],'manifest_sha256':file_sha(output/'manifest.json'),
                  'published_training':False,'completed_at':time.time()}
        atomic_json(output/'COMPLETE.json',complete);return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True);parser.add_argument('--source-view',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--asset-id',action='append',required=True)
    parser.add_argument('--evidence',action='append',required=True)
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--memory-limit',default='32GB')
    args=parser.parse_args()
    print(json.dumps(build(args.candidate,args.source_view,args.output,asset_ids=args.asset_id,evidence=args.evidence,
                           threads=args.threads,memory_limit=args.memory_limit)))


if __name__=='__main__':main()
