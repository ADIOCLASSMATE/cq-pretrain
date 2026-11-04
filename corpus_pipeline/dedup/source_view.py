"""Sequential native-file view of a training candidate keep/exclude index.

One uint8 per original metadata row: 0 exclude, 1 keep, 2 quarantine,
255 not indexed (permitted only for a bounded validation). No image is decoded,
no source row is rewritten, and a keep decision is not a quality warranty.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from corpus_pipeline.postprocess.reconcile import (
    atomic_json, artifact, bind_identity, file_sha, locked, quoted, require,
    stamp, validate_artifacts,
)
from .training_release import connection, parquet_sql

SCHEMA='corpus.training_candidate_source_view.v1'
CODES={'exclude':0,'keep':1,'quarantine':2,'not_indexed':255}
TASK_POLICY='corpus.task_policy.v1'
POLICY_ROUTES={1:'i2t',2:'t2i',3:'mixed',4:'text',5:'unknown'}
POLICY_CODES={str(code):{'processing_policy':TASK_POLICY,'task_route':route,'image_loss_allowed':bool(code % 2)}
              for number,route in POLICY_ROUTES.items() for code in (2*number,2*number+1)
              if not (route=='i2t' and code % 2)}


def source_bindings(candidate, manifest):
    result={}
    validate_artifacts(candidate,{'artifacts':manifest['native_metadata_bindings']})
    for item in manifest['native_metadata_bindings']:
        value=json.loads((candidate/item['path']).read_text());path=str(Path(value['path']).resolve(strict=True))
        require(path not in result and stamp(path)==value['stamp'],'Duplicate/changed native metadata binding')
        require(value.get('path')==path and len(value.get('sha256',''))==64,'Invalid native metadata binding')
        result[path]=value
    return result


def ordered_locators(candidate, manifest, output, *, threads, memory_limit):
    marker=output/'ORDERED_COMPLETE.json';target=output/'ordered.parquet'
    if marker.exists():
        value=json.loads(marker.read_text());validate_artifacts(output,value);return value
    paths=[]
    for bucket in manifest['buckets']:
        validate_artifacts(candidate,bucket)
        require({Path(x['path']).name for x in bucket['artifacts']}=={'keep.parquet','exclude.parquet','quarantine.parquet'},
                'Candidate bucket must contain all three views')
        paths.extend(candidate/item['path'] for item in bucket['artifacts'])
    con=connection(output,threads,memory_limit)
    try:
        source=parquet_sql(paths)
        columns={row[0] for row in con.execute('DESCRIBE SELECT * FROM '+source).fetchall()}
        policy_columns={'processing_policy','task_route','image_loss_allowed'}
        require(not (columns & policy_columns) or policy_columns <= columns,'Incomplete candidate task policy columns')
        policy_select=''
        if policy_columns <= columns:
            invalid_policy=con.execute(f'''SELECT count(*) FROM {source} WHERE processing_policy IS NOT NULL AND
                (processing_policy<>{quoted(TASK_POLICY)} OR task_route IS NULL OR
                 task_route NOT IN ('i2t','t2i','mixed','text','unknown') OR image_loss_allowed IS NULL OR
                 (task_route='i2t' AND image_loss_allowed))''').fetchone()[0]
            require(invalid_policy==0,'Invalid candidate task policy or I2T image loss permission')
            has_policy=con.execute(f'SELECT count(*) FROM {source} WHERE processing_policy={quoted(TASK_POLICY)}').fetchone()[0]>0
            if has_policy:
                policy_select=f''',CASE WHEN processing_policy={quoted(TASK_POLICY)} THEN
                    (CASE task_route WHEN 'i2t' THEN 2 WHEN 't2i' THEN 4 WHEN 'mixed' THEN 6
                       WHEN 'text' THEN 8 WHEN 'unknown' THEN 10 END)+CAST(image_loss_allowed AS UTINYINT)
                    ELSE 0 END::UTINYINT AS task_policy_code'''
        invalid=con.execute(f'''SELECT count(*) FROM {source} WHERE original_parquet_path IS NULL OR
            row_index IS NULL OR row_index<0 OR decision IS NULL OR decision NOT IN ('exclude','keep','quarantine')''').fetchone()[0]
        require(invalid==0,'Invalid source locator or selection decision')
        temporary=output/'ordered.partial.parquet'
        con.execute(f'''COPY (SELECT original_parquet_path,row_index,
            CASE decision WHEN 'exclude' THEN 0 WHEN 'keep' THEN 1 ELSE 2 END::UTINYINT AS selection{policy_select}
            FROM {source} ORDER BY original_parquet_path,row_index)
            TO {quoted(temporary)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 65536)''')
        require(pq.ParquetFile(temporary).metadata.num_rows==manifest['records'],'Sorted locator coverage mismatch')
        os.replace(temporary,target)
        value={'complete':True,'records':manifest['records'],'artifacts':[artifact(target,output)]}
        atomic_json(marker,value);return value
    finally:con.close()


class MaskBuilder:
    def __init__(self, output, bindings, full_scope):
        self.output=output;self.bindings=bindings;self.full_scope=full_scope
        self.current=None;self.mask=None;self.previous=-1;self.files=[];self.seen=set()
        self.policy=None
    def finish(self):
        if self.current is None:return
        counts=np.bincount(self.mask,minlength=256)
        require(not self.full_scope or counts[255]==0,'Full release has unindexed native rows')
        identity=hashlib.sha256(self.current.encode()).hexdigest()
        path=self.output/'masks'/(identity+'.u8');marker=self.output/'masks'/(identity+'.json')
        binding=self.bindings[self.current]
        require(stamp(self.current)==binding['stamp'],'Native metadata changed during mask build')
        value={'source_path':self.current,'source_sha256':binding['sha256'],'source_stamp':binding['stamp'],
               'rows':len(self.mask),'counts':{name:int(counts[code]) for name,code in CODES.items()},
               'mask_sha256':hashlib.sha256(self.mask.tobytes()).hexdigest(),
               'mask_path':str(path.relative_to(self.output)),'mask_bytes':len(self.mask)}
        if self.policy is not None and np.any(self.policy):
            policy_path=self.output/'policies'/(identity+'.u8')
            value.update(task_policy_path=str(policy_path.relative_to(self.output)),
                         task_policy_bytes=len(self.policy),task_policy_sha256=hashlib.sha256(self.policy.tobytes()).hexdigest())
            if policy_path.exists():
                require(policy_path.stat().st_size==len(self.policy) and file_sha(policy_path)==value['task_policy_sha256'],
                        'Existing task policy sidecar changed')
            else:
                policy_path.parent.mkdir(parents=True,exist_ok=True);temporary=policy_path.with_suffix('.partial')
                with temporary.open('wb') as stream:
                    stream.write(self.policy.tobytes());stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,policy_path)
        if marker.exists():
            old=json.loads(marker.read_text())
            require(old==value and path.stat().st_size==len(self.mask) and file_sha(path)==value['mask_sha256'],
                    'Existing source mask is corrupt or belongs to another selection')
        else:
            path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.partial')
            with temporary.open('wb') as stream:
                stream.write(self.mask.tobytes());stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path);atomic_json(marker,value)
        self.files.append(value);self.current=None;self.mask=None;self.policy=None;self.previous=-1
    def consume(self,path,indices,values,policy_codes=None):
        if path!=self.current:
            self.finish()
            require(path in self.bindings and path not in self.seen,'Unknown/repeated native file in sorted input')
            self.seen.add(path);self.current=path
            reader=pq.ParquetFile(path);rows=reader.metadata.num_rows;reader.close()
            self.mask=np.full(rows,255,dtype=np.uint8)
        require(len(indices)>0 and indices[0]>self.previous and np.all(indices[1:]>indices[:-1]),
                'Duplicate or unsorted original row locator')
        require(indices[0]>=0 and indices[-1]<len(self.mask),'Original row locator outside file')
        require(np.all(values<=2),'Unknown source selection code')
        if policy_codes is not None:
            require(len(policy_codes)==len(indices) and np.all(np.isin(policy_codes,[0]+[int(k) for k in POLICY_CODES])),
                    'Invalid task policy sidecar code')
            if self.policy is None:self.policy=np.zeros(len(self.mask),dtype=np.uint8)
            self.policy[indices]=policy_codes
        self.mask[indices]=values;self.previous=int(indices[-1])
    def read(self,path,batch_size=65536):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            names=batch.column('original_parquet_path').to_numpy(zero_copy_only=False)
            indices=batch.column('row_index').to_numpy(zero_copy_only=False)
            values=batch.column('selection').to_numpy(zero_copy_only=False)
            policies=(batch.column('task_policy_code').to_numpy(zero_copy_only=False)
                      if 'task_policy_code' in batch.schema.names else None)
            boundaries=np.r_[0,np.flatnonzero(names[1:]!=names[:-1])+1,len(names)]
            for begin,end in zip(boundaries[:-1],boundaries[1:]):
                self.consume(str(names[begin]),indices[begin:end],values[begin:end],
                             None if policies is None else policies[begin:end])
        self.finish()
        require(self.seen==set(self.bindings),'Native metadata binding has no selected locator coverage')
        return self.files


def build(candidate,output,*,threads=2,memory_limit='32GB'):
    candidate=Path(candidate).resolve(strict=True);output=Path(output).resolve()
    complete=json.loads((candidate/'COMPLETE.json').read_text())
    require(complete.get('complete') is True,'Candidate index is incomplete')
    require(complete['manifest_sha256']==file_sha(candidate/'manifest.json'),'Candidate manifest SHA mismatch')
    manifest=json.loads((candidate/'manifest.json').read_text())
    require(manifest['records']==complete['records'],'Candidate record count differs')
    bindings=source_bindings(candidate,manifest)
    identity={'schema':SCHEMA,'code_sha256':file_sha(__file__),'candidate':str(candidate),
              'candidate_complete_sha256':file_sha(candidate/'COMPLETE.json'),
              'candidate_manifest_sha256':complete['manifest_sha256']}
    with locked(output):
        bind_identity(output,identity)
        if (output/'COMPLETE.json').exists():
            result=json.loads((output/'COMPLETE.json').read_text())
            require(result['manifest_sha256']==file_sha(output/'manifest.json'),'Source view manifest changed')
            for item in json.loads((output/'manifest.json').read_text())['files']:
                verify_mask(output,item);verify_policy_overlay(output,item)
            return result
        atomic_json(output/'PROGRESS.json',{'phase':'ordering_locators','updated_at':time.time()})
        ordered_locators(candidate,manifest,output,threads=threads,memory_limit=memory_limit)
        atomic_json(output/'PROGRESS.json',{'phase':'writing_selection_masks','updated_at':time.time()})
        files=MaskBuilder(output,bindings,manifest['scope']=='full_run').read(output/'ordered.parquet')
        counts=Counter({name:0 for name in CODES})
        for item in files:counts.update(item['counts'])
        indexed=sum(counts[name] for name in ('keep','exclude','quarantine'))
        require(indexed==manifest['records'],'Source mask coverage differs from candidate index')
        require(all(counts[name]==manifest['counts'].get(name,0) for name in ('keep','exclude','quarantine')),
                'Source mask policy counts differ from candidate index')
        result={'schema':SCHEMA,'complete':True,'scope':manifest['scope'],'records':indexed,
            'native_rows':sum(x['rows'] for x in files),'source_files':len(files),'mask_bytes':sum(x['mask_bytes'] for x in files),
            'counts':dict(counts),'codes':CODES,'files':files,'identity_sha256':file_sha(output/'IDENTITY.json'),
            'published_training':False,'source_certification':False,'original_records_modified':False,
            'image_bytes_decoded':False,'input_candidate':str(candidate),
            'input_candidate_manifest_sha256':identity['candidate_manifest_sha256']}
        if any('task_policy_path' in item for item in files):
            result['task_policy_overlay']={'schema':'corpus.source_view.task_policy.v1','codes':POLICY_CODES,
                                           'zero_code':'preserve_legacy_policy','source_files_modified':False}
        atomic_json(output/'manifest.json',result)
        final={key:result[key] for key in ('schema','complete','scope','records','native_rows','source_files','mask_bytes','counts','published_training')}
        final.update(manifest_sha256=file_sha(output/'manifest.json'),completed_at=time.time())
        atomic_json(output/'COMPLETE.json',final);return final


def verify_mask(view,item):
    path=(view/item['mask_path']).resolve(strict=True)
    require(path.is_relative_to(view),'Mask path escapes view')
    require(path.stat().st_size==item['mask_bytes']==item['rows'] and file_sha(path)==item['mask_sha256'],
            'Selection mask byte count or SHA mismatch')
    mask=np.fromfile(path,dtype=np.uint8)
    counts=np.bincount(mask,minlength=256)
    require(sum(counts[code] for code in CODES.values())==len(mask),'Invalid mask selection code')
    require(all(int(counts[code])==item['counts'][name] for name,code in CODES.items()),'Mask selection counts differ')
    return mask


def verify_policy_overlay(view,item):
    """Verify the optional per-native-row task policy without reading images."""
    fields={'task_policy_path','task_policy_bytes','task_policy_sha256'}
    require(not (fields & item.keys()) or fields <= item.keys(),'Incomplete task policy sidecar binding')
    if 'task_policy_path' not in item:return None
    view=Path(view).resolve(strict=True);path=(view/item['task_policy_path']).resolve(strict=True)
    require(path.is_relative_to(view),'Task policy sidecar escapes view')
    require(path.stat().st_size==item['task_policy_bytes']==item['rows'] and
            file_sha(path)==item['task_policy_sha256'],'Task policy sidecar byte count or SHA mismatch')
    codes=np.fromfile(path,dtype=np.uint8)
    require(np.all(np.isin(codes,[0]+[int(k) for k in POLICY_CODES])),'Invalid task policy sidecar code')
    return codes


def _overlay_task_policy(batch,codes):
    """Attach derived permissions to returned rows; native bytes stay intact."""
    active=codes!=0
    for name,kind in [('processing_policy',pa.string()),('task_route',pa.string()),('image_loss_allowed',pa.bool_())]:
        previous=batch.column(name).to_pylist() if name in batch.schema.names else [None]*batch.num_rows
        values=[POLICY_CODES[str(int(code))][name] if enabled else old
                for code,enabled,old in zip(codes,active,previous)]
        array=pa.array(values,type=kind)
        if name in batch.schema.names:batch=batch.set_column(batch.schema.get_field_index(name),name,array)
        else:batch=batch.append_column(name,array)
    return batch


def iter_native_batches(view,batch_size=8192,epoch_seed=None,*,worker_id=0,num_workers=1):
    """Stream SHA-verified source rows with optional derived task permissions.

    File order may be seeded and sharded across data-loader workers. Rows within
    a native file keep their original order. All original metadata fields remain
    intact, including task direction, segments, image slots, and annotations.
    """
    require(type(batch_size) is int and batch_size>0,'batch_size must be positive')
    require(type(num_workers) is int and type(worker_id) is int and 0<=worker_id<num_workers,'Invalid reader worker partition')
    view=Path(view).resolve(strict=True);complete=json.loads((view/'COMPLETE.json').read_text())
    require(complete.get('complete') is True and complete['manifest_sha256']==file_sha(view/'manifest.json'),
            'Source view manifest is incomplete or changed')
    manifest=json.loads((view/'manifest.json').read_text());files=list(manifest['files'])
    require(manifest['codes']==CODES,'Unsupported selection mask encoding')
    overlay=manifest.get('task_policy_overlay')
    if overlay is not None:
        require(overlay.get('schema')=='corpus.source_view.task_policy.v1' and overlay.get('codes')==POLICY_CODES,
                'Unsupported source-view task policy overlay')
    if epoch_seed is not None:random.Random(epoch_seed).shuffle(files)
    for item in files[worker_id::num_workers]:
        if item['counts']['keep']==0:continue
        mask=verify_mask(view,item);path=Path(item['source_path'])
        policies=verify_policy_overlay(view,item)
        require(policies is None or overlay is not None,'Task policy sidecar lacks its manifest contract')
        require(stamp(path)==item['source_stamp'],'Native source metadata changed')
        require(file_sha(path)==item['source_sha256'],'Native source SHA mismatch')
        require(stamp(path)==item['source_stamp'],'Native source changed while hashing')
        reader=pq.ParquetFile(path);offset=0;selected=0
        try:
            require(reader.metadata.num_rows==len(mask),'Source/mask row count mismatch')
            for batch in reader.iter_batches(batch_size=batch_size):
                stop=offset+batch.num_rows;keep=mask[offset:stop]==1
                if np.any(keep):
                    result=batch.filter(pa.array(keep))
                    if overlay is not None:
                        codes=policies[offset:stop][keep] if policies is not None else np.zeros(result.num_rows,dtype=np.uint8)
                        result=_overlay_task_policy(result,codes)
                    selected+=result.num_rows;yield result
                offset=stop
            require(offset==len(mask) and selected==item['counts']['keep'],'Sequential source reader lost rows')
            require(stamp(path)==item['source_stamp'],'Native source changed during read')
        finally:reader.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--memory-limit',default='32GB')
    args=parser.parse_args();print(json.dumps(build(args.candidate,args.output,threads=args.threads,memory_limit=args.memory_limit)))

if __name__=='__main__':main()
