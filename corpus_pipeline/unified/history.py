"""Read-only batch lookup of sealed historical frontpass evidence by image SHA.

Registration binds small manifests; it never creates 99M per-asset cache files.
Queries prune to requested SHA buckets, verify touched Parquet/vector artifacts,
and use a single original observation with the exact resident-worker identity.
Reconciliation max scores are deliberately not used as model outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter,OrderedDict
import json
import math
import os
from pathlib import Path
import threading

import numpy as np

from corpus_pipeline.postprocess.reconcile import atomic_json,canonical,digest,file_sha,require,stamp,quoted
from .cache import HEX,normalize_evidence,prediction,available_models,valid_models,file_lock,verification_stamp

SCHEMA='corpus.unified.historical_image_evidence.v1'


def bound(path,expected=None):
    path=Path(path).resolve(strict=True);before=stamp(path);checksum=file_sha(path)
    require(expected is None or checksum==expected,'Historical input SHA mismatch: '+str(path))
    require(stamp(path)==before,'Historical input changed during hashing')
    return {'path':str(path),'sha256':checksum,'size_bytes':before['size'],'stamp':before}


def artifact_binding(root,item):
    root=Path(root).resolve();path=(root/item['path']).resolve()
    require(path.is_relative_to(root),'Historical artifact escapes its completed root')
    require(HEX.fullmatch(item['sha256']) and type(item['size_bytes']) is int,'Invalid historical artifact binding')
    return {'path':str(path),'sha256':item['sha256'],'size_bytes':item['size_bytes']}


def register_history(frontpass_root,reconciliation_root,output,*,decisions_manifest=None,quality_overlay=None,catalog_root=None):
    front=Path(frontpass_root).resolve(strict=True);recon=Path(reconciliation_root).resolve(strict=True)
    complete_bound=bound(front/'FRONTPASS_COMPLETE.json');complete=json.loads(Path(complete_bound['path']).read_text())
    require(complete.get('complete') is True,'Frontpass is incomplete')
    index_bound=bound(front/'frontpass_index.json',complete['index_sha256']);index=json.loads(Path(index_bound['path']).read_text())
    require(index.get('complete') is True and index['counts']==complete['counts'],'Frontpass index completion differs')
    rc_bound=bound(recon/'COMPLETE.json');rc=json.loads(Path(rc_bound['path']).read_text())
    require(rc.get('complete') is True and rc['input_index_sha256']==index_bound['sha256'],'Reconciliation belongs to different frontpass')
    buckets=[]
    for b in rc['buckets']:
        receipt=bound(recon/b['receipt_path'],b['receipt_sha256']);value=json.loads(Path(receipt['path']).read_text())
        require(value['bucket']==b['bucket'],'Reconciliation bucket differs')
        assets=[x for x in value['artifacts'] if Path(x['path']).name=='assets.parquet']
        require(len(assets)==1,'Missing historical asset index')
        buckets.append({'bucket':b['bucket'],'receipt':receipt,'assets':artifact_binding(recon,assets[0])})
    require(len({x['bucket'] for x in buckets})==len(buckets),'Repeated historical SHA bucket')
    extra={}
    if decisions_manifest:
        binding=bound(decisions_manifest);value=json.loads(Path(binding['path']).read_text())
        require(value.get('complete') is True,'Historical screening is incomplete')
        extra['decisions_manifest']=binding
    if quality_overlay:
        area=Path(quality_overlay).resolve();binding=bound(area/'COMPLETE.json');value=json.loads(Path(binding['path']).read_text())
        require(value.get('complete') is True,'Historical quality overlay is incomplete')
        extra['quality_overlay_complete']=binding;extra['quality_overlay_manifest']=bound(area/'manifest.json',value['manifest_sha256'])
    if catalog_root:
        area=Path(catalog_root).resolve()
        if (area/'catalog/COMPLETE.json').exists():area=area/'catalog'
        binding=bound(area/'COMPLETE.json');value=json.loads(Path(binding['path']).read_text())
        require(value.get('complete') is True and value.get('input_sha256')==rc_bound['sha256'],
                'Historical SSCD catalog belongs to a different reconciliation')
        extra['catalog_complete']=binding
    result={'schema':SCHEMA,'complete':True,'frontpass_root':str(front),'reconciliation_root':str(recon),
        'frontpass_complete':complete_bound,'frontpass_index':index_bound,'reconciliation_complete':rc_bound,
        'buckets':buckets,'model_identity_sha256s':sorted({p['model_identity_sha256'] for p in index['parts']}),
        'indexed_assets':rc['counts']['assets'],'key':'exact_original_encoded_sha256_and_complete_worker_identity',
        'vector_source':'committed_frontpass_sscd.f32_with_full_file_sha_and_offset',
        'scores_source':'one_committed_original_observation_not_reconciliation_maxima',
        'screening_is_separate_policy_evidence':True,'auxiliary_manifests':extra,
        'per_asset_files_created':0,'registration_code_sha256':file_sha(__file__)}
    output=Path(output).resolve()
    with file_lock(output/'.register.lock'):
        marker=output/'manifest.json'
        if marker.exists():require(json.loads(marker.read_text())==result,'Historical registration identity changed')
        else:atomic_json(marker,result)
        completion={'schema':SCHEMA,'complete':True,'manifest_sha256':file_sha(marker),'indexed_assets':result['indexed_assets']}
        atomic_json(output/'COMPLETE.json',completion)
    return completion


class HistoryIndex:
    register=staticmethod(register_history)
    def __init__(self,root,*,threads=1,memory_limit='1GB',max_verified_files=32768,max_batch_size=4096):
        self.root=Path(root).resolve(strict=True);complete=json.loads((self.root/'COMPLETE.json').read_text())
        require(complete.get('complete') is True and file_sha(self.root/'manifest.json')==complete['manifest_sha256'],
                'Historical registry manifest changed')
        self.manifest=json.loads((self.root/'manifest.json').read_text());require(self.manifest['schema']==SCHEMA,'Unsupported history schema')
        require(type(max_verified_files) is int and 1<=max_verified_files<=131072,'Invalid historical verification memory bound')
        require(type(max_batch_size) is int and 1<=max_batch_size<=4096,'Invalid historical lookup batch bound')
        require(len(self.manifest['buckets'])<=256,'Historical exact SHA index has too many buckets')
        self.max_verified_files=max_verified_files;self.max_batch_size=max_batch_size;self.verified=OrderedDict();self.fixed_verified={};self._lock=threading.Lock()
        self._verify(self.manifest['frontpass_complete'],fixed=True);self._verify(self.manifest['frontpass_index'],fixed=True);self._verify(self.manifest['reconciliation_complete'],fixed=True)
        index=json.loads(Path(self.manifest['frontpass_index']['path']).read_text())
        self.parts={x['task']['task_id']:x for x in index['parts']}
        require(len(self.parts)==len(index['parts']),'Duplicate historical committed task')
        self.buckets={x['bucket']:x for x in self.manifest['buckets']}
        import duckdb
        self.con=duckdb.connect();self.con.execute(f'SET threads={int(threads)}');self.con.execute('SET memory_limit='+quoted(memory_limit))
        self.con.execute('SET preserve_insertion_order=false');self.last_report={};self.stats=Counter()
    def _verify(self,binding,*,fixed=False):
        # A random 512-SHA lookup touches most of the 256 buckets. Keep their
        # verification separate from the task-file LRU to avoid rehashing tens
        # of GB each window. This fixed set is bounded by 3 + 2*256 entries.
        cache=self.fixed_verified if fixed else self.verified
        path=Path(binding['path']);current=verification_stamp(path);key=(binding['sha256'],binding['size_bytes'],current)
        if cache.get(str(path))!=key:
            require(current['size']==binding['size_bytes'] and file_sha(path)==binding['sha256'] and verification_stamp(path)==current,
                    'Historical artifact changed: '+str(path))
            cache[str(path)]=key
        if not fixed:
            self.verified.move_to_end(str(path))
            while len(self.verified)>self.max_verified_files:self.verified.popitem(last=False)
        return path
    def _part_artifacts(self,task_id):
        part=self.parts[task_id];result={}
        for item in part['artifacts']:
            name=Path(item['path']).name
            if name in ('images.parquet','sscd.f32','summary.json'):
                result[name]=artifact_binding(self.manifest['frontpass_root'],item)
        require(set(result)=={'images.parquet','sscd.f32','summary.json'},'Historical committed task is incomplete')
        summary_path=self._verify(result['summary.json']);summary=json.loads(summary_path.read_text())
        require(summary['task']==part['task'] and summary['model_identity_sha256']==part['model_identity_sha256'],
                'Historical observation model/task binding changed')
        return part,result
    def lookup_many(self,shas,identity_sha256,models=('cf384','unifd','sscd')):
        shas=list(dict.fromkeys(shas));models=valid_models(models)
        require(len(shas)<=self.max_batch_size,'Historical lookup exceeds bounded batch size')
        require(all(isinstance(s,str) and HEX.fullmatch(s) for s in shas) and HEX.fullmatch(identity_sha256),'Invalid historical identity key')
        with self._lock:return self._lookup(shas,identity_sha256,models)
    def _lookup(self,shas,identity_sha256,models):
        report=Counter(requested=len(shas));result={}
        if identity_sha256 not in self.manifest['model_identity_sha256s']:
            report['identity_incompatible']=len(shas);self.last_report=dict(report);self.stats.update(report);return result
        selected=[]
        for prefix in sorted({sha[:2] for sha in shas}):
            if prefix not in self.buckets:continue
            bucket=self.buckets[prefix];self._verify(bucket['receipt'],fixed=True);path=self._verify(bucket['assets'],fixed=True)
            wanted=[sha for sha in shas if sha[:2]==prefix]
            sql='SELECT asset_id,task_id,sample_id,image_index,model_identity_sha256,sscd_file,sscd_byte_offset,sscd_file_sha256,sscd_file_size_bytes,original_width,original_height,pixel_sha256,has_invalid_observation,pixel_sha256_conflict,dimension_count FROM read_parquet('+quoted(path)+') WHERE asset_id IN ('+','.join(quoted(x) for x in wanted)+')'
            columns=[x[0] for x in self.con.execute('DESCRIBE '+sql).fetchall()]
            rows=[dict(zip(columns,row)) for row in self.con.execute(sql).fetchall()]
            self._verify(bucket['assets'],fixed=True)
            require(len(rows)==len({x['asset_id'] for x in rows}),'Historical asset map is ambiguous')
            for row in rows:
                if row['model_identity_sha256']!=identity_sha256:report['identity_incompatible']+=1;continue
                if row['has_invalid_observation'] or row['pixel_sha256_conflict'] or row['dimension_count']!=1:
                    report['invalid_historical_evidence']+=1;continue
                selected.append(row)
        by_task={}
        for row in selected:by_task.setdefault(row['task_id'],[]).append(row)
        for task_id,rows in by_task.items():
            part,artifacts=self._part_artifacts(task_id)
            require(part['model_identity_sha256']==identity_sha256,'Historical model identity differs from committed task')
            path=self._verify(artifacts['images.parquet']);wanted=','.join(quoted(x['asset_id']) for x in rows)
            import pyarrow as pa
            self.con.register('wanted_history',pa.Table.from_pylist([
                {k:r[k] for k in ('asset_id','sample_id','image_index')} for r in rows]))
            query='SELECT asset_id,sample_id,image_index,status,original_width,original_height,pixel_sha256,cf384_prob,unifd_prob,sscd_byte_offset,error_json,original_ref_json FROM read_parquet('+quoted(path)+') JOIN wanted_history USING(asset_id,sample_id,image_index) WHERE asset_id IN ('+wanted+')'
            columns=[x[0] for x in self.con.execute('DESCRIBE '+query).fetchall()]
            observations={}
            # At most one committed task is materialized here; no global observations table is loaded.
            for record in self.con.execute(query).fetchall():
                row=dict(zip(columns,record));key=(row['asset_id'],row['sample_id'],row['image_index'])
                require(key not in observations,'Duplicate historical image slot');observations[key]=row
            self.con.unregister('wanted_history');self._verify(artifacts['images.parquet'])
            for asset in rows:
                aid=asset['asset_id'];row=observations.get((aid,asset['sample_id'],asset['image_index']))
                require(row is not None,'Reconciled asset is missing its committed observation')
                for key in ('original_width','original_height','pixel_sha256','sscd_byte_offset'):
                    require(row[key]==asset[key],'Reconciled evidence differs from original observation')
                errors=json.loads(row['error_json']);ref=json.loads(row['original_ref_json'])
                require(isinstance(errors,dict) and ref.get('sha256')==aid,'Malformed historical decode evidence')
                if errors.get('error_stage')=='decode':report['invalid_historical_evidence']+=1;continue
                value={'image_sha256':aid,'asset_id':aid,'model_identity_sha256':identity_sha256,
                       'original_width':row['original_width'],'original_height':row['original_height'],
                       'status':'ok','scores':{}}
                if row['status']=='rejected_too_small':
                    value['status']='too_small'
                else:
                    value['pixel_sha256']=row['pixel_sha256']
                    for name in ('cf384','unifd'):
                        prob=row[name+'_prob']
                        if name in models and name not in errors.get('models',{}) and prob is not None and math.isfinite(prob) and 0<=prob<=1:
                            value['scores'][name]={'status':'ok','prob':prob}
                    offset=row['sscd_byte_offset'];binding=artifacts['sscd.f32']
                    if 'sscd' in models and 'sscd' not in errors.get('models',{}) and offset is not None:
                        require(str(Path(asset['sscd_file']).resolve())==binding['path'] and asset['sscd_file_sha256']==binding['sha256'] and asset['sscd_file_size_bytes']==binding['size_bytes'],
                                'Historical vector file binding differs from committed task')
                        require(type(offset) is int and offset>=0 and offset%2048==0 and offset+2048<=binding['size_bytes'],'Historical SSCD offset outside file')
                        vector_path=self._verify(binding)
                        with vector_path.open('rb') as stream:stream.seek(offset);data=stream.read(2048)
                        self._verify(binding)
                        require(len(data)==2048,'Historical vector truncated')
                        value['sscd']={'status':'ok','dimensions':512,'embedding':np.frombuffer(data,dtype='<f4').copy()}
                if value['status']=='ok' and not value['scores'] and 'sscd' not in value:
                    report['missing_model_evidence']+=1;continue
                entry=normalize_evidence(value,aid,identity_sha256)
                result[aid]=prediction(entry)
                report['complete_hits' if value['status']=='too_small' or set(models)<=available_models(entry[0]) else 'partial_hits']+=1
        report['not_found']=len(shas)-len(result)-report['identity_incompatible']-report['invalid_historical_evidence']-report['missing_model_evidence']
        self.last_report=dict(report);self.stats.update(report);return result
    def close(self):self.con.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--frontpass-root',required=True)
    parser.add_argument('--reconciliation-root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--decisions-manifest');parser.add_argument('--quality-overlay');parser.add_argument('--catalog-root')
    args=parser.parse_args();print(canonical(register_history(args.frontpass_root,args.reconciliation_root,args.output,
        decisions_manifest=args.decisions_manifest,quality_overlay=args.quality_overlay,catalog_root=args.catalog_root)))


if __name__=='__main__':main()
