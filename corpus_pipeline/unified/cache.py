"""Cross-run, exact-byte inference evidence cache; never a routing decision.

The complete resident-worker identity (including device, dtype, weights,
preprocessing and numerical settings) is part of the key. SQLite shards avoid
one file per image. Every returned payload is checksummed and validated.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager,closing
from copy import deepcopy
import fcntl
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import re
import sqlite3
import socket
import threading
import uuid

import numpy as np

from corpus_pipeline.postprocess.reconcile import atomic_json,canonical,digest,require,file_sha,stamp,artifact

SCHEMA='corpus.unified.image_evidence_cache.v1'
MODELS=('cf384','unifd','sscd')
HEX=re.compile(r'[0-9a-f]{64}\Z')
DECODE_FIELDS=('original_width','original_height','oriented_width','oriented_height','pixel_sha256')


class CacheCorruptionError(ValueError):
    pass


def asset_sha(asset):
    value=asset.get('sha256',asset.get('image_sha256',asset.get('asset_id')))
    require(isinstance(value,str) and HEX.fullmatch(value),'Expected exact encoded-image SHA256')
    return value


def valid_models(names):
    names=tuple(names)
    require(names and len(set(names))==len(names) and set(names)<=set(MODELS),'Invalid requested models')
    return names


def normalize_evidence(result,sha,identity_sha256):
    """Strip source/status policy, retain only independently valid components.

    Failed/incomplete live results must be retried, not promoted into a cache.
    Historical adapters may construct an ok result from successful individual
    stages, with missing models deliberately absent.
    """
    if result.get('status') not in ('ok','too_small'):return None
    require(result.get('image_sha256')==sha and result.get('model_identity_sha256')==identity_sha256,
            'Inference evidence identity mismatch')
    for name in ('original_width','original_height'):
        require(type(result.get(name)) is int and 0<result[name]<=100_000_000,'Invalid original image dimensions')
    require(result['original_width']*result['original_height']<=100_000_000,'Excessive original pixel count')
    payload={'schema':SCHEMA,'image_sha256':sha,'model_identity_sha256':identity_sha256,
             'status':result['status'],'scores':{},'original_width':result['original_width'],
             'original_height':result['original_height']}
    if result['status']=='too_small':
        require(min(result['original_width'],result['original_height'])<128,'False too-small evidence')
        payload.update(size_status='rejected_too_small',reason='original_image_short_edge_lt_128')
        return payload,None
    require(min(result['original_width'],result['original_height'])>=128,'Passing evidence below size floor')
    payload['size_status']='pass'
    for name in ('oriented_width','oriented_height'):
        if name in result:
            require(type(result[name]) is int and result[name]>0,'Invalid oriented image dimensions')
            payload[name]=result[name]
    if result.get('pixel_sha256') is not None:
        require(isinstance(result['pixel_sha256'],str) and HEX.fullmatch(result['pixel_sha256']),'Invalid pixel identity')
        payload['pixel_sha256']=result['pixel_sha256']
    for name in ('cf384','unifd'):
        score=result.get('scores',{}).get(name)
        if not score:continue
        require(score.get('status')=='ok','Failed detector evidence cannot be cached')
        probability=score.get('prob')
        require(type(probability) in (int,float) and math.isfinite(probability) and 0<=probability<=1,'Invalid detector probability')
        value={'status':'ok','prob':float(probability)}
        if score.get('logit') is not None:
            require(type(score['logit']) in (int,float) and math.isfinite(score['logit']),'Invalid detector logit')
            value['logit']=float(score['logit'])
        payload['scores'][name]=value
    vector=None
    if result.get('sscd') is not None:
        value=result['sscd'];require(value.get('status')=='ok','Failed SSCD evidence cannot be cached')
        array=np.asarray(value.get('embedding'),dtype='<f4')
        require(array.shape==(512,) and np.isfinite(array).all() and abs(float(np.linalg.norm(array))-1)<.001,
                'Invalid normalized 512-float SSCD vector')
        require(HEX.fullmatch(payload.get('pixel_sha256','')) is not None,'SSCD evidence lacks decoded pixel identity')
        vector=array.tobytes();payload['sscd']={'status':'ok','dimensions':512,'dtype':'<f4'}
    require(payload['scores'] or vector is not None,'No successful model evidence')
    return payload,vector


def available_models(payload):
    return set(payload.get('scores',{})) | ({'sscd'} if payload.get('sscd') else set())


def prediction(entry,models=None):
    payload,vector=entry;result=deepcopy(payload);result.pop('schema',None)
    sha=result['image_sha256'];result.update(asset_id=sha,sha256=sha)
    if result.get('sscd'):
        result['sscd']={'status':'ok','dimensions':512,'embedding':np.frombuffer(vector,dtype='<f4').copy()}
    if models is not None and result['status']!='too_small':
        wanted=set(models);result['scores']={k:v for k,v in result['scores'].items() if k in wanted}
        if 'sscd' not in wanted:result.pop('sscd',None)
        require(wanted<=available_models(payload),'Requested evidence is incomplete')
    return result


def checksum(payload,vector):
    return hashlib.sha256(canonical(payload).encode()+b'\0'+(vector or b'')).hexdigest()


def verification_stamp(path):
    value=Path(path).stat()
    return {'size':value.st_size,'mtime_ns':value.st_mtime_ns,'ctime_ns':value.st_ctime_ns,
            'inode':value.st_ino,'device':value.st_dev}


def unpack(row,sha,identity_sha256):
    if row is None:return None
    try:
        payload=json.loads(row[0]);vector=bytes(row[1]) if row[1] is not None else None
        require(payload.get('schema')==SCHEMA and checksum(payload,vector)==row[2],'Evidence checksum mismatch')
        normalized=normalize_evidence(prediction((payload,vector)),sha,identity_sha256)
        require(normalized is not None and normalized[0]==payload and normalized[1]==vector,'Evidence semantic mismatch')
        return payload,vector
    except (ValueError,TypeError,KeyError) as error:
        raise CacheCorruptionError(f'Corrupt evidence for {sha}: {error}') from error


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


class EvidenceCache:
    """Host-owned WAL shards; other hosts read only immutable sealed Parquet.

    No other host's live SQLite/WAL is opened. A moved worker may recompute
    evidence that was never sealed; this is safer than multi-host WAL access.
    """
    def __init__(self,root,identity,*,timeout=120,writer_id=None):
        self.identity=deepcopy(identity);self.identity_sha256=digest(identity)
        self.root=Path(root).resolve()/self.identity_sha256;self.timeout=timeout;self._verified_seals={}
        boot=Path('/proc/sys/kernel/random/boot_id')
        owner={'hostname':socket.gethostname(),'boot_id':boot.read_text().strip() if boot.exists() else 'unavailable'}
        self.writer_id=writer_id or digest(owner)[:24]
        require(re.fullmatch('[a-zA-Z0-9_-]{1,100}',self.writer_id),'Invalid writer namespace')
        self.writer_root=self.root/'writers'/self.writer_id
        with file_lock(self.root/'.identity.lock'):
            marker=self.root/'IDENTITY.json';value={'schema':SCHEMA,'model_identity_sha256':self.identity_sha256,'identity':identity}
            if marker.exists():require(json.loads(marker.read_text())==value,'Cache model identity changed')
            else:atomic_json(marker,value)
        with file_lock(self.writer_root/'.owner.lock'):
            marker=self.writer_root/'OWNER.json'
            if marker.exists():require(json.loads(marker.read_text())==owner,'Writer SQLite namespace belongs to another host')
            else:atomic_json(marker,owner)
    def path(self,sha):return self.writer_root/(sha[:2]+'.sqlite')
    def _open(self,path,write=False):
        con=sqlite3.connect(str(path) if write else path.as_uri()+'?mode=ro',uri=not write,timeout=self.timeout)
        if write:
            con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=FULL')
            con.execute('CREATE TABLE IF NOT EXISTS evidence(sha TEXT PRIMARY KEY,payload TEXT NOT NULL,vector BLOB,checksum TEXT NOT NULL)')
        return con
    def get_many(self,shas,*,models=None):
        wanted=set(valid_models(MODELS if models is None else models))
        groups={};result={}
        for sha in sorted(set(shas)):
            require(isinstance(sha,str) and HEX.fullmatch(sha),'Invalid cache asset key')
            groups.setdefault(sha[:2],[]).append(sha)
        for values in groups.values():
            path=self.path(values[0])
            if not path.exists():continue
            with file_lock(self.writer_root/(values[0][:2]+'.lock')),closing(self._open(path)) as con:
                for begin in range(0,len(values),512):
                    part=values[begin:begin+512]
                    for row in con.execute('SELECT sha,payload,vector,checksum FROM evidence WHERE sha IN ('+','.join('?' for _ in part)+')',part):
                        result[row[0]]=unpack(row[1:],row[0],self.identity_sha256)
        # Sealed bundles are immutable and are the only cross-host cache path.
        import pyarrow.parquet as pq
        for complete in sorted((self.root/'sealed').glob('*/COMPLETE.json')):
            value=json.loads(complete.read_text())
            require(value.get('complete') is True and value['model_identity_sha256']==self.identity_sha256,'Invalid sealed cache identity')
            for item in value['files']:
                if item['bucket'] not in groups:continue
                missing=[sha for sha in groups[item['bucket']] if sha not in result or
                         (result[sha][0]['status']!='too_small' and not wanted<=available_models(result[sha][0]))]
                if not missing:continue
                path=complete.parent/item['path'];require(path.resolve().is_relative_to(complete.parent),'Sealed cache path escapes bundle')
                current=verification_stamp(path);bound=(item['sha256'],item['size_bytes'],current)
                if self._verified_seals.get(str(path))!=bound:
                    require(current['size']==item['size_bytes'] and file_sha(path)==item['sha256'] and verification_stamp(path)==current,'Sealed cache artifact changed')
                    self._verified_seals[str(path)]=bound
                table=pq.read_table(path,filters=[('sha','in',missing)])
                for row in table.to_pylist():
                    new=unpack((row['payload'],row['vector'],row['checksum']),row['sha'],self.identity_sha256)
                    result[row['sha']]=self._merge(result.get(row['sha']),new)
        return result
    @staticmethod
    def _merge(old,new):
        if old is None:return new
        for key in ('status',*DECODE_FIELDS):
            if key in old[0] and key in new[0]:require(old[0][key]==new[0][key],'Conflicting decoded evidence for identical bytes')
        merged=deepcopy(old[0]);merged.update({k:v for k,v in new[0].items() if k not in merged})
        merged['scores'].update({k:v for k,v in new[0]['scores'].items() if k not in merged['scores']})
        if not old[0].get('sscd') and new[0].get('sscd'):merged['sscd']=new[0]['sscd']
        return merged,old[1] if old[1] is not None else new[1]
    def put_many(self,results):
        """One bounded transaction per SHA bucket, never one connection per image."""
        results=list(islice(iter(results),4097));require(len(results)<=4096,'Cache write exceeds bounded batch size')
        groups={};accepted=[]
        for result in results:
            sha=asset_sha(result);new=normalize_evidence(result,sha,self.identity_sha256);accepted.append(new is not None)
            if new is not None:groups.setdefault(sha[:2],[]).append((sha,new))
        for prefix,rows in groups.items():
          with file_lock(self.writer_root/(prefix+'.lock')):
            con=self._open(self.path(rows[0][0]),True)
            try:
                con.execute('BEGIN IMMEDIATE')
                # A single query loads existing rows; repeats within this write merge in order.
                old={}
                for begin in range(0,len(rows),512):
                    shas=list(dict.fromkeys(sha for sha,_ in rows[begin:begin+512]))
                    for row in con.execute('SELECT sha,payload,vector,checksum FROM evidence WHERE sha IN ('+','.join('?' for _ in shas)+')',shas):
                        old[row[0]]=unpack(row[1:],row[0],self.identity_sha256)
                for sha,new in rows:old[sha]=self._merge(old.get(sha),new)
                con.executemany('INSERT OR REPLACE INTO evidence VALUES(?,?,?,?)',
                                [(sha,canonical(entry[0]),entry[1],checksum(*entry)) for sha,entry in old.items()])
                con.commit()
            except BaseException:con.rollback();raise
            finally:con.close()
        return accepted
    def put(self,result):return self.put_many([result])[0]
    def seal(self):
        """Commit a bounded-streaming snapshot for read-only reuse on any host."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        root=self.root/'sealed'/uuid.uuid4().hex;root.mkdir(parents=True)
        schema=pa.schema([('sha',pa.string()),('payload',pa.string()),('vector',pa.binary()),('checksum',pa.string())])
        files=[];total=0
        for source in sorted(self.writer_root.glob('??.sqlite')):
            target=root/(source.stem+'.parquet');count=0
            with file_lock(self.writer_root/(source.stem+'.lock')),closing(self._open(source)) as con,pq.ParquetWriter(target,schema,compression='zstd') as writer:
                cursor=con.execute('SELECT sha,payload,vector,checksum FROM evidence ORDER BY sha')
                while rows:=cursor.fetchmany(1024):
                    values=[]
                    for row in rows:
                        unpack(row[1:],row[0],self.identity_sha256)
                        values.append(dict(zip(schema.names,row)))
                    writer.write_table(pa.Table.from_pylist(values,schema=schema));count+=len(values)
            total+=count;files.append({**artifact(target,root),'bucket':source.stem,'rows':count})
        value={'schema':SCHEMA,'complete':True,'model_identity_sha256':self.identity_sha256,
               'writer_id':self.writer_id,'records':total,'files':files}
        atomic_json(root/'COMPLETE.json',value);return {'path':str(root/'COMPLETE.json'),'sha256':file_sha(root/'COMPLETE.json'),**value}


class CachedInferenceWorker:
    """Single-consumer wrapper; multi-process workers share transactional cache.

    On hits, verify_input_bytes=True checks the current encoded bytes without
    decoding. False is only for an upstream ingestion contract that already
    verified the SHA; it never changes the exact-SHA cache key.
    """
    def __init__(self,delegate,cache_dir,*,histories=(),verify_input_bytes=True,lookup_window=None):
        self.delegate=delegate;self.identity=deepcopy(delegate.identity)
        self.identity_sha256=delegate.identity_sha256
        require(digest(self.identity)==self.identity_sha256,'Delegate model identity checksum mismatch')
        self.batch_size=delegate.batch_size;self.names=tuple(delegate.names)
        self.lookup_window=min(4096,4*self.batch_size) if lookup_window is None else lookup_window
        require(type(self.lookup_window) is int and self.batch_size<=self.lookup_window<=4096,'lookup_window must cover one batch and be at most 4096')
        self.cache=EvidenceCache(cache_dir,self.identity);self.verify_input_bytes=verify_input_bytes
        from .history import HistoryIndex
        self.histories=[];self._owned_histories=[]
        for item in histories:
            history=HistoryIndex(item) if isinstance(item,(str,Path)) else item
            self.histories.append(history)
            if isinstance(item,(str,Path)):self._owned_histories.append(history)
        self.cache_stats=Counter();self.stats=getattr(delegate,'stats',{})
        self.execution_identity={**getattr(delegate,'execution_identity',{}),'evidence_cache':SCHEMA,
                                 'strict_model_identity':True,'verify_input_bytes_on_hits':verify_input_bytes,
                                 'cache_lookup_window':self.lookup_window}
        self._active=threading.Lock();self.closed=False
    def __getattr__(self,name):return getattr(self.delegate,name)
    def _count(self,key,count=1):
        self.cache_stats[key]+=count;self.stats['evidence_cache_'+key]=self.cache_stats[key]
    def _check_current(self,asset,entry):
        payload=entry[0]
        for name in ('width','height'):
            declared=asset.get('original_'+name,asset.get(name))
            if declared is not None and declared!=payload['original_'+name]:
                return {'asset_id':asset_sha(asset),'image_sha256':asset_sha(asset),'model_identity_sha256':self.identity_sha256,
                        'status':'error','scores':{},'error_stage':'decode','error_type':'ValueError','error':'declared original image dimensions mismatch'}
        if self.verify_input_bytes:
            try:
                from corpus_pipeline.cluster.inference import _read_blob
                _read_blob(asset);self._count('encoded_bytes_verified')
            except Exception as error:
                return {'asset_id':asset_sha(asset),'image_sha256':asset_sha(asset),'model_identity_sha256':self.identity_sha256,
                        'status':'error','scores':{},'error_stage':'decode','error_type':type(error).__name__,'error':str(error)[:400]}
        return None
    def _batch(self,assets,names):
        # I2T's explicit SSCD-only decode admits small images. Legacy evidence
        # uses a 128px floor and a SHA-only cache key, so its too-small result
        # cannot be reused or merged into this separate request contract.
        bypass,ordinary=[],[]
        for index,asset in enumerate(assets):
            allowed=asset.get('allow_small_for_dedup',False)
            require(type(allowed) is bool,'allow_small_for_dedup must be a boolean')
            if allowed:
                require(names==('sscd',),'Small-image dedup bypass requires SSCD-only inference')
                asset_sha(asset)
                bypass.append((index,asset))
            else:ordinary.append((index,asset))
        if not bypass:return self._cached_batch(assets,names)
        output=[None]*len(assets)
        if ordinary:
            values=self._cached_batch([asset for _,asset in ordinary],names)
            for (index,_),value in zip(ordinary,values):output[index]=value
        items=[asset for _,asset in bypass]
        values=list(self.delegate.iter_results(items,models=names))
        require(len(values)==len(items) and [value.get('image_sha256') for value in values]==
                [asset_sha(asset) for asset in items],'Delegate returned incomplete or reordered results')
        self._count('dedup_only_bypass_images',len(items));self._count('dedup_only_bypass_calls')
        for (index,_),value in zip(bypass,values):output[index]=value
        return output
    def _cached_batch(self,assets,names):
        shas=[asset_sha(a) for a in assets];unique=list(dict.fromkeys(shas));found=self.cache.get_many(unique,models=names)
        for history in self.histories:
            missing=[sha for sha in unique if sha not in found or (found[sha][0]['status']!='too_small' and not set(names)<=available_models(found[sha][0]))]
            if not missing:break
            historical=history.lookup_many(missing,self.identity_sha256,models=names)
            require(set(historical)<=set(missing),'History returned an unrequested asset')
            accepted=self.cache.put_many(historical.values());self._count('historical_assets',sum(accepted))
            found.update(self.cache.get_many(missing,models=names))
        results={};groups={};representative={}
        for asset,sha in zip(assets,shas):representative.setdefault(sha,asset)
        for sha in unique:
            entry=found.get(sha);asset=representative[sha]
            if entry is not None:
                error=self._check_current(asset,entry)
                if error is not None:results[sha]=error;self._count('input_integrity_failures');continue
                missing=tuple(name for name in names if name not in available_models(entry[0]))
                if entry[0]['status']=='too_small' or not missing:
                    results[sha]=prediction(entry,names);self._count('hits');continue
            else:missing=names
            groups.setdefault(missing,[]).append(asset)
        for missing,items in groups.items():
            self._count('miss_assets',len(items));self._count('delegate_calls')
            values=list(self.delegate.iter_results(items,models=missing))
            require(len(values)==len(items) and [p.get('image_sha256') for p in values]==[asset_sha(a) for a in items],
                    'Delegate returned incomplete or reordered results')
            accepted=self.cache.put_many(values)
            refreshed=self.cache.get_many([asset_sha(a) for a,ok in zip(items,accepted) if ok],models=names)
            found.update(refreshed)
            for asset,result,ok in zip(items,values,accepted):
                sha=asset_sha(asset)
                if ok:
                    results[sha]=prediction(refreshed[sha],names)
                else:
                    # Preserve the actual failure; do not turn a cached SSCD stage into a detector pass.
                    results[sha]=result;self._count('uncached_failures')
        output=[]
        for asset,sha in zip(assets,shas):
            value=deepcopy(results[sha])
            # Repeated declared SHA may refer to different occurrences/bytes; validate every cache hit occurrence.
            if asset is not representative[sha] and sha in found:
                error=self._check_current(asset,found[sha])
                if error is not None:value=error;self._count('input_integrity_failures')
            output.append(value)
        return output
    def iter_batches(self,assets,*,models=None):
        require(not self.closed,'Worker is closed')
        names=valid_models(self.names if models is None else models)
        require(set(names)<=set(self.names),'Requested models are not resident')
        if not self._active.acquire(blocking=False):raise RuntimeError('inference worker is single-consumer')
        try:
            iterator=iter(assets)
            # Pass multiple batches to the delegate so its CPU decode/prefetch
            # can overlap accelerator inference. Preserve the public batch size.
            while window:=list(islice(iterator,self.lookup_window)):
                values=self._batch(window,names)
                for begin in range(0,len(window),self.batch_size):
                    yield window[begin:begin+self.batch_size],values[begin:begin+self.batch_size]
        finally:self._active.release()
    def iter_results(self,assets,*,models=None):
        stream=self.iter_batches(assets,models=models)
        try:
            for _,values in stream:yield from values
        finally:stream.close()
    def infer_batch(self,assets,*,models=None):
        batch=list(islice(iter(assets),self.batch_size+1));require(len(batch)<=self.batch_size,'infer_batch exceeds configured batch_size')
        return list(self.iter_results(batch,models=models))
    def close(self):
        if not self.closed:
            try:self.delegate.close()
            finally:
                for history in self._owned_histories:history.close()
                self.closed=True
    def seal_cache(self):return self.cache.seal()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
