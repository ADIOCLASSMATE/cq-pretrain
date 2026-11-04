"""Recover DenseFusion-4V captions by exact PNG bytes from pinned ZIP ranges.

Only annotations, fingerprints and verification receipts are persisted. Official
image bytes are decompressed in memory and must equal a mounted PNG SHA256.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import struct
import threading
import time
import zipfile
import zlib

from corpus_pipeline.production.common import atomic_json, canonical, file_sha
from .pixmo_recovery import _claim, stamp

CONTRACT='genund.dense4v_exact_png_range_binding.v1'
REVISION='b935d5ad34311e351bc2f3eb33667451e331e167'
ANNOTATION_SHA='2022f14f9dfeb4b5db8f156f60b72e2c609b7ec7b4be5186c29080e37cf99476'
PREFIX='DenseFusion-4V-100K'
MAX_MEMBER=64<<20


class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,size):
        import requests
        self.url,self.size,self.position=url,size,0
        self.session=requests.Session();self.session.trust_env=False
        self.transferred=0
        self.resolved_url=None

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if not 0<=target<=self.size:raise ValueError('ZIP seek outside original file')
        self.position=target;return target

    def read_at(self,offset,length):
        length=min(length,self.size-offset)
        if not 0<=offset<=self.size or not 0<=length<=MAX_MEMBER:raise ValueError('ZIP range byte budget')
        if length==0:return b''
        end=offset+length-1
        for attempt in range(4):
            try:
                # A CDN URL authorizes the pinned immutable object. Reuse it for
                # different exact ranges, refreshing only when it expires.
                with self.session.get(self.resolved_url or self.url,
                    params=None if self.resolved_url else {'download':'true','range_start':offset,'range_end':end},
                    headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(8,25)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{self.size}':
                        raise ValueError('official ZIP server did not honor exact byte range')
                    blocks=[];count=0
                    for block in response.iter_content(1<<18):
                        count+=len(block)
                        if count>length:raise ValueError('ZIP range exceeded requested bytes')
                        blocks.append(block)
                    if count!=length:raise ValueError('truncated ZIP range')
                    self.resolved_url=response.url
                    self.transferred+=count;return b''.join(blocks)
            except Exception:
                self.resolved_url=None
                if attempt==3:raise
                time.sleep(attempt+1)

    def read(self,length=-1):
        if length<0:length=self.size-self.position
        value=self.read_at(self.position,length);self.position+=len(value);return value

    def close(self):
        self.session.close();super().close()


def read_member(reader,info):
    """Validate ZIP headers, inflate, CRC and length; return bytes only in RAM."""
    if info.flag_bits&1 or info.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
        raise ValueError('encrypted or unsupported ZIP member')
    if not 0<info.file_size<=MAX_MEMBER or not 0<info.compress_size<=MAX_MEMBER-65536:
        raise ValueError('ZIP member byte budget')
    initial=30+len(info.filename.encode('utf8'))+len(info.extra)+info.compress_size
    blob=reader.read_at(info.header_offset,initial)
    if len(blob)<30:raise ValueError('truncated ZIP local header')
    signature,version,flags,method,mtime,mdate,crc,compressed,size,name_len,extra_len=struct.unpack('<4s5H3I2H',blob[:30])
    if signature!=b'PK\x03\x04' or flags&1 or method!=info.compress_type:
        raise ValueError('ZIP local and central headers disagree')
    header=30+name_len+extra_len;required=header+info.compress_size
    if required>len(blob):blob+=reader.read_at(info.header_offset+len(blob),required-len(blob))
    name=blob[30:30+name_len].decode('utf8' if flags&0x800 else 'cp437')
    if name!=info.filename:raise ValueError('ZIP member filename differs from central directory')
    data=blob[header:required]
    if info.compress_type==zipfile.ZIP_DEFLATED:
        inflater=zlib.decompressobj(-15);data=inflater.decompress(data,info.file_size+1)
        if not inflater.eof or inflater.unconsumed_tail:raise ValueError('ZIP member inflation exceeds declared size')
    if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:
        raise ValueError('ZIP original PNG length or CRC mismatch')
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('official member is not PNG')
    return data,{'header_offset':info.header_offset,'compressed_size':info.compress_size,'size':info.file_size,
                 'method':info.compress_type,'crc32':info.CRC,'requested_range':[info.header_offset,info.header_offset+required-1]}


def verify_local_candidates(root,candidates,original):
    expected=hashlib.sha256(original).hexdigest();matches=[]
    for relative in candidates:
        path=(Path(root)/relative).resolve(strict=True)
        if Path(root).resolve() not in path.parents:raise ValueError('candidate image escapes mounted root')
        before=stamp(path)
        if before['size']!=len(original):continue
        actual=file_sha(path)
        if before!=stamp(path):raise ValueError('mounted candidate changed during SHA verification')
        if actual==expected:matches.append(relative)
    return expected,sorted(matches)


def _zip_worker(spec,rows,candidates,root,output,state):
    shard=Path(spec['rfilename']).stem;directory=output/shard;directory.mkdir(exist_ok=True)
    receipt=directory/'COMPLETE.json';database=directory/'results.sqlite'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if file_sha(database)!=value['sha256']:raise ValueError('completed Dense4V result shard changed')
        state[shard]={'status':'complete','counts':value['counts'],'completed':value['rows']};return value
    url=f"https://huggingface.co/datasets/BAAI/DenseFusion-1M/resolve/{REVISION}/{spec['rfilename']}"
    reader=HTTPRangeFile(url,spec['size']);db=sqlite3.connect(database)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS bindings (original_path TEXT PRIMARY KEY,status TEXT,result_json TEXT)')
        done={row[0] for row in db.execute('SELECT original_path FROM bindings')}
        counts=Counter(dict(db.execute('SELECT status,count(*) FROM bindings GROUP BY status')))
        state[shard]={'status':'running','completed':len(done),'total':len(rows),'counts':dict(counts)}
        with zipfile.ZipFile(reader) as archive:
            members={}
            for info in archive.infolist():
                if info.is_dir():continue
                name=PurePosixPath(info.filename)
                if name.is_absolute() or '..' in name.parts or name.suffix!='.png':raise ValueError('unexpected original ZIP member path')
                if name.stem in members:raise ValueError('ambiguous original image ID within ZIP')
                members[name.stem]=info
        atomic_json(directory/'ZIP_INDEX.json',{'revision':REVISION,'archive':spec,'members':len(members),
                    'central_directory_obtained_by_http_range':True,'full_archive_sha_validated':False})
        for row in rows:
            original_path=row['image_path']
            if original_path in done:continue
            result={'original_path':original_path,'image_id':row['image_id'],'status':'unbound'}
            proposed=candidates.get(row['image_id'],[])
            try:
                if not proposed:result['reason']='no_local_image_id_candidate'
                elif row['image_id'] not in members:result['reason']='official_zip_member_missing'
                else:
                    info=members[row['image_id']];data,evidence=read_member(reader,info)
                    sha,matches=verify_local_candidates(root,proposed,data)
                    if matches:
                        zip_evidence={**evidence,'zip_compression':evidence['method']}
                        del zip_evidence['method']
                        result.update(status='bound',relative_path=matches[0],local_sha256=sha,equivalent_local_paths=matches,
                            proof={'method':'official_png_identical_encoded_sha256','original_sha256':sha,
                                   'revision':REVISION,'archive_path':spec['rfilename'],'archive_lfs_sha256':spec['lfs']['sha256'],
                                   'full_archive_sha_validated':False,'member':info.filename,**zip_evidence})
                    else:result.update(reason='original_png_sha_has_no_exact_local_candidate',original_sha256=sha,candidate_paths=proposed)
            except Exception as error:
                result.update(reason='official_image_verification_failed',error_type=type(error).__name__,error=str(error)[:500])
            db.execute('INSERT INTO bindings VALUES (?,?,?)',(original_path,result['status'],canonical(result)));db.commit()
            counts[result['status']]+=1;state[shard]={'status':'running','completed':sum(counts.values()),'total':len(rows),'counts':dict(counts),'network_bytes':reader.transferred}
        db.close();db=None
        value={'status':'complete','rows':len(rows),'counts':dict(counts),'path':str(database),'sha256':file_sha(database),
               'archive':spec,'network_bytes':reader.transferred}
        atomic_json(receipt,value);state[shard]={**state[shard],'status':'complete'};return value
    finally:
        reader.close()
        if db is not None:db.close()


def run(root,annotations,api_metadata,output,threads=24,previous_implementation_sha=None):
    root,annotations,api_metadata,output=map(Path,(root,annotations,api_metadata,output))
    root=root.resolve(strict=True);output=output.resolve()
    if root==output or root in output.parents:raise ValueError('recovery output must be outside source root')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'COMPLETE.json').exists():return
    claim=_claim(output/'.claim');started=time.time();stop=threading.Event();state={};status={'status':'running','phase':'loading_annotations','pid':os.getpid()}
    def heartbeat():
        while not stop.is_set():
            value={**status,'updated_unix':time.time(),'elapsed_seconds':time.time()-started,
                   'shards':dict(state),'completed_images':sum(x.get('completed',0) for x in list(state.values()))}
            atomic_json(output/'HEARTBEAT.json',value);atomic_json(claim/'HEARTBEAT.json',value);stop.wait(15)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        if file_sha(annotations)!=ANNOTATION_SHA:raise ValueError('Dense4V official annotation SHA mismatch')
        api=json.loads(api_metadata.read_text())
        if api['sha']!=REVISION:raise ValueError('Dense4V API revision mismatch')
        specs={Path(s['rfilename']).stem:s for s in api['siblings'] if s['rfilename'].startswith('images/'+PREFIX+'/') and s['rfilename'].endswith('.zip')}
        groups=defaultdict(list);wanted=set()
        with annotations.open() as stream:
            for line in stream:
                row=json.loads(line);parts=PurePosixPath(row['image_path']).parts
                if len(parts)!=3 or parts[0]!=PREFIX or parts[1] not in specs or Path(parts[2]).stem!=row['image_id']:
                    raise ValueError('unregistered Dense4V annotation image path')
                groups[parts[1]].append({'image_id':row['image_id'],'image_path':row['image_path']});wanted.add(row['image_id'])
        candidates=defaultdict(list);directory_stamps={};status['phase']='indexing_id_candidates'
        for directory in sorted((root/'DenseFusion-1M').iterdir()):
            if not directory.is_dir():raise ValueError('unexpected local Dense root member')
            before=stamp(directory);directory_stamps[directory.name]=before
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.endswith('.png'):raise ValueError('unexpected local Dense image extension')
                    key=entry.name[:-4]
                    if key in wanted:candidates[key].append('DenseFusion-1M/'+directory.name+'/'+entry.name)
            if before!=stamp(directory):raise ValueError('local image directory changed during candidate inventory')
        identity={'contract':CONTRACT,'root':str(root),'annotation_sha256':ANNOTATION_SHA,'api_sha256':file_sha(api_metadata),
                  'implementation_sha256':file_sha(__file__),'directories':directory_stamps,'revision':REVISION,
                  'annotation_rows':sum(map(len,groups.values())),'distinct_image_ids':len(wanted),'candidate_image_ids':len(candidates)}
        identity_path=output/'IDENTITY.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            previous=json.loads(identity_path.read_text())
            if not previous_implementation_sha or previous.get('implementation_sha256')!=previous_implementation_sha or {
                k:v for k,v in previous.items() if k!='implementation_sha256'}!={k:v for k,v in identity.items() if k!='implementation_sha256'}:
                raise ValueError('Dense4V recovery identity changed')
            atomic_json(output/('CODE_MIGRATION_'+identity['implementation_sha256']+'.json'),{
                'previous_identity':previous,'current_identity':identity,
                'reason':'reuse same pinned CDN object across exact ranges; PNG length/CRC/SHA verification unchanged'})
        atomic_json(identity_path,identity);status['phase']='verifying_official_png_ranges'
        with ThreadPoolExecutor(threads) as pool:
            futures=[pool.submit(_zip_worker,specs[shard],rows,candidates,root,output,state) for shard,rows in sorted(groups.items())]
            results=[future.result() for future in as_completed(futures)]
        target=output/'bindings.sqlite';temporary=target.with_suffix('.building');temporary.unlink(missing_ok=True)
        db=sqlite3.connect(temporary);db.execute('CREATE TABLE bindings (original_path TEXT PRIMARY KEY,status TEXT,result_json TEXT)');counts=Counter()
        for result in results:
            part=sqlite3.connect('file:'+result['path']+'?mode=ro',uri=True)
            try:
                for row in part.execute('SELECT original_path,status,result_json FROM bindings'):
                    db.execute('INSERT INTO bindings VALUES (?,?,?)',row);counts[row[1]]+=1
            finally:part.close()
        db.commit();db.close();os.replace(temporary,target)
        receipt={'status':'complete','contract':CONTRACT,'path':str(target),'sha256':file_sha(target),'fingerprint':stamp(target),
                 'counts':dict(counts),'annotation_rows':identity['annotation_rows'],'identity':identity,'shards':results,
                 'image_payloads_persisted':False,'source_files_modified':False}
        atomic_json(output/'COMPLETE.json',receipt);status.update(status='complete',phase='complete')
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error));atomic_json(output/'FAILED.json',status);raise
    finally:
        stop.set();thread.join();atomic_json(output/'HEARTBEAT.json',{**status,'updated_unix':time.time(),'shards':dict(state)});shutil.rmtree(claim)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','annotations','api_metadata','output'):parser.add_argument('--'+name.replace('_','-'),required=True)
    parser.add_argument('--threads',type=int,default=24);parser.add_argument('--previous-implementation-sha')
    run(**vars(parser.parse_args()))

if __name__=='__main__':main()
