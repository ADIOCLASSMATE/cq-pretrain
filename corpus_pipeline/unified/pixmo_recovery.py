"""Exact, resumable binding of re-encoded mounted PixMo images to official QA.

Source URL bytes are transient and must match the official SHA256. A binding
requires identical decoded pixels or an exactly reproduced JPEG byte stream;
filenames, row positions and approximate image similarity are never evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import threading
import time
from urllib.parse import urlsplit
import uuid

from PIL import Image, ImageOps, JpegImagePlugin, __version__ as PIL_VERSION
from corpus_pipeline.production.common import atomic_json, canonical, digest, file_sha

SOURCE = 'pixmo-ask-model-anything'
CONTRACT = 'genund.pixmo_verified_reencoding.v1'
MAX_BYTES = 32 << 20
MAX_PIXELS = 100_000_000


def stamp(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def pixels(image):
    image = ImageOps.exif_transpose(image)
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise ValueError('image exceeds pixel budget')
    return hashlib.sha256(f'RGBA:{width}:{height}\0'.encode() + image.convert('RGBA').tobytes()).hexdigest()


def describe(blob):
    with Image.open(io.BytesIO(blob)) as image:
        if image.width * image.height > MAX_PIXELS or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('unsupported image geometry/frame count')
        pixel_sha = pixels(image)
        profile = None
        if image.format == 'JPEG':
            profile = {'qtables': image.quantization, 'subsampling': JpegImagePlugin.get_sampling(image),
                       'progressive': bool(image.info.get('progressive'))}
        return {'local_sha256': hashlib.sha256(blob).hexdigest(), 'pixel_sha256': pixel_sha,
                'width': image.width, 'height': image.height, 'bytes': len(blob),
                'profile_json': canonical(profile) if profile else None}


def _index_directory(root, directory, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    receipt = output / (directory.name + '.COMPLETE.json')
    parquet = output / (directory.name + '.parquet')
    before = stamp(directory)
    if receipt.exists():
        value = json.loads(receipt.read_text())
        if value['source_fingerprint'] != before or file_sha(parquet) != value['sha256']:
            raise ValueError('completed PixMo image directory changed')
        return value
    rows, cache, errors = [], {}, []
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if path.suffix != '.jpg' or path.is_symlink():
            raise ValueError('unregistered PixMo image member')
        relative = path.relative_to(root).as_posix()
        try:
            first = stamp(path)
            if not 0 < first['size'] <= MAX_BYTES:
                raise ValueError('image byte budget')
            blob = path.read_bytes()
            if first != stamp(path):
                raise ValueError('mounted image changed during indexing')
            sha = hashlib.sha256(blob).hexdigest()
            info = cache.get(sha)
            if info is None:
                info = describe(blob)
                if len(cache) >= 512:
                    cache.clear()
                cache[sha] = info
            rows.append({**info, 'relative_path': relative})
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({'relative_path': relative, 'error_type': type(error).__name__, 'error': str(error)})
    if before != stamp(directory):
        raise ValueError('PixMo image directory changed during indexing')
    schema = pa.schema([('local_sha256',pa.string()),('pixel_sha256',pa.string()),('width',pa.int64()),
                        ('height',pa.int64()),('bytes',pa.int64()),('profile_json',pa.string()),('relative_path',pa.string())])
    temporary = parquet.with_suffix('.partial')
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),temporary,compression='zstd')
    os.replace(temporary,parquet)
    result = {'complete': True, 'source_fingerprint': before, 'rows': len(rows), 'invalid_images': errors,
              'sha256': file_sha(parquet), 'path': str(parquet)}
    atomic_json(receipt,result)
    return result


def _merge_index(output, directories):
    import pyarrow.parquet as pq
    target = output / 'images.sqlite'; receipt = output / 'IMAGE_INDEX_COMPLETE.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    if not all((output/'image_index'/(directory.name+'.COMPLETE.json')).exists() for directory in directories):
        return None
    claim = output / '.merge-index.claim'
    try: _claim(claim)
    except RuntimeError: return None
    try:
        temporary = target.with_suffix('.building'); temporary.unlink(missing_ok=True)
        db = sqlite3.connect(temporary)
        db.execute('CREATE TABLE images (local_sha256 TEXT,pixel_sha256 TEXT,relative_path TEXT,width INTEGER,height INTEGER,profile_json TEXT)')
        rows = errors = 0
        for directory in directories:
            item = json.loads((output/'image_index'/(directory.name+'.COMPLETE.json')).read_text())
            if file_sha(item['path']) != item['sha256']:
                raise ValueError('PixMo image index shard changed')
            values = pq.read_table(item['path']).to_pylist()
            db.executemany('INSERT INTO images VALUES (?,?,?,?,?,?)',[(v['local_sha256'],v['pixel_sha256'],v['relative_path'],v['width'],v['height'],v['profile_json']) for v in values])
            rows += len(values); errors += len(item['invalid_images']); db.commit()
        db.execute('CREATE INDEX encoded ON images(local_sha256)'); db.execute('CREATE INDEX decoded ON images(pixel_sha256)'); db.execute('CREATE INDEX geometry ON images(width,height)')
        profiles = [json.loads(row[0]) for row in db.execute('SELECT profile_json,count(*) n FROM images WHERE profile_json IS NOT NULL GROUP BY profile_json ORDER BY n DESC LIMIT 8')]
        distinct = db.execute('SELECT count(DISTINCT local_sha256) FROM images').fetchone()[0]
        db.commit();db.close();os.replace(temporary,target)
        result = {'complete':True,'contract':CONTRACT,'path':str(target),'sha256':file_sha(target),'rows':rows,
                  'distinct_encoded_images':distinct,'invalid_images':errors,'profiles':profiles,'pil_version':PIL_VERSION}
        atomic_json(receipt,result);return result
    finally:
        shutil.rmtree(claim)


def _lookup(db, column, value):
    rows = db.execute('SELECT local_sha256,relative_path,pixel_sha256 FROM images WHERE '+column+'=? ORDER BY relative_path',(value,)).fetchall()
    if not rows:
        return None
    return {'local_sha256':rows[0][0],'relative_path':rows[0][1],'pixel_sha256':rows[0][2],
            'equivalent_local_paths':[row[1] for row in rows]}


def verify_blob(blob, expected_sha, db, profiles):
    """Pure verifier: never binds a source image with a changed upstream hash."""
    observed = hashlib.sha256(blob).hexdigest()
    if observed != expected_sha:
        return {'status':'source_changed','observed_source_sha256':observed}
    match = _lookup(db,'local_sha256',expected_sha)
    if match:
        return {'status':'bound',**match,'proof':{'method':'identical_encoded_sha256','source_sha256':expected_sha}}
    with Image.open(io.BytesIO(blob)) as source:
        if source.width*source.height>MAX_PIXELS or getattr(source,'n_frames',1)!=1:
            raise ValueError('unsupported original image geometry/frame count')
        pixel_sha = pixels(source)
        match = _lookup(db,'pixel_sha256',pixel_sha)
        if match:
            return {'status':'bound',**match,'proof':{'method':'identical_oriented_rgba_geometry_sha256','source_sha256':expected_sha,'pixel_sha256':pixel_sha}}
        seen = set()
        for transpose in ((False,True) if source.getexif().get(274,1)!=1 else (False,)):
            rgb = (ImageOps.exif_transpose(source) if transpose else source).convert('RGB')
            # Exact byte comparisons make these hypotheses safe: an unsuccessful
            # hypothesis is never treated as an approximate match.
            for bound in (None,2048,1024,512):
                if bound is not None and max(rgb.size)<=bound:
                    continue
                for resampling in ([None] if bound is None else ['LANCZOS','BICUBIC','BILINEAR']):
                    image = rgb.copy()
                    if bound:
                        image.thumbnail((bound,bound),getattr(Image.Resampling,resampling))
                    key=(transpose,image.size,resampling)
                    if key in seen:continue
                    seen.add(key)
                    if db.execute('SELECT 1 FROM images WHERE width=? AND height=? LIMIT 1',image.size).fetchone() is None:
                        continue
                    for profile in profiles:
                        if profile['subsampling'] not in (0,1,2):continue
                        qtables={int(k):v for k,v in profile['qtables'].items()}
                        for optimize in (False,True):
                            encoded=io.BytesIO();image.save(encoded,format='JPEG',qtables=qtables,subsampling=profile['subsampling'],
                                                         progressive=profile['progressive'],optimize=optimize)
                            checksum=hashlib.sha256(encoded.getvalue()).hexdigest()
                            match=_lookup(db,'local_sha256',checksum)
                            if match:
                                proof={'method':'exact_reencoded_sha256','source_sha256':expected_sha,'local_sha256':checksum,
                                       'exif_transpose':transpose,'max_side':bound,'resampling':resampling,'convert':'RGB',
                                       'jpeg_profile':profile,'optimize':optimize,'pil_version':PIL_VERSION}
                                return {'status':'bound',**match,'proof':proof}
    return {'status':'unbound','reason':'no_exact_pixel_or_reencoded_match','source_sha256':expected_sha}


def _public_url(url):
    parsed=urlsplit(url)
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('unsupported original image URL')
    host=parsed.hostname.lower()
    if host=='localhost' or host.endswith(('.localhost','.local','.internal')):
        raise ValueError('non-public original image URL')
    try:
        if not ipaddress.ip_address(host).is_global:raise ValueError('non-public original image address')
    except ValueError as error:
        if 'non-public' in str(error):raise
    return url


def fetch_original(url):
    import requests
    session=requests.Session();session.trust_env=False
    for _ in range(6):
        _public_url(url)
        with session.get(url,allow_redirects=False,stream=True,timeout=(6,15)) as response:
            if response.is_redirect:
                from urllib.parse import urljoin
                url=urljoin(url,response.headers['Location']);continue
            response.raise_for_status();parts=[];size=0
            for block in response.iter_content(1<<18):
                size+=len(block)
                if size>MAX_BYTES:raise ValueError('original image byte budget exceeded')
                parts.append(block)
            return b''.join(parts)
    raise ValueError('original image redirect limit')


def _save_official_image(blob, expected_sha, recovered_root, recovered_dataset_id):
    if hashlib.sha256(blob).hexdigest()!=expected_sha:
        raise ValueError('recovered original image SHA differs from official annotation')
    relative=expected_sha[:2]+'/'+expected_sha+'.bin'
    target=Path(recovered_root)/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            temporary.write_bytes(blob)
            if file_sha(temporary)!=expected_sha:raise ValueError('recovered original write SHA differs')
            os.replace(temporary,target)
        finally:temporary.unlink(missing_ok=True)
    if file_sha(target)!=expected_sha:raise ValueError('previously recovered original image changed')
    return {'status':'bound','local_sha256':expected_sha,'relative_path':relative,
            'asset_root':str(Path(recovered_root).resolve()),'asset_dataset_id':recovered_dataset_id,
            'recovered_bytes':len(blob),'proof':{'method':'recovered_official_encoded_sha256','source_sha256':expected_sha,
            'local_sha256':expected_sha,'original_bytes_preserved':True}}


def _recover(item,index,profiles,recovered_root=None,recovered_dataset_id=None):
    sha,urls=item;errors=[]
    db=sqlite3.connect('file:'+index+'?mode=ro',uri=True)
    try:
        for url in urls[:3]:
            try:
                blob=fetch_original(url)
                result=verify_blob(blob,sha,db,profiles)
                if recovered_root and result['status']=='unbound' and result.get('source_sha256')==sha:
                    result=_save_official_image(blob,sha,recovered_root,recovered_dataset_id)
                result.update(original_sha256=sha,source_url=url)
                if result['status']=='bound':return result
                errors.append(result)
            except Exception as error:
                errors.append({'status':'fetch_or_decode_failed','error_type':type(error).__name__})
        return {'status':'unbound','original_sha256':sha,'attempts':errors,'reason':'no_verified_local_binding'}
    finally:db.close()


def _claim(path):
    try:path.mkdir()
    except FileExistsError:
        heartbeat=path/'HEARTBEAT.json'
        # A freshly created claim may not have written its first heartbeat yet.
        modified=heartbeat.stat().st_mtime if heartbeat.exists() else path.stat().st_mtime
        if time.time()-modified<300:
            raise RuntimeError('worker claim remains active')
        archive=path.with_name(path.name+'.stale.'+uuid.uuid4().hex)
        path.rename(archive);path.mkdir()
    atomic_json(path/'OWNER.json',{'token':uuid.uuid4().hex,'host':socket.gethostname(),'pid':os.getpid()})
    return path


def finalize_bindings(output,workers):
    """Publish a sealed all-source binding table only after all partitions finish."""
    output=Path(output);receipt=output/'COMPLETE.json'
    if receipt.exists():return json.loads(receipt.read_text())
    if not all((output/f'worker-{i}'/'COMPLETE.json').exists() for i in range(workers)):return None
    claim=output/'.merge-results.claim'
    try:_claim(claim)
    except RuntimeError:return None
    try:
        target=output/'bindings.sqlite';temporary=target.with_suffix('.building');temporary.unlink(missing_ok=True)
        db=sqlite3.connect(temporary)
        db.execute('CREATE TABLE bindings (original_sha256 TEXT PRIMARY KEY,status TEXT,result_json TEXT)')
        receipts=[];counts=Counter()
        for i in range(workers):
            item=json.loads((output/f'worker-{i}'/'COMPLETE.json').read_text());receipts.append(item)
            if file_sha(item['result_path'])!=item['result_sha256']:raise ValueError('completed PixMo result partition changed')
            part=sqlite3.connect('file:'+item['result_path']+'?mode=ro',uri=True)
            try:
                for row in part.execute('SELECT original_sha256,status,result_json FROM results'):
                    db.execute('INSERT INTO bindings VALUES (?,?,?)',row);counts[row[1]]+=1
            finally:part.close()
            db.commit()
        if len({item['annotations_sha256'] for item in receipts})!=1 or len({item['image_index_sha256'] for item in receipts})!=1:
            raise ValueError('PixMo recovery partitions used different inputs')
        db.close();os.replace(temporary,target)
        result={'status':'complete','contract':CONTRACT,'path':str(target),'sha256':file_sha(target),'fingerprint':stamp(target),
                'original_images':sum(counts.values()),'counts':dict(counts),'workers':workers,
                'annotations_sha256':receipts[0]['annotations_sha256'],'image_index_sha256':receipts[0]['image_index_sha256'],
                'worker_receipts':receipts,'unbound_policy':'quarantine_original_ordered_QA_without_invented_image_binding'}
        atomic_json(receipt,result);return result
    finally:shutil.rmtree(claim)


def run(root,annotations,output,worker_index,workers=4,network_threads=32,image_index_receipt=None,
        recovered_root=None,recovered_dataset_id=None):
    import pyarrow.parquet as pq
    root,annotations,output=Path(root).resolve(strict=True),Path(annotations).resolve(strict=True),Path(output).resolve()
    if not 0<=worker_index<workers or root==output or root in output.parents:
        raise ValueError('invalid recovery worker/output')
    output.mkdir(parents=True,exist_ok=True);(output/'image_index').mkdir(exist_ok=True)
    if bool(recovered_root)!=bool(recovered_dataset_id):raise ValueError('recovered image root and dataset binding must both be explicit')
    if recovered_root:
        from corpus_pipeline.unified.genund_loose import _options
        _options(recovered_dataset_id,('i2t',),1)
        recovered_root=Path(recovered_root).resolve()
        if recovered_root==root or root in recovered_root.parents:raise ValueError('recovered originals must not alter mounted source')
    work=output/f'worker-{worker_index}';work.mkdir(exist_ok=True)
    if (work/'COMPLETE.json').exists():
        finalize_bindings(output,workers);return
    claim=_claim(work/'.claim');stop=threading.Event();state={'status':'running','phase':'indexing','pid':os.getpid(),'worker':worker_index}
    owner=json.loads((claim/'OWNER.json').read_text())
    def check_owner():
        if not (claim/'OWNER.json').exists() or json.loads((claim/'OWNER.json').read_text())!=owner:
            raise RuntimeError('worker lease was replaced; refusing further output')
    def beat():
        while not stop.is_set():
            check_owner()
            value={**state,'updated_unix':time.time()};atomic_json(work/'HEARTBEAT.json',value);atomic_json(claim/'HEARTBEAT.json',value);stop.wait(15)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:
        directories=sorted(path for path in (root/SOURCE).iterdir() if path.is_dir())
        identity={'contract':CONTRACT,'root':str(root),'annotations':str(annotations),'annotations_sha256':file_sha(annotations),
                  'workers':workers,'implementation_sha256':file_sha(__file__),'pil_version':PIL_VERSION,
                  'directories':{path.name:stamp(path) for path in directories}}
        if recovered_root:identity.update(recovered_root=str(recovered_root),recovered_dataset_id=recovered_dataset_id)
        if image_index_receipt:identity['image_index_receipt_sha256']=file_sha(image_index_receipt)
        identity_path=work/'IDENTITY.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise ValueError('recovery worker input/code identity changed')
        atomic_json(identity_path,identity)
        if image_index_receipt:
            index=json.loads(Path(image_index_receipt).read_text())
            if index['contract']!=CONTRACT or file_sha(index['path'])!=index['sha256']:
                raise ValueError('previously completed PixMo image index changed')
        else:
            selected=[path for i,path in enumerate(directories) if i%workers==worker_index]
            with ThreadPoolExecutor(4) as pool:
                for _ in pool.map(lambda directory:_index_directory(root,directory,output/'image_index'),selected):pass
            check_owner()
            state['phase']='waiting_for_image_index'
            index=None
            while index is None:
                check_owner()
                index=_merge_index(output,directories)
                if index is None:stop.wait(5)
        state['phase']='recovering_original_image_bindings'
        rows=pq.read_table(annotations,columns=['image_sha256','image_url']).to_pylist();images={}
        for row in rows:
            sha=row['image_sha256']
            if int(sha[:8],16)%workers!=worker_index:continue
            urls=images.setdefault(sha,[])
            if row['image_url'] not in urls:urls.append(row['image_url'])
        db=sqlite3.connect(work/'results.sqlite')
        db.execute('CREATE TABLE IF NOT EXISTS results (original_sha256 TEXT PRIMARY KEY,status TEXT,result_json TEXT)')
        done={row[0] for row in db.execute('SELECT original_sha256 FROM results')};pending=[item for item in images.items() if item[0] not in done]
        counts=Counter(dict(db.execute('SELECT status,count(*) FROM results GROUP BY status')))
        state.update(total_original_images=len(images),completed=len(done),counts=dict(counts))
        with ThreadPoolExecutor(network_threads) as pool:
            # Keep a bounded window instead of 73k image byte futures in memory.
            for start in range(0,len(pending),network_threads*4):
                futures=[pool.submit(_recover,item,index['path'],index['profiles'],recovered_root,recovered_dataset_id) for item in pending[start:start+network_threads*4]]
                for future in as_completed(futures):
                    result=future.result();check_owner();db.execute('INSERT INTO results VALUES (?,?,?)',(result['original_sha256'],result['status'],canonical(result)));db.commit()
                    counts[result['status']]+=1;state.update(completed=sum(counts.values()),counts=dict(counts))
        db.close();result={'complete':True,'status':'complete','worker':worker_index,'contract':CONTRACT,'original_images':len(images),
                          'counts':dict(counts),'annotations_sha256':file_sha(annotations),'image_index_sha256':index['sha256'],
                          'result_path':str(work/'results.sqlite'),'result_sha256':file_sha(work/'results.sqlite'),
                          'identity':identity}
        atomic_json(work/'COMPLETE.json',result);state.update(status='complete',phase='complete')
        finalize_bindings(output,workers)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error));atomic_json(work/'FAILED.json',state);raise
    finally:
        stop.set();thread.join()
        if (claim/'OWNER.json').exists() and json.loads((claim/'OWNER.json').read_text())==owner:
            atomic_json(work/'HEARTBEAT.json',{**state,'updated_unix':time.time()});shutil.rmtree(claim)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--worker-index',type=int,required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--network-threads',type=int,default=32)
    parser.add_argument('--image-index-receipt');parser.add_argument('--recovered-root');parser.add_argument('--recovered-dataset-id')
    args=parser.parse_args();run(**vars(args))

if __name__=='__main__':main()
