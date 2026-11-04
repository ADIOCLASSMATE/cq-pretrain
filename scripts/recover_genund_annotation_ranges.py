"""Parallel ranged annotation recovery and a metadata-only FineVision projection."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import requests
import pyarrow.parquet as pq

ROOT = Path('/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/genund_six_with_dcvlm_20261103_v1/annotations')
_LOCAL=threading.local()
_URL_CACHE={}
_URL_LOCK=threading.Lock()


def fetch(url, start, stop):
    if stop-start > 8<<20:
        first_stop=min(stop,start+(4<<20))
        first=fetch(url,start,first_stop)
        ranges=[(value,min(stop,value+(4<<20))) for value in range(first_stop,stop,4<<20)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            rest=list(pool.map(lambda pair:fetch(url,*pair),ranges))
        return b''.join([first,*rest])
    last = None
    if not hasattr(_LOCAL,'session'): _LOCAL.session=requests.Session()
    for attempt in range(8):
        try:
            with _URL_LOCK: cached=_URL_CACHE.get(url)
            selected=cached or (url.replace('https://huggingface.co/','https://hf-mirror.com/') if attempt%2 else url)
            response = _LOCAL.session.get(selected,headers={'Range':f'bytes={start}-{stop-1}'},timeout=(15,75))
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{stop-1}/'):
                raise ValueError('server did not honor exact metadata range')
            if len(response.content) != stop-start: raise ValueError('partial metadata range')
            with _URL_LOCK: _URL_CACHE[url]=response.url
            return response.content
        except (requests.RequestException,ValueError) as exc:
            last = exc
            with _URL_LOCK: _URL_CACHE.pop(url,None)
            if attempt < 7: time.sleep(2)
    raise last


class RangeFile(io.RawIOBase):
    def __init__(self,url,size): self.url=url; self.size=size; self.position=0; self.bytes_read=0; self.requests=0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position = offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def read(self,size=-1):
        size = self.size-self.position if size<0 else min(size,self.size-self.position)
        if size <= 0: return b''
        if size > 64<<20: raise ValueError('metadata projection requested an excessive byte range')
        data = fetch(self.url,self.position,self.position+size); self.position+=len(data)
        self.bytes_read+=len(data); self.requests+=1; return data


def fine_probe():
    root=ROOT/'FineVision'; info=json.loads((root/'hub_info.json').read_text()); results=[]
    for subset in ['CoSyn_400k_circuit','nlvr2']:
        item=next(x for x in info['siblings'] if x['rfilename'].startswith(subset+'/') and x['rfilename'].endswith('.parquet'))
        url=f"https://huggingface.co/datasets/HuggingFaceM4/FineVision/resolve/{info['sha']}/{item['rfilename']}?download=true"
        with RangeFile(url,item['size']) as stream:
            reader=pq.ParquetFile(stream)
            columns=[column.path for column in reader.schema if not column.path.endswith('.bytes')]
            table=reader.read_row_group(0,columns=columns)
            target=root/(subset+'.metadata_probe.parquet'); pq.write_table(table,target)
            results.append({'subset':subset,'source':item,'revision':info['sha'],'columns':columns,
                'rows':reader.metadata.num_rows,'row_groups':reader.metadata.num_row_groups,
                'projected_rows':table.num_rows,'schema':str(table.schema),'sample_rows':table.slice(0,3).to_pylist(),
                'network_bytes':stream.bytes_read,'range_requests':stream.requests,'output':str(target)})
            (root/'projection_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    return results


def scale_download():
    root=ROOT/'ScaleCap-450k'; info=json.loads((root/'hub_info.json').read_text())
    item=next(x for x in info['siblings'] if x['rfilename']=='ScaleCap450k.json'); size=item['size']
    url=f"https://huggingface.co/datasets/long-xing1/ScaleCap-450k/resolve/{info['sha']}/ScaleCap450k.json?download=true"
    directory=root/'ranges'; directory.mkdir(exist_ok=True); block=8<<20
    def part(start):
        stop=min(size,start+block); path=directory/f'{start:012d}.part'
        if path.exists() and path.stat().st_size==stop-start: return path
        data=fetch(url,start,stop); temporary=path.with_suffix('.partial'); temporary.write_bytes(data); os.replace(temporary,path)
        return path
    with ThreadPoolExecutor(max_workers=12) as pool: paths=list(pool.map(part,range(0,size,block)))
    target=root/'ScaleCap450k.json'; temporary=root/'ScaleCap450k.assembling'; checksum=hashlib.sha256()
    with temporary.open('wb') as stream:
        for path in paths:
            data=path.read_bytes(); checksum.update(data); stream.write(data)
    if checksum.hexdigest()!=item['lfs']['sha256']: raise ValueError('assembled annotation SHA differs from pinned upstream LFS')
    os.replace(temporary,target)
    result={'complete':True,'repo':'long-xing1/ScaleCap-450k','revision':info['sha'],
            'annotation':{'path':str(target),'bytes':size,'sha256':checksum.hexdigest()},'source_modified':False}
    (root/'COMPLETE.json').write_text(json.dumps(result,indent=2)); return result


def safe(name,fn):
    try: return {'task':name,'result':fn()}
    except Exception as exc:
        result={'task':name,'error_type':type(exc).__name__,'error':str(exc)[:500]}
        (ROOT/(name+'_range_error.json')).write_text(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(safe,'finevision_projection',fine_probe),pool.submit(safe,'scalecap_download',scale_download)]
        result=[job.result() for job in jobs]
    (ROOT/'RANGE_RECOVERY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); print(json.dumps(result,ensure_ascii=False))
