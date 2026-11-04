"""Resumable byte-exact local FineVision asset lookup, one fixed writer/subset."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import time

from corpus_pipeline.production.common import atomic_json,digest,file_sha

SUFFIXES={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.gif'}


def build_subset(plan,area,subset,workers=8):
    directory=area/'local_assets'/subset; directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'COMPLETE.json'
    if marker.exists(): return json.loads(marker.read_text())
    temporary=directory/'index.building.sqlite'; target=directory/'index.sqlite'
    if target.exists() and not temporary.exists(): os.replace(target,temporary)
    db=sqlite3.connect(temporary)
    db.execute('CREATE TABLE IF NOT EXISTS assets(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL,size INTEGER NOT NULL,mtime_ns INTEGER NOT NULL)')
    existing={row[0]:(row[1],row[2]) for row in db.execute('SELECT path,size,mtime_ns FROM assets')}
    root=Path(plan['root']); source=root/'FineVision'/subset; paths=[]; count=0; source_bytes=0
    for parent,dirs,names in os.walk(source,followlinks=False):
        for name in names:
            path=Path(parent)/name
            if path.suffix.lower() not in SUFFIXES: continue
            if path.is_symlink(): raise ValueError('FineVision source asset symlink is not supported')
            rel=str(path.relative_to(root)); stat=path.stat(); count+=1; source_bytes+=stat.st_size
            if rel in existing:
                if existing.pop(rel)!=(stat.st_size,stat.st_mtime_ns): raise ValueError('indexed original image changed')
            else: paths.append((rel,stat.st_size,stat.st_mtime_ns))
    if existing: raise ValueError('indexed original image disappeared')
    def checksum(item):
        rel,size,mtime=item; path=root/rel; sha=file_sha(path); after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=(size,mtime): raise ValueError('original image changed while hashing')
        return rel,sha,size,mtime
    processed=0; started=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(checksum,paths):
            db.execute('INSERT INTO assets VALUES(?,?,?,?)',row); processed+=1
            if processed%1000==0:
                db.commit(); atomic_json(directory/'PROGRESS.json',{'subset':subset,'hashed':count-len(paths)+processed,
                    'files':count,'source_bytes':source_bytes,'updated_at':time.time(),'seconds':time.time()-started})
    db.execute('CREATE INDEX IF NOT EXISTS asset_sha ON assets(sha256)'); db.commit()
    actual=db.execute('SELECT COUNT(*) FROM assets').fetchone()[0]
    if actual!=count: raise ValueError('local asset index cardinality differs')
    db.close(); os.replace(temporary,target)
    result={'complete':True,'subset':subset,'files':count,'source_bytes':source_bytes,'index':str(target),
            'index_sha256':file_sha(target),'source_modified':False,'seconds':time.time()-started}
    atomic_json(marker,result); return result


def build_partition(plan,area,partition,partitions,status):
    selected=[name for name in plan['subsets'] if int(digest({'subset':name}),16)%partitions==partition]
    selected.sort(key=lambda name:(name not in {'nlvr2','spot_the_diff'},plan['expected_rows'][name]))
    for subset in selected:
        status['local_asset_subset']=subset
        try: build_subset(plan,area,subset)
        except Exception as exc:
            atomic_json(area/'local_assets'/subset/'ERROR.json',{'error_type':type(exc).__name__,'error':str(exc),
                'at':time.time(),'subset':subset}); status['local_asset_error']=subset
            return
    status['local_asset_partition_complete']=True


def lookup(area,subset,sha):
    marker=area/'local_assets'/subset/'COMPLETE.json'
    if not marker.exists(): raise RuntimeError('complete local asset SHA index required before missing-image recovery')
    path=area/'local_assets'/subset/'index.sqlite'
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    try:
        row=db.execute('SELECT path,size,mtime_ns FROM assets WHERE sha256=? ORDER BY path LIMIT 1',(sha,)).fetchone()
        return row
    finally: db.close()
