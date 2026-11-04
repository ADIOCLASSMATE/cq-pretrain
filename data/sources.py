"""Indexed JSONL and published Parquet sources share the consumed block cursor."""
from array import array
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

from data.catalog import Catalog, digest


class JsonlCatalog:
    def __init__(self, path, block_rows, source):
        self.path, self.block_rows, self.source = Path(path).resolve(), block_rows, source
        self.offsets = array('Q'); h = hashlib.sha256()
        with self.path.open('rb') as handle:
            while True:
                offset = handle.tell(); line = handle.readline()
                if not line: break
                h.update(line)
                if line.strip(): self.offsets.append(offset)
        self.source_root, self.source_sizes = self.path.parent, {}
        self.meta = dict(identity=digest(dict(sha256=h.hexdigest(), source=source)),
            units=(len(self.offsets)+block_rows-1)//block_rows, documents=len(self.offsets),
            tokenizer={}, development=False)

    def check(self, allow_development=False):
        return dict(identity=self.meta['identity'], complete=True, development=False)

    def block(self, index):
        start = index*self.block_rows
        return 0, start, min(self.block_rows, len(self.offsets)-start)

    def read_rows(self, shard, start, count):
        rows=[]
        with self.path.open('rb') as handle:
            for index in range(start, start+count):
                handle.seek(self.offsets[index]); row=json.loads(handle.readline())
                rows.append(self.normalize(row, index))
        return rows

    def normalize(self, row, index):
        task = row.get('task', self.source.get('task', '')).lower()
        if task not in ('t2v','v2t','t2i','i2t'):
            raise ValueError('JSONL records require task=t2i/i2t/t2v/v2t')
        video = task.endswith('v') or task.startswith('v')
        key = 'videos' if video else 'images'
        media = row.get(key)
        if media is None:
            media = [dict(path=row.get('video_path' if video else 'image_path'),
                          **{k:row[k] for k in ('frames_path','fps','start_seconds','num_frames') if k in row})]
        media = [dict(x) for x in media]
        for item in media:
            for name in ('path','frames_path'):
                if item.get(name): item[name]=str((self.path.parent/item[name]).resolve())
            if video:
                item['video']=True
                item['settings']=self.source.get('video', {'mode':'native'})
        messages=row.get('messages')
        if messages is None:
            caption=row[self.source.get('caption_field','caption')]
            visual=dict(type='video' if video else 'image', index=0)
            text=dict(type='text',text=caption)
            messages=([dict(role='user',content=[text]),dict(role='assistant',content=[visual])]
                if task.startswith('t') else
                [dict(role='user',content=[visual]),dict(role='assistant',content=[text])])
        segments=[]; seen=[]
        for turn,message in enumerate(messages):
            content=message['content']
            if isinstance(content,str):content=[dict(type='text',text=content)]
            for item in content:
                segment=dict(role=message['role'],turn=turn)
                if item['type']=='text':segment.update(kind='text',text=item['text'])
                elif item['type'] in ('image','video'):
                    idx=item.get('index',0)
                    if 'video_id' in item:
                        idx=next(i for i,m in enumerate(media) if m.get('id','v0')==item['video_id'])
                    if not 0 <= idx < len(media):raise ValueError('Visual reference out of range')
                    segment.update(kind='image',image_index=idx);seen.append(message['role'])
                else:raise ValueError(f'Unsupported content type: {item["type"]}')
                segments.append(segment)
        target=task.startswith('t')
        if not seen or any(role != ('assistant' if target else 'user') for role in seen):
            raise ValueError('Generation targets belong in assistant; understanding visuals belong in user')
        return dict(row, sample_id=str(row.get('sample_id',f'{self.path.stem}:{index}')),
                    images=media, segments=segments, format='conversation', direction=task.upper(),
                    source_shard='', text_tokens=0, image_loss_allowed=target and row.get('image_loss_allowed',True))

    def close(self): pass


class MixedCatalog:
    def __init__(self, path, sources, block_rows, cache_bytes):
        self.path=Path(path).resolve(); self.catalogs=[]; self.ends=[]; total=0
        self.source_root=Path('/'); self.source_sizes={}
        for source in sources:
            root=(self.path.parent/source['path']).resolve()
            catalog=(Catalog(root, source.get('source_root'), block_rows, cache_bytes) if root.is_dir()
                     else JsonlCatalog(root,block_rows,{k:v for k,v in source.items() if k not in ('path','source_root')}))
            self.catalogs.append(catalog); total+=catalog.meta['units']; self.ends.append(total)
            self.source_sizes.update({str(catalog.source_root/k).lstrip('/'):v for k,v in catalog.source_sizes.items()})
        self.meta=dict(identity=digest([c.meta['identity'] for c in self.catalogs]),units=total,
            documents=sum(c.meta['documents'] for c in self.catalogs),tokenizer={},
            development=any(c.meta['development'] for c in self.catalogs))

    def check(self, allow_development=False):
        for c in self.catalogs:c.check(allow_development)
        return dict(identity=self.meta['identity'],complete=True,development=self.meta['development'])

    def block(self,index):
        source=bisect_right(self.ends,index); previous=self.ends[source-1] if source else 0
        shard,start,count=self.catalogs[source].block(index-previous)
        return (source,shard),start,count

    def read_rows(self,shard,start,count):
        source,index=shard; catalog=self.catalogs[source]
        rows=catalog.read_rows(index,start,count)
        for row in rows:
            row['sample_id']=f'{source}:{row["sample_id"]}'
            if row['source_shard']:row['source_shard']=str(catalog.source_root/row['source_shard']).lstrip('/')
        return rows

    def close(self):
        for c in self.catalogs:c.close()


def open_catalog(root, source_root=None, block_rows=2048, cache_bytes=128<<20):
    path=Path(root)
    if path.is_dir():return Catalog(path,source_root,block_rows,cache_bytes)
    if path.suffix=='.jsonl':return JsonlCatalog(path,block_rows,{})
    import yaml
    sources=yaml.safe_load(path.read_text())['sources']
    return MixedCatalog(path,sources,block_rows,cache_bytes)
