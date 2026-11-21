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
        task = (row.get('task') or self.source.get('task') or '').lower()
        native = 'segments' in row
        messages=row.get('messages')
        if not native and messages is None and task not in ('t2v','v2t','t2i','i2t'):
            raise ValueError('Caption shorthand requires a direction or explicit segments')
        video = not native and task in ('t2v','v2t')
        key = 'videos' if video else 'images'
        media = row.get(key)
        if media is None and native:
            media=[]
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
        common=dict(sample_id=str(row.get('sample_id',f'{self.path.stem}:{index}')),images=media,
                    source_shard='',text_tokens=None)
        if native:
            return dict(row, **common, format=row.get('format','interleaved'))
        if messages is None:
            caption=row[self.source.get('caption_field','caption')]
            visual=dict(type='video' if video else 'image', index=0)
            if 'generation_target' in row:
                visual['generation_target'] = row['generation_target']
            text=dict(type='text',text=caption)
            messages=([dict(role='user',content=[text]),dict(role='assistant',content=[visual])]
                if task.startswith('t') else
                [dict(role='user',content=[visual]),dict(role='assistant',content=[text])])
        segments=[]
        for turn,message in enumerate(messages):
            content=message['content']
            if isinstance(content,str):content=[dict(type='text',text=content)]
            for item in content:
                segment=dict(role=message.get('role'),turn=turn)
                if item['type']=='text':segment.update(item,kind='text')
                elif item['type'] in ('image','video'):
                    idx=item.get('index',0)
                    if 'video_id' in item:
                        idx=next(i for i,m in enumerate(media) if m.get('id','v0')==item['video_id'])
                    if not 0 <= idx < len(media):raise ValueError('Visual reference out of range')
                    segment.update(item,kind='image',image_index=idx)
                else:raise ValueError(f'Unsupported content type: {item["type"]}')
                segments.append(segment)
        return dict(row, **common, segments=segments, format='conversation', direction=task.upper() or None)

    def close(self): pass


class MixedCatalog:
    def __init__(self, path, sources, block_rows, cache_bytes):
        self.path=Path(path).resolve(); self.catalogs=[]; self.ends=[]; total=0
        self.source_root=Path('/'); self.source_sizes={}
        for source in sources:
            root=(self.path.parent/source['path']).resolve()
            catalog=(open_catalog(root, source.get('source_root'), block_rows, cache_bytes) if root.is_dir()
                     else JsonlCatalog(root,block_rows,{k:v for k,v in source.items() if k not in ('path','source_root')}))
            self.catalogs.append(catalog); total+=catalog.meta['units']; self.ends.append(total)
            self.source_sizes.update({str(catalog.source_root/k).lstrip('/'):v for k,v in catalog.source_sizes.items()})
        self.meta=dict(identity=digest([c.meta['identity'] for c in self.catalogs]),units=total,
            documents=sum(c.meta['documents'] for c in self.catalogs),tokenizer={},
            development=any(c.meta['development'] for c in self.catalogs))
        if any(c.meta.get('caption_sampling') for c in self.catalogs):
            from data.captions import CONTRACT
            self.meta['caption_sampling'] = CONTRACT
        if any(c.meta.get('task_sampling') for c in self.catalogs):
            from data.task_groups import CONTRACT
            self.meta['task_sampling'] = CONTRACT

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


def release_entry(root):
    path = Path(root).resolve()
    if path.is_dir() and (path/'sources.yaml').is_file():
        return path/'sources.yaml'
    return path


def release_geometry(root):
    """Read only the entry declaration, before constructing the shuffled block index."""
    path = release_entry(root)
    if path.is_dir():
        path = path/'manifest.json'
    if path.name == 'manifest.json' and path.is_file():
        entry = json.loads(path.read_text())
    elif path.suffix in ('.yaml', '.yml') and path.is_file():
        import yaml
        entry = yaml.safe_load(path.read_text())
    else:
        return {}
    geometry = entry.get('training_geometry', {})
    for key in ('sequence_length','shuffle_block_rows','vision_max_pixels','generation_max_pixels'):
        if key in geometry and (type(geometry[key]) is not int or geometry[key] < 1):
            raise ValueError(f'training_geometry.{key} must be a positive integer')
    return geometry


def open_catalog(root, source_root=None, block_rows=2048, cache_bytes=128<<20):
    path=release_entry(root)
    if path.is_dir() or path.name == 'manifest.json':
        if path.is_file():
            path = path.parent
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('schema_version') == 'corpus.combined_training.v1':
            from corpus_pipeline.postprocess.reconcile import file_sha, validate_artifacts
            completion=json.loads((path/'COMPLETE.json').read_text())
            if not completion.get('complete') or completion['manifest_sha256'] != file_sha(path/'manifest.json'):
                raise ValueError('Combined training publication changed')
            validate_artifacts(path, {'artifacts':manifest['children']})
            sources=[dict(s,source_root=source_root) if source_root is not None else s for s in manifest['sources']]
            catalog = MixedCatalog(path/'manifest.json',sources,block_rows,cache_bytes)
            if 'training_geometry' in manifest:
                catalog.meta['training_geometry'] = manifest['training_geometry']
            return catalog
        if manifest.get('schema') == 'corpus.baseline_union.v1':
            from corpus_pipeline.unified.baseline_union import verify_reference_release
            from corpus_pipeline.postprocess.reconcile import validate_artifacts
            manifest=verify_reference_release(path,verify_payloads=False)
            validate_artifacts(path, {'artifacts': [manifest['training_catalog']]})
            path=path/manifest['training_catalog']['path']
            path=path.parent
            manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('schema_version') == 'corpus.training_selection.v1':
            from data.selection_catalog import SelectionCatalog
            return SelectionCatalog(path,source_root,block_rows,cache_bytes)
        if manifest.get('schema_version') in ('corpus.caption_groups.v1','corpus.caption_groups.v2'):
            from data.grouped_catalog import GroupedCaptionCatalog
            return GroupedCaptionCatalog(path,source_root,block_rows,cache_bytes)
        if manifest.get('schema_version') in ('corpus.openuni_image_captions.v1','corpus.openuni_image_captions.v2'):
            from data.grouped_catalog import ImageCaptionCatalog
            return ImageCaptionCatalog(path,source_root,block_rows,cache_bytes)
        if manifest.get('schema_version') in ('corpus.task_groups.v1','corpus.task_groups.v2','corpus.task_groups.v3'):
            from data.grouped_catalog import TaskGroupCatalog
            return TaskGroupCatalog(path,source_root,block_rows,cache_bytes)
        return Catalog(path,source_root,block_rows,cache_bytes)
    if path.suffix=='.jsonl':return JsonlCatalog(path,block_rows,{})
    import yaml
    entry=yaml.safe_load(path.read_text())
    if 'catalog' in entry:
        manifest = (path.parent/entry['catalog']).resolve(strict=True)
        if ('sources' in entry or manifest.name != 'manifest.json' or
                hashlib.sha256(manifest.read_bytes()).hexdigest() != entry['catalog_sha256']):
            raise ValueError('Training entry does not match its pinned catalog manifest')
        catalog = open_catalog(manifest, source_root, block_rows, cache_bytes)
    else:
        catalog=MixedCatalog(path,entry['sources'],block_rows,cache_bytes)
    field = entry.get('generation_target_field', 'generation_target')
    if field not in ('generation_target','image_loss_allowed'):
        raise ValueError('Invalid generation_target_field in the release entry')
    catalog.meta['generation_target_field'] = field
    if 'training_geometry' in entry:
        existing = catalog.meta.get('training_geometry', {})
        geometry = entry['training_geometry']
        if any(geometry[k] != existing[k] for k in existing.keys() & geometry.keys()):
            raise ValueError('Training geometry conflicts with the pinned catalog')
        catalog.meta['training_geometry'] = {**existing, **geometry}
    return catalog
