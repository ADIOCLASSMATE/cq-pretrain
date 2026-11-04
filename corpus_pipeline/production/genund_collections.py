"""Explicit, portable adapters for OpenUni, ScaleCap and FineVision.

All persisted image references are relative to a pinned dataset root. Annotation
chunks are derived metadata only; source image/annotation files are never edited.
OpenUni uses documented image/annotation joins, never a global basename join.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tarfile
import tempfile

from .common import atomic_json, canonical, digest, file_sha

CONTRACT = 'genund.named_collections.v1'
TASK_POLICY = 'corpus.task_policy.v1'
MAX_METADATA = 32 << 20
MAX_IMAGE = 256 << 20
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
OPENUNI_FIELDS = {
    'cc12m': ('caption', 'detailed_caption'),
    'laion6m': ('caption', 'detailed_caption', 're_caption'),
    'megalith-10m': ('caption_internlm2', 'caption_internlm2_short', 'caption_florence2',
                     'caption_florence2_short', 'caption_sharecap', 'caption_sharecap_short'),
    'redcaps5m': ('caption', 'original_caption', 're_caption'),
    'redcaps5m_resized': ('caption', 'original_caption', 're_caption'),
    'text-to-image-2M': ('prompt',),
}
PROVENANCE_FIELDS = ('image_origin', 'image_provenance', 'image_generation_status', 'is_ai_generated',
                     'is_generated', 'ai_generated', 'is_ai_edited', 'edit_source_origin', 'source_image_origin')


class UnusableImage(ValueError):
    """A per-record integrity failure, never permission to discard its whole unit."""


def stamp(path):
    value = Path(path).stat()
    return {'size': value.st_size, 'mtime_ns': value.st_mtime_ns}


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\0' in value or '://' in value:
        raise ValueError('invalid source-relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('source-relative path escapes the declared dataset')
    return path.as_posix()


def bound(root, value, *, exists=True):
    root = Path(root).resolve(strict=True)
    path = (root/relative(value)).resolve(strict=exists)
    if not path.is_relative_to(root):
        raise ValueError('source path escapes the declared dataset')
    return path


def iter_json_records(path):
    """Stream an exact JSON array or JSONL; malformed/truncated tails fail closed."""
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8-sig') as stream:
            for number, line in enumerate(stream):
                if len(line.encode()) > MAX_METADATA:
                    raise ValueError('annotation record exceeds metadata budget')
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict): raise ValueError('annotation record must be an object')
                    yield number, value
        return
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8-sig') as stream:
        buffer = ''; eof = False; number = 0
        def more():
            nonlocal buffer, eof
            block = stream.read(1 << 20); eof = not block; buffer += block
        def whitespace():
            nonlocal buffer
            while not buffer.strip() and not eof: more()
            buffer = buffer.lstrip()
        whitespace()
        if not buffer.startswith('['): raise ValueError('annotation file must contain a JSON array')
        buffer = buffer[1:]; whitespace()
        if buffer.startswith(']'):
            buffer = buffer[1:]
        else:
            while True:
                while True:
                    try: value, end = decoder.raw_decode(buffer); break
                    except json.JSONDecodeError:
                        if eof: raise ValueError('truncated or malformed annotation array')
                        if len(buffer.encode()) > MAX_METADATA: raise ValueError('annotation record exceeds metadata budget')
                        more()
                if not isinstance(value, dict): raise ValueError('annotation record must be an object')
                yield number, value; number += 1
                buffer = buffer[end:]; whitespace()
                if buffer.startswith(']'):
                    buffer = buffer[1:]; break
                if not buffer.startswith(','): raise ValueError('missing annotation array separator')
                buffer = buffer[1:]; whitespace()
                if buffer.startswith(']'): raise ValueError('trailing annotation array comma')
        whitespace()
        if buffer or not eof and stream.read().strip(): raise ValueError('trailing data after annotation array')


def prepare_annotation_units(path, staging, *, base, rows_per_unit=8192):
    """Split annotation metadata once, with resumable immutable per-file receipts."""
    path = Path(path).resolve(strict=True); staging = Path(staging).resolve()
    if rows_per_unit < 1: raise ValueError('rows_per_unit must be positive')
    identity = {'source_path': str(path), 'source_sha256': file_sha(path), 'source_fingerprint': stamp(path),
                'base': base, 'rows_per_unit': rows_per_unit, 'adapter_sha256': file_sha(__file__)}
    directory = staging/digest(identity); marker = directory/'COMPLETE.json'
    if marker.exists():
        value = json.loads(marker.read_text())
        if value['identity'] != identity: raise ValueError('annotation preparation identity changed')
        for unit in value['units']:
            if stamp(unit['path']) != unit['fingerprint'] or file_sha(unit['path']) != unit['sha256']:
                raise ValueError('prepared annotation chunk changed')
        return value['units']
    directory.mkdir(parents=True, exist_ok=True)
    units = []; stream = None; count = 0; start = 0
    def finish():
        nonlocal stream
        if stream is None: return
        stream.flush(); os.fsync(stream.fileno()); stream.close(); stream = None
        target = directory/f'{len(units):06d}.jsonl'; os.replace(target.with_suffix('.partial'), target)
        unit = {**base, 'path': str(target), 'fingerprint': stamp(target), 'sha256': file_sha(target),
                'annotation_path': str(path), 'annotation_sha256': identity['source_sha256'],
                'annotation_fingerprint': identity['source_fingerprint'], 'row_start': start, 'row_end': count,
                'relative_path': base.get('source_shard', path.name)+f'#{start}:{count}', 'scope': 'full_declared_annotation'}
        unit['unit_id'] = digest(unit); units.append(unit)
    try:
        for number, value in iter_json_records(path):
            if stream is None:
                start = count; stream = (directory/f'{len(units):06d}.partial').open('w', encoding='utf-8')
            stream.write(canonical({'row_index': number, 'record': value})+'\n'); count += 1
            if count-start >= rows_per_unit: finish()
        finish()
    finally:
        if stream is not None: stream.close()
    if stamp(path) != identity['source_fingerprint']: raise ValueError('annotation changed during preparation')
    if not units: raise ValueError('empty annotation collection')
    atomic_json(marker, {'complete': True, 'identity': identity, 'records': count, 'units': units})
    return units


class Storage:
    """Read exact loose paths or source-local TAR members with bounded caches."""
    def __init__(self, root, dataset_id, scratch=None):
        self.root = Path(root).resolve(strict=True); self.dataset_id = dataset_id
        self.temporary = tempfile.TemporaryDirectory(prefix='genund-members-', dir=scratch)
        self.database = sqlite3.connect(str(Path(self.temporary.name)/'members.sqlite'))
        self.database.execute('CREATE TABLE members(archive TEXT,name TEXT,offset INTEGER,length INTEGER,PRIMARY KEY(archive,name))')
        self.archives = {}; self.handles = OrderedDict(); self.hashes = OrderedDict(); self.related = {}; self.metadata = {}

    def close(self):
        for store in self.related.values(): store.close()
        for stream in self.handles.values(): stream.close()
        self.database.close(); self.temporary.cleanup()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _archive(self, archive):
        path = bound(self.root, archive)
        if archive not in self.archives:
            before = stamp(path)
            with tarfile.open(path, 'r:') as tar:
                for member in tar:
                    if member.isfile():
                        name = relative(member.name)
                        try: self.database.execute('INSERT INTO members VALUES(?,?,?,?)', (archive,name,member.offset_data,member.size))
                        except sqlite3.IntegrityError as exc: raise ValueError('duplicate TAR member: '+name) from exc
                    tar.members.clear()
            self.database.commit()
            if stamp(path) != before: raise ValueError('TAR changed during member indexing')
            self.archives[archive] = {'archive_sha256': file_sha(path), 'archive_size': before['size'], 'fingerprint': before}
        if stamp(path) != self.archives[archive]['fingerprint']: raise ValueError('TAR changed after member indexing')
        return path

    def locate(self, directory, name, *, cc12m=False):
        """Only the explicit shard namespace may map a relative path into a TAR."""
        directory, name = relative(directory), relative(name)
        loose = bound(self.root, directory+'/'+name, exists=False)
        if loose.is_file(): return {'kind': 'file', 'path': str(loose.relative_to(self.root)), 'member': str(loose.relative_to(self.root))}
        parts = PurePosixPath(name).parts
        if len(parts) < 2: raise FileNotFoundError('missing source-local shard path: '+directory+'/'+name)
        shard, inside = parts[0], '/'.join(parts[1:])
        archive_names = [directory+'/'+shard+'.tar']
        if cc12m: archive_names.append(directory+'/cc12m-train-'+shard+'.tar')
        candidates = []
        for archive in archive_names:
            path = bound(self.root, archive, exists=False)
            if not path.is_file(): continue
            self._archive(archive)
            matches = self.database.execute('SELECT name,offset,length FROM members WHERE archive=? AND name IN (?,?)',
                                             (archive,name,inside)).fetchall()
            for member, offset, length in matches:
                candidates.append({'kind': 'tar', 'path': archive, 'member': member, 'offset': offset, 'length': length,
                                   **{k:v for k,v in self.archives[archive].items() if k != 'fingerprint'}})
        if len(candidates) != 1: raise FileNotFoundError('missing or ambiguous exact shard member: '+directory+'/'+name)
        return candidates[0]

    def read(self, location, maximum=MAX_METADATA):
        path = bound(self.root, location['path']); size = location.get('length', path.stat().st_size)
        if not 0 < size <= maximum: raise ValueError('source component exceeds byte budget')
        key = location['path']
        if key not in self.handles: self.handles[key] = path.open('rb')
        handle = self.handles[key]; self.handles.move_to_end(key)
        while len(self.handles) > 32: self.handles.popitem(last=False)[1].close()
        before = os.fstat(handle.fileno()); handle.seek(location.get('offset', 0)); data = handle.read(size)
        after = os.fstat(handle.fileno())
        if len(data) != size or (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise ValueError('source component changed or was truncated')
        return data

    def image(self, location):
        key = canonical(location)
        if key in self.hashes:
            self.hashes.move_to_end(key); return dict(self.hashes[key])
        size=location.get('length',bound(self.root,location['path']).stat().st_size)
        if size==0: raise UnusableImage('corrupt_image')
        if size>MAX_IMAGE: raise UnusableImage('image_payload_exceeds_ingestion_byte_budget')
        data = self.read(location, MAX_IMAGE)
        ref = {**location, 'dataset_id': self.dataset_id, 'sha256': hashlib.sha256(data).hexdigest(),
               'offset': location.get('offset', 0), 'length': len(data)}
        ref.update(offset_data=ref['offset'], size=ref['length'])
        self.hashes[key] = ref
        while len(self.hashes) > 8192: self.hashes.popitem(last=False)
        return dict(ref)


def segment(kind, *, text=None, role=None, turn=None, index=None, member=None):
    return {'kind': kind, 'text': text, 'role': role, 'turn': turn, 'image_index': index, 'source_member': member}


def conversation_segments(turns, refs, *, implicit_images=False):
    """Preserve exact turn text and every explicit image position; no rewritten QA."""
    normalized = []; placeholders = 0
    for turn in turns:
        role = turn.get('from', turn.get('role')); text = turn.get('value', turn.get('content'))
        role = {'human': 'user', 'gpt': 'assistant'}.get(role, role)
        if role not in {'user', 'assistant', 'system'} or not isinstance(text, str):
            raise ValueError('unsupported original conversation role/content')
        placeholders += text.count('<image>'); normalized.append((role,text))
    issues = []; parts = []
    implicit = implicit_images and placeholders == 0 and bool(refs)
    if placeholders != len(refs) and not implicit:
        issues.append('unresolved_original_image_binding')
        return [segment('text',text=text,role=role,turn=i) for i,(role,text) in enumerate(normalized)], issues
    cursor = 0; inserted = False
    for number,(role,text) in enumerate(normalized):
        if implicit and role == 'user' and not inserted:
            parts.extend(segment('image',role='user',turn=number,index=i,member=ref['member']) for i,ref in enumerate(refs))
            inserted = True
        pieces = text.split('<image>')
        for index,piece in enumerate(pieces):
            if piece or len(pieces)==1: parts.append(segment('text',text=piece,role=role,turn=number))
            if index < len(pieces)-1:
                if role != 'user': issues.append('non_user_image_in_i2t_conversation')
                parts.append(segment('image',role=role,turn=number,index=cursor,member=refs[cursor]['member'])); cursor += 1
    if implicit and not inserted: issues.append('unresolved_original_image_binding')
    return parts, sorted(set(issues))


def packet(source, source_shard, key, annotation, refs, parts, direction, locator, *, issues=(), variant=None,
           image_origin=None, caption_origin='unknown', format='caption'):
    issues = sorted(set(issues)); raw = {'source': source, 'source_shard': source_shard, 'source_key': key,
        'format': format, 'direction': direction, 'image_count': len(refs), 'segments': parts,
        'source_annotations': annotation, 'caption_origin': caption_origin,
        'processing_policy': TASK_POLICY, 'task_route': direction, 'image_loss_allowed': direction != 'i2t'}
    for field in PROVENANCE_FIELDS:
        if field in annotation: raw[field] = annotation[field]
    if image_origin is not None: raw['image_origin'] = image_origin
    raw['sample_id'] = digest({'contract': CONTRACT, 'source': source, 'shard': source_shard, 'key': key,
                              'annotation': annotation, 'images': [r['sha256'] for r in refs], 'direction': direction, 'variant': variant})
    metadata = {'adapter': CONTRACT, 'processing_policy': TASK_POLICY, 'direction': direction,
                'task_route': direction, 'image_loss_allowed': direction != 'i2t', 'issues': issues,
                'pairing_state': 'unresolved' if issues else 'verified',
                'prepared_adapter_status': 'quarantine' if issues else 'ready',
                'caption_fields_inspected': format == 'caption', 'caption_variant_field': variant,
                'pairing_evidence_scope': 'explicit source-local image/annotation key and exact encoded image SHA256'}
    if 'unresolved_original_image_binding' in issues:
        metadata['binding_contract'] = 'unresolved_original_image_binding.v1'
    return {'raw': raw, 'source_image_refs': refs, 'locator': {'adapter': CONTRACT, **locator}, 'metadata': metadata}


def _captions(annotation, fields):
    result = []; seen = set()
    for field in fields:
        value = annotation.get(field)
        if value is None: continue
        if not isinstance(value, str): raise ValueError('non-string registered caption: '+field)
        # URL metadata is not image supervision. Whitespace in real captions is preserved.
        if not value.strip() or re.fullmatch(r'https?://\S+', value.strip()): continue
        if value not in seen: result.append((field,value)); seen.add(value)
    return result


def caption_packets(source, shard, key, annotation, refs, captions, directions, locator, *, issues=(), image_origin=None):
    if not captions:
        captions = [('missing_annotation', '')]; issues = [*issues, 'annotation_not_recovered']
    for field,text in captions:
        for direction in directions:
            if direction not in {'i2t','t2i'}: raise ValueError('explicit caption direction required')
            images = [segment('image',index=i,member=ref['member']) for i,ref in enumerate(refs)]
            words = [segment('text',text=text,member=field)]
            yield packet(source,shard,key,annotation,refs,images+words if direction=='i2t' else words+images,
                         direction,locator,issues=issues,variant=field,image_origin=image_origin,caption_origin='ai')


def openuni_record(unit, row, row_index, store):
    subset = unit['subset']; fields = OPENUNI_FIELDS[subset]
    image = relative(row['image']); annotation_path = relative(row['annotation'])
    if PurePosixPath(image).with_suffix('') != PurePosixPath(annotation_path).with_suffix(''):
        raise ValueError('OpenUni image/annotation keys differ')
    refs = []; issues = []; annotation = {}
    try: refs = [store.image(store.locate(unit['image_root'], image, cc12m=subset=='cc12m'))]
    except FileNotFoundError: issues.append('image_reference_unresolved')
    except UnusableImage as exc: issues.append(str(exc))
    try:
        location = store.locate(unit['annotation_root'],annotation_path)
        annotation = json.loads(store.read(location).decode('utf-8-sig'))
        if not isinstance(annotation,dict): raise ValueError('OpenUni annotation is not an object')
    except FileNotFoundError: issues.append('annotation_not_recovered')
    if 'key' in annotation and str(annotation['key']) != PurePosixPath(image).stem:
        issues.append('image_annotation_key_mismatch')
    origin = None
    # These exact generator namespaces are documented by the pinned original
    # T2I2M source; unrelated occurrences of "flux" in caption text have no effect.
    if subset == 'text-to-image-2M' and re.match(r'^(?:flux|dalle|dall-e)_',PurePosixPath(image).name,re.I):
        origin = 'ai_generated'
    locator = {'source_shard':unit['source_shard'],'annotation_row_index':row_index,
               'manifest_record':row,'annotation_sha256':digest(annotation),'subset':subset,
               'task_direction_evidence':'https://github.com/wusize/OpenUni/blob/main/docs/datasets/'+
                   {'megalith-10m':'megalith10m','text-to-image-2M':'text2image2m','redcaps5m_resized':'redcaps5m'}.get(subset,subset)+'.md'}
    yield from caption_packets('openuni',unit['source_shard'],image,annotation,refs,_captions(annotation,fields),
                               unit.get('directions',['t2i']),locator,issues=issues,image_origin=origin)


def scalecap_record(unit, row, row_index, store):
    paths = row.get('images'); turns = row.get('conversations')
    if not isinstance(paths,list) or len(paths)!=1 or not isinstance(turns,list):
        raise ValueError('ScaleCap requires its original one-image conversations schema')
    refs = []; issues = []; recovered_evidence=[]; recovery=None
    if unit.get('image_recovery_map'):
        key=unit['image_recovery_map']
        if key not in store.metadata:
            if file_sha(key)!=unit['image_recovery_map_sha256']: raise ValueError('ScaleCap recovery binding map changed')
            value=json.loads(Path(key).read_text())
            if not value.get('complete') or value['binding']['dataset_id']!=unit.get('recovered_dataset_id'):
                raise ValueError('ScaleCap recovered dataset binding differs')
            store.metadata[key]=value
        recovery=store.metadata[key]
    for image in paths:
        name = relative(image)
        if not name.startswith('images/'): name = 'images/'+name
        selected='ScaleCap450k_image/'+name
        try:
            if recovery and selected in recovery['images']:
                item=recovery['images'][selected]; dataset=item['dataset_id']
                if dataset!=unit.get('recovered_dataset_id'): raise ValueError('ScaleCap recovered image uses unknown dataset')
                if dataset not in store.related: store.related[dataset]=Storage(unit['recovered_root'],dataset)
                ref=store.related[dataset].image({'kind':'file','path':item['path'],'member':item['path']})
                if ref['sha256']!=item['sha256']: raise ValueError('ScaleCap recovered original image SHA differs')
                refs.append(ref); recovered_evidence.append(item)
            else: refs.append(store.image(store.locate('ScaleCap450k_image',name)))
        except FileNotFoundError: issues.append('image_reference_unresolved')
        except UnusableImage as exc: issues.append(str(exc))
    locator = {'source_shard':unit['source_shard'],'annotation_row_index':row_index,'annotation_sha256':digest(row),
               'image_recovery_evidence':recovered_evidence}
    key = str(row.get('id',row_index)); directions = unit.get('directions',['i2t','t2i'])
    if 'i2t' in directions:
        parts,binding_issues = conversation_segments(turns,refs)
        yield packet('ScaleCap450k_image',unit['source_shard'],key,row,refs,parts,'i2t',locator,
                     issues=[*issues,*binding_issues],caption_origin='ai',format='conversation')
    if 't2i' in directions:
        captions = [(f'conversations[{i}].value',turn.get('value',turn.get('content'))) for i,turn in enumerate(turns)
                    if turn.get('from',turn.get('role')) in {'gpt','assistant'}]
        if any(not isinstance(text,str) for _,text in captions): raise ValueError('invalid ScaleCap caption content')
        yield from caption_packets('ScaleCap450k_image',unit['source_shard'],key,row,refs,captions,['t2i'],locator,issues=issues)


def finevision_record(unit, row, row_index, store):
    paths = row.get('images'); texts = row.get('texts')
    if not isinstance(paths,list) or not isinstance(texts,list):
        raise ValueError('FineVision requires original images and texts arrays')
    refs = []; issues = []
    for image in paths:
        # A metadata recovery job must supply original paths or a verified SHA
        # binding sidecar. An image ordinal alone is never a path mapping.
        if isinstance(image,dict):
            if not isinstance(image.get('path'),str) or not re.fullmatch('[0-9a-f]{64}',str(image.get('sha256'))):
                raise ValueError('FineVision extracted-image mapping requires path and verified SHA256')
            name,expected = image['path'],image['sha256']
        elif isinstance(image,str): name,expected = image,None
        else: raise ValueError('FineVision embedded bytes need explicit verified extraction bindings')
        name = relative(name); selected_store=store
        if isinstance(image,dict) and image.get('dataset_id') not in {None,store.dataset_id}:
            if image['dataset_id'] != unit.get('recovered_dataset_id') or not unit.get('recovered_root'):
                raise ValueError('FineVision recovered image dataset is not bound in the unit')
            dataset_id=image['dataset_id']
            if dataset_id not in store.related: store.related[dataset_id]=Storage(unit['recovered_root'],dataset_id)
            selected_store=store.related[dataset_id]
        if not name.startswith('FineVision/'): name = 'FineVision/'+name
        try:
            ref = selected_store.image({'kind':'file','path':name,'member':name})
            if expected is not None and expected != ref['sha256']: raise ValueError('FineVision image binding SHA differs')
            refs.append(ref)
        except FileNotFoundError: issues.append('image_reference_unresolved')
    turns = []
    for text in texts:
        if not isinstance(text,dict) or not isinstance(text.get('user'),str) or not isinstance(text.get('assistant'),str):
            raise ValueError('unsupported FineVision texts schema')
        turns.extend([{'role':'user','content':text['user']},{'role':'assistant','content':text['assistant']}])
    parts,binding_issues = conversation_segments(turns,refs,implicit_images=True)
    locator = {'source_shard':unit['source_shard'],'annotation_row_index':row_index,
               'annotation_sha256':digest(row),'subset':unit.get('subset'),
               'binding_contract':'FineVision.images_order_before_first_user.v1'}
    yield packet('FineVision',unit['source_shard'],str(row.get('id',row_index)),row,refs,parts,'i2t',locator,
                 issues=[*issues,*binding_issues],format='conversation')


def enumerate_units(spec, staging_dir):
    """Return bounded work units; large manifests are streamed into derived JSONL."""
    root = Path(spec['root']).resolve(strict=True); staging = Path(staging_dir).resolve()
    if staging == root or staging.is_relative_to(root): raise ValueError('derived metadata must be outside the source root')
    adapter = spec['adapter']; dataset_id = spec['dataset_id']
    base = {'adapter':adapter,'collection':'genund','root':str(root),'dataset_id':dataset_id}
    if spec.get('recovered_dataset_id'):
        if adapter != 'genund_finevision' or not spec.get('recovered_root'):
            raise ValueError('recovered assets require an explicit FineVision root binding')
        base.update(recovered_dataset_id=spec['recovered_dataset_id'],recovered_root=spec['recovered_root'],
                    portable_datasets=[dataset_id,spec['recovered_dataset_id']],image_payloads_copied=True,
                    image_payload_copy_scope='only_missing_original_images_from_sha_verified_upstream_parquet')
    units = []; rows_per_unit = spec.get('rows_per_unit',8192)
    if adapter == 'genund_openuni':
        subsets = spec.get('subsets',list(OPENUNI_FIELDS))
        if not subsets or any(x not in OPENUNI_FIELDS for x in subsets): raise ValueError('unknown OpenUni subset')
        for subset in subsets:
            prefix = 'openuni/data/'+subset
            names = ({'cc12m':['data.json'],'laion6m':['data.json'],'megalith-10m':['megalith10m_all.json'],
                      'redcaps5m_resized':['redcaps5m_data.json'],
                      'text-to-image-2M':['data/data_1024_10K.json','data/data_512_2M.json']}).get(subset,[])
            if subset == 'redcaps5m':
                directory = bound(root,prefix+'/raw'); shards = {}
                for path in directory.iterdir():
                    if path.is_dir() and re.fullmatch(r'\d+',path.name): shards[path.name] = path
                    elif path.is_file() and path.suffix == '.tar' and re.fullmatch(r'\d+',path.stem): shards.setdefault(path.stem,path)
                if not shards: raise ValueError('OpenUni RedCaps has no image shards')
                for shard,path in sorted(shards.items()):
                    descriptor = {**base,'source':'openuni','subset':subset,'raw_shard':shard,
                        'raw_carrier':str(path.relative_to(root)),'raw_carrier_fingerprint':stamp(path),
                        'image_root':prefix+'/raw','annotation_root':prefix+'/raw','directions':['t2i'],
                        'source_shard':prefix+'/raw/'+shard,'mode':'exact_stem_raw_shard'}
                    target = staging/'raw_shards'/(digest(descriptor)+'.json'); target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists() and json.loads(target.read_text()) != descriptor: raise ValueError('raw shard descriptor changed')
                    if not target.exists(): atomic_json(target,descriptor)
                    unit = {**descriptor,'path':str(target),'fingerprint':stamp(target),'sha256':file_sha(target),
                            'relative_path':descriptor['source_shard'],'scope':'full_declared_source_shard'}
                    unit['unit_id'] = digest(unit); units.append(unit)
                continue
            for name in names:
                manifest = bound(root,prefix+'/'+name)
                raw_root = prefix+'/raw'
                if subset == 'text-to-image-2M': raw_root += '/'+Path(name).stem
                item = {**base,'source':'openuni','subset':subset,'source_shard':prefix+'/'+name,
                        'image_root':raw_root,'annotation_root':prefix+'/captions' if subset in
                         {'cc12m','laion6m','megalith-10m'} else raw_root,'directions':['t2i']}
                units += prepare_annotation_units(manifest,staging/'annotations',base=item,rows_per_unit=rows_per_unit)
    elif adapter in {'genund_scalecap','genund_finevision'}:
        paths = spec.get('annotation_paths')
        if not isinstance(paths,list) or not paths: raise ValueError('explicit recovered original annotation_paths required')
        source = 'ScaleCap450k_image' if adapter == 'genund_scalecap' else 'FineVision'
        for value in paths:
            item = value if isinstance(value,dict) else {'path':value}
            path = Path(item['path']).resolve(strict=True)
            if item.get('sha256') and file_sha(path) != item['sha256']: raise ValueError('recovered annotation SHA mismatch')
            source_shard = item.get('source_shard',source+'/'+path.name)
            unit_base = {**base,'source':source,'source_shard':source_shard,'subset':item.get('subset'),
                         'directions':spec.get('directions',['i2t','t2i'] if adapter=='genund_scalecap' else ['i2t'])}
            if adapter == 'genund_finevision' and unit_base['directions'] != ['i2t']: raise ValueError('FineVision original conversations are I2T')
            units += prepare_annotation_units(path,staging/'annotations',base=unit_base,rows_per_unit=rows_per_unit)
    else: raise ValueError('unsupported named gen-und collection adapter')
    return units


def iter_records(unit, rules=None):
    if stamp(unit['path']) != unit['fingerprint']: raise ValueError('prepared annotation unit changed')
    if unit.get('mode') == 'exact_stem_raw_shard':
        path = bound(unit['root'],unit['raw_carrier'])
        if stamp(path) != unit['raw_carrier_fingerprint']: raise ValueError('raw shard changed after enumeration')
        with Storage(unit['root'],unit['dataset_id'],unit.get('scratch')) as store:
            if path.is_dir():
                names = (entry.name for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES)
            else:
                store._archive(unit['raw_carrier'])
                names = (row[0] for row in store.database.execute('SELECT name FROM members WHERE archive=? ORDER BY name',(unit['raw_carrier'],)))
            number = 0
            for name in names:
                if PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES: continue
                parts = PurePosixPath(name).parts
                if len(parts)>1 and parts[0] != unit['raw_shard']: raise ValueError('raw image escaped its explicit shard namespace')
                image = name if len(parts)>1 else unit['raw_shard']+'/'+name
                row = {'image':image,'annotation':str(PurePosixPath(image).with_suffix('.json'))}
                yield from openuni_record(unit,row,number,store); number += 1
        return
    if stamp(unit['annotation_path']) != unit['annotation_fingerprint']: raise ValueError('source annotation changed')
    parser = {'genund_openuni':openuni_record,'genund_scalecap':scalecap_record,'genund_finevision':finevision_record}[unit['adapter']]
    with Storage(unit['root'],unit['dataset_id'],unit.get('scratch')) as store:
        for number, wrapped in iter_json_records(unit['path']):
            if unit.get('record_layout') == 'original_jsonl':
                if unit['adapter'] != 'genund_finevision': raise ValueError('original JSONL layout requires recovered FineVision')
                yield from parser(unit,wrapped,number,store)
            else:
                yield from parser(unit,wrapped['record'],wrapped['row_index'],store)
