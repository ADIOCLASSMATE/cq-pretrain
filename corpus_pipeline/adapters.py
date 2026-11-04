"""Read-only, bounded-memory discovery and record adapters for corpus stage one.

Discovery and archive joins use temporary SQLite databases rather than retaining
millions of names. No image is decoded, extracted or downloaded. Container parse
errors fail the unit; callers must record the failure, never declare it complete.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import tempfile
from functools import lru_cache
from typing import Iterator

IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.avif', '.tif', '.tiff', '.heic'})
TEXT_EXTS = frozenset({'.txt', '.caption', '.text'})
JSON_EXTS = frozenset({'.json', '.jsonl', '.ndjson'})
SKIP_DIRS = frozenset({'.git', '.cache', '__pycache__', '.ipynb_checkpoints'})
SKIP_NAMES = frozenset({'dataset_info.json', 'dataset_infos.json', 'state.json', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', '_source.json'})
SKIP_PREFIXES = ('readme', 'license', 'licence', 'notice', 'citation', 'requirements', 'copying')
WRAPPERS = ('data', 'records', 'annotations', 'examples', 'samples', 'train', 'validation', 'test')
MAX_METADATA_BYTES = 32 * 1024 * 1024
IMAGE_PROVENANCE_KEYS = frozenset({
    'image_origin', 'image_provenance', 'image_generation_status',
    'is_ai_generated', 'is_generated', 'ai_generated', 'is_ai_edited',
    'is_synthetic_image', 'image_is_synthetic',
    'edit_source_origin', 'source_image_origin', 'original_image_origin',
    'edit_source_provenance',
})


@contextmanager
def _scratch_db():
    with tempfile.TemporaryDirectory(prefix='corpus-stage1-') as scratch:
        db = sqlite3.connect(str(Path(scratch) / 'index.sqlite'))
        try:
            db.execute('PRAGMA temp_store=FILE')
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            yield db
        finally:
            db.close()


def _stat(path):
    st = Path(path).stat()
    return {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def _unit(collection, source, path, adapter, config=None):
    path = str(Path(path).absolute())
    result = {'collection': collection, 'source': source, 'path': path,
              'adapter': adapter, 'config': config or {}, 'fingerprint': _stat(path)}
    result['unit_id'] = hashlib.sha256(f'{collection}\0{source}\0{adapter}\0{path}'.encode()).hexdigest()
    return result


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def discover_dcvlm(root) -> Iterator[dict]:
    """Discover only validated published catalog entries, never build scratch files.

    Catalog and manifest are hashed; Parquet payloads are NOT rehashed. Their
    published digest plus current size/mtime identify the snapshot for resume.
    """
    root = Path(root).resolve()
    manifest_path, catalog_path = root / 'manifest.json', root / 'catalog.json'
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if manifest.get('schema_version') != 'dcvlm.pretrain.v1':
        raise ValueError('DCVLM manifest schema_version must be dcvlm.pretrain.v1')
    if manifest.get('phase') != 'complete':
        raise ValueError('DCVLM manifest phase is not complete')
    with (root / 'COMPLETE.json').open(encoding='utf-8') as stream:
        complete = json.load(stream)
    catalog_digest = _sha_file(catalog_path)
    if complete.get('validation_passed') is not True or complete.get('catalog_sha256') != catalog_digest:
        raise ValueError('DCVLM COMPLETE validation/catalog checksum mismatch')
    identity = {'manifest_sha256': _sha_file(manifest_path), 'catalog_sha256': catalog_digest,
                'complete_sha256': _sha_file(root / 'COMPLETE.json')}
    config = {'collection_root': str(root), 'source_pool': manifest.get('source_pool'),
              'schema_version': manifest.get('schema_version')}
    for item in _json_records(catalog_path):
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe DCVLM catalog path')
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError('DCVLM catalog path escapes dataset root')
        unit = _unit('dcvlm', item['source'], path, 'dcvlm_parquet',
                     {**config, 'source_shard': item.get('source_shard')})
        if 'bytes' in item and unit['fingerprint']['size'] != item['bytes']:
            raise ValueError(f'DCVLM published file size mismatch: {path}')
        unit['fingerprint'].update({k: item[k] for k in ('sha256', 'rows') if k in item})
        unit['fingerprint'].update(identity)
        yield unit


def _kind(name):
    lower = name.lower()
    if lower in SKIP_NAMES or lower.startswith(SKIP_PREFIXES):
        return None
    if lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz')):
        return 'tar'
    ext = Path(lower).suffix
    if ext in IMAGE_EXTS:
        return 'loose_image'
    if ext in TEXT_EXTS:
        return 'text'
    if ext in JSON_EXTS:
        return 'jsonl' if ext in {'.jsonl', '.ndjson'} else 'json'
    if ext == '.parquet':
        return 'parquet'
    if ext in {'.csv', '.tsv'}:
        return 'csv'
    if ext in {'.zip', '.arrow', '.feather', '.7z', '.rar', '.h5', '.hdf5', '.lmdb', '.tfrecord', '.tfrecords', '.zst'}:
        return 'unsupported'
    return None


def _walk_files(root):
    # Recursive scandir holds at most one iterator per depth, not a huge file list.
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink():
                # No following cycles or paths outside the declared collection.
                yield Path(entry.path), 'symlink'
            elif entry.is_dir(follow_symlinks=False):
                if entry.name not in SKIP_DIRS:
                    yield from _walk_files(entry.path)
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path), 'file'



def _bounded_walk(root, requested, scopes, source):
    entry_limit = min(20000, max(512, requested * 64))
    directory_limit = 64
    scope = {'mode': 'bounded_filesystem_prefix', 'entries_examined': 0, 'directories_examined': 0,
             'entry_limit': entry_limit, 'directory_limit': directory_limit,
             'representative_sample': False, 'complete': False}
    scopes[source] = scope

    def descend(directory):
        if scope['directories_examined'] >= directory_limit:
            return
        scope['directories_examined'] += 1
        # Recurse immediately when a directory is observed. Finishing a huge
        # parent's scandir first would exhaust the global budget before any leaf.
        with os.scandir(directory) as entries:
            for entry in entries:
                if scope['entries_examined'] >= entry_limit:
                    return
                scope['entries_examined'] += 1
                if entry.is_symlink():
                    yield Path(entry.path), 'symlink'
                elif entry.is_dir(follow_symlinks=False):
                    if entry.name not in SKIP_DIRS:
                        yield from descend(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    yield Path(entry.path), 'file'

    yield from descend(Path(root))
    # Scope remains partial even if this bounded traversal happened to exhaust.


def discover_genund(root, max_files_per_source=None) -> Iterator[dict]:
    """Yield stable sorted work units. The explicit limit is per source *unit*.

    A loose image and same-stem TXT/JSON are one unit. Global annotation files
    remain independent; image-only inventory rows do not become training pairs.
    With a limit, inventory traversal is also bounded and explicitly marked partial.
    Limited discovery is a filesystem-prefix diagnostic, not a uniform sample.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Collection root is unavailable: {root}')
    if max_files_per_source is not None and max_files_per_source < 1:
        raise ValueError('max_files_per_source must be positive or None')
    with _scratch_db() as db:
        db.execute('CREATE TABLE files(path TEXT PRIMARY KEY, source TEXT, stem TEXT, kind TEXT)')
        db.execute('CREATE TABLE sources(name TEXT PRIMARY KEY)')
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False) and entry.name not in SKIP_DIRS:
                    db.execute('INSERT OR IGNORE INTO sources VALUES (?)', (entry.name,))
        source_scopes = {}
        if max_files_per_source is None:
            scans = [(None, _walk_files(root))]
        else:
            # Sampling does not traverse a 91-TiB / hundreds-of-millions-file tree.
            # Scandir prefix order is filesystem-defined, deliberately NOT claimed random.
            scans = [(name, _bounded_walk(root / name, max_files_per_source, source_scopes, name))
                     for (name,) in db.execute('SELECT name FROM sources ORDER BY name')]
            if not scans:
                scans = [('__root__', _bounded_walk(root, max_files_per_source, source_scopes, '__root__'))]
        for scan_source, scan in scans:
            for index, (path, node_kind) in enumerate(scan):
                rel = path.relative_to(root)
                source = scan_source or (rel.parts[0] if len(rel.parts) > 1 else '__root__')
                kind = 'unsupported_symlink' if node_kind == 'symlink' else _kind(path.name)
                if kind:
                    db.execute('INSERT OR IGNORE INTO files VALUES (?,?,?,?)',
                               (str(path), source, str(path.with_suffix('')), kind))
                    db.execute('INSERT OR IGNORE INTO sources VALUES (?)', (source,))
                    if max_files_per_source is not None and kind in {'text', 'json'}:
                        stem = str(path.with_suffix(''))
                        for suffix in sorted(IMAGE_EXTS):
                            candidate = stem + suffix
                            if _exists(candidate):
                                changed = db.execute('INSERT OR IGNORE INTO files VALUES (?,?,?,?)',
                                                     (candidate, source, stem, 'loose_image')).rowcount
                                if changed:
                                    scope = source_scopes.get(source, {})
                                    scope['paired_image_followups'] = scope.get('paired_image_followups', 0) + 1
                if index % 10000 == 0:
                    db.commit()
        db.commit()
        db.execute('CREATE INDEX stem_kind ON files(stem,kind)')
        db.execute('CREATE INDEX source_path ON files(source,path)')
        for (source,) in db.execute('SELECT name FROM sources ORDER BY name'):
            emitted = 0
            pending_loose = []
            source_root = root / source if source != '__root__' else root
            config_base = {'collection_root': str(root), 'source_root': str(source_root),
                           'discovery_limit_per_source': max_files_per_source,
                           'discovery_scope': source_scopes.get(source, {'mode': 'full'})}
            for path, stem, kind in db.execute("SELECT path,stem,kind FROM files WHERE source=? ORDER BY CASE kind WHEN 'json' THEN 0 WHEN 'jsonl' THEN 0 WHEN 'parquet' THEN 1 WHEN 'tar' THEN 2 WHEN 'loose_image' THEN 3 WHEN 'text' THEN 4 WHEN 'csv' THEN 5 ELSE 9 END,path", (source,)):
                # Sidecar annotations are consumed through their paired image.
                if kind in {'text', 'json'} and db.execute(
                    "SELECT 1 FROM files WHERE stem=? AND kind='loose_image' LIMIT 1", (stem,)).fetchone():
                    continue
                config = dict(config_base)
                if kind == 'loose_image':
                    sidecars = {p for (p,) in db.execute(
                        "SELECT path FROM files WHERE stem=? AND kind IN ('text','json') ORDER BY path", (stem,))}
                    # A sampled directory prefix may end between an image and its sidecar.
                    sidecars.update(stem + suffix for suffix in sorted(TEXT_EXTS | {'.json'}) if _exists(stem + suffix))
                    config['sidecars'] = sorted(sidecars)
                    config['ambiguous_image_stem'] = db.execute(
                        "SELECT count(*) FROM files WHERE stem=? AND kind='loose_image'", (stem,)).fetchone()[0] > 1
                if kind == 'unsupported_symlink':
                    # Broken symlinks cannot stat; surface an explicit unit.
                    unit = {'collection': 'genund', 'source': source, 'path': path,
                            'adapter': 'unsupported', 'config': {**config, 'reason': 'symlink_not_followed'},
                            'fingerprint': {'symlink_target': os.readlink(path)}}
                    unit['unit_id'] = hashlib.sha256(f'genund\0{source}\0unsupported\0{path}'.encode()).hexdigest()
                else:
                    unit = _unit('genund', source, path, kind, config)
                    if config.get('sidecars'):
                        unit['fingerprint']['sidecars'] = {p: _stat(p) for p in config['sidecars']}
                if kind != 'loose_image' and pending_loose:
                    yield _bundle_loose(pending_loose)
                    pending_loose = []
                    emitted += 1
                    if max_files_per_source is not None and emitted >= max_files_per_source:
                        break
                if kind == 'loose_image':
                    pending_loose.append(unit)
                    if len(pending_loose) < 1024:
                        continue
                    unit = _bundle_loose(pending_loose)
                    pending_loose = []
                yield unit
                emitted += 1
                if max_files_per_source is not None and emitted >= max_files_per_source:
                    break
            if pending_loose and (max_files_per_source is None or emitted < max_files_per_source):
                yield _bundle_loose(pending_loose)
                emitted += 1
            if not emitted:
                unit = _unit('genund', source, source_root, 'unsupported',
                             {**config_base, 'reason': 'no_supported_files_discovered'})
                yield unit



def _bundle_loose(units):
    first = units[0]
    rows = [{'path': u['path'], 'sidecars': u['config'].get('sidecars', []),
             'ambiguous_image_stem': u['config'].get('ambiguous_image_stem', False),
             'fingerprint': u['fingerprint']} for u in units]
    inputs = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    unit = {**first, 'adapter': 'loose_batch',
            'config': {k: v for k, v in first['config'].items() if k not in {'sidecars', 'ambiguous_image_stem'}}}
    unit['config']['rows'] = rows
    unit['fingerprint'] = {'file_count': len(rows), 'inputs_sha256': hashlib.sha256(inputs).hexdigest()}
    names = '\0'.join(row['path'] for row in rows)
    unit['unit_id'] = hashlib.sha256(f"{first['collection']}\0{first['source']}\0loose_batch\0{names}".encode()).hexdigest()
    return unit


def _iter_loose_batch(unit):
    for i, row in enumerate(unit['config']['rows']):
        child = {**unit, 'path': row['path'], 'adapter': 'loose_image',
                 'config': {**{k: v for k, v in unit['config'].items() if k != 'rows'},
                            'sidecars': row['sidecars'],
                            'ambiguous_image_stem': row['ambiguous_image_stem']}}
        for record in _iter_loose(child):
            record['locator']['batch_row_index'] = i
            yield record

def _envelope(unit, raw, locator, issues=None, image_refs=None):
    issues = list(issues or [])
    if image_refs == [] and any(isinstance(raw.get(key), str) for key in ('caption', 'recaption')):
        issues.append('caption_image_reference_missing')
    if image_refs and any(str(ref.get('field') or '').startswith(('messages.', 'messages[', 'conversations.', 'conversations[')) for ref in image_refs) and any(
            str(ref.get('field') or '').split('.', 1)[0].split('[', 1)[0] in {'image', 'images', 'image_path', 'image_url'} for ref in image_refs):
        issues.append('ambiguous_image_reference_mapping')
    result = {'collection': unit['collection'], 'source': unit['source'],
              'locator': {'path': unit['path'], **locator}, 'raw': raw,
              'issues': list(dict.fromkeys(issues)), 'source_config': unit.get('config', {})}
    if image_refs is not None:
        result['image_refs'] = image_refs
    return result


def _bounded_read(path, limit=MAX_METADATA_BYTES):
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'Metadata record exceeds {limit} bytes: {path}')
    return data.decode('utf-8-sig')


def _object_records(value):
    if isinstance(value, list):
        yield from value
    elif isinstance(value, dict):
        for name in WRAPPERS:
            if name in value and isinstance(value[name], list):
                yield from value[name]
                return
        # ID -> record maps, not dictionaries of scalar annotations.
        if value and all(isinstance(v, dict) for v in value.values()) and not any(
                k in value for k in ('image', 'images', 'messages', 'conversations', 'caption', 'text')):
            for key, record in value.items():
                yield {'_mapping_key': key, **record}
        else:
            yield value
    else:
        yield value


def _stream_json_array(stream):
    """Incremental top-level arrays, keeping at most one bounded record in RAM."""
    decoder = json.JSONDecoder()
    buffer, pos, eof = '', 0, False
    def refill():
        nonlocal buffer, pos, eof
        buffer = buffer[pos:]
        pos = 0
        chunk = stream.read(65536)
        eof = not chunk
        buffer += chunk
    refill()
    while True:
        while pos < len(buffer) and buffer[pos].isspace(): pos += 1
        if pos < len(buffer): break
        if eof: raise ValueError('Empty JSON array input')
        refill()
    if buffer[pos] != '[': raise ValueError('Expected top-level JSON array')
    pos += 1
    need_comma = False
    while True:
        while True:
            while pos < len(buffer) and buffer[pos].isspace(): pos += 1
            if pos < len(buffer): break
            if eof: raise ValueError('Truncated JSON array')
            refill()
        if buffer[pos] == ']':
            pos += 1
            if buffer[pos:].strip(): raise ValueError('Trailing data after JSON array')
            for trailing in iter(lambda: stream.read(65536), ''):
                if trailing.strip(): raise ValueError('Trailing data after JSON array')
            return
        if need_comma:
            if buffer[pos] != ',': raise ValueError('Missing JSON array separator')
            pos += 1
            while True:
                while pos < len(buffer) and buffer[pos].isspace(): pos += 1
                if pos < len(buffer): break
                if eof: raise ValueError('Truncated JSON array')
                refill()
            if buffer[pos] == ']': raise ValueError('Trailing comma in JSON array')
        while True:
            try:
                value, end = decoder.raw_decode(buffer, pos)
                # A scalar number/string at chunk boundary may be incomplete.
                if end == len(buffer) and not eof:
                    refill()
                    continue
                pos = end
                break
            except json.JSONDecodeError:
                if eof: raise
                if len(buffer) - pos > MAX_METADATA_BYTES:
                    raise ValueError('JSON record exceeds metadata byte limit')
                refill()
        yield value
        need_comma = True


def _json_records(path):
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        first = stream.read(4096).lstrip()[:1]
        stream.seek(0)
        if first == '[':
            yield from _stream_json_array(stream)
            return
    if Path(path).stat().st_size <= MAX_METADATA_BYTES:
        yield from _object_records(json.loads(_bounded_read(path)))
        return
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError('Large JSON object requires ijson; install corpus_pipeline requirements') from exc
    # Identify known wrappers without loading their arrays.
    with Path(path).open('rb') as stream:
        wrapper = None
        for prefix, event, value in ijson.parse(stream):
            if prefix in WRAPPERS and event == 'start_array':
                wrapper = prefix
                break
            if prefix and '.' not in prefix and event not in {'map_key', 'start_map', 'end_map'}:
                if prefix not in WRAPPERS:
                    break
    with Path(path).open('rb') as stream:
        if wrapper:
            yield from ijson.items(stream, wrapper + '.item', use_float=True)
        else:
            for key, value in ijson.kvitems(stream, '', use_float=True):
                if not isinstance(value, dict):
                    raise ValueError('Large JSON object is not a supported record mapping/wrapper')
                yield {'_mapping_key': key, **value}


@lru_cache(maxsize=16384)
def _exists(path):
    return Path(path).is_file()


def _ref(value, unit, field):
    if not isinstance(value, str) or not value.strip(): return None
    value = value.strip()
    if value.startswith(('http://', 'https://', 's3://', 'gs://', 'oss://')):
        return {'kind': 'url', 'path': value, 'field': field, 'exists': False}
    if value.startswith('data:image/'):
        return {'kind': 'embedded', 'path': unit['path'], 'field': field,
                'exists': bool(value.partition(',')[2]), 'encoding': 'data_uri'}
    if len(value) > 8192 or '\n' in value: return None
    config = unit.get('config', {})
    for old, new in config.get('path_prefix_map', {}).items():
        if value == old or value.startswith(old.rstrip('/') + '/'):
            value = new.rstrip('/') + value[len(old.rstrip('/')):]
            break
    relative = Path(value.removeprefix('file://'))
    bases = [Path(unit['path']).parent]
    bases.extend(Path(config[k]) for k in ('source_root', 'collection_root') if config.get(k))
    paths = [relative] if relative.is_absolute() else [b / relative for b in bases]
    for candidate in paths:
        candidate = candidate.absolute()
        if _exists(str(candidate)):
            return {'kind': 'file', 'path': str(candidate), 'field': field, 'exists': True}
    return {'kind': 'file', 'path': str(paths[0].absolute()), 'original_path': value,
            'field': field, 'exists': False}


def _generic_refs(raw, unit):
    refs = []
    def provenance(value):
        # Preserve image-local evidence even when resolving nested image_url/path
        # objects. A lineage object is metadata, not another training image.
        metadata = {key: child for key, child in value.items() if key in IMAGE_PROVENANCE_KEYS}
        if isinstance(value.get('metadata'), dict):
            nested = provenance(value['metadata'])
            if nested:
                metadata['metadata'] = nested
        return metadata

    def retain_parent(local, parent):
        if not local:
            return parent
        if not parent:
            return local
        return {**local, 'metadata': retain_parent(local.get('metadata', {}), parent)}

    def visit(value, field='', image_context=False, image_metadata=None):
        if isinstance(value, dict):
            image_metadata = dict(image_metadata or {})
            if image_context:
                local_metadata = provenance(value)
                if image_metadata and local_metadata:
                    # Keep both scopes, including contradictory evidence; a
                    # nested location must not overwrite a parent's AI flag.
                    local_metadata = retain_parent(local_metadata, image_metadata)
                image_metadata = local_metadata or image_metadata
            for key, child in value.items():
                canonical = str(key).lower().replace('_', '')
                context = canonical in {'image','images','img','imagepath','imagepaths','imagefile','imagefilename','imageurl','imageurls','image1','image2','inputimage','outputimage','sourceimage','targetimage'}
                if image_context and canonical in {'path', 'url'}: context = True
                if context or canonical in {'messages','conversations','content'}:
                    visit(child, f'{field}.{key}' if field else str(key), context, image_metadata)
        elif isinstance(value, list):
            for i, child in enumerate(value): visit(child, f'{field}[{i}]', image_context, image_metadata)
        elif image_context:
            ref = _ref(value, unit, field)
            if ref:
                if image_metadata:
                    ref['metadata'] = image_metadata
                refs.append(ref)
    visit(raw)
    if not refs and isinstance(raw, dict) and isinstance(raw.get('url'), str) and any(
            key in raw for key in ('caption', 'prompt', 'width', 'height', 'image_width', 'image_height')):
        ref = _ref(raw['url'], unit, 'url')
        if ref: refs.append(ref)
    return refs


def _as_raw(value):
    return value if isinstance(value, dict) else {'text': value} if isinstance(value, str) else {'_value': value}


def _iter_parquet(unit):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('Parquet adapter requires pyarrow') from exc
    file = pq.ParquetFile(unit['path'])
    selected, binary_columns = [], []
    for i in range(len(file.schema)):
        column = file.schema.column(i)
        logical = str(column.logical_type)
        if column.physical_type in {'BYTE_ARRAY', 'FIXED_LEN_BYTE_ARRAY'} and logical not in {'String', 'JSON'}:
            binary_columns.append((i, column.path))
        else:
            selected.append(column.path)
    row_index = 0
    config = unit.get('config', {})
    for rg in range(file.num_row_groups):
        for batch in file.iter_batches(batch_size=256, row_groups=[rg], columns=selected):
            for raw in batch.to_pylist():
                issues, refs = [], []
                if unit['adapter'] == 'dcvlm_parquet':
                    shard = raw.get('source_shard') or config.get('source_shard')
                    pool = config.get('source_pool')
                    tar_path = str((Path(pool) / shard).absolute()) if pool and shard else None
                    for image in raw.get('images') or []:
                        ref = {**image, 'kind': 'tar', 'path': tar_path,
                               'exists': _exists(tar_path) if tar_path else False}
                        refs.append(ref)
                    if refs and not tar_path: issues.append('dcvlm_source_pool_unresolved')
                else:
                    refs = _generic_refs(raw, unit)
                    for column_index, field in binary_columns:
                        # Only image-like binary fields count; hashes/embeddings do not.
                        if not any(word in field.lower() for word in ('image', 'jpg', 'jpeg', 'png', 'webp')):
                            continue
                        stats = file.metadata.row_group(rg).column(column_index).statistics
                        exists = True if stats is not None and stats.null_count == 0 else None
                        refs.append({'kind': 'embedded', 'path': unit['path'], 'field': field,
                                     'row_index': row_index, 'row_group': rg, 'exists': exists})
                        if exists is None: issues.append('embedded_presence_unverified')
                yield _envelope(unit, raw, {'row_index': row_index, 'row_group': rg}, issues, refs)
                row_index += 1


def _merge_sidecar(raw, content, name, issues):
    if isinstance(content, dict):
        for key, value in content.items():
            if key in raw and raw[key] != value:
                issues.append('conflicting_sidecar_fields')
                raw.setdefault('_sidecar_conflicts', []).append({'member': name, 'field': key, 'value': value})
            else:
                raw[key] = value
    elif isinstance(content, str):
        if 'caption' not in raw:
            raw['caption'] = content
        elif raw['caption'] != content:
            raw.setdefault('_alternative_captions', []).append({'member': name, 'text': content})
    else:
        issues.append('unsupported_sidecar_value')


def _iter_loose(unit):
    raw, issues = {}, []
    for sidecar in unit.get('config', {}).get('sidecars', []):
        text = _bounded_read(sidecar)
        content = json.loads(text) if Path(sidecar).suffix.lower() == '.json' else text
        _merge_sidecar(raw, content, sidecar, issues)
    if unit.get('config', {}).get('ambiguous_image_stem'):
        issues.append('ambiguous_image_stem')
    if not raw: issues.append('image_only_unpaired')
    ref = {'kind': 'file', 'path': unit['path'], 'exists': _exists(unit['path'])}
    yield _envelope(unit, raw, {'row_index': 0}, issues, [ref])


def _tar_key(name):
    # WDS keys may include dots; known annotation suffixes are composite.
    for suffix in ('.conversations.txt', '.caption.txt', '.captions.json', '.caption.json'):
        if name.lower().endswith(suffix): return name[:-len(suffix)]
    return str(PurePosixPath(name).with_suffix(''))


def _iter_tar(unit):
    compressed = not unit['path'].lower().endswith('.tar')
    with _scratch_db() as db:
        db.execute('CREATE TABLE members(seq INTEGER PRIMARY KEY, key TEXT, name TEXT, kind TEXT, offset INTEGER, length INTEGER, content TEXT)')
        # Stream compressed TARs; uncompressed TAR uses seek to skip image payloads.
        mode = 'r|*' if compressed else 'r:'
        with tarfile.open(unit['path'], mode=mode) as archive:
            for seq, member in enumerate(archive):
                if not member.isfile(): continue
                kind = _kind(PurePosixPath(member.name).name)
                if kind not in {'loose_image', 'text', 'json'}: continue
                text = None
                if kind in {'text', 'json'}:
                    if member.size > MAX_METADATA_BYTES:
                        raise ValueError(f'TAR metadata member exceeds limit: {member.name}')
                    handle = archive.extractfile(member)
                    if handle is None: raise ValueError(f'Unreadable TAR member: {member.name}')
                    text = handle.read().decode('utf-8-sig')
                db.execute('INSERT INTO members VALUES (?,?,?,?,?,?,?)',
                           (seq, _tar_key(member.name), member.name, kind, member.offset_data, member.size, text))
                if seq % 10000 == 0: db.commit()
                # tarfile otherwise caches every TarInfo, defeating bounded memory.
                archive.members.clear()
        db.commit()
        db.execute('CREATE INDEX member_key ON members(key,seq)')
        for row_index, (key,) in enumerate(db.execute('SELECT DISTINCT key FROM members ORDER BY key')):
            raw, refs, issues, names = {}, [], [], set()
            for seq, name, kind, offset, length, content in db.execute(
                    'SELECT seq,name,kind,offset,length,content FROM members WHERE key=? ORDER BY seq', (key,)):
                if name in names: issues.append('duplicate_tar_member')
                names.add(name)
                if kind == 'loose_image':
                    refs.append({'kind': 'tar', 'path': unit['path'], 'member': name,
                                 'offset': offset, 'length': length, 'exists': True,
                                 'offset_space': 'decompressed_tar' if compressed else 'file'})
                else:
                    value = json.loads(content) if kind == 'json' else content
                    if name.lower().endswith('.conversations.txt'):
                        try: value = {'conversations': json.loads(content)}
                        except json.JSONDecodeError: issues.append('unparsed_conversations_sidecar')
                    _merge_sidecar(raw, value, name, issues)
            if refs and not raw: issues.append('image_only_unpaired')
            if len(refs) > 1: issues.append('ambiguous_tar_image_group')
            if not refs: refs = _generic_refs(raw, unit)
            yield _envelope(unit, raw, {'row_index': row_index, 'member_key': key}, issues, refs)


def iter_records(unit, max_records=None) -> Iterator[dict]:
    """Stream raw envelopes. Limits are explicit; the runner records partial work."""
    if max_records is not None and max_records < 0:
        raise ValueError('max_records must be nonnegative or None')
    if max_records == 0: return
    adapter = unit['adapter']
    if adapter in {'parquet', 'dcvlm_parquet'}:
        iterator = _iter_parquet(unit)
    elif adapter == 'loose_image':
        iterator = _iter_loose(unit)
    elif adapter == 'loose_batch':
        iterator = _iter_loose_batch(unit)
    elif adapter == 'tar':
        iterator = _iter_tar(unit)
    elif adapter == 'unsupported':
        reason = unit.get('config', {}).get('reason', 'unsupported_container')
        raise ValueError(f'{reason}: {unit["path"]}')
    else:
        iterator = _iter_metadata(unit)
    try:
        for i, envelope in enumerate(iterator):
            yield envelope
            if max_records is not None and i + 1 >= max_records: break
    finally:
        close = getattr(iterator, 'close', None)
        if close: close()


def _iter_metadata(unit):
    adapter, path = unit['adapter'], Path(unit['path'])
    if adapter == 'json':
        values = _json_records(path)
        for i, value in enumerate(values):
            raw = _as_raw(value)
            yield _envelope(unit, raw, {'row_index': i}, [], _generic_refs(raw, unit))
    elif adapter == 'jsonl':
        with path.open('rb') as stream:
            line_number = 0
            while True:
                line = stream.readline(MAX_METADATA_BYTES + 1)
                if not line: break
                line_number += 1
                if len(line) > MAX_METADATA_BYTES: raise ValueError('JSONL record exceeds metadata limit')
                text = line.decode('utf-8-sig' if line_number == 1 else 'utf-8')
                if not text.strip(): continue
                raw = _as_raw(json.loads(text))
                yield _envelope(unit, raw, {'row_index': line_number - 1, 'line_number': line_number}, [], _generic_refs(raw, unit))
    elif adapter == 'csv':
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            for i, raw in enumerate(csv.DictReader(stream, delimiter='\t' if path.suffix.lower() == '.tsv' else ',')):
                issues = ['malformed_csv_columns'] if None in raw else []
                yield _envelope(unit, raw, {'row_index': i}, issues, _generic_refs(raw, unit))
    elif adapter == 'text':
        # Standalone TXT is one document. Oversized documents need an explicit chunking rule.
        issues = [] if unit.get('config', {}).get('allow_standalone_text') else ['orphan_text_sidecar_requires_schema']
        yield _envelope(unit, {'text': _bounded_read(path)}, {'row_index': 0}, issues, [])
    else:
        raise ValueError(f'Unknown adapter: {adapter}')
