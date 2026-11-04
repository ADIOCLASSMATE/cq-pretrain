"""Read-only, resumable explicit adapters into a corpus.native.v2 snapshot.

The snapshot stores metadata and bound image ranges, never image copies. A full
run enumerates every declared input; diagnostics are explicitly scoped prefixes.
Unsupported schemas fail before COMPLETE, rather than inventing supervision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tarfile
import tempfile

from corpus_pipeline.dedup.training_release import text_fingerprint
from corpus_pipeline.production.common import atomic_json, canonical, digest, file_sha
from corpus_pipeline.production import genund_ingest
from corpus_pipeline.production.records import snapshot

SCHEMA = 'corpus.native.v2'
ADAPTERS = {'dcvlm_balanced_wds': 'dcvlm', 'dcvlm_native': 'dcvlm',
            'genund_caption_tar': 'genund', 'genund_native': 'genund'}
BALANCED_CONTRACT = 'dcvlm.balanced_webdataset.v1'
MAX_METADATA = 32 << 20
MAX_IMAGE = 256 << 20
MAX_BUFFER_BYTES = 64 << 20
IMAGE_SUFFIXES = {'jpg', 'jpeg', 'png', 'webp'}


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def _stamp(path):
    value = Path(path).stat()
    return {'size': value.st_size, 'mtime_ns': value.st_mtime_ns}


def _bound(root, path):
    root, given = Path(root).resolve(strict=True), Path(path)
    if '..' in given.parts:
        raise ValueError('input path traversal')
    resolved = (root / given).resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError('input path escapes declared root or is not a file')
    return resolved


def _member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('unsafe TAR member path')
    key, sep, suffix = path.name.partition('.')
    if not key or not sep:
        raise ValueError('unsupported WebDataset member name')
    return str(path.parent / key), suffix


def _range_sha(stream, offset, size):
    stream.seek(offset)
    result = hashlib.sha256()
    remaining = size
    while remaining:
        block = stream.read(min(1 << 20, remaining))
        if not block:
            raise ValueError('truncated original TAR component')
        result.update(block)
        remaining -= len(block)
    return result.hexdigest()


def _canonical_ref(ref):
    value = {**json.loads(ref.get('metadata_json', '{}')), **ref}
    value.pop('metadata_json', None)
    if value.get('dataset_id'):
        from corpus_pipeline.production.portable_refs import resolve_ref, _range
        resolve_ref(value)
        offset,size=_range(value)
        return {'kind':value['kind'],'path':value['path'],'member':value['member'],
                'offset_data':offset,'size':size,'offset':offset,'length':size,
                'sha256':value['sha256'],'width':value.get('width'),'height':value.get('height'),
                'metadata_json':canonical(value)}
    if value.get('kind', 'tar') != 'tar':
        raise ValueError('only explicit uncompressed TAR image ranges are supported')
    path = Path(value['path'])
    if not path.is_absolute() or path.suffix.lower() != '.tar' or not path.is_file():
        raise ValueError('image range requires an existing absolute TAR path')
    _member(value['member'])
    offset, size = value.get('offset_data', value.get('offset')), value.get('size', value.get('length'))
    if type(offset) is not int or offset < 0 or type(size) is not int or not 0 < size <= MAX_IMAGE:
        raise ValueError('invalid image byte range')
    if offset + size > path.stat().st_size or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', '')):
        raise ValueError('image range or SHA256 invalid')
    for left, right in (('offset', offset), ('length', size)):
        if left in value and value[left] != right:
            raise ValueError('conflicting image range aliases')
    return {'kind': 'tar', 'path': str(path.resolve()), 'member': value['member'],
            'offset_data': offset, 'size': size, 'offset': offset, 'length': size,
            'sha256': value['sha256'], 'width': value.get('width'), 'height': value.get('height'),
            'metadata_json': canonical(value)}


def _row(collection, original, refs, locator, metadata=None):
    segments = original['segments']
    refs = [_canonical_ref(ref) for ref in refs]
    indexes = [part.get('image_index') for part in segments if part['kind'] == 'image']
    quarantine = (metadata or {}).get('binding_contract') == 'unresolved_original_image_binding.v1'
    if quarantine and ((metadata or {}).get('pairing_state') != 'unresolved'
                       or (metadata or {}).get('prepared_adapter_status') != 'quarantine'
                       or 'unresolved_original_image_binding' not in (metadata or {}).get('issues', [])
                       or indexes or any(part.get('image_index') is not None for part in segments)):
        raise ValueError('invalid unresolved original binding quarantine')
    if original['image_count'] != len(refs) or (not quarantine and set(indexes) != set(range(len(refs)))):
        raise ValueError('incomplete original image bindings')
    for part in segments:
        if part['kind'] == 'image':
            index = part['image_index']
            if type(index) is not int or not 0 <= index < len(refs) or part['source_member'] != refs[index]['member']:
                raise ValueError('image order/member mismatch')
        elif part['kind'] != 'text' or not isinstance(part.get('text'), str):
            raise ValueError('unsupported original segment')
    fingerprint = text_fingerprint(segments)
    if original.get('text_hash') is not None and original['text_hash'] != fingerprint:
        raise ValueError('original text_hash does not match ordered supervision')
    return {'collection': collection, 'source': original['source'], 'sample_id': original['sample_id'],
            'source_shard': original['source_shard'], 'source_key': original['source_key'],
            'format': original['format'], 'segments': segments, 'images': refs,
            'image_count': len(refs), 'text_hash': fingerprint,
            'original_supervision_json': canonical(original), 'original_locator_json': canonical(locator),
            'adapter_metadata_json': canonical(metadata or {})}


def arrow_schema():
    import pyarrow as pa
    segment = pa.struct([('kind', pa.string()), ('text', pa.large_string()), ('role', pa.string()),
                         ('turn', pa.int64()), ('image_index', pa.int64()), ('source_member', pa.string())])
    image = pa.struct([(name, pa.string()) for name in ('kind', 'path', 'member')] +
                      [(name, pa.int64()) for name in ('offset_data', 'size', 'offset', 'length')] +
                      [('sha256', pa.string()), ('width', pa.int64()), ('height', pa.int64()),
                       ('metadata_json', pa.large_string())])
    return pa.schema([(name, pa.string()) for name in ('collection', 'source', 'sample_id', 'source_shard', 'source_key', 'format')] +
                     [('segments', pa.large_list(segment)), ('images', pa.large_list(image)),
                      ('image_count', pa.int64()), ('text_hash', pa.string())] +
                     [(name, pa.large_string()) for name in ('original_supervision_json', 'original_locator_json', 'adapter_metadata_json')])


def _balanced_record(path, relative, key, members, stream, declared_source):
    mandatory = {'conversations.txt', 'data_type.txt', 'ds_name.txt', 'orig_url.txt'}
    if not mandatory <= members.keys():
        raise ValueError('balanced sample lacks required original annotations')
    annotations, components, image_members = {}, [], {}
    for suffix, item in members.items():
        name, offset, size = item
        match = re.fullmatch(r'(\d+)\.(jpg|jpeg|png|webp)', suffix, re.IGNORECASE)
        if match:
            index = int(match[1])
            if index in image_members or not 0 < size <= MAX_IMAGE:
                raise ValueError('ambiguous or oversized balanced image member')
            image_members[index] = item
        elif suffix in mandatory | {'ntok.txt', 'mdp.txt'}:
            if size > MAX_METADATA:
                raise ValueError('balanced metadata exceeds byte budget')
            stream.seek(offset)
            blob = stream.read(size)
            if len(blob) != size:
                raise ValueError('truncated balanced metadata')
            annotations[suffix] = blob.decode('utf-8')
        else:
            raise ValueError('unsupported balanced member suffix: ' + suffix)
        components.append({'member': name, 'offset_data': offset, 'size': size,
                           'sha256': _range_sha(stream, offset, size)})
    if set(image_members) != set(range(len(image_members))):
        raise ValueError('balanced image indices must be contiguous from zero')
    source = annotations['ds_name.txt'].strip()
    data_type = annotations['data_type.txt'].strip()
    if not source or data_type not in {'instruction', 'text', 'captioning', 'mmdoc_mt'}:
        raise ValueError('unknown balanced source/data_type contract')
    if declared_source is not None and source != declared_source:
        raise ValueError('balanced ds_name differs from declared source; no implicit filtering')
    # ``instruction`` is a supervision/task label, not an image-count promise.
    # The pinned baseline contains text-only instruction examples (including
    # visualwebinstruct, allava, and tqa) with no image member or placeholder.
    # Their original label and conversations remain untouched; the serialized
    # format below follows the actual members. Placeholder/range validation
    # still proves every image binding, regardless of the task label.
    if ((data_type == 'text' and image_members)
            or (data_type in {'captioning', 'mmdoc_mt'} and not image_members)):
        raise ValueError('balanced data_type and image membership disagree')
    refs = []
    component_map = {item['member']: item for item in components}
    for index in range(len(image_members)):
        name, offset, size = image_members[index]
        refs.append({'kind': 'tar', 'path': str(path), 'member': name, 'offset_data': offset,
                     'size': size, 'sha256': component_map[name]['sha256']})
    segments, conversations, cursor = [], [], 0
    member = members['conversations.txt'][0]
    for turn in annotations['conversations.txt'].split('<EOCL>\n'):
        if turn == '':
            continue
        role, colon, value = turn.partition(':')
        if not colon or role not in {'human', 'gpt'}:
            raise ValueError('unsupported balanced conversation role')
        ordinal = len(conversations)
        conversations.append({'from': role, 'value': value})
        role = {'human': 'user', 'gpt': 'assistant'}[role]
        pieces = value.split('<image>')
        for index, piece in enumerate(pieces):
            if piece or len(pieces) == 1:
                segments.append({'kind': 'text', 'text': piece, 'role': role, 'turn': ordinal,
                                 'image_index': None, 'source_member': member})
            if index + 1 < len(pieces):
                if cursor >= len(refs):
                    raise ValueError('balanced image placeholder exceeds members')
                segments.append({'kind': 'image', 'text': None, 'role': role, 'turn': ordinal,
                                 'image_index': cursor, 'source_member': refs[cursor]['member']})
                cursor += 1
    if cursor != len(refs) or not conversations:
        raise ValueError('balanced conversation leaves unbound images or turns')
    locator = {'adapter': BALANCED_CONTRACT, 'path': str(path), 'source_shard': relative,
               'source_key': key, 'components': components, 'archive_fingerprint': _stamp(path)}
    original = {'sample_id': digest({'contract': BALANCED_CONTRACT, 'path': str(path), 'key': key,
                                    'components': components}), 'source': source, 'source_shard': relative,
                'source_key': key, 'format': 'interleaved' if data_type == 'mmdoc_mt' else
                ('conversation' if refs else 'text_conversation'), 'image_count': len(refs),
                'segments': segments, 'conversations': conversations, 'source_annotations': annotations,
                'data_type': data_type, 'ds_name': source, 'orig_url': annotations['orig_url.txt']}
    return _row('dcvlm', original, refs, locator, {'adapter': BALANCED_CONTRACT,
                'pairing_state': 'verified', 'prepared_adapter_status': 'ready'})


def _balanced_rows(unit, record_parser=None):
    record_parser = record_parser or _balanced_record
    path = Path(unit['path'])
    # A small disk-backed uniqueness index avoids retaining a shard's keys in RAM.
    with tempfile.TemporaryDirectory(prefix='balanced-keys-') as temporary, path.open('rb') as stream:
        db = sqlite3.connect(str(Path(temporary) / 'keys.sqlite'))
        try:
            db.execute('CREATE TABLE keys (key TEXT PRIMARY KEY)')
            key, members = None, {}
            with tarfile.open(path, 'r:') as archive:
                for member in archive:
                    if member.isdir():
                        archive.members.clear()
                        continue
                    if not member.isfile():
                        raise ValueError('unsupported nonregular TAR member')
                    current, suffix = _member(member.name)
                    if current != key:
                        if key is not None:
                            yield record_parser(path, unit['relative_path'], key, members, stream, unit['source'])
                        try:
                            db.execute('INSERT INTO keys VALUES (?)', (current,))
                        except sqlite3.IntegrityError as error:
                            raise ValueError('noncontiguous or duplicate WebDataset key') from error
                        key, members = current, {}
                    if suffix in members:
                        raise ValueError('duplicate balanced sample member')
                    members[suffix] = (member.name, member.offset_data, member.size)
                    archive.members.clear()
                if key is not None:
                    yield record_parser(path, unit['relative_path'], key, members, stream, unit['source'])
        finally:
            db.close()


def _native_rows(unit):
    import pyarrow.parquet as pq
    entry, pool = unit['native_entry'], unit.get('source_pool')
    index = 0
    for batch in pq.ParquetFile(unit['path']).iter_batches(batch_size=128):
        for raw in batch.to_pylist():
            if raw['source'] != entry['source'] or raw['source_shard'] != entry['source_shard']:
                raise ValueError('native input row/catalog identity mismatch')
            if raw.get('collection', 'dcvlm') != unit['collection']:
                raise ValueError('native adapter cannot relabel another collection')
            if unit['adapter']=='genund_native':
                if unit['native_schema']!=SCHEMA:
                    raise ValueError('genund native requires corpus.native.v2')
                yield raw
                index += 1
                continue
            refs = raw['images']
            if unit['native_schema'] == 'dcvlm.pretrain.v1':
                tar = _bound(pool, raw['source_shard'])
                refs = [{**ref, 'kind': 'tar', 'path': str(tar)} for ref in refs]
            locator = {'adapter': 'dcvlm.native.v1', 'path': unit['path'], 'row_index': index,
                       'source_shard': raw['source_shard'], 'source_key': raw['source_key'],
                       'parquet_fingerprint': _stamp(unit['path'])}
            yield _row('dcvlm', raw, refs, locator, json.loads(raw.get('adapter_metadata_json', '{}')))
            index += 1
    if index != entry['rows']:
        raise ValueError('native input row count differs from catalog')


def _genund_rows(unit, rules):
    manifest = genund_ingest.build_manifest(unit['root'], unit['source'], [unit['path']],
                                           **unit.get('caption_adapter_options', {}))
    snap = genund_ingest.snapshot_for_manifest(manifest)
    for record in genund_ingest.iter_genund_records(manifest, snap, rules):
        metadata = {key: record[key] for key in ('pairing_state', 'pairing_evidence_scope',
                    'prepared_adapter_status', 'caption_fields_inspected', 'caption_contract',
                    'prepared_semantic_task', 'caption_candidates')}
        metadata.update(adapter=genund_ingest.CONTRACT, issues=record['normalized']['issues'])
        if 'adapter_options' in manifest or 'processing_policy' in record['raw']:
            metadata.update(direction=record['raw']['direction'],
                            image_loss_allowed=record['raw']['image_loss_allowed'],
                            caption_variant_fields=record['raw'].get('caption_variant_fields', []))
        if 'processing_policy' in record['raw']:
            metadata['processing_policy'] = record['raw']['processing_policy']
        yield _row('genund', record['raw'], record['source_image_refs'], record['locator'], metadata)


def enumerate_inputs(inputs):
    if not isinstance(inputs, list) or not inputs:
        raise ValueError('inputs must be a nonempty list')
    units, seen = [], set()
    for spec in inputs:
        adapter, collection = spec['adapter'], spec['collection']
        if adapter not in ADAPTERS or ADAPTERS[adapter] != collection:
            raise ValueError('unsupported collection/adapter contract')
        caption_options = None
        source_directions = None
        if 'source_directions' in spec:
            if 'direction' in spec:
                raise ValueError('direction and source_directions are mutually exclusive')
            mapping = spec['source_directions']
            if (not isinstance(mapping, dict) or not mapping
                    or any(not isinstance(key, str) or not key or value not in ('i2t', 't2i')
                           for key, value in mapping.items())):
                raise ValueError('source_directions requires a nonempty source-to-i2t/t2i mapping')
            source_directions = dict(sorted(mapping.items()))
        if any(key in spec for key in ('preserve_all_captions', 'direction', 'source_directions')):
            if adapter != 'genund_caption_tar':
                raise ValueError('explicit caption options require the genund caption adapter')
            caption_options = genund_ingest.adapter_options(
                preserve_all_captions=spec.get('preserve_all_captions', False), direction=spec.get('direction', 't2i'))
        root = Path(spec['root']).resolve(strict=True)
        if not root.is_dir():
            raise ValueError('input root must be a mounted directory')
        source = spec.get('source')
        selected = spec.get('paths')
        if selected is not None and (not isinstance(selected, list) or not selected):
            raise ValueError('explicit paths must be a nonempty list')
        if adapter in ('dcvlm_native','genund_native'):
            snap = snapshot(root)
            available = {path: entry for path, entry in snap['catalog_index'].items()
                         if source is None or entry['source'] == source}
            if source is not None and not available:
                raise ValueError('unknown requested native source')
            paths = [_bound(root, path) for path in selected] if selected is not None else [Path(path) for path in available]
            if any(str(path) not in available for path in paths):
                raise ValueError('selected native path is not in the declared source catalog')
        elif adapter == 'dcvlm_balanced_wds':
            shard_root = root / 'shards'
            paths = [_bound(root, path) for path in selected] if selected is not None else sorted(shard_root.rglob('*.tar'))
            if selected is None and any(path.is_file() and path.suffix.lower() != '.tar' for path in shard_root.rglob('*')):
                raise ValueError('unsupported carrier in balanced shards; explicit adapter required')
            if any(shard_root.resolve() not in path.resolve().parents for path in paths):
                raise ValueError('balanced input must use official shards/, never preview')
        else:
            sources = [source] if source is not None else sorted(path.name for path in root.iterdir() if path.is_dir())
            if not sources or any(value not in genund_ingest.SUPPORTED_SOURCES for value in sources):
                raise ValueError('unsupported gen-und source contract; explicit adapter required')
            if source_directions is not None and set(source_directions) != set(sources):
                raise ValueError('source_directions must exactly cover the enumerated gen-und sources')
            if selected is None:
                for value in sources:
                    for candidate in (root / value).rglob('*'):
                        if candidate.is_file() and candidate.suffix.lower() != '.tar' and candidate.name.lower() not in {
                                'readme.md', 'readme.txt', 'license', 'license.txt', 'dataset_info.json', '.gitattributes'}:
                            raise ValueError('unsupported gen-und file carrier; full source enumeration cannot skip ' + str(candidate))
            paths = [_bound(root, path) for path in selected] if selected is not None else [path for value in sources for path in sorted((root / value).rglob('*.tar'))]
            # These contracts consume TARs only, never silently skip extracted image datasets.
            if selected is None and any(not list((root / value).rglob('*.tar')) for value in sources):
                raise ValueError('gen-und source has no supported TAR carrier; extracted/Parquet adapter required')
        if not paths:
            raise ValueError('input selection has no supported source files')
        for path in paths:
            path = _bound(root, path)
            if str(path) in seen:
                raise ValueError('duplicate input source file')
            seen.add(str(path))
            selected_source = source
            if adapter == 'genund_caption_tar':
                selected_source = path.relative_to(root).parts[0]
                if selected_source not in sources:
                    raise ValueError('gen-und selected path outside declared source')
                genund_ingest._bound_path(root, selected_source, path)
            if adapter != 'dcvlm_native' and path.suffix.lower() != '.tar':
                raise ValueError('adapter requires uncompressed TAR inputs')
            unit = {'adapter': adapter, 'collection': collection, 'root': str(root), 'source': selected_source,
                    'path': str(path), 'relative_path': str(path.relative_to(root)), 'fingerprint': _stamp(path),
                    'scope': 'selected_paths' if selected is not None else ('selected_source' if source else 'full_declared_root')}
            if caption_options is not None:
                unit['caption_adapter_options'] = {**caption_options}
                if source_directions is not None:
                    unit['source_directions'] = source_directions
                    unit['caption_adapter_options']['direction'] = source_directions[selected_source]
            if adapter in ('dcvlm_native','genund_native'):
                unit['snapshot_fingerprint'] = snap['fingerprint']
                unit['native_entry'] = available[str(path)]
                unit['native_schema'] = snap['manifest']['schema_version']
                unit['source_pool'] = snap['manifest'].get('source_pool')
                unit['allowed_image_roots'] = snap['manifest'].get('allowed_image_roots', [unit['source_pool']])
                unit['portable_datasets'] = snap['manifest'].get('portable_datasets',[])
            unit['unit_id'] = digest(unit)
            units.append(unit)
    return units


def _verify_receipt(output, unit, receipt, limit):
    if receipt.get('unit') != unit or receipt.get('limit') != limit or receipt.get('complete') is not True:
        raise ValueError('ingest receipt identity mismatch')
    if _stamp(unit['path']) != unit['fingerprint'] or file_sha(unit['path']) != receipt['source_sha256']:
        raise ValueError('original source changed since ingest receipt')
    for item in receipt['catalog']:
        path = _bound(output, item['path'])
        if path.stat().st_size != item['bytes'] or file_sha(path) != item['sha256']:
            raise ValueError('committed normalized Parquet changed')
    return receipt


def _process_unit(output, unit, rules, limit, rows_per_file):
    import pyarrow as pa
    import pyarrow.parquet as pq
    directory = output / 'units' / unit['unit_id']
    receipt_path = directory / 'COMPLETE.json'
    if receipt_path.exists():
        return _verify_receipt(output, unit, json.loads(receipt_path.read_text()), limit)
    if directory.exists():
        shutil.rmtree(directory)  # Only uncommitted output owned by this input unit.
    directory.mkdir(parents=True)
    if _stamp(unit['path']) != unit['fingerprint']:
        raise ValueError('original source changed after input enumeration')
    source_sha = file_sha(unit['path'])
    if unit['adapter'] in ('dcvlm_native','genund_native') and unit['native_schema'] == SCHEMA:
        if source_sha != unit['native_entry']['sha256']:
            raise ValueError('native v2 original Parquet SHA256 differs from published catalog')
    buffers, buffer_sizes, catalog = defaultdict(list), Counter(), []
    rows = slots = buffered = buffered_bytes = 0
    formats, sources, adapter_statuses = Counter(), Counter(), Counter()

    def flush(key):
        nonlocal buffered, buffered_bytes
        values = buffers.pop(key)
        target = directory / f'{len(catalog):06d}.parquet'
        temporary = target.with_suffix('.partial')
        table = pa.Table.from_pylist(values, schema=arrow_schema())
        pq.write_table(table, temporary, compression='zstd', row_group_size=min(128, rows_per_file))
        os.replace(temporary, target)
        catalog.append({'path': str(target.relative_to(output)), 'rows': len(values),
                        'bytes': target.stat().st_size, 'sha256': file_sha(target),
                        'collection': key[0], 'source': key[1], 'source_shard': key[2]})
        buffered -= len(values)
        buffered_bytes -= buffer_sizes.pop(key)

    if unit['adapter']=='genund_loose':
        from .genund_loose import iter_unit_rows
        iterator=iter_unit_rows(unit)
    elif unit['adapter'] in ('genund_openuni','genund_scalecap','genund_finevision'):
        from corpus_pipeline.production.genund_collections import iter_records
        iterator=(_row('genund',r['raw'],r['source_image_refs'],r['locator'],r['metadata'])
                  for r in iter_records(unit,rules))
    else:
        iterator = (_balanced_rows(unit) if unit['adapter'] == 'dcvlm_balanced_wds' else
                    _native_rows(unit) if unit['adapter'] in ('dcvlm_native','genund_native') else _genund_rows(unit, rules))
    exhausted = False
    try:
        while limit is None or rows < limit:
            try:
                row = next(iterator)
            except StopIteration:
                exhausted = True
                break
            key = (row['collection'], row['source'], row['source_shard'])
            buffers[key].append(row)
            row_bytes = sum(len(row[name].encode()) for name in ('original_supervision_json', 'original_locator_json', 'adapter_metadata_json'))
            row_bytes += sum(len((part.get('text') or '').encode()) + 256 for part in row['segments'])
            row_bytes += sum(len(ref['metadata_json'].encode()) + len(ref['path'].encode()) + 512 for ref in row['images']) + 1024
            buffer_sizes[key] += row_bytes
            buffered_bytes += row_bytes
            rows += 1
            slots += row['image_count']
            buffered += 1
            formats[row['format']] += 1
            sources[row['collection'] + ':' + row['source']] += 1
            adapter_statuses[json.loads(row['adapter_metadata_json']).get('prepared_adapter_status', 'ready')] += 1
            while buffered >= rows_per_file or buffered_bytes >= MAX_BUFFER_BYTES:
                flush(max(buffers, key=lambda item: buffer_sizes[item]))
        for key in sorted(buffers):
            flush(key)
    finally:
        iterator.close()
    if _stamp(unit['path']) != unit['fingerprint']:
        raise ValueError('original source changed during ingest')
    value = {'schema_version': 'corpus.native.ingest_unit.v1', 'complete': True, 'unit': unit,
             'source_sha256': source_sha, 'limit': limit, 'source_exhausted': exhausted,
             'records': rows, 'image_slots': slots, 'formats': dict(formats), 'sources': dict(sources),
             'adapter_statuses': dict(adapter_statuses), 'catalog': catalog}
    atomic_json(receipt_path, value)
    return value


def build(inputs, output, rules=None, max_records=None, rows_per_file=8192):
    """Build or resume an immutable, SHA-bound metadata snapshot; return COMPLETE.

    max_records is a global diagnostic prefix, never a full-collection result.
    The unit of recovery is one input TAR/Parquet. No image is copied or decoded.
    """
    _positive(rows_per_file, 'rows_per_file')
    if max_records is not None:
        _positive(max_records, 'max_records')
    output = Path(output).resolve()
    units = enumerate_inputs(inputs)
    if any(Path(unit['root']) == output or Path(unit['root']) in output.parents for unit in units):
        raise ValueError('output must be outside read-only input roots')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = {'schema_version': 'corpus.native.ingest_plan.v1', 'units': units,
                'max_records': max_records, 'rows_per_file': rows_per_file, 'rules_sha256': digest(rules or {}),
                'max_buffer_bytes': MAX_BUFFER_BYTES,
                'code_sha256': file_sha(__file__), 'genund_adapter_sha256': file_sha(genund_ingest.__file__),
                'text_fingerprint_helper_sha256': hashlib.sha256(inspect.getsource(text_fingerprint).encode()).hexdigest()}
        plan_path = output / 'PLAN.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError('ingest plan changed; use a new output root')
        if not plan_path.exists():
            atomic_json(plan_path, plan)
        catalog, receipts = [], []
        allowed_image_roots = set()
        for unit in units:
            if unit['adapter'] in ('dcvlm_native','genund_native'):
                allowed_image_roots.update(unit['allowed_image_roots'])
            else:
                allowed_image_roots.add(unit['root'])
        if None in allowed_image_roots:
            raise ValueError('native input lacks bound image storage roots')
        rows = slots = 0
        formats, sources, statuses = Counter(), Counter(), Counter()
        for unit in units:
            if max_records is not None and rows >= max_records:
                break
            limit = None if max_records is None else max_records - rows
            receipt = _process_unit(output, unit, rules or {}, limit, rows_per_file)
            rows += receipt['records']
            slots += receipt['image_slots']
            catalog.extend(receipt['catalog'])
            formats.update(receipt['formats'])
            sources.update(receipt['sources'])
            statuses.update(receipt['adapter_statuses'])
            path = output / 'units' / unit['unit_id'] / 'COMPLETE.json'
            receipts.append({'path': str(path.relative_to(output)), 'sha256': file_sha(path)})
        if not rows:
            raise ValueError('no records decoded; cannot publish empty snapshot')
        scope = 'bounded_prefix' if max_records is not None else ('selected_inputs' if any(unit['scope'] != 'full_declared_root' for unit in units) else 'full_declared_inputs')
        manifest = {'schema_version': SCHEMA, 'phase': 'complete', 'collections': sorted({item['collection'] for item in catalog}),
                    'allowed_image_roots': sorted(allowed_image_roots),
                    'portable_datasets': sorted({dataset for unit in units for dataset in unit.get('portable_datasets',[])}),
                    'scope': scope, 'source_files_modified': False, 'image_payloads_copied': False,
                    'enumeration_scope': 'all_declared_mounted_files_not_upstream_release_completeness',
                    'plan_sha256': file_sha(plan_path), 'unit_receipts': receipts,
                    'statistics': {'samples': rows, 'image_slots': slots, 'formats': dict(formats),
                                   'sources': dict(sources), 'adapter_statuses': dict(statuses),
                                   'input_units_planned': len(units), 'input_units_processed': len(receipts)}}
        if (output / 'COMPLETE.json').exists():
            complete = json.loads((output / 'COMPLETE.json').read_text())
            if (json.loads((output / 'manifest.json').read_text()) != manifest or
                    json.loads((output / 'catalog.json').read_text()) != catalog or
                    complete.get('manifest_sha256') != file_sha(output / 'manifest.json') or
                    complete.get('catalog_sha256') != file_sha(output / 'catalog.json')):
                raise ValueError('completed native snapshot binding mismatch')
            return complete
        atomic_json(output / 'catalog.json', catalog)
        atomic_json(output / 'manifest.json', manifest)
        complete = {'schema_version': SCHEMA, 'complete': True, 'validation_passed': True,
                    'validation_scope': 'original_structure_and_byte_bindings_not_image_decode_or_quality',
                    'catalog_sha256': file_sha(output / 'catalog.json'), 'manifest_sha256': file_sha(output / 'manifest.json'),
                    'plan_sha256': file_sha(plan_path), 'scope': scope, 'statistics': manifest['statistics']}
        atomic_json(output / 'COMPLETE.json', complete)
        return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, help='JSON list of explicit input contracts')
    parser.add_argument('--output', required=True)
    parser.add_argument('--rules')
    parser.add_argument('--max-records', type=int)
    parser.add_argument('--rows-per-file', type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(build(json.loads(Path(args.inputs).read_text()), args.output,
                           rules=json.loads(Path(args.rules).read_text()) if args.rules else None,
                           max_records=args.max_records, rows_per_file=args.rows_per_file), ensure_ascii=False))


if __name__ == '__main__':
    main()
