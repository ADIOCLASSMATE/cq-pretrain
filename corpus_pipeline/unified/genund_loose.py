"""Read-only adapters for the mounted loose-file gen-und sources.

BLIP plans walk the huge raw directory once, using bounded name buffers. Dense
plans stream the authoritative JSONL once; workers consume only their bound byte
range. PixMo binds upstream QA to an explicitly SHA-indexed local image, never a
numeric filename or row order. Plans/rows contain references, not image copies.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import time

from corpus_pipeline.dedup.training_release import text_fingerprint
from corpus_pipeline.production.common import atomic_json, atomic_text, canonical, digest, file_sha

CONTRACT = 'genund.loose_original_supervision.v1'
ADAPTER = 'genund_loose'
POLICY = 'corpus.task_policy.v1'
BLIP = 'BLIP3o-Pretrain-Long-Caption'
DENSE = 'DenseFusion-1M'
PIXMO = 'pixmo-ask-model-anything'
MAX_METADATA = 32 << 20
MAX_IMAGE = 256 << 20


def _stamp(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\0' in value:
        raise ValueError('unsafe original relative path')
    value.encode('utf-8')  # Surrogate filenames require an explicit byte-path contract.
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError('unsafe original relative path')
    return path


def _bound(root, relative):
    relative = _relative(relative)
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError('original file escapes dataset root')
    return path


def _options(dataset_id, directions, records_per_unit):
    if not re.fullmatch(r'[^/@\s]+/[^/@\s]+@(?:[0-9a-f]{40}|[0-9a-f]{64})', dataset_id or ''):
        raise ValueError('dataset_id must name a pinned dataset revision')
    directions = list(directions)
    if not directions or len(directions) != len(set(directions)) or any(x not in ('i2t', 't2i') for x in directions):
        raise ValueError('directions must explicitly select unique i2t/t2i views')
    if type(records_per_unit) is not int or records_per_unit < 1:
        raise ValueError('records_per_unit must be positive')
    return {'dataset_id': dataset_id, 'directions': sorted(directions), 'records_per_unit': records_per_unit}


def _setup(root, output, source, options, extra):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('plan output must be outside the read-only source root')
    output.mkdir(parents=True, exist_ok=True)
    identity = {'contract': CONTRACT, 'root': str(root), 'source': source, **options, **extra,
                'adapter_sha256': file_sha(__file__)}
    complete = output / 'PLAN.json'
    if complete.exists():
        plan = json.loads(complete.read_text())
        if plan['identity'] != identity:
            raise ValueError('loose-source plan identity changed; use a new output directory')
        for unit in plan['units']:
            if _stamp(unit['path']) != unit['fingerprint'] or file_sha(unit['path']) != unit['sha256']:
                raise ValueError('sealed work unit changed')
        return root, output, identity, plan['units']
    return root, output, identity, None


def _unit(path, identity, index, count, **extra):
    unit = {'adapter': ADAPTER, 'collection': 'genund', 'root': identity['root'],
            'source': identity['source'], 'source_shard': identity['source'] + f'/unit-{index:06d}',
            'path': str(path), 'fingerprint': _stamp(path), 'sha256': file_sha(path),
            'dataset_id': identity['dataset_id'], 'directions': identity['directions'],
            'record_count': count, 'scope': 'full_declared_root',
            **({'raw_directory_fingerprint': identity['raw_directory_fingerprint']}
               if 'raw_directory_fingerprint' in identity else {}),
            **{key: identity[key] for key in ('portable_datasets', 'recovered_root', 'recovered_asset_bindings')
               if key in identity}, **extra}
    unit['unit_id'] = digest(unit)
    return unit


def prepare_blip_units(root, output, *, dataset_id, directions=('i2t', 't2i'), records_per_unit=32768):
    """Seal a bounded manifest of stems; resumed workers never rewalk raw/.

    A completed plan is reusable without enumeration. If the initial directory
    walk is interrupted, it restarts; filesystem directory cookies are not used
    as durable checkpoints. This performs no per-image stat or image reads.
    """
    options = _options(dataset_id, directions, records_per_unit)
    raw = Path(root) / BLIP / 'raw'
    before = _stamp(raw)
    root, output, identity, previous = _setup(root, output, BLIP, options,
                                             {'raw_directory_fingerprint': before})
    if previous is not None:
        return previous
    counts, names, units = Counter(), [], []
    started = time.monotonic()

    def flush():
        if not names:
            return
        path = output / f'{len(units):06d}.jsonl'
        with atomic_text(path) as handle:
            for stem in names:
                handle.write(canonical({'stem': stem}) + '\n')
        units.append(_unit(path, identity, len(units), len(names), format='blip_stems'))
        names.clear()
        atomic_json(output / 'PROGRESS.json', {'phase': 'enumerating_names', 'sealed_units': len(units),
                    'name_counts': dict(counts), 'elapsed_seconds': time.monotonic() - started})

    with os.scandir(raw) as entries:
        for entry in entries:
            name = entry.name
            if name.endswith('.jpg') or name.endswith('.txt'):
                _relative(name)
                counts['.jpg' if name.endswith('.jpg') else '.txt'] += 1
                if name.endswith('.jpg'):
                    names.append(name[:-4])
                    if len(names) >= records_per_unit:
                        flush()
            else:
                counts['other'] += 1
    flush()
    if _stamp(raw) != before:
        raise ValueError('BLIP raw directory changed during enumeration')
    if not units or counts['.jpg'] != counts['.txt'] or counts['other']:
        raise ValueError('BLIP raw requires exactly JPG/TXT pairs; observed ' + canonical(dict(counts)))
    atomic_json(output / 'PLAN.json', {'identity': identity, 'units': units, 'name_counts': dict(counts),
                                     'image_payloads_read': False, 'source_files_modified': False})
    return units


def _binding_index(value, contract):
    if value.get('contract') != contract:
        raise ValueError('verified image binding index contract differs')
    index = {key: value[key] for key in ('contract', 'path', 'sha256', 'fingerprint')}
    index['path'] = str(Path(index['path']).resolve(strict=True))
    if _stamp(index['path']) != index['fingerprint'] or file_sha(index['path']) != index['sha256']:
        raise ValueError('verified image binding index changed')
    return index


def _binding_asset_declarations(index):
    """Seal recovered dataset permissions into both plan and work-unit identity.

    Native manifests aggregate unit declarations. A valid recovered reference
    therefore needs its pinned dataset ID here before workers emit any rows.
    Only positive bindings can authorize an extra root; quarantine evidence
    cannot add readable datasets.
    """
    bindings = {}
    database = _open_verified_index(index)
    try:
        for saved, in database.execute("SELECT result_json FROM bindings WHERE status='bound'"):
            value = json.loads(saved)
            dataset, location = value.get('asset_dataset_id'), value.get('asset_root')
            if dataset is None and location is None:
                continue
            if not dataset or not location:
                raise ValueError('recovered binding requires both pinned dataset and root')
            _options(dataset, ('i2t',), 1)
            root = Path(location)
            if not root.is_absolute() or not root.is_dir():
                raise ValueError('recovered binding root must be an existing absolute directory')
            root = str(root.resolve(strict=True))
            if dataset in bindings and bindings[dataset] != root:
                raise ValueError('recovered dataset maps to conflicting roots')
            bindings[dataset] = root
    finally:
        database.close()
    if not bindings:
        return {}
    roots = set(bindings.values())
    if len(roots) != 1:
        raise ValueError('one loose source plan may declare only one recovered asset root')
    return {'portable_datasets': sorted(bindings), 'recovered_root': roots.pop(),
            'recovered_asset_bindings': dict(sorted(bindings.items()))}


def prepare_dense_units(root, annotations, output, *, dataset_id, directions=('i2t', 't2i'), records_per_unit=8192,
                        binding_index=None):
    """Bind exact upstream JSONL byte ranges without copying the annotations."""
    options = _options(dataset_id, directions, records_per_unit)
    annotations = Path(annotations).resolve(strict=True)
    stamp = _stamp(annotations)
    extra = {'annotations': str(annotations), 'annotations_fingerprint': stamp}
    if binding_index is not None:
        extra['binding_index'] = _binding_index(binding_index, 'genund.dense4v_exact_png_range_binding.v1')
        extra.update(_binding_asset_declarations(extra['binding_index']))
    root, output, identity, previous = _setup(root, output, DENSE, options, extra)
    if previous is not None:
        return previous
    units, count, start, ordinal = [], 0, 0, 0
    overall, part = hashlib.sha256(), hashlib.sha256()
    with annotations.open('rb') as stream:
        def flush(end):
            nonlocal count, start, part, ordinal
            descriptor = {'annotation_path': str(annotations), 'annotation_fingerprint': stamp,
                          'offset': start, 'length': end - start, 'first_row': ordinal,
                          'records': count, 'sha256': part.hexdigest()}
            variant = {}
            if binding_index is not None:
                descriptor['binding_index'] = extra['binding_index']
                variant['source_shard'] = DENSE + f'/4v/unit-{len(units):06d}'
            path = output / f'{len(units):06d}.json'
            atomic_json(path, descriptor)
            units.append(_unit(path, identity, len(units), count, format='dense_jsonl', **variant))
            ordinal += count
            count, start, part = 0, end, hashlib.sha256()
        while True:
            line = stream.readline(MAX_METADATA + 1)
            if not line:
                break
            if len(line) > MAX_METADATA or not line.strip():
                raise ValueError('oversized or empty DenseFusion annotation line')
            overall.update(line); part.update(line); count += 1
            if count >= records_per_unit:
                flush(stream.tell())
        if count:
            flush(stream.tell())
    if _stamp(annotations) != stamp or not units:
        raise ValueError('DenseFusion annotation changed or is empty')
    atomic_json(output / 'PLAN.json', {'identity': identity, 'units': units,
                                     'annotations_sha256': overall.hexdigest(), 'annotation_rows': ordinal})
    return units


def prepare_dense_repair_units(root, annotations, binding_index, output, *, dataset_id, directions=('i2t', 't2i')):
    """Read repaired originals at their unchanged 1M annotation byte offsets."""
    index = _binding_index(binding_index, 'genund.dense1m_recovered_png_binding.v1')
    annotations = Path(annotations).resolve(strict=True)
    root, output, identity, previous = _setup(root, output, DENSE, _options(dataset_id, directions, 1),
        {'annotations': str(annotations), 'annotations_fingerprint': _stamp(annotations),
         'binding_index': index, 'annotation_variant': 'repaired_original_1m',
         **_binding_asset_declarations(index)})
    if previous is not None:
        return previous
    units = []
    db = _open_verified_index(index)
    try:
        for saved, in db.execute('SELECT result_json FROM bindings ORDER BY original_path'):
            binding = json.loads(saved); locator = binding['annotation_locator']
            if locator['annotation_path'] != str(annotations):
                raise ValueError('repaired Dense annotation belongs to a different original file')
            descriptor = {'annotation_path': str(annotations), 'annotation_fingerprint': _stamp(annotations),
                          'offset': locator['offset'], 'length': locator['length'], 'records': 1,
                          'first_row': locator['row_index'], 'sha256': locator['annotation_sha256'],
                          'binding_index': index, 'annotation_variant': 'repaired_original_1m'}
            path = output / f'{len(units):06d}.json'; atomic_json(path, descriptor)
            units.append(_unit(path, identity, len(units), 1, format='dense_jsonl', scope='verified_recovery_subset',
                               source_shard=DENSE + f'/repaired_original_1m/unit-{len(units):06d}'))
    finally:
        db.close()
    atomic_json(output / 'PLAN.json', {'identity': identity, 'units': units,
                                      'annotation_rows': len(units), 'logical_parent': DENSE})
    return units


def build_pixmo_image_index(root, output):
    """Resume at completed image-directory boundaries, never guess QA bindings.

    Re-encoded local JPGs will not match an upstream encoded SHA and must use an
    independently verified annotation/path map. No row-position fallback exists.
    """
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('image index must be outside the read-only source root')
    output.parent.mkdir(parents=True, exist_ok=True)
    source = root / PIXMO
    directories = [(directory, _stamp(directory)) for directory in sorted(source.iterdir()) if directory.is_dir()]
    identity = canonical({'root': str(root), 'source': PIXMO,
                          'directories': {path.name: stamp for path, stamp in directories}})
    if not directories:
        raise ValueError('PixMo image directory list is empty')
    temporary = output.with_suffix('.partial')
    db = sqlite3.connect(output if output.exists() else temporary)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS images (sha256 TEXT, relative_path TEXT UNIQUE, size INTEGER, mtime_ns INTEGER)')
        db.execute('CREATE TABLE IF NOT EXISTS completed_directories (name TEXT PRIMARY KEY, fingerprint TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        saved = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if saved and saved[0] != identity:
            raise ValueError('PixMo image index input directories changed')
        if not saved:
            db.execute('INSERT INTO metadata VALUES (?,?)', ('identity', identity)); db.commit()
        for directory, directory_stamp in directories:
            done = db.execute('SELECT fingerprint FROM completed_directories WHERE name=?', (directory.name,)).fetchone()
            if done:
                if done[0] != canonical(directory_stamp):
                    raise ValueError('completed PixMo image directory changed')
                continue
            if output.exists():
                raise ValueError('published PixMo image index is incomplete')
            # One transaction per directory: an interrupted directory rolls back.
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.endswith('.jpg'):
                        raise ValueError('unregistered PixMo image shard member')
                    path = _bound(root, str(Path(entry.path).relative_to(root)))
                    before = _stamp(path)
                    checksum = file_sha(path)
                    if _stamp(path) != before or not 0 < before['size'] <= MAX_IMAGE:
                        raise ValueError('PixMo image changed or is oversized')
                    db.execute('INSERT INTO images VALUES (?,?,?,?)',
                               (checksum, str(path.relative_to(root)), before['size'], before['mtime_ns']))
            if _stamp(directory) != directory_stamp:
                raise ValueError('PixMo image directory changed during indexing')
            db.execute('INSERT INTO completed_directories VALUES (?,?)', (directory.name, canonical(directory_stamp)))
            db.commit()
        db.execute('CREATE INDEX IF NOT EXISTS image_sha ON images(sha256)')
        db.commit()
    finally:
        db.close()
    if not output.exists():
        os.replace(temporary, output)
    return {'path': str(output), 'sha256': file_sha(output), 'fingerprint': _stamp(output)}


def prepare_pixmo_units(root, annotations, image_index, output, *, dataset_id, binding_index=None):
    """Each official Parquet row group is one resumable QA unit."""
    import pyarrow.parquet as pq
    options = _options(dataset_id, ('i2t',), 1)
    paths = [Path(path).resolve(strict=True) for path in annotations]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('explicit PixMo annotation shards must be nonempty and unique')
    if binding_index is not None:
        if image_index is not None:
            raise ValueError('PixMo requires exactly one image binding mode')
        index = _binding_index(binding_index, 'genund.pixmo_verified_reencoding.v1')
        index_key = 'binding_index'
    else:
        index = {'path': str(Path(image_index['path']).resolve(strict=True)),
                 'sha256': image_index['sha256'], 'fingerprint': image_index['fingerprint']}
        if _stamp(index['path']) != index['fingerprint'] or file_sha(index['path']) != index['sha256']:
            raise ValueError('PixMo image index changed')
        index_key = 'image_index'
    root, output, identity, previous = _setup(root, output, PIXMO, options,
        {'annotations': [{'path': str(path), 'fingerprint': _stamp(path)} for path in paths], index_key: index,
         **(_binding_asset_declarations(index) if binding_index is not None else {})})
    if previous is not None:
        return previous
    units = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        if not {'image_url', 'image_sha256', 'question', 'answer'} <= set(parquet.schema_arrow.names):
            raise ValueError('PixMo requires the official image SHA and question/answer fields')
        checksum = file_sha(path)
        for group in range(parquet.num_row_groups):
            descriptor = {'annotation_path': str(path), 'annotation_fingerprint': _stamp(path),
                          'annotation_sha256': checksum, 'row_group': group, index_key: index}
            target = output / f'{len(units):06d}.json'
            atomic_json(target, descriptor)
            units.append(_unit(target, identity, len(units), parquet.metadata.row_group(group).num_rows,
                               format='pixmo_parquet'))
    atomic_json(output / 'PLAN.json', {'identity': identity, 'units': units})
    return units


def _read_bound(path, maximum):
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not 0 <= before.st_size <= maximum:
            raise ValueError('original component exceeds byte budget')
        value = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns) or len(value) != before.st_size:
        raise ValueError('original component changed while reading')
    return value


def _image_ref(unit, relative, expected_sha=None):
    path = _bound(unit['root'], relative)
    blob = _read_bound(path, MAX_IMAGE)
    if not blob:
        raise ValueError('empty original image')
    checksum = hashlib.sha256(blob).hexdigest()
    if expected_sha is not None and checksum != expected_sha:
        raise ValueError('original image does not match annotation SHA')
    value = {'kind': 'file', 'dataset_id': unit['dataset_id'], 'path': relative, 'member': relative,
             'offset': 0, 'offset_data': 0, 'length': len(blob), 'size': len(blob), 'sha256': checksum,
             'width': None, 'height': None}
    return {**value, 'metadata_json': canonical(value)}


def _segment(kind, *, text=None, role=None, turn=None, source_member=None):
    return {'kind': kind, 'text': text, 'role': role, 'turn': turn,
            'image_index': 0 if kind == 'image' else None, 'source_member': source_member}


def _native(unit, ref, segments, locator, direction, source_key, caption=False):
    locator = {**locator, 'adapter': CONTRACT, 'source': unit['source'], 'view_direction': direction,
               'processing_policy': POLICY, 'dataset_id': unit['dataset_id']}
    fingerprint = text_fingerprint(segments)
    sample_id = digest({'source': unit['source'], 'dataset_id': unit['dataset_id'], 'annotation': locator,
                        'image_sha256': ref['sha256'], 'ordered_text_hash': fingerprint})
    original = {'sample_id': sample_id, 'source': unit['source'], 'source_shard': unit['source_shard'],
                'source_key': source_key, 'format': 'caption' if caption else 'conversation',
                'direction': direction, 'processing_policy': POLICY, 'image_loss_allowed': direction == 't2i',
                'image_count': 1, 'segments': segments, 'source_annotation_locator': locator,
                'caption_origin': 'ai' if caption else 'human'}
    metadata = {'adapter': CONTRACT, 'direction': direction, 'processing_policy': POLICY,
                'image_loss_allowed': direction == 't2i', 'pairing_state': 'verified',
                'prepared_adapter_status': 'ready', 'caption_fields_inspected': caption,
                'pairing_evidence_scope': 'original_path_or_sha_and_annotation_binding_not_semantic_QC'}
    if caption:
        target = next(i for i, segment in enumerate(segments) if segment['kind'] == 'text')
        metadata['caption_contract'] = {'adapter': CONTRACT, 'target_segment_index': target, 'image_index': 0,
            'allowed_training_lengths': ['short', 'medium', 'long'],
            'language_policy': 'preserve_original_target_language_and_explicit_user_language_constraints',
            'preserve_original_user_instruction': True}
        metadata['caption_candidates'] = [{'text': segments[target]['text'], 'field': locator['field'],
            'origin': 'ai', 'annotation_kind': 'existing_annotation', 'image_index': 0, 'granularity': 'original',
            'source_ref': {'locator': locator, 'field': locator['field']}}]
    return {'collection': 'genund', **{key: original[key] for key in ('source', 'sample_id', 'source_shard', 'source_key', 'format')},
            'segments': segments, 'images': [ref], 'image_count': 1, 'text_hash': fingerprint,
            'original_supervision_json': canonical(original), 'original_locator_json': canonical(locator),
            'adapter_metadata_json': canonical(metadata)}


def _caption_views(unit, ref, text, locator, source_key):
    if not isinstance(text, str):
        raise ValueError('caption must retain its original string value')
    # Empty targets remain explicit caption-recovery work, never fabricated text.
    for direction in unit['directions']:
        image = _segment('image', source_member=ref['member'])
        caption = _segment('text', text=text, source_member=locator['field'])
        yield _native(unit, ref, [image, caption] if direction == 'i2t' else [caption, image],
                      locator, direction, source_key, caption=True)


def _blip_rows(unit):
    directory = Path(unit['root']) / BLIP / 'raw'
    if _stamp(directory) != unit['raw_directory_fingerprint']:
        raise ValueError('BLIP raw directory changed after planning')
    with Path(unit['path']).open() as stream:
        for line in stream:
            stem = json.loads(line)['stem']
            _relative(stem)
            relative = BLIP + '/raw/' + stem
            text_path = _bound(unit['root'], relative + '.txt')
            blob = _read_bound(text_path, MAX_METADATA)
            locator = {'annotation_kind': 'text_file', 'annotation_path': relative + '.txt',
                       'annotation_sha256': hashlib.sha256(blob).hexdigest(), 'field': relative + '.txt'}
            ref = _image_ref(unit, relative + '.jpg')
            yield from _caption_views(unit, ref, blob.decode('utf-8-sig'), locator, stem)
    if _stamp(directory) != unit['raw_directory_fingerprint']:
        raise ValueError('BLIP raw directory changed during ingest')


def _open_verified_index(index):
    if _stamp(index['path']) != index['fingerprint']:
        raise ValueError('verified binding index changed')
    _verify_file_sha(index['path'], index['sha256'])
    return sqlite3.connect('file:' + index['path'] + '?mode=ro', uri=True)


def _legacy_dense4v_zip_evidence(proof, original_path, contract):
    """Read sealed v1 evidence whose ZIP method overwrote the method label.

    The original index SHA and the per-image SHA checks remain mandatory.
    Historical receipts are left intact; incomplete ZIP evidence is rejected.
    """
    if contract != 'genund.dense4v_exact_png_range_binding.v1':
        return False
    method = proof.get('method')
    if type(method) is not int or method not in (0, 8):
        return False
    parts = _relative(original_path).parts
    if len(parts) != 3 or parts[0] != 'DenseFusion-4V-100K':
        return False
    if proof.get('archive_path') != 'images/DenseFusion-4V-100K/' + parts[1] + '.zip':
        return False
    if Path(proof.get('member', '')).name != parts[2]:
        return False
    for key, length in [('revision', 40), ('archive_lfs_sha256', 64)]:
        value = proof.get(key)
        if not isinstance(value, str) or len(value) != length or any(c not in '0123456789abcdef' for c in value):
            return False
    for key in ('header_offset', 'compressed_size', 'size', 'crc32'):
        if type(proof.get(key)) is not int:
            return False
    offset, compressed, size, crc = (proof[k] for k in ('header_offset', 'compressed_size', 'size', 'crc32'))
    requested = proof.get('requested_range')
    return bool(offset >= 0 and 0 < compressed <= MAX_IMAGE and 0 < size <= MAX_IMAGE
                and 0 <= crc <= 0xffffffff and proof.get('full_archive_sha_validated') is False
                and isinstance(requested, list) and len(requested) == 2
                and all(type(value) is int for value in requested)
                and requested[0] == offset and requested[1] >= offset + 30 + compressed - 1
                and (method != 0 or compressed == size))


def _dense_rows(unit, descriptor):
    path = Path(descriptor['annotation_path'])
    checksum = hashlib.sha256()
    index = descriptor.get('binding_index')
    db = _open_verified_index(index) if index else None
    try:
        with path.open('rb') as stream:
            stream.seek(descriptor['offset'])
            for ordinal in range(descriptor['records']):
                offset = stream.tell()
                line = stream.readline(MAX_METADATA + 1)
                if not line or len(line) > MAX_METADATA:
                    raise ValueError('DenseFusion annotation byte range truncated or oversized')
                checksum.update(line)
                value = json.loads(line)
                relative, image_id = value['image_path'], value['image_id']
                repaired = index and index.get('contract') == 'genund.dense1m_recovered_png_binding.v1'
                prefix = 'DenseFusion-4V-100K' if index and not repaired else DENSE
                if not isinstance(image_id, str) or not image_id or str(_relative(relative)).split('/')[0] != prefix or Path(relative).stem != image_id:
                    raise ValueError('DenseFusion image_path and image_id binding disagree')
                locator = {'annotation_kind': 'jsonl_line', 'annotation_path': str(path), 'offset': offset,
                           'length': len(line), 'annotation_sha256': hashlib.sha256(line).hexdigest(),
                           'row_index': descriptor['first_row'] + ordinal, 'field': 'caption'}
                if repaired:
                    locator['annotation_variant'] = 'repaired_original_1m'
                reason, binding, expected_sha = None, None, None
                if db is not None:
                    saved = db.execute('SELECT status,result_json FROM bindings WHERE original_path=?', (relative,)).fetchone()
                    if saved is None:
                        raise ValueError('Dense4V binding table does not cover original annotation')
                    binding = json.loads(saved[1])
                    if binding.get('original_path') != relative or saved[0] != binding.get('status'):
                        raise ValueError('Dense4V binding key differs from original annotation')
                    locator['image_binding'] = binding
                    if saved[0] != 'bound':
                        reason = ('official_dense1m_repair_unresolved' if repaired
                                  else 'official_dense4v_image_binding_unresolved')
                    else:
                        proof = binding['proof']; expected_sha = binding['local_sha256']
                        required_method = 'recovered_official_zip_png' if repaired else 'official_png_identical_encoded_sha256'
                        method_verified = (proof.get('method') == required_method or
                                           (not repaired and _legacy_dense4v_zip_evidence(proof, relative, index['contract'])))
                        if not method_verified or proof.get('original_sha256') != expected_sha:
                            raise ValueError('Dense4V binding lacks original PNG SHA evidence')
                        relative = binding['relative_path']
                if reason is None:
                    try:
                        asset_unit = unit
                        if binding and binding.get('asset_dataset_id'):
                            _options(binding['asset_dataset_id'], ('i2t', 't2i'), 1)
                            asset_unit = {**unit, 'root': binding['asset_root'], 'dataset_id': binding['asset_dataset_id']}
                        ref = _image_ref(asset_unit, relative, expected_sha)
                    except (FileNotFoundError, ValueError) as error:
                        if isinstance(error, ValueError) and str(error) != 'empty original image':
                            raise
                        reason = ('official_dense_image_path_not_mounted' if isinstance(error, FileNotFoundError)
                                  else 'official_dense_image_file_is_empty')
                if reason is not None:
                    from .genund_quarantine import row as quarantine_row
                    for direction in unit['directions']:
                        image = _segment('image', source_member=value['image_path'])
                        caption = _segment('text', text=value['caption'], source_member='caption')
                        sequence = [image, caption] if direction == 'i2t' else [caption, image]
                        yield quarantine_row(unit, sequence, locator, direction, image_id, reason, value)
                    continue
                yield from _caption_views(unit, ref, value['caption'], locator, image_id)
            if stream.tell() != descriptor['offset'] + descriptor['length'] or checksum.hexdigest() != descriptor['sha256']:
                raise ValueError('DenseFusion annotation byte range SHA mismatch')
    finally:
        if db is not None:
            db.close()


@lru_cache(maxsize=64)
def _cached_file_sha(path, size, mtime_ns, ctime_ns, expected):
    if file_sha(path) != expected:
        raise ValueError('bound metadata file SHA mismatch')
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (size, mtime_ns, ctime_ns):
        raise ValueError('bound metadata file changed during verification')


def _verify_file_sha(path, expected):
    stat = Path(path).stat()
    _cached_file_sha(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)


def _pixmo_rows(unit, descriptor):
    import pyarrow.parquet as pq
    recovered = 'binding_index' in descriptor
    index = descriptor['binding_index' if recovered else 'image_index']
    db = _open_verified_index(index)
    _verify_file_sha(descriptor['annotation_path'], descriptor['annotation_sha256'])
    ordinal = 0
    try:
        parquet = pq.ParquetFile(descriptor['annotation_path'])
        for batch in parquet.iter_batches(batch_size=128, row_groups=[descriptor['row_group']]):
            for value in batch.to_pylist():
                image_sha = value['image_sha256']
                if not isinstance(image_sha, str) or not re.fullmatch('[0-9a-f]{64}', image_sha):
                    raise ValueError('invalid upstream PixMo image SHA')
                valid_qa = all(isinstance(value[key], str) and value[key].strip() for key in ('question', 'answer'))
                if not valid_qa and not recovered:
                    raise ValueError('PixMo requires original nonempty question and answer')
                locator = {'annotation_kind': 'parquet_row', 'annotation_path': descriptor['annotation_path'],
                           'row_group': descriptor['row_group'], 'row_in_group': ordinal,
                           'annotation_sha256': digest(value), 'field': 'question,answer',
                           'original_image_url': value['image_url']}
                binding = None
                if recovered:
                    saved = db.execute('SELECT status,result_json FROM bindings WHERE original_sha256=?', (image_sha,)).fetchone()
                    if saved is None:
                        raise ValueError('PixMo binding table does not cover original annotation')
                    binding = json.loads(saved[1]);locator['image_binding'] = binding
                    if binding.get('original_sha256') != image_sha or saved[0] != binding.get('status'):
                        raise ValueError('PixMo binding key differs from original annotation')
                    if saved[0] == 'bound':
                        proof = binding['proof']
                        if proof.get('source_sha256') != image_sha or proof.get('method') not in (
                            'identical_encoded_sha256', 'identical_oriented_rgba_geometry_sha256', 'exact_reencoded_sha256',
                            'recovered_official_encoded_sha256'):
                            raise ValueError('PixMo binding lacks verified original image SHA evidence')
                        relative, expected_sha = binding['relative_path'], binding['local_sha256']
                    else:
                        relative, expected_sha = 'original_image_sha256:' + image_sha, None
                else:
                    matches = db.execute('SELECT relative_path FROM images WHERE sha256=? ORDER BY relative_path', (image_sha,)).fetchall()
                    if not matches:
                        raise ValueError('PixMo image SHA has no verified local image binding: ' + image_sha)
                    relative, expected_sha = matches[0][0], image_sha
                segments = [_segment('image', role='user', turn=0, source_member=relative),
                            _segment('text', text=value['question'], role='user', turn=0, source_member='question'),
                            _segment('text', text=value['answer'], role='assistant', turn=1, source_member='answer')]
                if recovered and (binding['status'] != 'bound' or not valid_qa):
                    from .genund_quarantine import row as quarantine_row
                    yield quarantine_row(unit, segments, locator, 'i2t', image_sha + ':' + str(ordinal),
                                         ('official_pixmo_image_binding_unresolved' if binding['status'] != 'bound'
                                          else 'official_pixmo_question_or_answer_empty'), value)
                else:
                    asset_unit = unit
                    if recovered and binding.get('asset_dataset_id'):
                        _options(binding['asset_dataset_id'], ('i2t',), 1)
                        asset_unit = {**unit, 'root': binding['asset_root'], 'dataset_id': binding['asset_dataset_id']}
                    ref = _image_ref(asset_unit, relative, expected_sha)
                    yield _native(unit, ref, segments, locator, 'i2t', image_sha + ':' + str(ordinal))
                ordinal += 1
    finally:
        db.close()


def iter_unit_rows(unit):
    """Native.v2 reader contract, consumed directly by the shared ingest writer."""
    if unit.get('adapter') != ADAPTER or unit.get('source') not in (BLIP, DENSE, PIXMO):
        raise ValueError('unsupported loose-file source contract')
    _options(unit['dataset_id'], unit['directions'], unit['record_count'])
    if unit.get('unit_id') != digest({key: value for key, value in unit.items() if key != 'unit_id'}):
        raise ValueError('loose-file unit identity changed')
    if _stamp(unit['path']) != unit['fingerprint'] or file_sha(unit['path']) != unit['sha256']:
        raise ValueError('sealed loose-file work unit changed')
    if unit['format'] == 'blip_stems' and unit['source'] == BLIP:
        yield from _blip_rows(unit)
        return
    descriptor = json.loads(Path(unit['path']).read_text())
    annotation = descriptor['annotation_path']
    if _stamp(annotation) != descriptor['annotation_fingerprint']:
        raise ValueError('original annotation file changed')
    if unit['format'] == 'dense_jsonl' and unit['source'] == DENSE:
        yield from _dense_rows(unit, descriptor)
    elif unit['format'] == 'pixmo_parquet' and unit['source'] == PIXMO and unit['directions'] == ['i2t']:
        yield from _pixmo_rows(unit, descriptor)
    else:
        raise ValueError('loose-file unit format/source/direction mismatch')
    if _stamp(annotation) != descriptor['annotation_fingerprint']:
        raise ValueError('original annotation file changed during ingest')


def read_original_annotation(locator, *, dataset_roots=None):
    """Recover the complete original annotation bound to a derived view."""
    kind = locator['annotation_kind']
    if kind == 'text_file':
        if not dataset_roots or locator['dataset_id'] not in dataset_roots:
            raise ValueError('original TXT annotation requires its dataset root mapping')
        blob = _read_bound(_bound(dataset_roots[locator['dataset_id']], locator['annotation_path']), MAX_METADATA)
        checksum, value = hashlib.sha256(blob).hexdigest(), blob.decode('utf-8-sig')
    elif kind == 'jsonl_line':
        with Path(locator['annotation_path']).open('rb') as stream:
            stream.seek(locator['offset']); blob = stream.read(locator['length'])
        checksum, value = hashlib.sha256(blob).hexdigest(), json.loads(blob)
    elif kind == 'parquet_row':
        import pyarrow.parquet as pq
        value = pq.ParquetFile(locator['annotation_path']).read_row_group(locator['row_group']).slice(locator['row_in_group'], 1).to_pylist()[0]
        checksum = digest(value)
    else:
        raise ValueError('unsupported original annotation locator')
    if checksum != locator['annotation_sha256']:
        raise ValueError('original annotation SHA mismatch')
    return value
