"""Explicit gen-und caption TAR contracts; no generic task or pairing inference.

Supported source-local formats: BLIP3o Long/Short exact-stem image+TXT, and
LAION-aesthetics image+TXT/JSON.caption. Uncompressed TAR payloads stay in place;
headers and bounded annotation text are indexed in a temporary SQLite database.
This adapter has fixture validation; live gen-und mount validation is separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tarfile
import tempfile
import unicodedata

from corpus_pipeline.policy import normalize, evaluate, resolve_source_config
from .common import atomic_json, digest

SCHEMA = 'production.genund_tar.v1'
CONTRACT = 'genund.exact_stem_caption_tar.v1'
TASK_POLICY = 'corpus.task_policy.v1'
SUPPORTED_SOURCES = {
    'BLIP3o-Pretrain-Long-Caption': {'txt': True, 'json_caption': False, 'caption_origin': 'ai'},
    'BLIP3o-Pretrain-Short-Caption': {'txt': True, 'json_caption': False, 'caption_origin': 'ai'},
    'laion-aesthetics-12m': {'txt': True, 'json_caption': True, 'caption_origin': 'unknown'},
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 128 * 1024 * 1024
PROVENANCE_FIELDS = {
    'image_origin', 'image_provenance', 'image_generation_status', 'is_ai_generated', 'is_generated',
    'ai_generated', 'is_ai_edited', 'is_synthetic_image', 'image_is_synthetic',
    'edit_source_origin', 'source_image_origin', 'original_image_origin', 'edit_source_provenance',
}


def _usable_caption_field(text):
    """Only lexical placeholder checks; this is not a visual quality verdict."""
    if not isinstance(text, str) or not text.strip():
        return False
    value = text.strip()
    return not (re.fullmatch(r'https?://[^\s<>]+', value, re.IGNORECASE)
                or re.fullmatch(r'(?:/|\./|\.\./)[^\s]+\.(?:jpe?g|png|webp|gif|bmp|tiff?|json|txt|parquet)', value, re.IGNORECASE))


def _fingerprint(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _bound_path(root, source, path):
    if source not in SUPPORTED_SOURCES:
        raise ValueError('unsupported gen-und source contract; metadata repair required')
    source_root = (root / source).resolve()
    if root not in source_root.parents:
        raise ValueError('source root escapes collection')
    path = Path(path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if source_root not in resolved.parents or resolved.suffix.lower() != '.tar' or not resolved.is_file():
        raise ValueError('gen-und adapter requires an existing uncompressed TAR below its declared source')
    return resolved


def _identity(manifest):
    value = {'schema_version': manifest['schema_version'], 'collection': manifest['collection'],
            'snapshot_root': str(Path(manifest['snapshot_root']).resolve()), 'contract': manifest['contract'],
            'shards': manifest['shards']}
    if 'adapter_options' in manifest:
        value['adapter_options'] = manifest['adapter_options']
    return value


def adapter_options(*, preserve_all_captions=False, direction='t2i'):
    if type(preserve_all_captions) is not bool or direction not in {'t2i', 'i2t'}:
        raise ValueError('explicit caption options require a boolean preserve_all_captions and t2i/i2t direction')
    return {'preserve_all_captions': preserve_all_captions, 'direction': direction}


def build_manifest(root, source, tar_paths, *, max_records_per_shard=None,
                   preserve_all_captions=False, direction='t2i'):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError('gen-und collection is not mounted')
    if max_records_per_shard is not None and (isinstance(max_records_per_shard, bool) or not isinstance(max_records_per_shard, int) or max_records_per_shard < 1):
        raise ValueError('max_records_per_shard must be positive')
    paths = [_bound_path(root, source, path) for path in tar_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('explicit TAR list must be nonempty and unique')
    value = {'schema_version': SCHEMA, 'collection': 'genund', 'contract': CONTRACT,
             'snapshot_root': str(root), 'all_images_required': True, 'min_short_edge': 128,
             'scope': 'explicit_selected_caption_tar_contract_not_collection_wide_support',
             'shards': [{'source': source, 'tar_path': str(path.relative_to(root)),
                         'fingerprint': _fingerprint(path), 'max_records': max_records_per_shard} for path in paths]}
    options = adapter_options(preserve_all_captions=preserve_all_captions, direction=direction)
    if preserve_all_captions or direction != 't2i':
        value['adapter_options'] = options
    value['snapshot_fingerprint'] = {'declared_shards_sha256': digest(_identity(value))}
    return value


def snapshot_for_manifest(manifest):
    if manifest.get('schema_version') != SCHEMA or manifest.get('collection') != 'genund' or manifest.get('contract') != CONTRACT:
        raise ValueError('unsupported gen-und input manifest')
    root = Path(manifest['snapshot_root']).resolve()
    if not root.is_dir():
        raise FileNotFoundError('gen-und collection is not mounted')
    if 'adapter_options' in manifest:
        if adapter_options(**manifest['adapter_options']) != manifest['adapter_options']:
            raise ValueError('invalid caption adapter options')
    seen = set()
    for shard in manifest['shards']:
        path = _bound_path(root, shard['source'], shard['tar_path'])
        if path in seen:
            raise ValueError('duplicate gen-und TAR input')
        seen.add(path)
        if _fingerprint(path) != shard['fingerprint']:
            raise ValueError('gen-und TAR changed after manifest creation')
        limit = shard.get('max_records')
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError('invalid gen-und per-shard limit')
    fingerprint = {'declared_shards_sha256': digest(_identity(manifest))}
    if not seen or fingerprint != manifest['snapshot_fingerprint']:
        raise ValueError('gen-und input identity mismatch')
    return {'root': str(root), 'collection': 'genund', 'manifest': manifest, 'fingerprint': fingerprint}


def _index_tar(path, db):
    db.execute('CREATE TABLE members(seq INTEGER PRIMARY KEY, key TEXT, name TEXT, suffix TEXT, offset INTEGER, length INTEGER, content TEXT)')
    with tarfile.open(path, mode='r:') as archive:
        for index, member in enumerate(archive):
            if not member.isfile():
                archive.members.clear()
                continue
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('unsafe TAR member path')
            suffix = name.suffix.lower()
            if suffix in IMAGE_SUFFIXES | {'.txt', '.json'}:
                content = None
                if suffix in {'.txt', '.json'}:
                    if member.size > MAX_METADATA_BYTES:
                        raise ValueError('TAR annotation exceeds metadata budget')
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError('unreadable TAR annotation')
                    with handle:
                        content = handle.read().decode('utf-8-sig')
                db.execute('INSERT INTO members VALUES (?,?,?,?,?,?,?)',
                           (index, str(name.with_suffix('')), member.name, suffix, member.offset_data, member.size, content))
                if index % 10000 == 0:
                    db.commit()
            # tarfile otherwise retains TarInfo objects for the whole archive.
            archive.members.clear()
    db.execute('CREATE INDEX member_key ON members(key,seq)')
    db.commit()


def _record(path, shard, key, members, rules):
    source = shard['source']
    contract = SUPPORTED_SOURCES[source]
    options = shard.get('adapter_options', {})
    direction = adapter_options(**options)['direction']
    processing_policy = TASK_POLICY if rules.get('task_routing') == TASK_POLICY else None
    issues, refs, captions, annotations, source_meta, components = [], [], [], {}, {}, []
    if any(count > 1 for count in Counter(row[0] for row in members).values()):
        issues.append('duplicate_tar_member')
    with path.open('rb') as stream:
        for name, suffix, offset, length, content in members:
            if not 0 <= length <= (MAX_IMAGE_BYTES if suffix in IMAGE_SUFFIXES else MAX_METADATA_BYTES):
                raise ValueError('TAR component exceeds byte budget')
            stream.seek(offset)
            blob = stream.read(length)
            if len(blob) != length:
                raise ValueError('truncated TAR component')
            component_sha = hashlib.sha256(blob).hexdigest()
            components.append({'member': name, 'suffix': suffix, 'offset': offset, 'length': length, 'sha256': component_sha})
            if suffix in {'.txt', '.json'} and blob.decode('utf-8-sig') != content:
                raise ValueError('TAR annotation changed after index creation')
            if suffix in IMAGE_SUFFIXES:
                if not 0 < length <= MAX_IMAGE_BYTES:
                    raise ValueError('image member exceeds byte budget')
                refs.append({'kind': 'tar', 'path': str(path), 'member': name, 'offset': offset,
                             'length': length, 'sha256': component_sha, 'exists': True,
                             'offset_space': 'file'})
            elif suffix == '.txt':
                annotations[name] = content
                captions.append({'text': content, 'field': name, 'origin': contract['caption_origin'],
                                 'annotation_kind': 'existing_annotation' if contract['caption_origin'] == 'ai' else 'original'})
            elif suffix == '.json':
                try:
                    value = json.loads(content)
                except json.JSONDecodeError:
                    issues.append('malformed_json_sidecar')
                    continue
                annotations[name] = value
                if not isinstance(value, dict):
                    issues.append('unsupported_json_sidecar_schema')
                    continue
                for field in PROVENANCE_FIELDS:
                    if field in value:
                        if field in source_meta and source_meta[field] != value[field]:
                            issues.append('conflicting_image_provenance')
                        source_meta[field] = value[field]
                if any(field in value for field in ('conversations', 'messages', 'texts', 'question', 'answer')):
                    issues.append('unsupported_task_annotation_requires_adapter')
                if contract['json_caption'] and 'caption' in value:
                    if not isinstance(value['caption'], str):
                        issues.append('non_string_caption')
                    else:
                        captions.append({'text': value['caption'], 'field': name + '#caption',
                                         'origin': contract['caption_origin'], 'annotation_kind': 'original'})
                elif 'caption' in value:
                    issues.append('unregistered_caption_field_requires_adapter')
    if len(refs) != 1:
        issues.append('ambiguous_or_absent_image_pair')
    if not captions:
        issues.append('annotation_not_recovered')
    # Empty annotations are retained as evidence and quarantined for repair. No
    # synthetic stand-in text is inserted to manufacture a classified record.
    # A URL-only TXT must not shadow an actual JSON.caption. This is field
    # recovery, not a model quality judgment; all original fields stay recorded.
    selected = next((item for item in captions if _usable_caption_field(item['text'])), captions[0] if captions else None)
    text = selected['text'] if selected else ''
    segments = [{'kind': 'text', 'text': text, 'role': None, 'turn': None, 'image_index': None,
                 'source_member': selected['field'].rsplit('#caption', 1)[0] if selected else None}]
    segments.extend({'kind': 'image', 'text': None, 'role': None, 'turn': None, 'image_index': index,
                     'source_member': ref['member']} for index, ref in enumerate(refs))
    if direction == 'i2t':
        segments = segments[1:] + segments[:1]
    locator = {'adapter': CONTRACT, 'source': source, 'path': str(path), 'source_shard': shard['tar_path'],
               'member_key': key, 'components': components, 'archive_fingerprint': _fingerprint(path)}
    if options:
        locator['adapter_options'] = options
    if processing_policy:
        locator['processing_policy'] = processing_policy
    sample_id = digest({'collection': 'genund', 'source': source, 'shard': shard['tar_path'], 'key': key,
                        'image_sha256': [ref['sha256'] for ref in refs], 'annotations': annotations})
    raw = {'sample_id': sample_id, 'source': source, 'source_shard': shard['tar_path'], 'source_key': key,
           'format': 'caption', 'direction': direction, 'image_count': len(refs), 'segments': segments,
           'source_annotations': annotations, 'selected_caption_field': selected['field'] if selected else None,
           'caption_origin': contract['caption_origin'], **source_meta}
    if options or processing_policy:
        # This is the task's permission; a T2I record still needs screening to be kept.
        raw['image_loss_allowed'] = direction == 't2i'
    if processing_policy:
        raw['processing_policy'] = processing_policy
    envelope = {'collection': 'genund', 'source': source, 'locator': locator, 'raw': raw,
                'image_refs': refs, 'issues': sorted(set(issues))}
    config = resolve_source_config('genund', source, locator, rules)
    normalized = evaluate(normalize(envelope, config), rules)
    if not issues and len(refs) == 1 and captions and not text.strip() and normalized['category'] == 'unclassified':
        previous = {'category': 'unclassified', 'reason_codes': list(normalized['reason_codes'])}
        recovered = normalize(envelope, config)
        recovered['verified_empty_caption_direction'] = direction
        normalized = evaluate(recovered, rules)
        normalized['adapter_original_classification'] = previous
        normalized['normalization_notes'].append('empty_caption_task_bound_by_registered_genund_' + direction + '_contract')
    valid = not issues and normalized['category'] == 'single_' + direction
    caption_contract = {'adapter': CONTRACT, 'target_segment_index': 0 if direction == 't2i' else len(refs), 'image_index': 0,
                        'allowed_training_lengths': ['short', 'medium', 'long'],
                        'language_policy': 'preserve_original_target_language_and_explicit_user_language_constraints',
                        'preserve_original_user_instruction': True} if valid else None
    return {'collection': 'genund', 'original_sample_id': sample_id, 'source': source, 'locator': locator,
            'source_record_sha256': digest(raw), 'raw': raw, 'normalized': normalized,
            'source_image_refs': refs, 'domain': shard.get('domain', 'unknown'), 'audit_only': False,
            'selection_reason': 'explicit_source_exact_stem_caption_tar_contract',
            'pairing_state': 'verified' if valid else 'unresolved', 'caption_fields_inspected': valid,
            'prepared_adapter_status': 'ready' if valid else 'repair_metadata',
            'pairing_evidence_scope': 'exact_stem_single_image_and_registered_caption_fields_not_semantic_QC',
            'caption_contract': caption_contract, 'prepared_semantic_task': 'caption',
            'caption_candidates': [{**item, 'image_index': 0, 'granularity': 'original',
                                    'source_ref': {'locator': locator, 'field': item['field']}} for item in captions]}


def _caption_groups(record):
    """Only exact native text equivalence; do not merge semantic paraphrases."""
    groups = {}
    for candidate in record['caption_candidates']:
        text = candidate['text']
        if not _usable_caption_field(text):
            continue
        key = unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n'))
        group = groups.setdefault(key, {'text': text, 'fields': []})
        group['fields'].append(candidate['field'])
    return list(groups.values())


def _caption_variant(record, group, rules):
    """Make an actual supervised row, while binding all original annotations."""
    result = deepcopy(record)
    raw = result['raw']
    original_id = raw['sample_id']
    variant = {'fields': group['fields'], 'text_sha256': digest(group['text'])}
    locator = {**result['locator'], 'caption_variant': variant}
    raw['sample_id'] = digest({'original_sample_id': original_id, 'direction': raw['direction'],
                              'caption_variant': variant})
    raw['original_sample_id'] = original_id
    raw['selected_caption_field'] = group['fields'][0]
    raw['caption_variant_fields'] = group['fields']
    for segment in raw['segments']:
        if segment['kind'] == 'text':
            segment['text'] = group['text']
            segment['source_member'] = group['fields'][0].rsplit('#caption', 1)[0]
    envelope = {'collection': 'genund', 'source': result['source'], 'locator': locator,
                'raw': raw, 'image_refs': result['source_image_refs'],
                'issues': result['normalized']['issues']}
    config = resolve_source_config('genund', result['source'], locator, rules)
    result.update(raw=raw, locator=locator, original_sample_id=raw['sample_id'],
                  source_record_sha256=digest(raw), normalized=evaluate(normalize(envelope, config), rules))
    return result


def original_record(locator):
    """Rebuild exactly the indexed original from individually SHA-bound TAR ranges."""
    if locator.get('adapter') != CONTRACT or locator.get('source') not in SUPPORTED_SOURCES:
        raise ValueError('unsupported gen-und original locator')
    path = Path(locator['path'])
    if path.suffix.lower() != '.tar' or _fingerprint(path) != locator['archive_fingerprint']:
        raise ValueError('original gen-und TAR changed')
    members = []
    with path.open('rb') as stream:
        for component in locator['components']:
            offset, length, suffix = component['offset'], component['length'], component['suffix']
            if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
                    or isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= MAX_IMAGE_BYTES
                    or suffix not in IMAGE_SUFFIXES | {'.txt', '.json'}):
                raise ValueError('invalid original gen-und component range')
            stream.seek(offset)
            blob = stream.read(length)
            if len(blob) != length or hashlib.sha256(blob).hexdigest() != component['sha256']:
                raise ValueError('original gen-und component SHA mismatch')
            content = blob.decode('utf-8-sig') if suffix in {'.txt', '.json'} else None
            members.append((component['member'], suffix, offset, length, content))
    shard = {'source': locator['source'], 'tar_path': locator['source_shard']}
    if 'adapter_options' in locator:
        shard['adapter_options'] = locator['adapter_options']
    reconstruction_rules = {}
    if 'processing_policy' in locator:
        if locator['processing_policy'] != TASK_POLICY:
            raise ValueError('unsupported caption processing policy')
        reconstruction_rules['task_routing'] = TASK_POLICY
    result = _record(path, shard, locator['member_key'], members, reconstruction_rules)
    if 'caption_variant' in locator:
        if not shard.get('adapter_options', {}).get('preserve_all_captions'):
            raise ValueError('caption variant requires the explicit all-caption contract')
        variants = [_caption_variant(result, group, reconstruction_rules) for group in _caption_groups(result)]
        selected = [item for item in variants if item['locator'] == locator]
        if len(selected) != 1:
            raise ValueError('caption variant is not bound to the original annotation fields')
        result = selected[0]
    if result['locator'] != locator:
        raise ValueError('original gen-und source binding changed')
    return result['raw']


def iter_genund_records(manifest, source_snapshot, rules, *, batch_size=128, max_records=None):
    # batch_size is a compatible consumer interface; each record is yielded alone.
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be positive')
    if max_records is not None and (isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1):
        raise ValueError('max_records must be positive')
    current = snapshot_for_manifest(manifest)
    if current['fingerprint'] != source_snapshot['fingerprint'] or current['root'] != source_snapshot['root']:
        raise ValueError('gen-und snapshot changed')
    root = Path(current['root'])
    emitted = 0
    for shard in manifest['shards']:
        path = _bound_path(root, shard['source'], shard['tar_path'])
        with tempfile.TemporaryDirectory(prefix='genund-caption-index-') as temporary:
            db = sqlite3.connect(str(Path(temporary) / 'members.sqlite'))
            try:
                db.execute('PRAGMA temp_store=FILE')
                _index_tar(path, db)
                for ordinal, (key,) in enumerate(db.execute('SELECT DISTINCT key FROM members ORDER BY key')):
                    if shard.get('max_records') is not None and ordinal >= shard['max_records']:
                        break
                    members = db.execute('SELECT name,suffix,offset,length,content FROM members WHERE key=? ORDER BY seq', (key,)).fetchall()
                    configured = {**shard, **({'adapter_options': manifest['adapter_options']}
                                             if 'adapter_options' in manifest else {})}
                    record = _record(path, configured, key, members, rules)
                    groups = _caption_groups(record) if manifest.get('adapter_options', {}).get('preserve_all_captions') else []
                    variants = (_caption_variant(record, group, rules) for group in groups) if groups else (record,)
                    for variant in variants:
                        yield variant
                        emitted += 1
                        if max_records is not None and emitted >= max_records:
                            return
            finally:
                db.close()
        if _fingerprint(path) != shard['fingerprint']:
            raise ValueError('gen-und TAR changed during iteration; discard incomplete run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', required=True, choices=sorted(SUPPORTED_SOURCES))
    parser.add_argument('--tar', action='append', required=True)
    parser.add_argument('--max-records-per-shard', type=int)
    parser.add_argument('--preserve-all-captions', action='store_true')
    parser.add_argument('--direction', choices=('t2i', 'i2t'), default='t2i')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = build_manifest(args.root, args.source, args.tar, max_records_per_shard=args.max_records_per_shard,
                           preserve_all_captions=args.preserve_all_captions, direction=args.direction)
    atomic_json(args.output, value)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'shards': len(value['shards']),
                      'scope': value['scope'], 'live_payload_validated': False}))


if __name__ == '__main__':
    main()
