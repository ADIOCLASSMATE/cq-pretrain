"""Lossless quarantine for baseline records outside the published image contract.

The original strict parser runs first and its successful result is returned
unchanged. We neither reinterpret literal tokens nor guess repeated bindings.
Only its two image-count errors have an explicit evidence-preserving fallback.
"""
from __future__ import annotations

import re

from . import ingest

BINDING_CONTRACT = 'unresolved_original_image_binding.v1'
ISSUE = 'unresolved_original_image_binding'
BINDING_ERRORS = frozenset({'balanced image placeholder exceeds members',
                            'balanced conversation leaves unbound images or turns'})


def record(path, relative, key, members, stream, declared_source):
    try:
        return ingest._balanced_record(path, relative, key, members, stream, declared_source)
    except ValueError as error:
        if str(error) not in BINDING_ERRORS:
            raise
        strict_error = str(error)
    # The strict parser has already validated every member, annotation, numbered
    # image, source and task label before either accepted exception can occur.
    annotations, components, images = {}, [], {}
    for suffix, (name, offset, size) in members.items():
        component = dict(member=name, offset_data=offset, size=size,
                         sha256=ingest._range_sha(stream, offset, size))
        components.append(component)
        match = re.fullmatch(r'(\d+)\.(jpg|jpeg|png|webp)', suffix, re.IGNORECASE)
        if match:
            images[int(match[1])] = dict(kind='tar', path=str(path), **component)
        else:
            stream.seek(offset)
            annotations[suffix] = stream.read(size).decode('utf-8')
    refs = [images[index] for index in range(len(images))]
    conversations, segments = [], []
    for turn in annotations['conversations.txt'].split('<EOCL>\n'):
        if not turn:
            continue
        role, colon, text = turn.partition(':')
        if not colon or role not in {'human', 'gpt'}:
            raise ValueError('unsupported balanced conversation role')
        ordinal = len(conversations)
        conversations.append({'from': role, 'value': text})
        segments.append(dict(kind='text', text=text,
                             role={'human': 'user', 'gpt': 'assistant'}[role], turn=ordinal,
                             image_index=None, source_member=members['conversations.txt'][0]))
    if not conversations:
        raise ValueError('balanced conversation has no turns')
    placeholders = sum(turn['value'].count('<image>') for turn in conversations)
    if placeholders == len(refs):
        raise ValueError('quarantine requires an actual image binding mismatch')
    source = annotations['ds_name.txt'].strip()
    data_type = annotations['data_type.txt'].strip()
    locator = dict(adapter=ingest.BALANCED_CONTRACT, path=str(path), source_shard=relative,
                   source_key=key, components=components, archive_fingerprint=ingest._stamp(path))
    original = dict(sample_id=ingest.digest(dict(contract=ingest.BALANCED_CONTRACT, path=str(path),
                                                 key=key, components=components)),
                    source=source, source_shard=relative, source_key=key,
                    format='interleaved' if data_type == 'mmdoc_mt' else
                           ('conversation' if refs else 'text_conversation'),
                    image_count=len(refs), segments=segments, conversations=conversations,
                    source_annotations=annotations, member_inventory=components,
                    data_type=data_type, ds_name=source, orig_url=annotations['orig_url.txt'])
    metadata = dict(adapter=ingest.BALANCED_CONTRACT, pairing_state='unresolved',
                    prepared_adapter_status='quarantine', binding_contract=BINDING_CONTRACT,
                    issues=[ISSUE], strict_parser_error=strict_error,
                    actual_image_members=len(refs), original_image_tokens=placeholders,
                    binding_repair_applied=False)
    return ingest._row('dcvlm', original, refs, locator, metadata)


def rows(unit):
    return ingest._balanced_rows(unit, record_parser=record)
