"""Freeze original source context for the 688-image human-review queue.

Only reads source TARs. Source labels are preserved as audit fields, never context.
Context is the contiguous WebDataset record matching the image's exact key.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from typing import Any

SCHEMA = 'corpus.context_review.input.v1'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
FIRST_REVIEWS = [191, 247, 309, 598, 597, 486, 168, 1]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def key_of(name: str) -> str:
    return Path(name).name.split('.', 1)[0]


def sequence_of(name: str) -> int | None:
    parts = Path(name).name.split('.')
    return int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None


def scan_headers(f, start: int, length: int, total: int) -> list[dict]:
    start = max(0, start // 512 * 512)
    f.seek(start)
    raw = f.read(min(length, total - start))
    found = []
    for pos in range(0, len(raw) - 511, 512):
        block = raw[pos:pos + 512]
        # All source shards use USTAR/PAX. Header checksum is verified by TarInfo.
        if block[257:262] != b'ustar':
            continue
        try:
            info = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
        except (tarfile.HeaderError, ValueError, UnicodeError):
            continue
        if info is None or not info.isreg():
            continue
        offset = start + pos + 512
        if info.size < 0 or offset + info.size > total:
            continue
        found.append({'member': info.name, 'header_offset': start + pos,
                      'offset': offset, 'length': info.size,
                      'padded_end': offset + (info.size + 511) // 512 * 512})
    return found


def contiguous_record(f, ref: dict, total: int) -> tuple[list[dict], dict]:
    """Find record boundaries without decoding or traversing image payloads."""
    key = key_of(ref['member'])
    image_start = int(ref['offset'])
    image_end = image_start + (int(ref['length']) + 511) // 512 * 512
    last = None
    for radius in (128 * 1024, 512 * 1024, 2 * 1024**2, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2):
        before_start = max(0, image_start - radius)
        headers = scan_headers(f, before_start, image_start - before_start, total)
        headers += scan_headers(f, image_end, radius, total)
        headers = sorted({h['header_offset']: h for h in headers}.values(), key=lambda h: h['header_offset'])
        own = [h for h in headers if key_of(h['member']) == key]
        exact = [h for h in own if h['member'] == ref['member'] and h['offset'] == image_start]
        if len(exact) != 1 or exact[0]['length'] != ref['length']:
            raise ValueError('Target TAR header does not match frozen member/offset/length')
        first, end = own[0]['header_offset'], max(h['padded_end'] for h in own)
        previous = [h for h in headers if key_of(h['member']) != key and h['header_offset'] < first]
        following = [h for h in headers if key_of(h['member']) != key and h['header_offset'] >= end]
        left = bool(previous) or before_start == 0
        right = bool(following)
        if not right and image_end + radius >= total:
            f.seek(end)
            right = f.read(1024) in (b'', b'\0' * 1024)
        last = {'method': 'bounded_tar_header_windows', 'radius_bytes': radius,
                'left_record_boundary_confirmed': left, 'right_record_boundary_confirmed': right,
                'record_key': key, 'record_first_header_offset': first, 'record_padded_end': end,
                'previous_record_key': key_of(previous[-1]['member']) if previous else None,
                'next_record_key': key_of(following[0]['member']) if following else None}
        if left and right:
            # Rewalk just the matched record by tar sizes to exclude any spurious
            # aligned header-like bytes inside an image payload.
            verified = []
            pos = first
            while pos < end:
                f.seek(pos)
                block = f.read(512)
                try:
                    info = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
                except (tarfile.HeaderError, ValueError, UnicodeError) as exc:
                    raise ValueError(f'Invalid contiguous TAR header at {pos}') from exc
                if info is None:
                    raise ValueError('Unexpected end of TAR inside record')
                data_offset = pos + 512
                next_pos = data_offset + (info.size + 511) // 512 * 512
                if info.isreg():
                    if key_of(info.name) != key:
                        raise ValueError('Different record key inside matched record boundaries')
                    verified.append({'member': info.name, 'header_offset': pos, 'offset': data_offset,
                                     'length': info.size, 'padded_end': next_pos})
                elif info.type not in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                    raise ValueError(f'Unsupported TAR extension member inside record: {info.type!r}')
                pos = next_pos
            if pos != end:
                raise ValueError('Contiguous TAR walk end mismatch')
            if len({h['member'] for h in verified}) != len(verified):
                raise ValueError('Duplicate TAR member names inside record')
            return verified, last
    return own, last


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v.strip()))


def extract_one(row: dict, human: dict) -> tuple[dict, dict]:
    ref = row['source_ref']
    path = Path(ref['path'])
    before = path.stat()
    image_length = int(ref['length'])
    limitations = []
    with path.open('rb') as f:
        f.seek(ref['offset'])
        raw_image = f.read(image_length)
        image_hash = sha(raw_image)
        if len(raw_image) != image_length or image_hash != row['item_id'] or image_hash != ref['sha256']:
            raise ValueError(f"#{row['review_number']}: target image bytes/hash mismatch")
        members, bounds = contiguous_record(f, ref, before.st_size)
        source_members = []
        raw_metadata = []
        texts = []
        images = []
        for member in members:
            name = member['member']
            ext = Path(name).suffix.lower()
            image = ext in IMAGE_SUFFIXES
            info = {k: v for k, v in member.items() if k != 'padded_end'}
            info['sequence_index'] = sequence_of(name)
            if image:
                info.update({'kind': 'image', 'is_target': name == ref['member'], 'displayed': name == ref['member']})
                if info['is_target']:
                    info['sha256'] = image_hash
                images.append(info.copy())
            elif ext in {'.txt', '.json'}:
                f.seek(member['offset'])
                raw = f.read(member['length'])
                if len(raw) != member['length']:
                    raise ValueError(f'Short source text read: {name}')
                info['sha256'] = sha(raw)
                try:
                    text = raw.decode('utf-8')
                except UnicodeError:
                    text = raw.decode('utf-8', errors='replace')
                    limitations.append(f'{name}: invalid UTF-8 replaced; original bytes hash retained')
                    info['raw_hex'] = raw.hex()
                info['text'] = text
                if ext == '.json':
                    try:
                        info['json'] = json.loads(text)
                        if not name.endswith('.filters.json'):
                            raw_metadata.append(info)
                    except json.JSONDecodeError:
                        limitations.append(f'{name}: invalid JSON; full raw text retained')
                elif not name.endswith(('.sha256.txt', '.phash.txt')):
                    texts.append(info)
                info['kind'] = 'metadata' if ext == '.json' else 'text'
            else:
                info['kind'] = 'binary_sidecar'
            source_members.append(info)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError('Source TAR changed during extraction')
    key = key_of(ref['member'])
    image_stem = ref['member'].rsplit('.', 1)[0]
    exact_meta = [m for m in raw_metadata if m['member'] == image_stem + '.json']
    metadata = exact_meta[0].get('json', {}) if exact_meta else {}
    if metadata.get('url') is not None and metadata.get('url') != ref.get('source_url'):
        raise ValueError(f"#{row['review_number']}: source metadata URL mismatch")
    caption_texts = []
    conversation_texts = []
    interleaved_segments = []
    region_annotations = []
    for m in raw_metadata:
        value = m.get('json')
        if not isinstance(value, dict):
            continue
        if isinstance(value.get('caption'), str):
            caption_texts.append(value['caption'])
        if 'region_caption' in value or ('questions' in value and 'answers' in value):
            region_annotations.append({'member': m['member'], **value})
            conversation_texts.append('Region annotation (bbox refers to target image, original JSON):\n' + json.dumps(value, ensure_ascii=False))
        for name in ('conversations', 'conversation', 'messages'):
            if value.get(name):
                conversation_texts.append(json.dumps(value[name], ensure_ascii=False))
    numbered_texts = [t for t in texts if re.fullmatch(re.escape(key) + r'\.\d+\.txt', t['member'])]
    is_interleaved = bool(numbered_texts) or len(images) > 1
    for t in texts:
        if t['member'].endswith('.conversations.txt'):
            conversation_texts.append(t['text'])
        elif t in numbered_texts:
            interleaved_segments.append({'member': t['member'], 'sequence_index': t['sequence_index'],
                                         'kind': 'text', 'text': t['text']})
        elif t['member'] in (key + '.txt', image_stem + '.txt', key + '..txt'):
            caption_texts.append(t['text'])
        elif t['text'].strip():
            limitations.append(f'Unclassified same-record text retained in evidence: {t["member"]}')
    caption_texts = unique(caption_texts)
    conversation_texts = unique(conversation_texts)
    interleaved_segments.sort(key=lambda t: t['sequence_index'])
    images.sort(key=lambda t: (t['sequence_index'] is None, t['sequence_index'] or 0, t['member']))
    if is_interleaved:
        task = 'interleaved'
    elif conversation_texts:
        task = 'conversation'
    elif caption_texts:
        task = 'caption_pair'
    else:
        task = 'unknown'
    complete = bounds['left_record_boundary_confirmed'] and bounds['right_record_boundary_confirmed']
    if not complete:
        limitations.append('Could not confirm complete contiguous record within bounded TAR windows')
    has_text = bool(caption_texts or conversation_texts or any(x['text'].strip() for x in interleaved_segments))
    status = 'partial' if limitations or not complete else ('complete' if has_text else 'missing')
    if not has_text:
        limitations.append('No caption/conversation/interleaved text recovered; do not infer or fabricate context')
    if len(images) > 1:
        limitations.append(f'Only target image {ref["member"]} is attached; {len(images)-1} other same-record image(s) are not displayed')
    summary = {k: metadata[k] for k in ('url', 'uid', 'key', 'face_bboxes', 'width', 'height', 'original_width', 'original_height', 'caption') if k in metadata}
    summary.update({'record_key': key, 'target_member': ref['member'], 'target_sequence_index': sequence_of(ref['member']),
                    'record_images': images, 'only_target_image_attached': True,
                    'target_metadata_url_checked': 'url' in metadata,
                    'target_metadata_url_matches': metadata.get('url') == ref.get('source_url') if 'url' in metadata else None})
    if region_annotations:
        summary['region_annotation_count'] = len(region_annotations)
    context = {'caption_texts': caption_texts, 'conversation_texts': conversation_texts,
               'interleaved_segments': interleaved_segments, 'metadata_summary': summary,
               'context_status': status, 'context_limitations': limitations}
    evidence = {'schema_version': 'corpus.context_review.source_evidence.v1', 'item_id': row['item_id'],
                'review_number': row['review_number'], 'source_name': row['source_name'], 'source_ref': ref,
                'source_bytes': image_length, 'source_mtime_ns': before.st_mtime_ns,
                'source_container_bytes': before.st_size, 'source_image_sha256': image_hash,
                'extraction': bounds, 'members': source_members,
                'context_sha256': sha(canonical(context))}
    candidate = {'schema_version': SCHEMA, 'item_id': row['item_id'], 'queue_order': None,
                 'review_number': row['review_number'], 'source_ref': ref, 'source_bytes': image_length,
                 'source_mtime_ns': before.st_mtime_ns, 'source_container_bytes': before.st_size,
                 'source_name': row['source_name'], 'task_type': task, 'source_context': context,
                 'human_review': human, 'source_evidence_sha256': sha(canonical(evidence))}
    return candidate, evidence


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / n).exists() for n in ('candidates.jsonl', 'source_evidence.jsonl', 'EXTRACTION.json')):
        raise SystemExit('Refusing to overwrite existing extraction artifacts')
    pointer_raw = (root / 'HUMAN_REVIEW_CURRENT.json').read_bytes()
    pointer = json.loads(pointer_raw)
    human_dir = Path(pointer['directory'])
    human_path = human_dir / 'human_overrides.jsonl.gz'
    human_raw = human_path.read_bytes()
    overrides = [json.loads(line) for line in gzip.decompress(human_raw).splitlines() if line]
    humans = {row['item_id']: row['human_review'] for row in overrides}
    queue_path = root / 'manual_review_ui/queue.jsonl'
    queue_raw = queue_path.read_bytes()
    queue = [json.loads(line) for line in queue_raw.splitlines() if line]
    if len(humans) != len(overrides) or len(set(row['item_id'] for row in queue)) != len(queue):
        raise ValueError('Duplicate human or queue item IDs')
    if set(humans) != {row['item_id'] for row in queue} or len(queue) != 688:
        raise ValueError('Expected exact 688-image human-review queue coverage')
    first = {n: index for index, n in enumerate(FIRST_REVIEWS)}
    queue.sort(key=lambda row: (0, first[row['review_number']]) if row['review_number'] in first else (1, row['review_number']))
    for row in queue:
        if humans[row['item_id']]['review_number'] != row['review_number']:
            raise ValueError('Human review number mismatch')
    results, errors = [], []
    def run(row):
        try:
            return extract_one(row, humans[row['item_id']]), None
        except Exception as exc:
            return None, {'item_id': row['item_id'], 'review_number': row['review_number'],
                          'error': f'{type(exc).__name__}: {exc}'}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (result, error) in enumerate(pool.map(run, queue)):
            if result:
                result[0]['queue_order'] = index
                results.append(result)
            else:
                errors.append(error)
            if (index + 1) % 50 == 0 or index + 1 == len(queue):
                print(json.dumps({'finished': index + 1, 'total': len(queue), 'errors': len(errors)}), flush=True)
    if errors:
        (output / 'EXTRACTION_ERRORS.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2) + '\n')
        raise SystemExit(f'Extraction failed for {len(errors)} rows; no complete manifest published')
    if (root / 'HUMAN_REVIEW_CURRENT.json').read_bytes() != pointer_raw:
        raise ValueError('Human review current pointer changed during extraction')
    for name, which in [('candidates.jsonl', 0), ('source_evidence.jsonl', 1)]:
        temp = output / ('.' + name + '.tmp')
        with temp.open('wb') as f:
            for result in results:
                f.write(canonical(result[which]) + b'\n')
            f.flush()
            os.fsync(f.fileno())
        temp.replace(output / name)
    candidates = [result[0] for result in results]
    report = {'schema_version': 'corpus.context_review.extraction.v1', 'complete': True,
              'created_at': datetime.now(timezone.utc).isoformat(), 'records': len(candidates),
              'source_queue': {'path': str(queue_path), 'sha256': sha(queue_raw)},
              'human_overrides': {'path': str(human_path), 'sha256': sha(human_raw)},
              'human_pointer_sha256': sha(pointer_raw),
              'code_sha256': sha(Path(__file__).read_bytes()),
              'first_review_numbers': FIRST_REVIEWS,
              'task_types': dict(Counter(c['task_type'] for c in candidates)),
              'context_statuses': dict(Counter(c['source_context']['context_status'] for c in candidates)),
              'source_counts': dict(Counter(c['source_name'] for c in candidates)),
              'unclear': [{'review_number': c['review_number'], 'item_id': c['item_id'],
                           'context_status': c['source_context']['context_status'],
                           'limitations': c['source_context']['context_limitations']}
                          for c in candidates if c['source_context']['context_status'] != 'complete'],
              'source_images_unchanged': True, 'human_decisions_unchanged': True,
              'all_target_image_hashes_verified': True,
              'context_policy': 'Original full source text retained. Only target image is attached. Other same-record image positions are listed; their pixels are not inspected. Human decisions are audit-only and must not be included in model prompts.',
              'artifacts': {n: {'path': str(output / n), 'bytes': (output / n).stat().st_size,
                                'sha256': sha((output / n).read_bytes())}
                            for n in ('candidates.jsonl', 'source_evidence.jsonl')}}
    (output / 'EXTRACTION.json').write_bytes(canonical(report) + b'\n')
    print(json.dumps({k: report[k] for k in ('records', 'task_types', 'context_statuses', 'unclear')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
