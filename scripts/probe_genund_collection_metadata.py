"""Bounded metadata-only live probe; never writes inside the mounted collection."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path


def shallow(path, limit=96):
    result = {'path': str(path), 'exists': path.is_dir(), 'entries': [], 'complete': True}
    if not result['exists']:
        return result
    with os.scandir(path) as iterator:
        for entry in iterator:
            if len(result['entries']) >= limit:
                result['complete'] = False
                break
            result['entries'].append({'name': entry.name, 'directory': entry.is_dir(follow_symlinks=False),
                                      'bytes': entry.stat(follow_symlinks=False).st_size if entry.is_file(follow_symlinks=False) else None})
    return result


def prefix(path):
    out = {'path': str(path), 'bytes': path.stat().st_size}
    text = path.open(encoding='utf-8-sig').read(2 << 20)
    decoder = json.JSONDecoder(); position = len(text)-len(text.lstrip())
    if text[position] == '[': position += 1
    rows = []
    for _ in range(3):
        while position < len(text) and (text[position].isspace() or text[position] == ','): position += 1
        if position >= len(text) or text[position] == ']': break
        value, position = decoder.raw_decode(text, position)
        if isinstance(value, dict):
            rows.append({k: v if not isinstance(v, str) else v[:500] for k, v in value.items()})
        else: rows.append(value)
    out['records'] = rows
    return out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.root); result = {'root': str(root), 'source_modified': False,
        'scope': 'bounded shallow metadata discovery; no image content reads', 'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    expected = ['annotations.json', 'annotations.jsonl', 'data.json', 'train.json', 'train.jsonl', 'train.parquet',
                'FineVision.json', 'ScaleCap450k.json']
    fine = shallow(root/'FineVision', 1000); result['finevision'] = fine
    fine['subsets'] = []
    for item in fine['entries']:
        if not item['directory']: continue
        path = root/'FineVision'/item['name']; row = shallow(path, 16)
        row['named_metadata'] = [str(path/name) for name in expected if (path/name).is_file()]
        fine['subsets'].append(row)
    result['scalecap'] = shallow(root/'ScaleCap450k_image', 128)
    result['scalecap']['images'] = shallow(root/'ScaleCap450k_image/images', 32)
    result['named_metadata'] = [str(base/name) for base in (root, root/'FineVision', root/'ScaleCap450k_image')
                                for name in expected if (base/name).is_file()]
    result['openuni'] = []
    for subset in ['cc12m', 'laion6m', 'megalith-10m', 'redcaps5m', 'redcaps5m_resized', 'text-to-image-2M']:
        path = root/'openuni/data'/subset; row = shallow(path, 64); row['manifests'] = []
        candidates = ['data.json', 'megalith10m_all.json', 'redcaps5m_data.json', 'data/data_1024_10K.json', 'data/data_512_2M.json']
        for name in candidates:
            if (path/name).is_file():
                try: row['manifests'].append(prefix(path/name))
                except Exception as exc: row['manifests'].append({'path': str(path/name), 'error': repr(exc)})
        result['openuni'].append(row)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(output), 'finevision_subsets': len(fine['subsets']),
                      'named_metadata': result['named_metadata']}))


if __name__ == '__main__': main()
