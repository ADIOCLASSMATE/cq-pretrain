"""Read-only source extraction into an auditable assistant review sidecar."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import sys
import time

from PIL import Image, ImageOps


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(run):
    run = Path(run).resolve()
    sys.path.insert(0, str(run / 'runtime'))
    from corpus_pipeline.production.records import read_asset_bytes
    root = run / 'operations/assistant_visual_review_20261029'
    root.mkdir(exist_ok=True)
    for name in ('previews', 'full_resolution'):
        (root / name).mkdir(exist_ok=True)
    identity = json.loads((run / 'api_review/INPUT.json').read_text())
    prompt = Path(identity['prompt_path'])
    db = sqlite3.connect(f'file:{run}/api_review/index.sqlite?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('BEGIN')
    rows = [dict(row) for row in db.execute("SELECT a.asset_id,a.input_sha,a.input_json,r.request_sha,r.state,r.attempts,r.last_error FROM assets a JOIN requests r USING(asset_id) WHERE r.state!='ok' ORDER BY a.asset_id")]
    db.rollback(); db.close()
    started = time.time()

    def prepare(item):
        row = json.loads(item['input_json']); aid = item['asset_id']
        ref = json.loads(row['original_ref_json'])
        raw = read_asset_bytes(ref, roots=run / 'source_roots.json')
        if sha(raw) != aid:
            raise ValueError('original_member_identity_changed')
        with Image.open(io.BytesIO(raw)) as original:
            original.load()
            original_mode, original_size, original_format = original.mode, original.size, original.format
            original_icc_length = len(original.info.get('icc_profile') or b'')
            picture = ImageOps.exif_transpose(original)
            rgba = picture.convert('RGBA')
            decoded_identity = sha(b'RGBA' + struct.pack('>II', *rgba.size) + rgba.tobytes())
            if decoded_identity != row['pixel_sha256']:
                raise ValueError('original_decoded_pixel_identity_changed')
            has_alpha = original.mode in ('RGBA', 'LA', 'P') or 'transparency' in original.info
            if has_alpha:
                picture = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
            else:
                picture = picture.convert('RGB')
            picture.info.clear()
            full_pixel_sha = sha(picture.tobytes())
            full_path = root / 'full_resolution' / (aid + '.png')
            if full_path.exists():
                raise ValueError('review_preview_already_exists')
            picture.save(full_path, format='PNG', icc_profile=None, exif=b'')
            full_size = picture.size
            picture.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            preview = root / 'previews' / (aid + '.png')
            picture.save(preview, format='PNG', icc_profile=None, exif=b'')
            with Image.open(preview) as check:
                check.load()
                if check.info.get('icc_profile') is not None or check.tobytes() != picture.tobytes():
                    raise ValueError('preview_serialization_changed_pixels')
            with Image.open(full_path) as check:
                check.load()
                if sha(check.tobytes()) != full_pixel_sha:
                    raise ValueError('full_resolution_serialization_changed_pixels')
        return {'asset_id': aid, 'original_sha256': aid, 'original_bytes': len(raw),
                'original_ref': ref, 'original_dimensions': list(original_size),
                'original_format': original_format, 'original_mode': original_mode,
                'original_icc_bytes': original_icc_length,
                'bound_decoded_rgba_geometry_sha256': decoded_identity,
                'decoded_identity_algorithm': 'sha256(RGBA + uint32_be_width_height + oriented_rgba_bytes)',
                'source_archive_and_member_sha256_verified': True,
                'input_sha256': item['input_sha'], 'request_sha256': item['request_sha'],
                'request_state_at_snapshot': item['state'],
                'preview_path': str(preview), 'preview_sha256': sha(preview.read_bytes()),
                'preview_dimensions': list(picture.size), 'preview_rgb_pixel_sha256': sha(picture.tobytes()),
                'full_resolution_path': str(full_path), 'full_resolution_dimensions': list(full_size),
                'full_resolution_sha256': sha(full_path.read_bytes()), 'full_resolution_rgb_pixel_sha256': full_pixel_sha,
                'transform': 'full_image_exif_transpose_rgb_white_alpha_icc_metadata_removed_preview_max1600',
                'crop': False, 'raw_data_changed': False, 'ledger_changed': False}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(prepare, rows))
    manifest = {'schema': 'corpus.assistant_visual_review_inputs.v1', 'snapshot_at': started,
                'completed_at': time.time(), 'images': images, 'count': len(images),
                'original_prompt_path': str(prompt), 'original_prompt_file_sha256': sha(prompt.read_bytes()),
                'frozen_input_sha256': sha((run / 'api_review/INPUT.json').read_bytes()),
                'source_script_sha256': sha(Path(__file__).read_bytes()),
                'reviewer': 'assistant_direct_visual_review_pending',
                'qwen_inference_claimed': False, 'ledger_changed': False}
    path = root / 'MANIFEST.json'
    with path.open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'manifest': str(path), 'count': len(images), 'previews': [x['preview_path'] for x in images]}))


if __name__ == '__main__':
    main(sys.argv[1])
