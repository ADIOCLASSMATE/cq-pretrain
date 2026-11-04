"""Portable, hash-bound DCVLM reader with separate selection and image loss.

Native columns are never rewritten. The default iterator returns selected rows;
``include_unselected=True`` returns all original rows for auditing. This module
does not change another collection's selection or deduplication policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import threading
import warnings

SCHEMA = 'corpus.dcvlm_loss_policy_source_view.v1'
READER_SCHEMA = 'corpus.dcvlm_loss_policy_reader.v1'
DATASET_ID = 'mlfoundations/dcvlm-baseline-200b@251523f5a04d7a852223454182cea63bc9f95063'
FLAGS = {
    'PRIOR_EXCLUDE': 1, 'PRIOR_QUARANTINE': 2, 'DUPLICATE': 4,
    'FRONTPASS_REJECTION': 8, 'SCREENING_EXCLUSION': 16,
    'UNRESOLVED_SCREENING': 32, 'INTEGRITY': 64, 'ORIGINAL_BINDING': 128,
    'SAFETY': 256, 'RECORD_QUALITY': 512, 'NO_IMAGE': 1024,
    'DUPLICATE_NOT_SELECTED': 2048, 'MISMATCH': 4096, 'REPAIRED': 8192,
    'USER_ADULT_REVIEW': 16384,
}
SAFETY_DENY_SHA256 = frozenset({
    '68f9c0a6ab44045b7d89ffad44090568ec9ea8585f84686ffc919a1ee375f0c4',
    '8181f93af4788732ea8718653d795079e90a92372a9be73e8a1f033d37eaac64',
    'a84ee87f06c7ec2b335482a8c215a3a6852797f97f4c0943825e638bb322b2c0',
    'b4dfaf147aec54742852854397190c83096e8f0111416a5e9316f9bf2fe6973f',
})
APPENDED_COLUMNS = (
    'dcvlm_image_loss_allowed', 'dcvlm_training_ready', 'dcvlm_selected',
    'dcvlm_policy_flags', 'dcvlm_previous_decision',
)
_INTEGRITY_BITS = FLAGS['PRIOR_QUARANTINE'] | FLAGS['INTEGRITY'] | FLAGS['ORIGINAL_BINDING']
_IMAGE_BLOCK_BITS = sum(FLAGS[name] for name in (
    'PRIOR_EXCLUDE', 'PRIOR_QUARANTINE', 'FRONTPASS_REJECTION',
    'SCREENING_EXCLUSION', 'UNRESOLVED_SCREENING', 'INTEGRITY',
    'ORIGINAL_BINDING', 'SAFETY', 'RECORD_QUALITY', 'NO_IMAGE',
))
_ALL_BITS = sum(FLAGS.values())
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_RELEASE_CACHE = {}
_RESOLVERS = {}
_POLICIES = {}
_DECODE_LOCK = threading.RLock()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stamp(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _json(path):
    before = stamp(path)
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict) and stamp(path) == before, 'JSON binding changed or is not an object')
    return value


def _relative(root, value):
    require(isinstance(value, str) and value and '\\' not in value and '\0' not in value,
            'Invalid relative artifact path')
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and '..' not in relative.parts
            and relative.as_posix() == value, 'Artifact path must be relative and contained')
    root = Path(root).resolve(strict=True)
    target = (root / value).resolve(strict=True)
    require(target.is_relative_to(root) and target.is_file(), 'Artifact escaped its binding root')
    return target


def _bound(root, item, dependencies):
    path = _relative(root, item['path'])
    before = stamp(path)
    if 'size_bytes' in item:
        require(type(item['size_bytes']) is int and item['size_bytes'] >= 0
                and before[2] == item['size_bytes'], 'Bound artifact size differs: ' + str(path))
    require(isinstance(item.get('sha256'), str) and _SHA.fullmatch(item['sha256'])
            and sha(path) == item['sha256'], 'Bound artifact SHA differs: ' + str(path))
    require(stamp(path) == before, 'Bound artifact changed while hashing')
    dependencies[path] = before
    return path


def _valid_flag_word(word):
    require(type(word) is int and 0 <= word <= _ALL_BITS and word & ~_ALL_BITS == 0,
            'Unknown policy flag bits')
    require(not (word & FLAGS['PRIOR_EXCLUDE'] and word & FLAGS['PRIOR_QUARANTINE']),
            'Previous exclusion and quarantine are mutually exclusive')
    require(not (word & FLAGS['USER_ADULT_REVIEW']) or bool(word & FLAGS['SAFETY']),
            'Adult-review override requires the historical safety flag')


def policy_arrays(flags):
    """Return (image_loss_allowed, training_ready, selected) boolean arrays.

    This pure predicate does not authorize an override: the release verifier
    separately requires its bound user-review and repair evidence.
    """
    import numpy as np
    values = np.asarray(flags)
    require(values.ndim == 1 and values.dtype.kind in 'iu', 'Expected a one-dimensional integer flag array')
    require(bool(np.all(values >= 0)) and bool(np.all(values <= _ALL_BITS)), 'Unknown policy flag bits')
    for word in np.unique(values):
        _valid_flag_word(int(word))
    values = values.astype(np.uint16, copy=False)
    ready = ((values & FLAGS['MISMATCH']) == 0)
    ready &= ((values & FLAGS['SAFETY']) == 0) | ((values & FLAGS['USER_ADULT_REVIEW']) != 0)
    ready &= ((values & _INTEGRITY_BITS) == 0) | ((values & FLAGS['REPAIRED']) != 0)
    selected = ready & ((values & FLAGS['DUPLICATE_NOT_SELECTED']) == 0)
    return selected & ((values & _IMAGE_BLOCK_BITS) == 0), ready, selected


def summarize_flag_counts(flag_counts):
    """Canonical counts shared with the builder; counts describe all rows."""
    result = dict.fromkeys((
        'records', 'selected_records', 'unselected_records', 'image_loss_allowed',
        'image_loss_disabled', 'training_ready', 'training_blocked',
        'previous_keep', 'previous_exclude', 'previous_quarantine',
    ), 0)
    for key, count in flag_counts.items():
        require(isinstance(key, (str, int)) and str(int(key)) == str(key), 'Invalid policy count key')
        word = int(key)
        _valid_flag_word(word)
        require(type(count) is int and count >= 0, 'Invalid policy count')
        ready = (not word & FLAGS['MISMATCH']
                 and (not word & FLAGS['SAFETY'] or bool(word & FLAGS['USER_ADULT_REVIEW']))
                 and (not word & _INTEGRITY_BITS or bool(word & FLAGS['REPAIRED'])))
        selected = ready and not word & FLAGS['DUPLICATE_NOT_SELECTED']
        allowed = selected and not word & _IMAGE_BLOCK_BITS
        result['records'] += count
        result['selected_records' if selected else 'unselected_records'] += count
        result['image_loss_allowed' if allowed else 'image_loss_disabled'] += count
        result['training_ready' if ready else 'training_blocked'] += count
        previous = ('exclude' if word & FLAGS['PRIOR_EXCLUDE'] else
                    'quarantine' if word & FLAGS['PRIOR_QUARANTINE'] else 'keep')
        result['previous_' + previous] += count
    return result


def _adult_override(root, binding, dependencies):
    if binding is None:
        return frozenset()
    path = _bound(root, binding, dependencies)
    value = _json(path)
    require(value.get('schema') == 'corpus.user_adult_review_override.v1'
            and value.get('decision') == 'adult_user_review'
            and value.get('user_statement') == '我经过审查了，没问题，是成年'
            and value.get('independent_age_verification') is False
            and value.get('allow_understanding') is True
            and value.get('allow_image_loss') is False
            and len(value.get('asset_ids', [])) == len(SAFETY_DENY_SHA256)
            and set(value['asset_ids']) == SAFETY_DENY_SHA256,
            'Invalid user adult-review binding')
    return frozenset(value['asset_ids'])


def _repair(root, binding, manifest, dependencies):
    if binding is None:
        return None
    import pyarrow.parquet as pq
    path = _bound(root, binding, dependencies)
    value = _json(path)
    require(value.get('schema') == 'corpus.dcvlm_integrity_repair.v1'
            and value.get('complete') is True
            and isinstance(value.get('parent_candidate_manifest_sha256'), str)
            and _SHA.fullmatch(value['parent_candidate_manifest_sha256']), 'Invalid repair manifest')
    if 'parent_candidate_manifest_sha256' in manifest:
        require(value['parent_candidate_manifest_sha256'] == manifest['parent_candidate_manifest_sha256'],
                'Repair applies to another parent candidate')
    paths = {}
    for name in ('records', 'image_policies'):
        item = value[name]
        bound = _bound(path.parent, item, dependencies)
        require(type(item.get('rows')) is int and item['rows'] >= 0, 'Invalid repair artifact row count')
        reader = pq.ParquetFile(bound)
        try:
            require(reader.metadata.num_rows == item['rows'], 'Repair artifact row coverage differs')
        finally:
            reader.close()
        paths[name] = bound
    evidence = value.get('evidence_artifacts', [])
    require(isinstance(evidence, list) and all(isinstance(item, dict) for item in evidence)
            and len({item['path'] for item in evidence}) == len(evidence),
            'Repair evidence artifact bindings are invalid or repeated')
    for item in evidence:
        _bound(path.parent, item, dependencies)
    return {'manifest': value, **paths}


def _release(view=None, *, _require_current=True):
    root = (Path(view) if view is not None else Path(__file__).resolve().parent.parent).resolve(strict=True)
    key = (str(root), str(Path(__file__).resolve()), _require_current)
    cached = _RELEASE_CACHE.get(key)
    if cached is not None and all(stamp(path) == old for path, old in cached['dependencies'].items()):
        return cached
    dependencies = {}
    if _require_current:
        current_path = root / 'CURRENT.json'
        dependencies[current_path] = stamp(current_path)
        current = _json(current_path)
        require(current.get('schema') == SCHEMA and current.get('scope') == 'full'
                and current.get('required_collections') == ['dcvlm']
                and current.get('root') == '.'
                and current.get('manifest', {}).get('path') == 'manifest.json'
                and current.get('complete', {}).get('path') == 'COMPLETE.json',
                'DCVLM loss-policy release is not published')
        done_path = _bound(root, current['complete'], dependencies)
    else:
        done_path = root / 'COMPLETE.json'
        dependencies[done_path] = stamp(done_path)
    done = _json(done_path)
    require(done.get('complete') is True, 'DCVLM loss-policy release is incomplete')
    if _require_current:
        require(current['manifest']['sha256'] == done['manifest_sha256'] == current['release_id'],
                'Current publication and completion identities differ')
    manifest_path = _bound(root, {'path': 'manifest.json', 'sha256': done['manifest_sha256']}, dependencies)
    manifest = _json(manifest_path)
    require(manifest.get('schema') == SCHEMA and manifest.get('complete') is True
            and manifest.get('scope') == 'full' and manifest.get('collection') == 'dcvlm'
            and manifest.get('dataset_id') == DATASET_ID
            and manifest.get('original_records_modified') is False
            and manifest.get('flags_encoding') == 'uint16_le'
            and manifest.get('flag_definitions') == FLAGS, 'Unsupported DCVLM loss-policy semantics')
    deny = manifest.get('safety_deny_sha256', [])
    require(isinstance(deny, list) and len(deny) == len(SAFETY_DENY_SHA256)
            and set(deny) == SAFETY_DENY_SHA256, 'Historical safety binding changed')
    binding_path = _bound(root, {'path': 'readers/READER_BINDING.json',
                                'sha256': manifest['reader_binding_sha256']}, dependencies)
    binding = _json(binding_path)
    names = {'dcvlm_loss_policy_reader.py', 'portable_refs.py'}
    require(binding.get('schema') == READER_SCHEMA and len(binding.get('files', [])) == 2
            and {item.get('name') for item in binding['files']} == names, 'Reader implementation binding incomplete')
    for item in binding['files']:
        _bound(binding_path.parent, {'path': item['name'], 'sha256': item['sha256']}, dependencies)
        if item['name'] == 'dcvlm_loss_policy_reader.py':
            running = Path(__file__).resolve()
            before = stamp(running)
            require(sha(running) == item['sha256'] and stamp(running) == before,
                    'Running reader does not match the bound implementation')
            dependencies[running] = before
    require(type(manifest.get('records')) is int and manifest['records'] >= 0
            and type(manifest.get('source_files')) is int, 'Invalid release coverage')
    files = manifest['files']
    require(isinstance(files, list) and len(files) == manifest['source_files'], 'Native file coverage differs')
    seen_sources, seen_flags, totals = set(), set(), Counter()
    for item in files:
        source = item['source_path']
        require(isinstance(source, str) and Path(source).is_absolute() and '..' not in Path(source).parts
                and source not in seen_sources, 'Invalid or duplicate native source')
        require(item['flags_path'] not in seen_flags, 'Duplicate policy flag artifact')
        seen_sources.add(source)
        seen_flags.add(item['flags_path'])
        require(type(item.get('rows')) is int and item['rows'] >= 0
                and item['flags_bytes'] == 2 * item['rows']
                and type(item.get('source_bytes')) is int and item['source_bytes'] > 0
                and _SHA.fullmatch(item.get('source_sha256', ''))
                and _SHA.fullmatch(item.get('flags_sha256', '')), 'Invalid native/flags binding')
        counts = summarize_flag_counts(item['flag_counts'])
        require(counts['records'] == item['rows'], 'Flag histogram does not cover native rows')
        totals.update({str(key): count for key, count in item['flag_counts'].items()})
    counts = summarize_flag_counts(totals)
    require(counts['records'] == manifest['records']
            and counts['selected_records'] == manifest['selected_records']
            and all(manifest['counts'].get(key) == count for key, count in counts.items()),
            'Policy counts differ from complete flag coverage')
    require(done.get('records') == manifest['records']
            and done.get('selected_records') == manifest['selected_records']
            and done.get('counts') == manifest['counts'], 'Release completion coverage differs')
    override = _adult_override(root, manifest.get('user_review'), dependencies)
    repair = _repair(root, manifest.get('repair_manifest'), manifest, dependencies)
    require(not any(int(word) & FLAGS['USER_ADULT_REVIEW'] for word in totals) or override,
            'Adult-review flag lacks a valid bound review')
    require(not any(int(word) & FLAGS['REPAIRED'] for word in totals) or repair is not None,
            'Repaired flag lacks bound repair evidence')
    require(all(stamp(path) == old for path, old in dependencies.items()), 'Release changed while verifying')
    result = {'root': root, 'manifest': manifest, 'dependencies': dependencies,
              'override': override, 'repair': repair}
    _RELEASE_CACHE[key] = result
    return result


def verify_release(view=None):
    """Verify published metadata, runtime and evidence; native files are lazy."""
    return _release(view)['manifest']


def _verify_release_payload(view):
    """Builder-only preflight before its final atomic CURRENT publication."""
    return _release(view, _require_current=False)['manifest']


def mapping(value):
    result = _json(value) if isinstance(value, (str, Path)) else value
    require(isinstance(result, dict) and result.get('schema') == 'corpus.source_root_mapping.v1'
            and isinstance(result.get('native_roots'), dict)
            and isinstance(result.get('dataset_roots'), dict), 'Explicit native and dataset roots required')
    for old, new in result['native_roots'].items():
        require(isinstance(old, str) and isinstance(new, str) and Path(old).is_absolute()
                and '..' not in Path(old).parts and Path(new).is_absolute(), 'Native roots must be absolute')
    return result


def relocated(original, roots):
    original = Path(original)
    require(original.is_absolute() and '..' not in original.parts, 'Invalid original native path')
    candidates = [(Path(old), Path(new)) for old, new in roots.items() if original.is_relative_to(Path(old))]
    require(len(candidates) == 1, 'Native source needs exactly one explicit root mapping')
    old, new = candidates[0]
    new = new.resolve(strict=True)
    target = (new / original.relative_to(old)).resolve(strict=True)
    require(target.is_relative_to(new) and target.is_file(), 'Native source escaped its mapped root')
    return target


def verify_flags(view, item):
    import numpy as np
    root = Path(view).resolve(strict=True)
    dependencies = {}
    path = _bound(root, {'path': item['flags_path'], 'sha256': item['flags_sha256']}, dependencies)
    require(path.stat().st_size == item['flags_bytes'] == 2 * item['rows'], 'Policy flag byte count differs')
    flags = np.fromfile(path, dtype='<u2')
    require(stamp(path) == dependencies[path], 'Policy flags changed during reading')
    words, counts = np.unique(flags, return_counts=True)
    observed = {str(int(word)): int(count) for word, count in zip(words, counts)}
    require(observed == item['flag_counts'], 'Policy flag histogram differs')
    summarize_flag_counts(observed)
    return flags


def iter_batches(view, roots, batch_size=8192, epoch_seed=None, *, worker_id=0,
                 num_workers=1, include_unselected=False):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    release = _release(view)
    root, manifest = release['root'], release['manifest']
    roots = mapping(roots)
    require(type(batch_size) is int and batch_size > 0, 'Positive batch size required')
    require(type(worker_id) is int and type(num_workers) is int and 0 <= worker_id < num_workers,
            'Invalid worker partition')
    require(type(include_unselected) is bool, 'include_unselected must be boolean')
    files = list(manifest['files'])
    if epoch_seed is not None:
        random.Random(epoch_seed).shuffle(files)
    for item in files[worker_id::num_workers]:
        flags = verify_flags(root, item)
        path = relocated(item['source_path'], roots['native_roots'])
        before = stamp(path)
        require(before[2] == item['source_bytes'] and sha(path) == item['source_sha256'],
                'Relocated native content differs')
        require(stamp(path) == before, 'Native metadata changed while hashing')
        reader = pq.ParquetFile(path)
        offset = emitted = 0
        try:
            require(reader.metadata.num_rows == item['rows'], 'Native/flags row coverage differs')
            require(not set(APPENDED_COLUMNS).intersection(reader.schema_arrow.names), 'Native column collides with policy columns')
            require('collection' in reader.schema_arrow.names, 'Native collection column is required')
            for batch in reader.iter_batches(batch_size=batch_size):
                require(bool(pc.all(pc.fill_null(pc.equal(batch.column('collection'), 'dcvlm'), False)).as_py()),
                        'Loss policy cannot apply to another collection')
                stop = offset + batch.num_rows
                words = flags[offset:stop]
                allowed, ready, selected = policy_arrays(words)
                previous = np.where(words & FLAGS['PRIOR_EXCLUDE'], 'exclude',
                                    np.where(words & FLAGS['PRIOR_QUARANTINE'], 'quarantine', 'keep'))
                additions = (pa.array(allowed), pa.array(ready), pa.array(selected),
                             pa.array(words, type=pa.uint16()), pa.array(previous))
                result = batch
                for name, values in zip(APPENDED_COLUMNS, additions):
                    result = result.append_column(name, values)
                offset = stop
                if not include_unselected:
                    result = result.filter(pa.array(selected))
                emitted += result.num_rows
                if result.num_rows:
                    yield result
            expected = item['rows'] if include_unselected else summarize_flag_counts(item['flag_counts'])['selected_records']
            require(offset == item['rows'] and emitted == expected, 'Reader lost ordered native rows')
            require(stamp(path) == before, 'Native metadata changed while reading')
        finally:
            reader.close()


def _resolver(release):
    path = release['root'] / 'readers' / 'portable_refs.py'
    key = (str(path), stamp(path))
    if key not in _RESOLVERS:
        spec = importlib.util.spec_from_file_location('frozen_dcvlm_refs_' + hashlib.sha256(str(path).encode()).hexdigest(), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RESOLVERS[key] = module
    return _RESOLVERS[key]


def read_image(ref, roots, *, view=None):
    """Read original encoded bytes after release, dataset and review checks."""
    release = _release(view)
    roots = mapping(roots)
    resolver = _resolver(release)
    logical = resolver.logical_ref(ref)
    require(logical.get('dataset_id') == DATASET_ID, 'Image is not the pinned DCVLM baseline dataset')
    asset = logical.get('sha256')
    require(asset not in SAFETY_DENY_SHA256 or asset in release['override'],
            'Image access requires the bound adult user review')
    return resolver.read_ref_bytes(ref, roots=roots['dataset_roots'])


def _image_policies(release):
    import pyarrow.parquet as pq
    if release['repair'] is None:
        return {}
    path = release['repair']['image_policies']
    key = (str(path), stamp(path))
    if key not in _POLICIES:
        rows = pq.read_table(path).to_pylist()
        policies = {}
        for item in rows:
            asset = item['asset_id']
            require(isinstance(asset, str) and _SHA.fullmatch(asset) and asset not in policies,
                    'Invalid or duplicate repair image policy')
            require(item['decode_policy'] in {'original', 'first_frame'} and item['bytes_verified'] is True
                    and item['decode_verified'] is True
                    and isinstance(item.get('frame0_rgb_sha256'), str) and _SHA.fullmatch(item['frame0_rgb_sha256'])
                    and type(item['frame_count']) is int and item['frame_count'] >= 1
                    and type(item['width']) is int and item['width'] > 0
                    and type(item['height']) is int and item['height'] > 0, 'Invalid verified image decode policy')
            policies[asset] = item
        require(stamp(path) == key[1], 'Image repair policies changed during reading')
        _POLICIES[key] = policies
    return _POLICIES[key]


def decode_image(ref, roots, *, view=None, max_pixels=200_000_000):
    """Decode a bounded RGB image; animations need a bound first-frame policy.

    Original TAR bytes and native references remain unchanged. No output file
    is created. The first-frame override is explicit release evidence.
    """
    from PIL import Image
    require(type(max_pixels) is int and 0 < max_pixels <= 200_000_000, 'Invalid decode pixel bound')
    release = _release(view)
    logical = _resolver(release).logical_ref(ref)
    blob = read_image(ref, roots, view=release['root'])
    require(len(blob) <= 256 << 20, 'Image exceeds the encoded byte bound')
    policy = _image_policies(release).get(logical['sha256'])
    with _DECODE_LOCK:
        old_bound = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = 200_000_000
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(blob)) as image:
                    require(0 < image.width * image.height <= max_pixels, 'Image exceeds the decode pixel bound')
                    frames = getattr(image, 'n_frames', 1)
                    if policy is not None:
                        require((image.width, image.height, frames) ==
                                (policy['width'], policy['height'], policy['frame_count']), 'Image dimensions/frame count differ from repair evidence')
                    require(frames == 1 or (policy is not None and policy['decode_policy'] == 'first_frame'),
                            'Animated image requires an explicit first-frame repair policy')
                    image.seek(0)
                    image.load()
                    result = image.convert('RGB')
                    if policy is not None:
                        require(hashlib.sha256(result.tobytes()).hexdigest() == policy['frame0_rgb_sha256'],
                                'Decoded first-frame pixels differ from repair evidence')
                    return result
        finally:
            Image.MAX_IMAGE_PIXELS = old_bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', required=True)
    parser.add_argument('--roots', required=True)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--include-unselected', action='store_true')
    args = parser.parse_args()
    rows = sum(batch.num_rows for batch in iter_batches(args.view, args.roots, args.batch_size,
                                                       include_unselected=args.include_unselected))
    print(json.dumps({'verified_rows': rows, 'include_unselected': args.include_unselected,
                      'native_columns_modified': False, 'images_rewritten': False}))
