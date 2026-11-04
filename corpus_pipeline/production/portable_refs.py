"""Resolve immutable dataset-relative TAR references without rewriting metadata.

CORPUS_DATASET_ROOTS names a JSON file mapping pinned dataset IDs to local roots.
Only this runtime mapping changes when the original archives move. Filesystem
identity is a process-local verification cache, never a persisted content ID.
"""
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading


MAX_IMAGE_BYTES = 256 << 20
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_DATASET = re.compile(r'(?:[^/@\s]+/[^/@\s]+@[0-9a-f]{40}|inspire/[^/@\s]+@[0-9a-f]{64})\Z')
_BINDING_FIELDS = ('kind', 'path', 'member', 'offset', 'offset_data', 'length',
                   'size', 'sha256', 'dataset_id', 'archive_sha256', 'archive_size')
_CACHE_LOCK = threading.Lock()
_ROOT_CACHE = OrderedDict()
_ARCHIVE_CACHE = OrderedDict()
# Keep the pinned baseline's 10,752 TAR fingerprints resident during SHA-ordered
# review. Entries contain only identity/stat metadata, never archive bytes.
_ARCHIVE_CACHE_LIMIT = 65536
_HASH_LOCKS = [threading.Lock() for _ in range(64)]


def _stamp(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def logical_ref(ref):
    """Return a copy with metadata fields exposed; preserve the logical path."""
    metadata = json.loads(ref.get('metadata_json') or '{}')
    if not isinstance(metadata, dict):
        raise ValueError('image metadata_json must be an object')
    portable = 'dataset_id' in ref or 'dataset_id' in metadata
    if portable:
        for key in _BINDING_FIELDS:
            if key in ref and key in metadata and ref[key] != metadata[key]:
                raise ValueError('conflicting portable image binding: ' + key)
    return {**metadata, **ref}


def _relative(value, description):
    if not isinstance(value, str) or not value or '\\' in value or '://' in value or '\0' in value:
        raise ValueError('invalid relative ' + description)
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.as_posix() != value:
        raise ValueError('unsafe relative ' + description)
    return path


def _range(ref):
    offset = ref.get('offset', ref.get('offset_data'))
    length = ref.get('length', ref.get('size'))
    if ('offset_data' in ref and ref['offset_data'] != offset) or ('size' in ref and ref['size'] != length):
        raise ValueError('conflicting image byte-range aliases')
    if type(offset) is not int or type(length) is not int or offset < 0 or not 0 < length <= MAX_IMAGE_BYTES:
        raise ValueError('invalid or oversized image byte range')
    if not isinstance(ref.get('sha256'), str) or not _HEX.fullmatch(ref['sha256']):
        raise ValueError('invalid image SHA256')
    return offset, length


def _roots(roots):
    if isinstance(roots, Mapping):
        return roots
    selected = roots if roots is not None else os.environ.get('CORPUS_DATASET_ROOTS')
    if not selected:
        raise ValueError('portable image requires CORPUS_DATASET_ROOTS or explicit roots')
    path = Path(selected).resolve(strict=True)
    with path.open('rb') as handle:
        before = _stamp(os.fstat(handle.fileno()))
        with _CACHE_LOCK:
            cached = _ROOT_CACHE.get(str(path))
            if cached and cached[0] == before:
                _ROOT_CACHE.move_to_end(str(path))
                return cached[1]
        value = json.load(handle)
        if before != _stamp(os.fstat(handle.fileno())):
            raise ValueError('dataset root mapping changed while reading')
    if not isinstance(value, dict):
        raise ValueError('dataset roots must be a JSON object')
    with _CACHE_LOCK:
        _ROOT_CACHE[str(path)] = (before, value)
        _ROOT_CACHE.move_to_end(str(path))
        while len(_ROOT_CACHE) > 16:
            _ROOT_CACHE.popitem(last=False)
    return value


def resolve_ref(ref, roots=None, *, allowed_datasets=None):
    """Validate a reference and return its ephemeral local TAR path.

    This checks containment, the declared archive size and range; full archive
    and member checksums are checked by read_ref_bytes before bytes are returned.
    """
    value = logical_ref(ref)
    kind = value.get('kind', 'tar')
    if kind not in ('tar','file'):
        raise ValueError('byte-range resolver accepts TAR references only')
    offset, length = _range(value)
    dataset = value.get('dataset_id')
    if dataset is None:
        if kind=='file':
            raise ValueError('loose original files require a pinned dataset mapping')
        if 'dataset_id' in value:
            raise ValueError('invalid pinned dataset_id')
        path = Path(value['path'])
        if path.suffix != '.tar':
            raise ValueError('byte-range resolver accepts uncompressed TAR only')
        return path
    if not isinstance(dataset, str) or not _DATASET.fullmatch(dataset):
        raise ValueError('invalid pinned dataset_id')
    if allowed_datasets is not None and dataset not in allowed_datasets:
        raise ValueError('portable dataset is not allowed by the native manifest')
    relative = _relative(value.get('path'), 'TAR path')
    _relative(value.get('member'), 'image member')
    if kind=='file' and (offset!=0 or value.get('member')!=value.get('path')):
        raise ValueError('loose image must bind its complete original file')
    archive_size, archive_sha = value.get('archive_size'), value.get('archive_sha256')
    if kind=='tar' and (relative.suffix != '.tar' or type(archive_size) is not int or archive_size <= 0
            or not isinstance(archive_sha, str) or not _HEX.fullmatch(archive_sha)
            or offset + length > archive_size):
        raise ValueError('invalid portable archive identity or image range')
    location = _roots(roots).get(dataset)
    if not isinstance(location, (str, os.PathLike)) or not Path(location).is_absolute():
        raise ValueError('dataset root missing or not absolute: ' + dataset)
    root = Path(location).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('dataset root is not a directory')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('portable TAR path escapes dataset root')
    observed = path.stat()
    expected_size=length if kind=='file' else archive_size
    if not stat.S_ISREG(observed.st_mode) or observed.st_size != expected_size:
        raise ValueError('portable archive size/type mismatch')
    return path


def _verify_archive(fd, path, value):
    """One full hash per archive identity/stat per process, also across threads."""
    key = (str(path), value['dataset_id'], value['archive_sha256'], value['archive_size'])
    with _HASH_LOCKS[hash(key) % len(_HASH_LOCKS)]:
        before = _stamp(os.fstat(fd))
        if before[2] != value['archive_size']:
            raise ValueError('portable archive size mismatch')
        with _CACHE_LOCK:
            cached = _ARCHIVE_CACHE.get(key)
            if cached == before:
                _ARCHIVE_CACHE.move_to_end(key)
                return before
        checksum = hashlib.sha256()
        position = 0
        while position < before[2]:
            chunk = os.pread(fd, min(8 << 20, before[2] - position), position)
            if not chunk:
                raise ValueError('truncated portable archive')
            checksum.update(chunk)
            position += len(chunk)
        if before != _stamp(os.fstat(fd)):
            raise ValueError('portable archive changed during verification')
        if checksum.hexdigest() != value['archive_sha256']:
            raise ValueError('portable archive SHA256 mismatch')
        with _CACHE_LOCK:
            _ARCHIVE_CACHE[key] = before
            _ARCHIVE_CACHE.move_to_end(key)
            while len(_ARCHIVE_CACHE) > _ARCHIVE_CACHE_LIMIT:
                _ARCHIVE_CACHE.popitem(last=False)
        return before


def read_ref_bytes(ref, roots=None):
    """Read original bytes with archive and member integrity, without mutation."""
    value = logical_ref(ref)
    path = resolve_ref(value, roots)
    offset, length = _range(value)
    portable = value.get('dataset_id') is not None
    flags = os.O_RDONLY | (getattr(os, 'O_NOFOLLOW', 0) if portable else 0)
    fd = os.open(path, flags)
    try:
        before = (_stamp(os.fstat(fd)) if value.get('kind')=='file'
                  else _verify_archive(fd, path, value)) if portable else None
        # Regular-file pread normally fills the request; tolerate short reads.
        pieces, remaining = [], length
        while remaining:
            chunk = os.pread(fd, remaining, offset + length - remaining)
            if not chunk:
                break
            pieces.append(chunk)
            remaining -= len(chunk)
        blob = b''.join(pieces)
        if portable and before != _stamp(os.fstat(fd)):
            raise ValueError('portable archive changed during image read')
    finally:
        os.close(fd)
    if len(blob) != length or hashlib.sha256(blob).hexdigest() != value['sha256']:
        raise ValueError('original image SHA256/length mismatch')
    return blob
