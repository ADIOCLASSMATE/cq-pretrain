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
# Restored local releases use a dated namespace; their archive SHA and size
# remain mandatory and are verified exactly like upstream pinned revisions.
_DATASET = re.compile(r'(?:[^/@\s]+/[^/@\s]+@[0-9a-f]{40}|inspire/[^/@\s]+@[0-9a-f]{64}|local/[^/@\s]+@[0-9]{8})\Z')
_BINDING_FIELDS = ('kind', 'path', 'member', 'offset', 'offset_data', 'length',
                   'size', 'sha256', 'dataset_id', 'archive_sha256', 'archive_size', 'parts')
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
    for location in value.values():
        if isinstance(location, dict) and 'layout_index' in location:
            index = _relative(location['layout_index'], 'release layout index')
            location['layout_index'] = str(path.parent / index)
    with _CACHE_LOCK:
        _ROOT_CACHE[str(path)] = (before, value)
        _ROOT_CACHE.move_to_end(str(path))
        while len(_ROOT_CACHE) > 16:
            _ROOT_CACHE.popitem(last=False)
    return value


def _asset_path(location, relative):
    if isinstance(location, dict):
        relative = _relative(location['archive_paths'][str(relative)], 'mounted archive path')
        location = location['root']
    if not isinstance(location, (str, os.PathLike)) or not Path(location).is_absolute():
        raise ValueError('dataset root missing or not absolute')
    root = Path(location).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not root.is_dir() or not path.is_relative_to(root):
        raise ValueError('portable asset path escapes dataset root')
    return path


_LAYOUT_LOCAL = threading.local()


def _layout_db(index, pid):
    # One connection per thread and process. The image pool resolves references
    # from several threads at once, and a sqlite3 connection shared between
    # them fails mid-query with "InterfaceError: bad parameter or other API
    # misuse" (three 64-rank restarts within ~10 steps each).
    import sqlite3
    connections = _LAYOUT_LOCAL.__dict__.setdefault('connections', {})
    key = (str(index), pid)
    if key not in connections:
        connections[key] = sqlite3.connect(Path(index).as_uri() + '?mode=ro&immutable=1', uri=True)
    return connections[key]


def _layout_source(ref, location):
    offset, length = _range(ref)
    if (ref.get('kind', 'tar') != 'tar' or ref.get('member') != ref['sha256']
            or not _DATASET.fullmatch(ref['dataset_id'])):
        raise ValueError('invalid layout TAR image binding')
    row = _layout_db(location['layout_index'], os.getpid()).execute(
        'select a.source,a.path,a.source_size,e.rg,e.row_in_group,e.file '
        'from archives a join entries e on e.id=a.id where a.name=? and a.size=? and a.sha=? and e.offset=?',
        (ref['path'], ref['archive_size'], ref['archive_sha256'], offset)).fetchone()
    if row is None or offset + length > ref['archive_size']:
        raise ValueError('layout TAR image absent from verified source index')
    source, archive, size, group, ordinal, loose = row
    path = _asset_path(location['mounts'][source], _relative(loose or archive, 'mounted layout source'))
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != (length if loose else size):
        raise ValueError('mounted layout source size/type mismatch')
    return path, group, ordinal


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
    if isinstance(location, dict) and 'layout_index' in location:
        return _layout_source(value, location)[0]
    path = _asset_path(location, relative)
    observed = path.stat()
    expected_size=length if kind=='file' else archive_size
    if not stat.S_ISREG(observed.st_mode) or observed.st_size != expected_size:
        raise ValueError('portable archive size/type mismatch')
    return path


def _archive_identity(fd, value):
    """Declared size and stat identity, without hashing the whole archive.

    What a reader needs is that the bytes it returns are the pinned image, and
    read_ref_bytes proves exactly that by hashing the member it extracted. The
    full-archive hash below proves something strictly weaker about the bytes in
    use -- that they arrived in the pinned container -- at a cost set by the
    container rather than the image: the stage-1 corpus draws about five images
    from a 2.3 GiB TAR before its shuffled stream moves on, so a 210 KiB image
    costs roughly 440 MiB of SHA256. That is the whole of the image pipeline's
    time on a 12-CPU-per-card node.

    So streaming readers take this path and keep the member hash, which is the
    check that actually constrains the data; release acceptance, which attests
    to archives rather than consuming them, keeps the full verification.
    """
    before = _stamp(os.fstat(fd))
    if before[2] != value['archive_size']:
        raise ValueError('portable archive size mismatch')
    return before


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


def read_ref_bytes(ref, roots=None, verify_archive=True):
    """Read original bytes with archive and member integrity, without mutation.

    `verify_archive=False` keeps every check that constrains the returned bytes
    -- containment, the declared archive size, stat stability across the read
    and the member's own SHA256 -- and drops only the full-archive hash. See
    _archive_identity for why a streaming reader should.
    """
    value = logical_ref(ref)
    if value.get('dataset_id') is not None:
        roots = _roots(roots)
        location = roots.get(value['dataset_id'])
        if isinstance(location, dict) and 'layout_index' in location:
            path, group, ordinal = _layout_source(value, location)
            if group is None:
                blob = path.read_bytes()
            else:
                import pyarrow.parquet as pq
                with pq.ParquetFile(path) as source:
                    image = source.read_row_group(group, columns=['image'], use_threads=False)[0][ordinal].as_py()
                blob = image if isinstance(image, bytes) else image['bytes']
            if len(blob) != value['length'] or hashlib.sha256(blob).hexdigest() != value['sha256']:
                raise ValueError('mounted layout image SHA256/length mismatch')
            return blob
    if value.get('kind') == 'original_ranges':
        return _read_original_ranges(value, roots)
    path = resolve_ref(value, roots)
    offset, length = _range(value)
    portable = value.get('dataset_id') is not None
    flags = os.O_RDONLY | (getattr(os, 'O_NOFOLLOW', 0) if portable else 0)
    fd = os.open(path, flags)
    try:
        if not portable:
            before = None
        elif value.get('kind') == 'file':
            before = _stamp(os.fstat(fd))
        elif verify_archive:
            before = _verify_archive(fd, path, value)
        else:
            before = _archive_identity(fd, value)
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


def _read_original_ranges(value, roots):
    """An original TAR member may cross physical split files; verify its bytes.

    These references pin the image itself, not an entire multi-terabyte TAR.
    Every read checks the complete original image SHA and every extent's bounds.
    """
    dataset = value.get('dataset_id')
    if not isinstance(dataset, str) or not _DATASET.fullmatch(dataset):
        raise ValueError('Invalid original-range dataset identity')
    _relative(value.get('member'), 'original image member')
    _, length = _range(dict(value, offset=0))
    location = _roots(roots).get(dataset)
    parts = value.get('parts')
    if not isinstance(parts, list) or not parts or sum(p['length'] for p in parts) != length:
        raise ValueError('Original image extents do not cover its declared length')
    chunks = []
    for part in parts:
        relative = _relative(part['path'], 'original archive part')
        path = _asset_path(location, relative)
        offset, count, size = part['offset'], part['length'], part['file_size']
        if (any(type(n) is not int for n in (offset,count,size)) or
                offset < 0 or count <= 0 or offset + count > size):
            raise ValueError('Invalid original archive extent')
        with path.open('rb', buffering=0) as f:
            before = _stamp(os.fstat(f.fileno()))
            if before[2] != size or not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise ValueError('Original archive part size/type changed')
            body = os.pread(f.fileno(), count, offset)
            if len(body) != count or _stamp(os.fstat(f.fileno())) != before:
                raise ValueError('Original archive part changed during read')
            chunks.append(body)
    body = b''.join(chunks)
    if hashlib.sha256(body).hexdigest() != value['sha256']:
        raise ValueError('Original image SHA256 mismatch')
    return body
