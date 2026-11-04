"""Index the pinned baseline release without changing or copying its raw TARs.

Stable artifacts contain logical dataset identities and relative archive paths.
Download roots are runtime arguments only. A missing download is a waiting state,
never evidence that the upstream release has fewer files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import errno
import fcntl
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import signal
import socket
import threading
import time

from corpus_pipeline.production import common
from corpus_pipeline.dedup import training_release
from . import ingest, baseline_compat

REPO = 'mlfoundations/dcvlm-baseline-200b'
REVISION = '251523f5a04d7a852223454182cea63bc9f95063'
EXPECTED_TRAINING_FILES = 10752
CONTRACT = 'dcvlm.baseline_webdataset.v1'
PLAN_SCHEMA = 'corpus.baseline.index_plan.v1'
UNIT_SCHEMA = 'corpus.baseline.index_unit.v1'
MIN_RESERVE_BYTES = 1 << 40
MAX_BUFFER_BYTES = 64 << 20
LEGACY_CODE_IDENTITY = dict(
    arrow_schema_sha256='2a843d8fe12222812e37099df88804a51f47b6aa3ab26b0e06808ff0e5fcf7d8',
    baseline_index_sha256='46c3c83c94ba6096941bdae0519cd279e3b5205964da0a48b8e35e887d59f4a2',
    common_sha256='22c31de5ef12f8196678fc5e0ee0c3e75fc82f7dc5bf6e18ab420cd7fd976236',
    ingest_sha256='f1c3fe0b1d1bf38535b68ac29284a396c2518c24f0d0e2bef1607ef896d8f854',
    pyarrow_version='24.0.0',
    text_fingerprint_module_sha256='b082d52765364ccb63ca85db00c78cfd67228fc48d133d1eef1fbabf426ea004')
# Audited v3 output includes both native v3 receipts and byte-preserved v2 receipts.
V3_CODE_IDENTITY = dict(
    arrow_schema_sha256='2a843d8fe12222812e37099df88804a51f47b6aa3ab26b0e06808ff0e5fcf7d8',
    baseline_compat_sha256='c790c42b310ccb77d1e6cf83df85d447b0251d8ea4eb7851459098c5cbf90d99',
    baseline_index_sha256='9495fdf638e7cc34f3dc27f558728ee89b3ddf7f7077277255ec7d77186e6bb4',
    common_sha256='22c31de5ef12f8196678fc5e0ee0c3e75fc82f7dc5bf6e18ab420cd7fd976236',
    ingest_sha256='db7e5231c012222f0fa81c77b73e8bda9d0690ac745dcea00c21c4fa4408c4d8',
    pyarrow_version='24.0.0',
    text_fingerprint_module_sha256='61fed185be0a454303ccde101943795b46f645b6f7f302d548c245ff1d312aa7')
AUDITED_CODE_IDENTITIES = (LEGACY_CODE_IDENTITY, V3_CODE_IDENTITY)
REUSE_CONTRACT = 'strict_success_rows_unchanged_quarantine_fallback.v1'
canonical, digest, file_sha, atomic_json = common.canonical, common.digest, common.file_sha, common.atomic_json


class PendingInputs(ValueError):
    """The fixed release or its indexes are not complete yet."""


class StopRequested(Exception):
    pass


def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def _relative(value):
    path = PurePosixPath(value)
    if not isinstance(value, str) or path.is_absolute() or '..' in path.parts or '\\' in value or not path.parts:
        raise ValueError('unsafe relative archive path')
    if str(path) != value:
        raise ValueError('noncanonical relative archive path')
    return value


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _load_download(download_control):
    control = Path(download_control).resolve(strict=True)
    manifest = _json(control / 'manifest.json')
    metadata_path = control / 'dataset_metadata.json'
    metadata = _json(metadata_path)
    if not manifest or manifest.get('repo') != REPO or manifest.get('revision') != REVISION:
        raise ValueError('download manifest is not the pinned baseline release')
    if (not metadata or metadata.get('id') != REPO or metadata.get('sha') != REVISION
            or manifest.get('metadata_sha256') != file_sha(metadata_path)):
        raise ValueError('official source metadata binding mismatch')
    official = {entry['rfilename']: entry for entry in metadata['siblings']}
    files, identifiers = {}, set()
    for entry in manifest['files']:
        relative = _relative(entry['path'])
        identifier = entry['id']
        source = official.get(relative)
        if (relative in files or not re.fullmatch('[0-9]{5,}', identifier) or identifier in identifiers
                or type(entry['size']) is not int or entry['size'] < 0 or source is None
                or entry['size'] != source['size']
                or entry.get('sha256') != (source.get('lfs') or {}).get('sha256')
                or entry.get('git_blob_oid') != source.get('blobId')):
            raise ValueError('download file differs from official source metadata')
        if not _sha(entry.get('sha256')) and not re.fullmatch('[0-9a-f]{40}', entry.get('git_blob_oid') or ''):
            raise ValueError('source file has no official checksum')
        files[relative] = entry
        identifiers.add(identifier)
    if (set(files) != set(official) or len(files) != manifest['total_files']
            or sum(entry['size'] for entry in files.values()) != manifest['total_bytes']):
        raise ValueError('download inventory is incomplete')
    units = []
    for relative, entry in sorted(files.items()):
        if not relative.startswith('shards/'):
            continue
        if not re.fullmatch(r'shards/[0-9]{3}/[0-9]{6}\.tar', relative) or not _sha(entry.get('sha256')):
            raise ValueError('unsupported baseline training carrier')
        identity = dict(repo=REPO, revision=REVISION, relative_path=relative,
                        size=entry['size'], sha256=entry['sha256'])
        units.append(dict(identity, unit_id=digest(identity)))
    if len(units) != EXPECTED_TRAINING_FILES:
        raise ValueError('fixed baseline training TAR count mismatch')
    inventory = [dict(path=path, size=entry['size'], sha256=entry.get('sha256'),
                      git_blob_oid=entry.get('git_blob_oid')) for path, entry in sorted(files.items())]
    return dict(control=control, root=Path(manifest['destination']).resolve(), manifest=manifest,
                files=files, units=units,
                inventory_sha256=digest(dict(repo=REPO, revision=REVISION, files=inventory)))


def _output(output, download):
    output = Path(output).resolve()
    for source in (download['root'], download['control']):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('index output must be separate from raw data and download control')
    return output


def _code_identity():
    import pyarrow as pa
    return dict(baseline_index_sha256=file_sha(__file__), ingest_sha256=file_sha(ingest.__file__),
                baseline_compat_sha256=file_sha(baseline_compat.__file__),
                common_sha256=file_sha(common.__file__),
                text_fingerprint_module_sha256=file_sha(training_release.__file__),
                arrow_schema_sha256=hashlib.sha256(ingest.arrow_schema().serialize().to_pybytes()).hexdigest(),
                pyarrow_version=pa.__version__)


def _safe_containers(output):
    for relative in ('state', 'state/locks', 'state/workers', 'state/errors', 'units', 'provenance'):
        path = output / relative
        if path.is_symlink() or output not in path.resolve().parents:
            raise ValueError('managed index directory must not be a symlink')


@contextmanager
def _lock(path, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('index lock file must not be a symlink')
    handle = path.open('a')
    acquired = False
    try:
        try:
            fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            acquired = True
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EACCES):
                raise
        yield acquired
    finally:
        handle.close()


def _compatible_plan(plan, download, rows_per_file):
    if (not plan or plan.get('schema_version') != PLAN_SCHEMA
            or plan.get('code_identity') not in AUDITED_CODE_IDENTITIES
            or plan.get('units') != download['units']
            or plan.get('source_inventory_sha256') != download['inventory_sha256']
            or plan.get('dataset_id') != REPO + '@' + REVISION
            or plan.get('contract') != CONTRACT or plan.get('rows_per_file') != rows_per_file
            or plan.get('max_buffer_bytes') != MAX_BUFFER_BYTES):
        raise ValueError('legacy index is not an audited compatible v2/v3 plan')


def _proof_path(output, relative):
    relative = _relative(relative)
    path = output / relative
    if path.is_symlink() or output not in path.resolve(strict=True).parents:
        raise ValueError('migration provenance escapes index output')
    return path


@lru_cache(maxsize=16)
def _cached_plan(path, expected_sha, stamp):
    # Stat includes ctime/inode: a changed parent plan invalidates cached evidence.
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('legacy plan provenance changed')
    plan = json.loads(data)
    return plan, {unit['unit_id']: unit for unit in plan['units']}


def _parent_plan(output, reuse):
    if (reuse.get('contract') != REUSE_CONTRACT
            or reuse.get('code_identity') not in AUDITED_CODE_IDENTITIES
            or not _sha(reuse.get('plan_sha256'))):
        raise ValueError('unsupported legacy compatibility binding')
    relative = 'provenance/plan-' + reuse['plan_sha256'] + '.json'
    path = _proof_path(output, relative)
    stat = path.stat()
    plan, units = _cached_plan(str(path), reuse['plan_sha256'],
                              (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
    if plan.get('code_identity') != reuse['code_identity']:
        raise ValueError('legacy plan code identity changed')
    return plan, units, path


def _legacy_plan(reuse_from, download, rows_per_file):
    origin = _output(reuse_from, download)
    _safe_containers(origin)
    plan = _json(origin / 'PLAN.json')
    current, seen = plan, set()
    while True:
        _compatible_plan(current, download, rows_per_file)
        if not current.get('reuse'):
            break
        reuse = current['reuse']
        if reuse.get('plan_sha256') in seen or len(seen) >= 4:
            raise ValueError('cyclic or excessive legacy plan ancestry')
        seen.add(reuse.get('plan_sha256'))
        current, _, _ = _parent_plan(origin, reuse)
    return origin, plan


def _migration_identity(output, plan, unit, receipt, receipt_sha, migration_path, depth=0):
    if receipt.get('code_identity') == plan['code_identity']:
        return True
    if depth >= 4:
        return False
    migration = _json(migration_path, {})
    reuse = plan.get('reuse', {})
    if (migration.get('contract') != REUSE_CONTRACT
            or migration.get('legacy_plan_sha256') != reuse.get('plan_sha256')
            or migration.get('original_receipt_sha256') != receipt_sha
            or migration.get('unit_id') != unit['unit_id']
            or migration.get('all_parquet_checksums_verified') is not True):
        return False
    parent, units, _ = _parent_plan(output, reuse)
    if (units.get(unit['unit_id']) != unit or any(parent.get(key) != plan.get(key)
            for key in ('schema_version', 'dataset_id', 'contract', 'source_inventory_sha256',
                        'rows_per_file', 'max_buffer_bytes'))):
        return False
    if receipt.get('code_identity') == parent['code_identity']:
        return 'parent_migration' not in migration
    proof = migration.get('parent_migration', {})
    if not _sha(proof.get('sha256')):
        return False
    path = _proof_path(output, proof.get('path', ''))
    if file_sha(path) != proof['sha256']:
        return False
    return _migration_identity(output, parent, unit, receipt, receipt_sha, path, depth + 1)


def prepare(download_control, output, *, rows_per_file=8192, reuse_from=None):
    """Freeze a portable release queue; no source TAR needs to exist yet."""
    ingest._positive(rows_per_file, 'rows_per_file')
    download = _load_download(download_control)
    output = _output(output, download)
    output.mkdir(parents=True, exist_ok=True)
    _safe_containers(output)
    plan = dict(schema_version=PLAN_SCHEMA, dataset_id=REPO + '@' + REVISION,
                repo=REPO, revision=REVISION, contract=CONTRACT, units=download['units'],
                source_inventory_sha256=download['inventory_sha256'],
                source_total_files=len(download['files']), source_total_bytes=download['manifest']['total_bytes'],
                rows_per_file=rows_per_file, max_buffer_bytes=MAX_BUFFER_BYTES,
                unresolved_binding_policy=baseline_compat.BINDING_CONTRACT,
                code_identity=_code_identity())
    with _lock(output / 'state/locks/prepare.lock', blocking=True):
        existing = _json(output / 'PLAN.json')
        if reuse_from is not None:
            origin, legacy = _legacy_plan(reuse_from, download, rows_per_file)
            if output == origin or output in origin.parents or origin in output.parents:
                raise ValueError('legacy and new index outputs must be separate')
            plan['reuse'] = dict(plan_sha256=file_sha(origin / 'PLAN.json'),
                                 code_identity=legacy['code_identity'], contract=REUSE_CONTRACT)
        elif existing and existing.get('reuse'):
            plan['reuse'] = existing['reuse']
        reuse = plan.get('reuse')
        if reuse:
            if (reuse.get('code_identity') not in AUDITED_CODE_IDENTITIES or reuse.get('contract') != REUSE_CONTRACT
                    or not _sha(reuse.get('plan_sha256'))):
                raise ValueError('unsupported legacy compatibility binding')
        if existing is not None and existing != plan:
            raise ValueError('baseline index plan changed; use a new output')
        if existing is None:
            atomic_json(output / 'PLAN.json', plan)
    return plan


def _context(download_control, output):
    download = _load_download(download_control)
    output = _output(output, download)
    _safe_containers(output)
    plan = _json(output / 'PLAN.json')
    if not plan or plan.get('schema_version') != PLAN_SCHEMA:
        raise ValueError('prepare the baseline index first')
    if (plan.get('units') != download['units'] or plan.get('source_inventory_sha256') != download['inventory_sha256']
            or plan.get('dataset_id') != REPO + '@' + REVISION or plan.get('contract') != CONTRACT
            or plan.get('code_identity') != _code_identity()):
        raise ValueError('baseline index source or frozen code identity changed')
    return download, output, plan


def _verified_file(download, entry):
    """Trust only the downloader's exact checksum receipt and unchanged final stat."""
    marker = _json(download['control'] / 'verified' / (entry['id'] + '.json'), {})
    target = download['root'] / entry['path']
    if (marker.get('revision') != REVISION or marker.get('path') != entry['path']
            or marker.get('source_sha256') != entry.get('sha256')
            or marker.get('source_git_blob_oid') != entry.get('git_blob_oid')
            or not _sha(marker.get('sha256'))
            or (entry.get('sha256') and marker['sha256'] != entry['sha256'])):
        return False
    try:
        resolved = target.resolve(strict=True)
        if download['root'] not in resolved.parents or not resolved.is_file():
            raise ValueError('source TAR escapes runtime dataset root')
        stat = target.stat()
    except FileNotFoundError:
        return False
    return stat.st_size == marker.get('size') == entry['size'] and stat.st_mtime_ns == marker.get('mtime_ns')


def _source_complete(download):
    complete = _json(download['control'] / 'COMPLETE.json', {})
    return (complete.get('revision') == REVISION and complete.get('files') == len(download['files'])
            and complete.get('bytes') == download['manifest']['total_bytes']
            and all(_verified_file(download, entry) for entry in download['files'].values()))


def _receipt(output, plan, unit, *, verify_files=False):
    path = output / 'units' / unit['unit_id'] / 'COMPLETE.json'
    value = _json(path)
    if value is None:
        return None
    identity_matches = _migration_identity(output, plan, unit, value, file_sha(path),
                                           path.parent / 'MIGRATION.json')
    if (value.get('schema_version') != UNIT_SCHEMA or value.get('complete') is not True
            or value.get('unit') != unit or not identity_matches
            or value.get('source_sha256') != unit['sha256'] or value.get('source_exhausted') is not True
            or value.get('dataset_id') != plan['dataset_id'] or value.get('records', 0) < 1
            or sum(entry['rows'] for entry in value.get('catalog', [])) != value['records']):
        raise ValueError('baseline unit receipt identity mismatch')
    if verify_files:
        for entry in value['catalog']:
            relative = _relative(entry['path'])
            target = (output / relative).resolve(strict=True)
            if (PurePosixPath(relative).parent != PurePosixPath('units') / unit['unit_id']
                    or target.parent != path.parent.resolve() or output not in target.parents
                    or target.stat().st_size != entry['bytes']
                    or file_sha(target) != entry['sha256']):
                raise ValueError('committed baseline Parquet changed')
    return value


def migrate(download_control, output, reuse_from, *, max_units=None, workers=1):
    """Verify and hard-link audited v2/v3 units with their complete ancestry."""
    if max_units is not None:
        ingest._positive(max_units, 'max_units')
    ingest._positive(workers, 'workers')
    if workers > 32:
        raise ValueError('migration workers must be at most 32')
    download, output, plan = _context(download_control, output)
    origin, legacy = _legacy_plan(reuse_from, download, plan['rows_per_file'])
    if output == origin or output in origin.parents or origin in output.parents:
        raise ValueError('legacy and new index outputs must be separate')
    if plan.get('reuse') != dict(plan_sha256=file_sha(origin / 'PLAN.json'),
                                code_identity=legacy['code_identity'], contract=REUSE_CONTRACT):
        raise ValueError('prepare must bind this exact legacy PLAN before migration')
    count = records = 0
    provenance = output / 'provenance' / ('plan-' + plan['reuse']['plan_sha256'] + '.json')
    with _lock(output / 'state/locks/migrate.lock', blocking=True):
        ancestors = [(origin / 'PLAN.json', plan['reuse']['plan_sha256'])]
        ancestor = legacy
        while ancestor.get('reuse'):
            ancestor, _, source = _parent_plan(origin, ancestor['reuse'])
            ancestors.append((source, file_sha(source)))
        for source, sha in ancestors:
            target = output / 'provenance' / ('plan-' + sha + '.json')
            if target.exists():
                if target.is_symlink() or file_sha(target) != sha:
                    raise ValueError('legacy plan provenance changed')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
        def one(unit):
            if not _verified_file(download, download['files'][unit['relative_path']]):
                return 0, 0
            with _lock(output / 'state/locks' / ('unit-' + unit['unit_id'] + '.lock')) as acquired:
                if not acquired:
                    return 0, 0
                if _receipt(output, plan, unit, verify_files=True) is not None:
                    return 0, 0
                original = _receipt(origin, legacy, unit, verify_files=True)
                if original is None:
                    return 0, 0
                _space(output, MIN_RESERVE_BYTES)
                directory = output / 'units' / unit['unit_id']
                if directory.is_symlink() or output not in directory.resolve().parents:
                    raise ValueError('unsafe migration unit directory')
                if directory.exists():
                    shutil.rmtree(directory)
                directory.mkdir(parents=True)
                for entry in original['catalog']:
                    os.link(origin / entry['path'], output / entry['path'])
                old_receipt = origin / 'units' / unit['unit_id'] / 'COMPLETE.json'
                migration = dict(contract=REUSE_CONTRACT,
                    legacy_plan_sha256=plan['reuse']['plan_sha256'], unit_id=unit['unit_id'],
                    original_receipt_sha256=file_sha(old_receipt), all_parquet_checksums_verified=True)
                if original['code_identity'] != legacy['code_identity']:
                    old_proof = old_receipt.parent / 'MIGRATION.json'
                    proof_sha = file_sha(old_proof)
                    proof_target = directory / ('migration-' + proof_sha + '.json')
                    os.link(old_proof, proof_target)
                    migration['parent_migration'] = dict(path=str(proof_target.relative_to(output)), sha256=proof_sha)
                    # Preserve every earlier proof at its already-bound logical path.
                    proof = _json(old_proof)
                    while proof.get('parent_migration'):
                        ref = proof['parent_migration']
                        source = _proof_path(origin, ref['path'])
                        if file_sha(source) != ref['sha256']:
                            raise ValueError('ancestor migration proof changed')
                        target = output / ref['path']
                        if target.parent != directory:
                            raise ValueError('ancestor proof is outside its unit')
                        if not target.exists():
                            os.link(source, target)
                        proof = _json(source)
                atomic_json(directory / 'MIGRATION.json', migration)
                os.link(old_receipt, directory / 'COMPLETE.json')
                _receipt(output, plan, unit, verify_files=True)
                return 1, original['records']
        # Each unit appears once in this executor. lockf still excludes other
        # processes; never submit the same unit twice within a process.
        pending = iter(plan['units'])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while max_units is None or count < max_units:
                width = workers if max_units is None else min(workers, max_units - count)
                batch = list(islice(pending, width))
                if not batch:
                    break
                for migrated, reused in pool.map(one, batch):
                    count += migrated
                    records += reused
    return dict(phase='migration_complete', units_migrated=count, records_reused=records)


def _portable_row(row, unit, dataset_id):
    """Change storage/identity fields only; keep all original supervision exactly."""
    row = dict(row)
    locator = json.loads(row['original_locator_json'])
    locator.pop('archive_fingerprint', None)
    locator.update(adapter=CONTRACT, dataset_id=dataset_id, path=unit['relative_path'],
                   archive_sha256=unit['sha256'], archive_size=unit['size'])
    sample_id = digest(dict(contract=CONTRACT, dataset_id=dataset_id, unit_id=unit['unit_id'],
                            source_key=row['source_key'], components=locator['components']))
    original = json.loads(row['original_supervision_json'])
    original['sample_id'] = sample_id
    images = []
    for source_ref in row['images']:
        ref = dict(source_ref)
        ref['path'] = unit['relative_path']
        metadata = json.loads(ref['metadata_json'])
        metadata.update({key: value for key, value in ref.items() if key != 'metadata_json'})
        metadata.update(dataset_id=dataset_id, archive_sha256=unit['sha256'], archive_size=unit['size'])
        ref['metadata_json'] = canonical(metadata)
        images.append(ref)
    metadata = json.loads(row['adapter_metadata_json'])
    metadata.update(adapter=CONTRACT, parser_contract=ingest.BALANCED_CONTRACT, dataset_id=dataset_id)
    row.update(sample_id=sample_id, images=images, original_supervision_json=canonical(original),
               original_locator_json=canonical(locator), adapter_metadata_json=canonical(metadata))
    return row


def _space(output, reserve_bytes):
    if shutil.disk_usage(output).free < reserve_bytes:
        raise OSError(errno.ENOSPC, 'baseline preprocessing free-space reserve reached')


def _process_unit(download, output, plan, unit, stop, reserve_bytes):
    import pyarrow as pa
    import pyarrow.parquet as pq
    entry = download['files'][unit['relative_path']]
    if not _verified_file(download, entry):
        raise PendingInputs('source TAR lacks current verified download receipt')
    existing = _receipt(output, plan, unit, verify_files=True)
    if existing:
        return existing
    source = download['root'] / unit['relative_path']
    before = source.stat()
    _space(output, reserve_bytes)
    if file_sha(source) != unit['sha256']:
        raise ValueError('source TAR SHA256 differs from pinned release')
    directory = output / 'units' / unit['unit_id']
    if directory.is_symlink() or output not in directory.resolve().parents:
        raise ValueError('unsafe unit output symlink')
    if directory.exists():
        shutil.rmtree(directory)  # This unit lock owns only its uncommitted derived files.
    directory.mkdir(parents=True)
    buffers, sizes, catalog = defaultdict(list), Counter(), []
    rows = slots = buffered = buffered_bytes = 0
    formats, sources, adapter_statuses = Counter(), Counter(), Counter()

    def flush(key):
        nonlocal buffered, buffered_bytes
        _space(output, reserve_bytes)
        values = buffers.pop(key)
        target = directory / ('%06d.parquet' % len(catalog))
        partial = target.with_suffix('.partial')
        table = pa.Table.from_pylist(values, schema=ingest.arrow_schema())
        pq.write_table(table, partial, compression='zstd', row_group_size=min(128, plan['rows_per_file']))
        with partial.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(partial, target)
        catalog.append(dict(path=str(target.relative_to(output)), rows=len(values), bytes=target.stat().st_size,
                            sha256=file_sha(target), collection='dcvlm', source=key, source_shard=unit['relative_path']))
        buffered -= len(values)
        buffered_bytes -= sizes.pop(key)

    iterator = baseline_compat.rows(dict(path=str(source), relative_path=unit['relative_path'], source=None))
    try:
        for raw in iterator:
            if stop.is_set():
                raise StopRequested()
            row = _portable_row(raw, unit, plan['dataset_id'])
            key = row['source']
            row_bytes = len(canonical(row).encode())
            buffers[key].append(row)
            sizes[key] += row_bytes
            buffered_bytes += row_bytes
            buffered += 1
            rows += 1
            slots += row['image_count']
            formats[row['format']] += 1
            sources['dcvlm:' + key] += 1
            adapter_statuses[json.loads(row['adapter_metadata_json'])['prepared_adapter_status']] += 1
            while buffered >= plan['rows_per_file'] or buffered_bytes >= plan['max_buffer_bytes']:
                flush(max(buffers, key=lambda value: sizes[value]))
        for key in sorted(buffers):
            flush(key)
    finally:
        iterator.close()
    after = source.stat()
    if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino)
            or not _verified_file(download, entry)):
        raise ValueError('source TAR changed during indexing')
    if not rows:
        raise ValueError('baseline TAR decoded no records')
    value = dict(schema_version=UNIT_SCHEMA, complete=True, unit=unit, dataset_id=plan['dataset_id'],
                 code_identity=plan['code_identity'], source_sha256=unit['sha256'], source_exhausted=True,
                 records=rows, image_slots=slots, formats=dict(formats), sources=dict(sources),
                 adapter_statuses=dict(adapter_statuses), catalog=catalog)
    atomic_json(directory / 'COMPLETE.json', value)
    return value


def _status(download, output, plan):
    ready = done = records = 0
    errors = Counter()
    for unit in plan['units']:
        ready += bool(_verified_file(download, download['files'][unit['relative_path']]))
        receipt = _receipt(output, plan, unit)
        if receipt is not None:
            done += 1
            records += receipt['records']
        else:
            error = _json(output / 'state/errors' / (unit['unit_id'] + '.json'))
            if error:
                errors[error['error_type']] += 1
    source_complete = _source_complete(download)
    complete = _json(output / 'COMPLETE.json')
    if complete is not None:
        if (complete.get('complete') is not True or complete.get('validation_passed') is not True
                or complete.get('manifest_sha256') != file_sha(output / 'manifest.json')
                or complete.get('catalog_sha256') != file_sha(output / 'catalog.json')
                or complete.get('plan_sha256') != file_sha(output / 'PLAN.json')
                or not source_complete or done != len(plan['units'])):
            raise ValueError('published baseline snapshot binding mismatch')
        phase = 'complete'
    elif done == len(plan['units']) and source_complete:
        phase = 'ready_to_finalize'
    elif ready == done:
        phase = 'waiting_for_download'
    else:
        phase = 'indexing'
    return dict(time=time.time(), phase=phase, dataset_id=plan['dataset_id'], source_complete=source_complete,
                units_total=len(plan['units']), units_source_ready=ready, units_complete=done,
                units_waiting_for_download=len(plan['units'])-ready, records_indexed=records,
                unit_errors=dict(errors),
                complete=phase == 'complete')


def status(download_control, output):
    """Inspect the fixed queue; this does not publish or alter its artifacts."""
    return _status(*_context(download_control, output))


def worker(download_control, output, worker, *, watch=False, max_units=None,
           reserve_bytes=MIN_RESERVE_BYTES, poll_seconds=30):
    """Claim ready TARs with GPFS record locks; independent processes share work."""
    name = str(worker)
    if name in ('.', '..') or not re.fullmatch('[A-Za-z0-9_.-]+', name):
        raise ValueError('invalid worker slot')
    if reserve_bytes < MIN_RESERVE_BYTES or poll_seconds < 1:
        raise ValueError('reserve must be at least 1 TiB and poll interval positive')
    if max_units is not None:
        ingest._positive(max_units, 'max_units')
    download, output, plan = _context(download_control, output)
    state = output / 'state'
    stop = threading.Event()
    with _lock(state / 'locks' / ('worker-' + name + '.lock')) as acquired:
        if not acquired:
            raise ValueError('baseline worker slot already running')
        identity = dict(worker=name, pid=os.getpid(), host=socket.gethostname(),
                        start_ticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        live = dict(identity, phase='indexing', unit_id=None, units_completed=0)

        def heartbeat():
            while not stop.is_set():
                atomic_json(state / 'workers' / (name + '.json'), dict(live, time=time.time()))
                if (state / 'STOP').exists() or (state / 'workers' / (name + '.STOP')).exists():
                    stop.set()
                    break
                stop.wait(5)

        old_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        order = list(plan['units'])
        random.Random(name).shuffle(order)
        try:
            while not stop.is_set():
                for unit in order:
                    if stop.is_set() or (max_units is not None and live['units_completed'] >= max_units):
                        break
                    if _receipt(output, plan, unit) is not None:
                        continue
                    if not _verified_file(download, download['files'][unit['relative_path']]):
                        continue
                    with _lock(state / 'locks' / ('unit-' + unit['unit_id'] + '.lock')) as claimed:
                        if not claimed or _receipt(output, plan, unit) is not None:
                            continue
                        live.update(phase='indexing', unit_id=unit['unit_id'])
                        try:
                            _process_unit(download, output, plan, unit, stop, reserve_bytes)
                            live['units_completed'] += 1
                            (state / 'errors' / (unit['unit_id'] + '.json')).unlink(missing_ok=True)
                        except StopRequested:
                            stop.set()
                            break
                        except PendingInputs:
                            pass
                        except Exception as error:
                            atomic_json(state / 'errors' / (unit['unit_id'] + '.json'),
                                        dict(time=time.time(), worker=name, unit_id=unit['unit_id'],
                                             relative_path=unit['relative_path'], error_type=type(error).__name__,
                                             error_message=str(error)[:400] if isinstance(error, ValueError) else None))
                            if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
                                live['phase'] = 'paused_space'
                                stop.wait(poll_seconds)
                                break
                        finally:
                            live['unit_id'] = None
                current = _status(download, output, plan)
                if (not watch or current['phase'] in ('complete', 'ready_to_finalize')
                        or (max_units is not None and live['units_completed'] >= max_units)):
                    return current
                live['phase'] = 'waiting_for_download' if current['phase'] == 'waiting_for_download' else 'waiting_for_units'
                stop.wait(poll_seconds)
            return _status(download, output, plan)
        finally:
            stop.set()
            monitor.join(timeout=10)
            live.update(phase='stopped', unit_id=None)
            atomic_json(state / 'workers' / (name + '.json'), dict(live, time=time.time()))
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def finalize(download_control, output):
    """Publish only the complete upstream release and every committed TAR index."""
    download, output, plan = _context(download_control, output)
    with _lock(output / 'state/locks/finalize.lock', blocking=True):
        current = _status(download, output, plan)
        if current['phase'] not in ('complete', 'ready_to_finalize'):
            raise PendingInputs('waiting for all training indexes and complete verified source release')
        catalog, receipts, migrations = [], [], []
        records = slots = 0
        formats, sources, adapter_statuses = Counter(), Counter(), Counter()
        for unit in plan['units']:
            value = _receipt(output, plan, unit, verify_files=True)
            records += value['records']
            slots += value['image_slots']
            formats.update(value['formats'])
            sources.update(value['sources'])
            adapter_statuses.update(value.get('adapter_statuses', {'ready': value['records']}))
            catalog.extend(value['catalog'])
            receipt = output / 'units' / unit['unit_id'] / 'COMPLETE.json'
            receipts.append(dict(path=str(receipt.relative_to(output)), sha256=file_sha(receipt)))
            if value['code_identity'] != plan['code_identity']:
                migration = receipt.parent / 'MIGRATION.json'
                migrations.append(dict(path=str(migration.relative_to(output)), sha256=file_sha(migration)))
        statistics = dict(samples=records, image_slots=slots, formats=dict(formats), sources=dict(sources),
                          adapter_statuses=dict(adapter_statuses), input_units_planned=len(plan['units']),
                          input_units_processed=len(receipts))
        manifest = dict(schema_version=ingest.SCHEMA, phase='complete', collections=['dcvlm'],
                        scope='full_pinned_upstream_release', enumeration_scope='official_pinned_training_tar_manifest',
                        portable_datasets=[plan['dataset_id']], allowed_image_roots=[],
                        source_files_modified=False, image_payloads_copied=False,
                        source_inventory_sha256=plan['source_inventory_sha256'],
                        source_validation=dict(repo=REPO, revision=REVISION, files=plan['source_total_files'],
                                               bytes=plan['source_total_bytes'], complete=True),
                        plan_sha256=file_sha(output / 'PLAN.json'), unit_receipts=receipts, statistics=statistics)
        if plan.get('reuse'):
            manifest['compatible_legacy_implementation'] = plan['reuse']
            provenance = output / 'provenance' / ('plan-' + plan['reuse']['plan_sha256'] + '.json')
            if migrations and (not provenance.is_file() or file_sha(provenance) != plan['reuse']['plan_sha256']):
                raise ValueError('migrated units lack their bound legacy plan provenance')
            if provenance.is_file():
                manifest['legacy_plan'] = dict(path=str(provenance.relative_to(output)),
                                               sha256=plan['reuse']['plan_sha256'])
            manifest['legacy_unit_migrations'] = migrations
            ancestry, ancestor = [], plan
            while ancestor.get('reuse') and provenance.is_file():
                ancestor, _, source = _parent_plan(output, ancestor['reuse'])
                ancestry.append(dict(path=str(source.relative_to(output)), sha256=file_sha(source)))
            manifest['legacy_plan_ancestry'] = ancestry
        existing = _json(output / 'COMPLETE.json')
        if existing is not None:
            if _json(output / 'manifest.json') != manifest or _json(output / 'catalog.json') != catalog:
                raise ValueError('published baseline manifest changed')
            return existing
        if not _source_complete(download):
            raise PendingInputs('source validation changed while finalizing')
        atomic_json(output / 'catalog.json', catalog)
        atomic_json(output / 'manifest.json', manifest)
        complete = dict(schema_version=ingest.SCHEMA, complete=True, validation_passed=True,
                        validation_scope='full_pinned_release_checksums_preserved_originals_and_explicit_binding_quarantine',
                        catalog_sha256=file_sha(output / 'catalog.json'), manifest_sha256=file_sha(output / 'manifest.json'),
                        plan_sha256=file_sha(output / 'PLAN.json'), scope=manifest['scope'], statistics=statistics)
        atomic_json(output / 'COMPLETE.json', complete)
        return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'migrate', 'worker', 'status', 'finalize'):
        child = subparsers.add_parser(name)
        child.add_argument('--download-control', required=True)
        child.add_argument('--output', required=True)
        if name == 'prepare':
            child.add_argument('--rows-per-file', type=int, default=8192)
            child.add_argument('--reuse-from')
        if name == 'migrate':
            child.add_argument('--reuse-from', required=True)
            child.add_argument('--max-units', type=int)
            child.add_argument('--workers', type=int, default=1)
        if name == 'worker':
            child.add_argument('--worker', required=True)
            child.add_argument('--watch', action='store_true')
            child.add_argument('--max-units', type=int)
            child.add_argument('--reserve-tib', type=float, default=1)
            child.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            plan = prepare(args.download_control, args.output, rows_per_file=args.rows_per_file,
                           reuse_from=args.reuse_from)
            result = dict(phase='prepared', dataset_id=plan['dataset_id'], units_total=len(plan['units']))
        elif args.command == 'migrate':
            result = migrate(args.download_control, args.output, args.reuse_from, max_units=args.max_units, workers=args.workers)
        elif args.command == 'worker':
            result = worker(args.download_control, args.output, args.worker, watch=args.watch,
                            max_units=args.max_units, reserve_bytes=int(args.reserve_tib * (1 << 40)),
                            poll_seconds=args.poll_seconds)
        else:
            result = globals()[args.command](args.download_control, args.output)
        print(canonical(result))
        return 0
    except PendingInputs:
        print(canonical(dict(error='waiting_for_inputs', complete=False)))
        return 2
    except Exception as error:
        print(canonical(dict(error_type=type(error).__name__, complete=False)))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
