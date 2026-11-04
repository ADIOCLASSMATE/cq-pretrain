"""Lossless TAR-local compaction of a complete pinned native metadata index.

Only Arrow metadata rows are rewritten. Images and original input files are read
only. A TAR's true row sources remain untouched and its output catalog carries a
closed mixed_source_tar.v1 membership list for downstream readers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
import time

from corpus_pipeline.production.common import atomic_json, digest, file_sha

NATIVE = 'corpus.native.v2'
CONTRACT = 'mixed_source_tar.v1'
PLAN_SCHEMA = 'corpus.native_compaction.plan.v1'
GROUP_SCHEMA = 'corpus.native_compaction.group.v1'
MIN_RESERVE_BYTES = 1 << 40
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class PendingGroups(ValueError):
    pass


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def _relative(value):
    require(isinstance(value, str) and value, 'Missing relative metadata path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in value
            and str(path) == value and value != '.', 'Unsafe relative metadata path')
    return value


def bound(root, relative):
    path = (root / _relative(relative)).resolve(strict=True)
    require(path.is_file() and path.is_relative_to(root), 'Metadata path escapes snapshot')
    return path


@contextmanager
def claim(path, blocking=False):
    """lockf for hosts, plus a thread lock because POSIX locks are process-wide."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCKS_GUARD:
        thread_lock = _LOCKS.setdefault(str(path.resolve()), threading.Lock())
    acquired = thread_lock.acquire(blocking=blocking)
    if not acquired:
        yield False
        return
    try:
        with path.open('a+') as stream:
            try:
                fcntl.lockf(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
            else:
                try:
                    yield True
                finally:
                    fcntl.lockf(stream, fcntl.LOCK_UN)
    finally:
        thread_lock.release()


def _stamp(path):
    value = path.stat()
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns}


def _space(output, reserve_bytes):
    require(reserve_bytes >= MIN_RESERVE_BYTES, 'At least 1 TiB reserve is required')
    if shutil.disk_usage(output).free < reserve_bytes:
        raise PendingGroups('Insufficient free space for the required 1 TiB reserve')


def _input(index):
    index = Path(index).resolve(strict=True)
    documents, bindings = {}, {}
    for name in ('PLAN.json', 'COMPLETE.json', 'manifest.json', 'catalog.json'):
        data = (index / name).read_bytes()
        documents[name] = json.loads(data)
        bindings[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    plan, complete, manifest, catalog = (documents[k] for k in
        ('PLAN.json', 'COMPLETE.json', 'manifest.json', 'catalog.json'))
    require(complete.get('schema_version') == manifest.get('schema_version') == NATIVE
            and complete.get('complete') is True and complete.get('validation_passed') is True
            and complete.get('scope') == manifest.get('scope') == 'full_pinned_upstream_release'
            and manifest.get('phase') == 'complete', 'A fully verified pinned native index is required')
    for key, name in (('plan_sha256', 'PLAN.json'), ('manifest_sha256', 'manifest.json'), ('catalog_sha256', 'catalog.json')):
        require(complete.get(key) == bindings[name]['sha256'], 'Native input receipt binding mismatch')
    require(manifest.get('plan_sha256') == bindings['PLAN.json']['sha256'], 'Native input plan binding mismatch')
    source = manifest.get('source_validation', {})
    require(source.get('complete') is True and source.get('repo') == plan.get('repo')
            and source.get('revision') == plan.get('revision')
            and source.get('files') == plan.get('source_total_files')
            and source.get('bytes') == plan.get('source_total_bytes')
            and manifest.get('source_inventory_sha256') == plan.get('source_inventory_sha256'),
            'Missing or inconsistent full source validation')
    require(isinstance(catalog, list) and catalog, 'A nonempty native input catalog is required')
    seen, totals, rows = set(), Counter(), 0
    for entry in catalog:
        path = _relative(entry.get('path'))
        require(path not in seen, 'Duplicate input catalog file')
        seen.add(path)
        require(entry.get('collection') in ('dcvlm', 'genund')
                and isinstance(entry.get('source'), str) and entry['source'] not in ('', '__mixed__')
                and isinstance(entry.get('source_shard'), str) and entry['source_shard'], 'Invalid source membership')
        _relative(entry['source_shard'])
        require(type(entry.get('rows')) is int and entry['rows'] > 0
                and type(entry.get('bytes')) is int and entry['bytes'] > 0
                and re.fullmatch('[0-9a-f]{64}', entry.get('sha256', '')) is not None,
                'Invalid input file checksum/count')
        rows += entry['rows']
        totals[(entry['collection'], entry['source'])] += entry['rows']
    require(rows == manifest.get('statistics', {}).get('samples') == complete.get('statistics', {}).get('samples'),
            'Native input catalog row coverage differs')
    require(set(manifest.get('collections', [])) == {entry['collection'] for entry in catalog},
            'Native input collection coverage differs')
    return index, documents, bindings, totals


def prepare(index, output, *, max_group_rows=65536, batch_rows=256):
    require(type(max_group_rows) is int and max_group_rows > 0
            and type(batch_rows) is int and 1 <= batch_rows <= 8192, 'Invalid bounded compaction limits')
    index, documents, bindings, totals = _input(index)
    output = Path(output).resolve()
    require(not output.is_relative_to(index) and not index.is_relative_to(output), 'Input and output must not overlap')
    groups, current, key, rows = [], [], None, 0
    def flush():
        if current:
            value = {'collection': key[0], 'source_shard': key[1], 'inputs': list(current), 'rows': rows,
                     'sources': sorted({item['source'] for item in current})}
            value['group_id'] = digest(value)
            groups.append(value)
    for entry in sorted(documents['catalog.json'], key=lambda item: (item['collection'], item['source_shard'], item['path'])):
        group_key = (entry['collection'], entry['source_shard'])
        require(entry['rows'] <= max_group_rows, 'An input file exceeds the maximum group row count')
        if current and (key != group_key or rows + entry['rows'] > max_group_rows):
            flush(); current, rows = [], 0
        key = group_key
        current.append(entry)
        rows += entry['rows']
    flush()
    plan = {'schema': PLAN_SCHEMA, 'contract': CONTRACT, 'input_root': str(index), 'inputs': bindings,
            'code_sha256': file_sha(__file__), 'max_group_rows': max_group_rows, 'batch_rows': batch_rows,
            'input_files': len(documents['catalog.json']), 'rows': sum(totals.values()), 'groups': groups,
            'per_source': [dict(collection=c, source=s, rows=n) for (c, s), n in sorted(totals.items())]}
    output.mkdir(parents=True, exist_ok=True)
    _space(output, MIN_RESERVE_BYTES)
    with claim(output / 'state/prepare.lock', blocking=True):
        path = output / 'PLAN.json'
        if path.exists():
            require(read(path) == plan, 'Compaction input or runtime changed; use a new output')
        else:
            require(not (output / 'COMPLETE.json').exists(), 'Completed output has no compaction plan')
            atomic_json(path, plan)
        lineage = output / 'lineage'
        require(not lineage.is_symlink(), 'Unsafe lineage directory')
        lineage.mkdir(exist_ok=True)
        for name, binding in bindings.items():
            target = lineage / name
            if target.exists():
                require(file_sha(target) == binding['sha256'], 'Copied source lineage changed')
            else:
                temporary = lineage / (name + '.partial')
                require(not temporary.is_symlink(), 'Unsafe lineage temporary path')
                shutil.copyfile(index / name, temporary)
                require(file_sha(temporary) == binding['sha256'], 'Source changed while copying metadata lineage')
                os.replace(temporary, target)
    return {'groups': len(groups), 'input_files': plan['input_files'], 'rows': plan['rows'], 'plan_sha256': file_sha(path)}


def _context(index, output):
    index, documents, bindings, _ = _input(index)
    output = Path(output).resolve(strict=True)
    plan = read(output / 'PLAN.json')
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('contract') == CONTRACT
            and plan.get('input_root') == str(index) and plan.get('inputs') == bindings
            and plan.get('code_sha256') == file_sha(__file__), 'Compaction plan/source/runtime binding changed')
    require(not output.is_relative_to(index) and not index.is_relative_to(output), 'Input/output overlap')
    planned = [entry for group in plan['groups'] for entry in group['inputs']]
    require(len({entry['path'] for entry in planned}) == len(planned) == plan['input_files']
            and sorted(planned, key=lambda entry: entry['path']) == sorted(documents['catalog.json'], key=lambda entry: entry['path']),
            'Compaction plan must cover every input file exactly once')
    for group in plan['groups']:
        body = {k: v for k, v in group.items() if k != 'group_id'}
        require(group['group_id'] == digest(body) and group['rows'] == sum(item['rows'] for item in group['inputs'])
                and group['sources'] == sorted({item['source'] for item in group['inputs']})
                and all(item['collection'] == group['collection'] and item['source_shard'] == group['source_shard'] for item in group['inputs']),
                'Compaction group membership changed')
    return index, output, plan, documents


def _receipt(output, plan_hash, group, verify=False):
    directory = output / 'groups' / group['group_id']
    marker = directory / 'COMPLETE.json'
    if not marker.exists():
        return None
    value = read(marker)
    expected = {'path': f'groups/{group["group_id"]}/data.parquet', 'collection': group['collection'],
                'source': '__mixed__', 'sources': group['sources'], 'source_shard': group['source_shard'],
                'contract': CONTRACT, 'rows': group['rows']}
    require(value.get('schema') == GROUP_SCHEMA and value.get('complete') is True
            and value.get('plan_sha256') == plan_hash and value.get('group_id') == group['group_id']
            and value.get('input_entries_sha256') == digest(group['inputs']), 'Group receipt binding changed')
    entry = value.get('catalog', {})
    require(all(entry.get(k) == v for k, v in expected.items()), 'Group catalog membership/count changed')
    path = bound(output, entry['path'])
    require(path.stat().st_size == entry.get('bytes')
            and (not verify or file_sha(path) == entry.get('sha256')), 'Committed compacted Parquet changed')
    expected_sources = Counter()
    for item in group['inputs']:
        expected_sources[item['source']] += item['rows']
    require(value.get('source_rows') == dict(expected_sources), 'Group source row coverage differs')
    return value


def _columns(batch):
    return {name: batch.column(batch.schema.get_field_index(name)).to_pylist()
            for name in ('collection', 'source', 'source_shard')}


def _group(index, output, plan, plan_hash, group, reserve_bytes):
    import pyarrow.parquet as pq
    if (output / 'STOP').exists():
        return {'phase': 'stopped', 'group_id': group['group_id']}
    with claim(output / 'state/locks' / (group['group_id'] + '.lock')) as acquired:
        if not acquired:
            return {'phase': 'claimed_elsewhere', 'group_id': group['group_id']}
        existing = _receipt(output, plan_hash, group, verify=True)
        if existing is not None:
            return {'phase': 'reused', 'group_id': group['group_id']}
        _space(output, reserve_bytes)
        directory = output / 'groups' / group['group_id']
        require(not directory.is_symlink() and directory.resolve().is_relative_to(output), 'Unsafe group directory')
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'data.parquet.partial'
        require(not temporary.is_symlink(), 'Unsafe compaction temporary path')
        writer, schema, total, sources, verified = None, None, 0, Counter(), []
        try:
            for item in group['inputs']:
                if (output / 'STOP').exists():
                    raise PendingGroups('Compaction stop requested')
                _space(output, reserve_bytes)
                path = bound(index, item['path']); before = _stamp(path)
                require(before['bytes'] == item['bytes'] and file_sha(path) == item['sha256'], 'Input Parquet checksum changed')
                parquet = pq.ParquetFile(path)
                require(parquet.metadata.num_rows == item['rows'], 'Input Parquet row count changed')
                if schema is None:
                    schema = parquet.schema_arrow
                    writer = pq.ParquetWriter(temporary, schema, compression='zstd')
                require(schema.equals(parquet.schema_arrow, check_metadata=True), 'Input Arrow schemas differ')
                counted = 0
                for batch in parquet.iter_batches(batch_size=plan['batch_rows'], use_threads=False):
                    values = _columns(batch)
                    require(set(values['collection']) == {item['collection']}
                            and set(values['source']) == {item['source']}
                            and set(values['source_shard']) == {item['source_shard']}, 'Input rows escape catalog membership')
                    # Direct Arrow write: never round-trip supervision or image locators through Python.
                    writer.write_batch(batch, row_group_size=plan['batch_rows'])
                    counted += batch.num_rows
                    sources[item['source']] += batch.num_rows
                require(counted == item['rows'] and _stamp(path) == before, 'Input changed while compacting')
                verified.append({'path': item['path'], 'sha256': item['sha256'], **before})
                total += counted
            require(total == group['rows'], 'Compacted row coverage mismatch')
            writer.close(); writer = None
            _space(output, reserve_bytes)
            target = directory / 'data.parquet'
            os.replace(temporary, target)
            catalog = {'path': str(target.relative_to(output)), 'rows': total, 'bytes': target.stat().st_size,
                'sha256': file_sha(target), 'collection': group['collection'], 'source': '__mixed__',
                'sources': sorted(sources), 'source_shard': group['source_shard'], 'contract': CONTRACT}
            receipt = {'schema': GROUP_SCHEMA, 'complete': True, 'group_id': group['group_id'],
                'plan_sha256': plan_hash, 'input_entries_sha256': digest(group['inputs']), 'verified_inputs': verified,
                'catalog': catalog, 'source_rows': dict(sources), 'original_row_fields_modified': False,
                'image_payloads_copied': False, 'source_files_modified': False}
            atomic_json(directory / 'COMPLETE.json', receipt)
            return {'phase': 'completed', 'group_id': group['group_id'], 'rows': total}
        finally:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)


def worker(index, output, *, workers=8, reserve_bytes=MIN_RESERVE_BYTES, max_groups=None):
    require(type(workers) is int and 1 <= workers <= 64, 'Invalid worker count')
    require(reserve_bytes >= MIN_RESERVE_BYTES, 'At least 1 TiB reserve is required')
    require(max_groups is None or type(max_groups) is int and max_groups > 0, 'Invalid group limit')
    index, output, plan, _ = _context(index, output)
    plan_hash = file_sha(output / 'PLAN.json')
    groups = iter(plan['groups'][:max_groups] if max_groups else plan['groups'])
    counts = Counter(); active = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def fill():
            while len(active) < workers * 2 and not (output / 'STOP').exists():
                group = next(groups, None)
                if group is None:
                    break
                future = pool.submit(_group, index, output, plan, plan_hash, group, reserve_bytes)
                active[future] = group['group_id']
        fill()
        while active:
            finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in finished:
                active.pop(future)
                counts[future.result()['phase']] += 1
            atomic_json(output / 'state' / f'worker-{os.getpid()}.json', {'time': time.time(), 'pid': os.getpid(),
                'counts': dict(counts), 'active_groups': list(active.values()), 'workers': workers})
            fill()
    return {'groups': len(plan['groups']), 'counts': dict(counts), 'stopped': (output / 'STOP').exists()}


def finalize(index, output):
    import pyarrow.parquet as pq
    index, output, plan, documents = _context(index, output)
    plan_hash = file_sha(output / 'PLAN.json')
    with claim(output / 'state/finalize.lock', blocking=True):
        _space(output, MIN_RESERVE_BYTES)
        catalog, receipts, totals, covered = [], [], Counter(), set()
        for group in plan['groups']:
            value = _receipt(output, plan_hash, group, verify=True)
            if value is None:
                raise PendingGroups('Compaction groups remain incomplete')
            expected = {item['path']: item for item in group['inputs']}
            observed = {item['path']: item for item in value['verified_inputs']}
            require(len(observed) == len(value['verified_inputs']) and set(observed) == set(expected), 'Input checksum coverage mismatch')
            for path, item in observed.items():
                require(path not in covered and item['sha256'] == expected[path]['sha256']
                        and _stamp(bound(index, path)) == {'bytes': item['bytes'], 'mtime_ns': item['mtime_ns']},
                        'Input file changed or was covered more than once')
                covered.add(path)
            entry = value['catalog']; source_rows = Counter(); rows = 0
            parquet = pq.ParquetFile(bound(output, entry['path']))
            for batch in parquet.iter_batches(batch_size=8192, columns=['collection', 'source', 'source_shard'], use_threads=False):
                columns = _columns(batch)
                require(set(columns['collection']) == {group['collection']}
                        and set(columns['source']).issubset(set(group['sources']))
                        and set(columns['source_shard']) == {group['source_shard']}, 'Compacted rows escape source/TAR membership')
                source_rows.update(columns['source']); rows += batch.num_rows
            require(rows == group['rows'] and dict(source_rows) == value['source_rows'], 'Compacted source rows differ')
            for source, count in source_rows.items():
                totals[(group['collection'], source)] += count
            catalog.append(entry)
            marker = output / 'groups' / group['group_id'] / 'COMPLETE.json'
            receipts.append({'path': str(marker.relative_to(output)), 'sha256': file_sha(marker)})
        expected_totals = {(row['collection'], row['source']): row['rows'] for row in plan['per_source']}
        require(dict(totals) == expected_totals and sum(totals.values()) == plan['rows']
                and len(covered) == plan['input_files'], 'Full compaction input/source/row coverage mismatch')
        for name, binding in plan['inputs'].items():
            require(file_sha(output / 'lineage' / name) == binding['sha256'], 'Source lineage copy changed')
        source_manifest = documents['manifest.json']
        statistics = dict(source_manifest['statistics'], input_metadata_files=plan['input_files'],
                          compacted_metadata_files=len(catalog))
        lineage = {'contract': 'complete_pinned_native_index.v1',
            'artifacts': {name: {'path': 'lineage/' + name, **value} for name, value in plan['inputs'].items()},
            'input_files': plan['input_files'], 'input_files_covered_once': len(covered),
            'per_source_rows': plan['per_source'], 'raw_source_validation': source_manifest['source_validation']}
        manifest = {'schema_version': NATIVE, 'phase': 'complete', 'scope': 'full_declared_inputs',
            'enumeration_scope': 'derived_full_pinned_source_metadata', 'catalog_contract': CONTRACT,
            'collections': sorted({entry['collection'] for entry in catalog}), 'statistics': statistics,
            'portable_datasets': source_manifest.get('portable_datasets', []),
            'allowed_image_roots': source_manifest.get('allowed_image_roots', []),
            'source_validation': source_manifest['source_validation'],
            'source_inventory_sha256': source_manifest['source_inventory_sha256'],
            'source_lineage': lineage, 'plan_sha256': plan_hash, 'unit_receipts': receipts,
            'source_files_modified': False, 'original_row_fields_modified': False, 'image_payloads_copied': False}
        marker = output / 'COMPLETE.json'
        if marker.exists():
            complete = read(marker)
            require(complete.get('schema_version') == NATIVE and complete.get('complete') is True
                    and complete.get('validation_passed') is True and complete.get('scope') == 'full_declared_inputs'
                    and complete.get('catalog_contract') == CONTRACT and complete.get('statistics') == statistics
                    and read(output / 'manifest.json') == manifest and read(output / 'catalog.json') == catalog
                    and complete.get('manifest_sha256') == file_sha(output / 'manifest.json')
                    and complete.get('catalog_sha256') == file_sha(output / 'catalog.json')
                    and complete.get('plan_sha256') == plan_hash, 'Published compacted snapshot changed')
            return complete
        atomic_json(output / 'catalog.json', catalog)
        atomic_json(output / 'manifest.json', manifest)
        complete = {'schema_version': NATIVE, 'complete': True, 'validation_passed': True,
            'scope': 'full_declared_inputs', 'catalog_contract': CONTRACT, 'statistics': statistics,
            'validation_scope': 'all_input_checksums_exact_once_rows_per_source_and_compacted_checksums',
            'plan_sha256': plan_hash, 'catalog_sha256': file_sha(output / 'catalog.json'),
            'manifest_sha256': file_sha(output / 'manifest.json')}
        atomic_json(marker, complete)
        return complete


def run(index, output, *, workers=8, max_group_rows=65536, batch_rows=256, reserve_bytes=MIN_RESERVE_BYTES):
    prepare(index, output, max_group_rows=max_group_rows, batch_rows=batch_rows)
    worker(index, output, workers=workers, reserve_bytes=reserve_bytes)
    return finalize(index, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'worker', 'finalize', 'run'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-group-rows', type=int, default=65536)
    parser.add_argument('--batch-rows', type=int, default=256)
    parser.add_argument('--reserve-tib', type=float, default=1)
    parser.add_argument('--max-groups', type=int)
    args = parser.parse_args(argv)
    try:
        if args.action == 'prepare':
            result = prepare(args.input, args.output, max_group_rows=args.max_group_rows, batch_rows=args.batch_rows)
        elif args.action == 'worker':
            result = worker(args.input, args.output, workers=args.workers, reserve_bytes=int(args.reserve_tib * MIN_RESERVE_BYTES), max_groups=args.max_groups)
        elif args.action == 'finalize':
            result = finalize(args.input, args.output)
        else:
            result = run(args.input, args.output, workers=args.workers, max_group_rows=args.max_group_rows,
                         batch_rows=args.batch_rows, reserve_bytes=int(args.reserve_tib * MIN_RESERVE_BYTES))
        print(json.dumps(result, sort_keys=True))
        return 0
    except PendingGroups as error:
        print(json.dumps({'complete': False, 'error': 'waiting_for_compaction', 'reason': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
